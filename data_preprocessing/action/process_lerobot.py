import argparse
import pathlib
from functools import partial
from multiprocessing import Pool

import decord
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zarr
from numcodecs import Blosc
import tqdm

NS_PER_SEC = 1_000_000_000


def _load_episodes_meta(dataset_dir: pathlib.Path) -> pd.DataFrame:
    files = sorted((dataset_dir / "meta" / "episodes").glob("**/file-*.parquet"))
    keep = [
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
        "videos/observation.images.topdown/chunk_index",
        "videos/observation.images.topdown/file_index",
        "videos/observation.images.topdown/from_timestamp",
        "videos/observation.images.topdown/to_timestamp",
        "videos/observation.images.left_wrist/chunk_index",
        "videos/observation.images.left_wrist/file_index",
        "videos/observation.images.left_wrist/from_timestamp",
        "videos/observation.images.left_wrist/to_timestamp",
        "videos/observation.images.right_wrist/chunk_index",
        "videos/observation.images.right_wrist/file_index",
        "videos/observation.images.right_wrist/from_timestamp",
        "videos/observation.images.right_wrist/to_timestamp",
    ]
    dfs = [pq.read_table(f, columns=keep).to_pandas() for f in files]
    return pd.concat(dfs).reset_index(drop=True).sort_values("episode_index").reset_index(drop=True)


def _load_tasks(dataset_dir: pathlib.Path) -> dict[int, str]:
    df = pq.read_table(dataset_dir / "meta" / "tasks.parquet").to_pandas()
    # task is in index, task_index is column
    return {int(row.task_index): str(task) for task, row in df.iterrows()}


def _decode_segment(video_path: pathlib.Path, from_ts: float, length: int, fps: int) -> tuple[np.ndarray, int]:
    """Decode a video segment, truncating to actual video length if the request runs past EOF.

    Returns (frames, effective_length) where effective_length <= length.
    """
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
    start = int(round(from_ts * fps))
    end = start + length
    if end > len(vr):
        # HF fs26: some later episodes claim frames past EOF — truncate.
        end = len(vr)
    effective = end - start
    if effective <= 0:
        raise ValueError(f"video {video_path.name}: requested start frame {start} >= total {len(vr)}")
    idx = list(range(start, end))
    return vr.get_batch(idx).asnumpy(), effective  # (T, H, W, 3) uint8, T


def _convert_safe(ep_row, **kwargs):
    """Wrap _convert to catch decord/codec failures so the pool doesn't die."""
    try:
        return _convert(ep_row, **kwargs)
    except Exception as e:
        ep_idx = int(ep_row.get("episode_index", -1))
        return f"FAIL ep {ep_idx}: {type(e).__name__}: {str(e)[:200]}"


def _convert(
    ep_row: pd.Series,
    *,
    dataset_dir: pathlib.Path,
    out_dir: pathlib.Path,
    fps: int,
    overwrite: bool,
    tasks: dict[int, str],
) -> str:
    # multi-task: episodes.parquet stores `tasks` as a list of task strings
    # (one per task this episode covers — typically just 1)
    ep_tasks = ep_row["tasks"]
    if isinstance(ep_tasks, str):
        ep_task_strs = [ep_tasks]
    elif hasattr(ep_tasks, "__iter__"):
        ep_task_strs = [str(t) for t in ep_tasks]
    else:
        ep_task_strs = [str(ep_tasks)]
    if len(ep_task_strs) != 1:
        msg = f"episode {ep_row['episode_index']}: expected exactly 1 task per episode, got {ep_task_strs}"
        raise ValueError(msg)
    task_text = ep_task_strs[0]
    # sanity check: the task string should be one of the known tasks
    known_task_strs = set(tasks.values())
    if task_text not in known_task_strs:
        msg = f"episode {ep_row['episode_index']}: task {task_text!r} not in known tasks {known_task_strs}"
        raise ValueError(msg)
    ep_idx = int(ep_row["episode_index"])
    out_path = out_dir / f"episode_{ep_idx:04d}.zarr"
    if out_path.exists() and not overwrite:
        return f"skip (exists): {out_path}"

    length = int(ep_row["length"])
    data_chunk = int(ep_row["data/chunk_index"])
    data_file = int(ep_row["data/file_index"])
    parquet_path = dataset_dir / "data" / f"chunk-{data_chunk:03d}" / f"file-{data_file:03d}.parquet"
    table = pq.read_table(parquet_path)

    # rows for this episode are contiguous; locate by exact frame_index window
    df = table.to_pandas()
    mask = df["episode_index"] == ep_idx
    sub = df.loc[mask].sort_values("frame_index").reset_index(drop=True)
    if len(sub) != length:
        # HF fs26 dataset has occasional mismatch between parquet row count and
        # meta length. Truncate both action/state and video segment to min.
        effective = min(len(sub), length)
        print(f"WARN episode {ep_idx}: parquet has {len(sub)} rows but meta says length={length}; truncating to {effective}")
        sub = sub.head(effective).reset_index(drop=True)
        length = effective

    # state/action are fixed_size_list[14]; pyarrow->pandas yields ndarray-of-lists
    state = np.stack(sub["observation.state"].to_numpy()).astype(np.float32)
    assert state.shape == (length, 14), state.shape
    action = np.stack(sub["action"].to_numpy()).astype(np.float32)
    assert action.shape == (length, 14), action.shape

    # videos. Train ingestion currently uses only topdown/workspace_rgb.
    cams = {
        "workspace_rgb": "observation.images.topdown",
    }
    decoded = {}
    effective_lengths = []
    for out_key, cam_key in cams.items():
        chunk = int(ep_row[f"videos/{cam_key}/chunk_index"])
        fid = int(ep_row[f"videos/{cam_key}/file_index"])
        from_ts = float(ep_row[f"videos/{cam_key}/from_timestamp"])
        vp = dataset_dir / "videos" / cam_key / f"chunk-{chunk:03d}" / f"file-{fid:03d}.mp4"
        decoded[out_key], eff = _decode_segment(vp, from_ts, length, fps)
        effective_lengths.append(eff)

    # If any camera was truncated (video ran past EOF), align all cams + state to the min.
    min_eff = min(effective_lengths)
    if min_eff < length:
        print(f"WARN episode {ep_idx}: video truncation, length {length} -> {min_eff}")
        for out_key in list(decoded.keys()):
            decoded[out_key] = decoded[out_key][:min_eff]
        state = state[:min_eff]
        action = action[:min_eff]
        sub = sub.head(min_eff).reset_index(drop=True)
        length = min_eff

    for out_key, arr in decoded.items():
        if arr.shape != (length, 480, 640, 3):
            msg = f"episode {ep_idx} {out_key}: shape {arr.shape}, expected ({length},480,640,3)"
            raise ValueError(msg)

    # synthesize timestamps from frame_index for tight monotonicity
    dt_ns = NS_PER_SEC / fps
    timestamps = (sub["frame_index"].to_numpy().astype(np.uint64) * 0) + (
        np.arange(length, dtype=np.uint64) * np.uint64(dt_ns)
    )
    if not np.all(np.diff(timestamps) > 0):
        msg = f"episode {ep_idx}: timestamps not strictly increasing"
        raise ValueError(msg)

    comp = Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)
    t_img = min(65, length)
    t_ld = min(1024, length)

    root = zarr.open(str(out_path), mode="w")

    # language (one constant string for this dataset; embedding added by precompute_t5.py)
    root.create_dataset(
        "language_instruction",
        shape=(1,), dtype=bytes, chunks=(1,), compressor=comp, overwrite=True,
    )[...] = np.array([task_text.encode()])
    root.create_dataset(
        "language_instruction_timestamps",
        shape=(1,), dtype="uint64", chunks=(1,), compressor=comp, overwrite=True,
    )[...] = np.array([0], dtype=np.uint64)

    # images
    for out_key, arr in decoded.items():
        root.create_dataset(
            out_key, shape=arr.shape, dtype=np.uint8,
            chunks=(t_img, *arr.shape[1:]), compressor=comp,
        )[...] = arr
        root.create_dataset(
            f"{out_key}_timestamps", shape=(length,), dtype="uint64", chunks=(length,),
        )[...] = timestamps

    # 14-dim bimanual joints and actions
    root.create_dataset(
        "joint_state_lowdim", shape=state.shape, dtype=np.float32,
        chunks=(t_ld, state.shape[1]), compressor=comp,
    )[...] = state
    root.create_dataset(
        "joint_state_lowdim_timestamps", shape=(length,), dtype="uint64", chunks=(length,),
    )[...] = timestamps
    root.create_dataset(
        "joint_action_lowdim", shape=action.shape, dtype=np.float32,
        chunks=(t_ld, action.shape[1]), compressor=comp,
    )[...] = action
    root.create_dataset(
        "joint_action_lowdim_timestamps", shape=(length,), dtype="uint64", chunks=(length,),
    )[...] = timestamps

    return f"ok: ep {ep_idx:04d} -> {out_path.name} (T={length})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", type=pathlib.Path, required=True,
                    help="lerobot v3 dataset root (contains data/, videos/, meta/)")
    ap.add_argument("--output-dir", type=pathlib.Path, required=True,
                    help="folder to write per-episode .zarr files into")
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="optional subset of episode indices to process (default: all)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ep_df = _load_episodes_meta(args.dataset_path)
    tasks = _load_tasks(args.dataset_path)
    print(f"loaded {len(tasks)} task(s): {tasks}")

    if args.episodes is not None:
        ep_df = ep_df[ep_df["episode_index"].isin(args.episodes)].reset_index(drop=True)

    rows = [r for _, r in ep_df.iterrows()]
    fn = partial(
        _convert_safe,
        dataset_dir=args.dataset_path,
        out_dir=args.output_dir,
        fps=args.fps,
        overwrite=args.overwrite,
        tasks=tasks,
    )
    if args.num_workers <= 1:
        for r in tqdm.tqdm(rows, desc="bi_yams -> zarr"):
            print(fn(r))
    else:
        with Pool(processes=args.num_workers) as pool:
            for msg in tqdm.tqdm(pool.imap_unordered(fn, rows), total=len(rows), desc="bi_yams -> zarr"):
                print(msg)


if __name__ == "__main__":
    main()
