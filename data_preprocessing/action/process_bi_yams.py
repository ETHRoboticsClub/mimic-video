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


def _decode_segment(video_path: pathlib.Path, from_ts: float, length: int, fps: int) -> np.ndarray:
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    start = int(round(from_ts * fps))
    idx = list(range(start, start + length))
    if idx[-1] >= len(vr):
        msg = f"requested frame {idx[-1]} but video {video_path.name} has only {len(vr)} frames"
        raise ValueError(msg)
    return vr.get_batch(idx).asnumpy()  # (T, H, W, 3) uint8


def _convert(
    ep_row: pd.Series,
    *,
    dataset_dir: pathlib.Path,
    out_dir: pathlib.Path,
    fps: int,
    overwrite: bool,
    task_text: str,
) -> str:
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
        msg = f"episode {ep_idx}: parquet has {len(sub)} rows but meta says length={length}"
        raise ValueError(msg)

    # action / state are fixed_size_list[14]; pyarrow→pandas yields ndarray-of-lists
    action = np.stack(sub["action"].to_numpy()).astype(np.float32)
    state = np.stack(sub["observation.state"].to_numpy()).astype(np.float32)
    assert action.shape == (length, 14), action.shape
    assert state.shape == (length, 14), state.shape

    # videos
    cams = {
        "workspace_rgb": "observation.images.topdown",
        "wrist_rgb_left": "observation.images.left_wrist",
        "wrist_rgb_right": "observation.images.right_wrist",
    }
    decoded = {}
    for out_key, cam_key in cams.items():
        chunk = int(ep_row[f"videos/{cam_key}/chunk_index"])
        fid = int(ep_row[f"videos/{cam_key}/file_index"])
        from_ts = float(ep_row[f"videos/{cam_key}/from_timestamp"])
        vp = dataset_dir / "videos" / cam_key / f"chunk-{chunk:03d}" / f"file-{fid:03d}.mp4"
        decoded[out_key] = _decode_segment(vp, from_ts, length, fps)
        if decoded[out_key].shape != (length, 480, 640, 3):
            msg = f"episode {ep_idx} {out_key}: shape {decoded[out_key].shape}, expected ({length},480,640,3)"
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

    # 14-dim joints
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
    ap.add_argument("--dataset-dir", type=pathlib.Path, required=True,
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

    ep_df = _load_episodes_meta(args.dataset_dir)
    tasks = _load_tasks(args.dataset_dir)
    # for this dataset there is exactly one task; assert that
    if len(tasks) != 1:
        msg = f"expected 1 task, got {len(tasks)}: {tasks}"
        raise ValueError(msg)
    task_text = next(iter(tasks.values()))

    if args.episodes is not None:
        ep_df = ep_df[ep_df["episode_index"].isin(args.episodes)].reset_index(drop=True)

    rows = [r for _, r in ep_df.iterrows()]
    fn = partial(
        _convert,
        dataset_dir=args.dataset_dir,
        out_dir=args.output_dir,
        fps=args.fps,
        overwrite=args.overwrite,
        task_text=task_text,
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
