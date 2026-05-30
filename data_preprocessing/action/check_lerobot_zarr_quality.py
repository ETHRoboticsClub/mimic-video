import argparse
import json
import pathlib
from dataclasses import dataclass

import decord
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zarr


CAMS = {
    "workspace_rgb": "observation.images.topdown",
}

ZARR_LENGTH_KEYS = [
    "workspace_rgb",
    "workspace_rgb_timestamps",
    "joint_state_lowdim",
    "joint_state_lowdim_timestamps",
    "joint_action_lowdim",
    "joint_action_lowdim_timestamps",
]


@dataclass
class EpisodeCheck:
    episode_index: int
    ok: bool
    expected_length: int | None
    zarr_length: int | None
    messages: list[str]


def _load_info(dataset_dir: pathlib.Path) -> dict:
    with (dataset_dir / "meta" / "info.json").open() as f:
        return json.load(f)


def _load_episodes_meta(dataset_dir: pathlib.Path) -> pd.DataFrame:
    files = sorted((dataset_dir / "meta" / "episodes").glob("**/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata parquet files found under {dataset_dir / 'meta' / 'episodes'}")
    return (
        pd.concat([pq.read_table(path).to_pandas() for path in files])
        .reset_index(drop=True)
        .sort_values("episode_index")
        .reset_index(drop=True)
    )


def _source_rows(dataset_dir: pathlib.Path, ep_row: pd.Series) -> pd.DataFrame:
    ep_idx = int(ep_row["episode_index"])
    chunk = int(ep_row["data/chunk_index"])
    file_idx = int(ep_row["data/file_index"])
    parquet_path = dataset_dir / "data" / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.parquet"
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    return df.loc[df["episode_index"] == ep_idx].sort_values("frame_index").reset_index(drop=True)


def _video_effective_length(dataset_dir: pathlib.Path, ep_row: pd.Series, cam_key: str, length: int, fps: int) -> int:
    chunk = int(ep_row[f"videos/{cam_key}/chunk_index"])
    file_idx = int(ep_row[f"videos/{cam_key}/file_index"])
    from_ts = float(ep_row[f"videos/{cam_key}/from_timestamp"])
    video_path = dataset_dir / "videos" / cam_key / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.mp4"
    reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
    start = int(round(from_ts * fps))
    end = min(start + length, len(reader))
    return max(0, end - start)


def _video_frame_window(dataset_dir: pathlib.Path, ep_row: pd.Series, cam_key: str, length: int, fps: int) -> tuple[pathlib.Path, int, int]:
    chunk = int(ep_row[f"videos/{cam_key}/chunk_index"])
    file_idx = int(ep_row[f"videos/{cam_key}/file_index"])
    from_ts = float(ep_row[f"videos/{cam_key}/from_timestamp"])
    video_path = dataset_dir / "videos" / cam_key / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.mp4"
    reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
    start = int(round(from_ts * fps))
    end = min(start + length, len(reader))
    if end <= start:
        raise ValueError(f"{video_path}: empty frame window start={start} end={end} total={len(reader)}")
    return video_path, start, end


def _sample_indices(length: int, samples: int | None) -> list[int]:
    if samples is None or samples >= length:
        return list(range(length))
    if samples <= 1:
        return [0]
    return sorted(set(int(round(i)) for i in np.linspace(0, length - 1, samples)))


def _compare_video_frames(
    dataset_dir: pathlib.Path,
    ep_row: pd.Series,
    root: zarr.Group,
    out_key: str,
    cam_key: str,
    length: int,
    fps: int,
    samples: int | None,
    tolerance: int,
    messages: list[str],
) -> None:
    video_path, start, end = _video_frame_window(dataset_dir, ep_row, cam_key, length, fps)
    available = end - start
    if available != length:
        messages.append(f"{out_key} source frame window length {available} != zarr length {length}")
        return

    local_indices = _sample_indices(length, samples)
    source_indices = [start + idx for idx in local_indices]
    reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
    source = reader.get_batch(source_indices).asnumpy()
    written = root[out_key].get_orthogonal_selection((local_indices, slice(None), slice(None), slice(None)))

    if source.shape != written.shape:
        messages.append(f"{out_key} sampled source shape {source.shape} != zarr shape {written.shape}")
        return

    delta = np.abs(source.astype(np.int16) - written.astype(np.int16))
    max_delta = int(delta.max()) if delta.size else 0
    mismatched = int(np.count_nonzero(delta > tolerance))
    if mismatched:
        first_bad = int(local_indices[int(np.argmax(delta.reshape(delta.shape[0], -1).max(axis=1) > tolerance))])
        messages.append(
            f"{out_key} frame content mismatch against {video_path.name}: "
            f"samples={len(local_indices)} max_delta={max_delta} pixels_over_tolerance={mismatched} "
            f"first_bad_local_frame={first_bad} source_frame={start + first_bad}"
        )


def _first_dim(root: zarr.Group, key: str) -> int:
    if key not in root:
        raise KeyError(f"missing zarr array {key!r}")
    shape = root[key].shape
    if not shape:
        raise ValueError(f"zarr array {key!r} is scalar")
    return int(shape[0])


def _check_timestamps(root: zarr.Group, key: str, expected_length: int, messages: list[str]) -> None:
    values = root[key][:]
    if len(values) != expected_length:
        messages.append(f"{key} length {len(values)} != expected {expected_length}")
        return
    if len(values) > 1 and not np.all(np.diff(values) > 0):
        messages.append(f"{key} is not strictly increasing")


def check_episode(
    dataset_dir: pathlib.Path,
    output_dir: pathlib.Path,
    ep_row: pd.Series,
    fps: int,
    frame_samples: int | None,
    frame_tolerance: int,
) -> EpisodeCheck:
    ep_idx = int(ep_row["episode_index"])
    messages: list[str] = []

    meta_length = int(ep_row["length"])
    source = _source_rows(dataset_dir, ep_row)
    source_length = min(meta_length, len(source))
    if len(source) != meta_length:
        messages.append(f"source parquet rows {len(source)} != meta length {meta_length}; converter uses {source_length}")

    for column in ("action", "observation.state"):
        if column not in source:
            messages.append(f"source parquet missing {column!r}")
            continue
        if len(source[column]) != len(source):
            messages.append(f"source {column} length {len(source[column])} != source rows {len(source)}")
        if len(source):
            width = len(source[column].iloc[0])
            if width != 14:
                messages.append(f"source {column} width {width} != 14")

    video_lengths = {
        out_key: _video_effective_length(dataset_dir, ep_row, cam_key, source_length, fps)
        for out_key, cam_key in CAMS.items()
    }
    expected_length = min([source_length, *video_lengths.values()])
    for out_key, video_length in video_lengths.items():
        if video_length != expected_length:
            messages.append(f"source video {out_key} effective length {video_length} != expected {expected_length}")

    zarr_path = output_dir / f"episode_{ep_idx:04d}.zarr"
    if not zarr_path.exists():
        return EpisodeCheck(ep_idx, False, expected_length, None, [*messages, f"missing {zarr_path}"])

    root = zarr.open(str(zarr_path), mode="r")
    zarr_lengths: dict[str, int] = {}
    for key in ZARR_LENGTH_KEYS:
        try:
            zarr_lengths[key] = _first_dim(root, key)
        except Exception as exc:
            messages.append(str(exc))

    unique_lengths = sorted(set(zarr_lengths.values()))
    if len(unique_lengths) != 1:
        messages.append(f"zarr arrays have inconsistent first dimensions: {zarr_lengths}")
        zarr_length = None
    else:
        zarr_length = unique_lengths[0]
        if zarr_length != expected_length:
            messages.append(f"zarr length {zarr_length} != expected source/action/video length {expected_length}")

    for image_key in CAMS:
        if image_key in root and root[image_key].shape[1:] != (480, 640, 3):
            messages.append(f"{image_key} shape {root[image_key].shape} has unexpected image dimensions")
    if "joint_state_lowdim" in root and root["joint_state_lowdim"].shape[1:] != (14,):
        messages.append(f"joint_state_lowdim shape {root['joint_state_lowdim'].shape} does not have 14 joints")
    if "joint_action_lowdim" in root and root["joint_action_lowdim"].shape[1:] != (14,):
        messages.append(f"joint_action_lowdim shape {root['joint_action_lowdim'].shape} does not have 14 joints")

    if zarr_length is not None:
        for key in (
            "workspace_rgb_timestamps",
            "joint_state_lowdim_timestamps",
            "joint_action_lowdim_timestamps",
        ):
            if key in root:
                _check_timestamps(root, key, zarr_length, messages)

        for out_key, cam_key in CAMS.items():
            if out_key in root:
                _compare_video_frames(
                    dataset_dir,
                    ep_row,
                    root,
                    out_key,
                    cam_key,
                    zarr_length,
                    fps,
                    frame_samples,
                    frame_tolerance,
                    messages,
                )

    return EpisodeCheck(ep_idx, not messages, expected_length, zarr_length, messages)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument(
        "--frame-samples",
        type=int,
        default=None,
        help="number of frames per camera/episode to compare; default compares every frame",
    )
    parser.add_argument(
        "--frame-tolerance",
        type=int,
        default=0,
        help="allowed absolute per-channel pixel difference when comparing source MP4 frames to Zarr frames",
    )
    args = parser.parse_args()

    info = _load_info(args.dataset_path)
    fps = args.fps or int(info["fps"])
    ep_df = _load_episodes_meta(args.dataset_path)
    if args.episodes is not None:
        ep_df = ep_df[ep_df["episode_index"].isin(args.episodes)].reset_index(drop=True)

    checks = [
        check_episode(args.dataset_path, args.output_dir, row, fps, args.frame_samples, args.frame_tolerance)
        for _, row in ep_df.iterrows()
    ]
    failures = [check for check in checks if not check.ok]

    print(f"checked episodes: {len(checks)}")
    print(f"passed: {len(checks) - len(failures)}")
    print(f"failed: {len(failures)}")

    for check in checks:
        status = "OK" if check.ok else "FAIL"
        print(
            f"{status} ep {check.episode_index:04d}: "
            f"expected={check.expected_length} zarr={check.zarr_length}"
        )
        for message in check.messages:
            print(f"  - {message}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
