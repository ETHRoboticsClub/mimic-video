#!/usr/bin/env python3
"""End-to-end test for the so101 preprocessor.

Given a training yaml, runs the preprocessor (idempotent — reuses the cache if
the revision dir already exists for this PIPELINE_VERSION), then loads one
preprocessed episode from disk and verifies:

  A. The output mp4 is encoded at SO101_FPS (= 10).
  B. The zarr's per-stream timestamps:
       - workspace_rgb_timestamps spacing == 1/SO101_FPS  (camera at 10 Hz)
       - joint_action_lowdim_timestamps spacing == 1/source_fps
         (actions kept at the source dataset's native rate)
       - joint_state_lowdim_timestamps same as joint_action
  C. Sample counts agree: workspace_rgb.shape[0] == len(workspace_rgb_timestamps),
     joint_state_lowdim.shape[0] == len(joint_state_lowdim_timestamps), etc.
  D. Cross-stream length ratio matches the resampling factor:
       len(lowdim_ts) / len(video_ts) ≈ source_fps / SO101_FPS

Usage:
    python scripts/test_so101_pipeline_preprocessing.py \
        --config configs/so101/video_finetune_subsampled_dual_task_low_rank.yaml

Optional:
    --stem source000_episode_000007   # pick a specific episode
    --tolerance-ns 5                  # ns tolerance for timestamp diffs
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr
from decord import VideoReader, cpu

# Make the repo importable when called directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.so101_pipeline import (  # noqa: E402
    SO101_FPS,
    download_repo_files,
    load_yaml,
    normalized_preprocessing_sources,
    preprocess_if_needed,
    read_info,
)


def _stem_to_source_idx(stem: str) -> int | None:
    """Multi-source mixes prefix stems with `sourceNNN_`. Return NNN or None."""
    if stem.startswith("source") and "_" in stem:
        token = stem.split("_", 1)[0]
        digits = token[len("source") :]
        if digits.isdigit():
            return int(digits)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="training yaml under configs/so101/")
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Episode stem to verify (e.g. source000_episode_000007). Default: first mp4 in video/ alphabetically.",
    )
    parser.add_argument(
        "--tolerance-ns",
        type=int,
        default=5,
        help="Tolerance in ns when comparing timestamp diffs to 1/fps (default 5).",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    revision_dir = preprocess_if_needed(config)
    print(f"\n=== validating revision dir: {revision_dir} ===\n")

    # Look up the source data rate per source repo (multi-source mixes index by sourceNNN_ prefix).
    sources = normalized_preprocessing_sources(config["preprocessing"])
    source_fps_by_idx: dict[int, int] = {}
    for idx, source in enumerate(sources):
        repo_id = source["repo_id"]
        revision = source.get("revision", "main")
        dataset_root = download_repo_files(repo_id, revision)
        info = read_info(dataset_root, source["video_feature"])
        source_fps_by_idx[idx] = int(info["fps"])
        print(f"  source[{idx}] {repo_id}@{revision}: info.json fps = {info['fps']}, _subsample_step = {info['_subsample_step']}")

    # Pick an episode.
    video_dir = revision_dir / "video_finetune" / "video"
    zarr_dir = revision_dir / "action_decoder"
    mp4_files = sorted(video_dir.glob("*.mp4"))
    if not mp4_files:
        raise SystemExit(f"No mp4s found under {video_dir}")
    stem = args.stem or mp4_files[0].stem
    mp4_path = video_dir / f"{stem}.mp4"
    zarr_path = zarr_dir / f"{stem}.zarr"
    if not mp4_path.exists():
        raise SystemExit(f"mp4 not found: {mp4_path}")
    if not zarr_path.exists():
        raise SystemExit(f"zarr not found: {zarr_path}")

    src_idx = _stem_to_source_idx(stem)
    if src_idx is not None and src_idx in source_fps_by_idx:
        source_fps = source_fps_by_idx[src_idx]
    elif len(source_fps_by_idx) == 1:
        source_fps = next(iter(source_fps_by_idx.values()))
    else:
        raise SystemExit(
            f"Cannot infer source fps for stem {stem!r}: stem has no sourceNNN_ prefix and "
            "multiple sources are present. Pass --stem explicitly."
        )

    print(f"\n  episode stem      : {stem}")
    print(f"  source fps        : {source_fps}")
    print(f"  expected video    : {SO101_FPS} fps")
    print(f"  expected lowdim   : {source_fps} fps  (kept at source rate)")

    # ---- A. mp4 fps ----
    print("\nA. mp4 fps")
    vr = VideoReader(str(mp4_path), ctx=cpu(0), num_threads=0)
    mp4_fps = float(vr.get_avg_fps())
    n_mp4_frames = len(vr)
    print(f"   {mp4_path.name}: {mp4_fps:.3f} fps, {n_mp4_frames} frames")
    if abs(mp4_fps - SO101_FPS) > 0.5:
        raise SystemExit(f"FAIL: mp4 fps {mp4_fps} differs from SO101_FPS={SO101_FPS} by more than 0.5")
    print("   ✓")

    # ---- B. zarr per-stream timestamps ----
    print("\nB. zarr per-stream timestamp spacing")
    root = zarr.open(str(zarr_path), mode="r")
    video_ts = root["workspace_rgb_timestamps"][:]
    lowdim_action_ts = root["joint_action_lowdim_timestamps"][:]
    lowdim_state_ts = root["joint_state_lowdim_timestamps"][:]

    expected_video_dt_ns = 1_000_000_000 // SO101_FPS
    expected_lowdim_dt_ns = 1_000_000_000 // source_fps

    def _check_dt(name: str, ts: np.ndarray, expected_dt_ns: int) -> None:
        if len(ts) < 2:
            print(f"   {name}: only {len(ts)} sample — skipping spacing check")
            return
        diffs = np.diff(ts)
        if not np.all(diffs > 0):
            raise SystemExit(f"FAIL: {name} not monotonic")
        max_dev = int(np.max(np.abs(diffs - expected_dt_ns)))
        mean_dt = int(diffs.mean())
        print(f"   {name}: {len(ts)} samples, mean dt = {mean_dt} ns, expected {expected_dt_ns} ns, max |Δ| = {max_dev} ns")
        if max_dev > args.tolerance_ns:
            raise SystemExit(
                f"FAIL: {name} timestamp spacing deviates from {expected_dt_ns} ns by {max_dev} ns "
                f"(> tolerance {args.tolerance_ns})"
            )

    _check_dt("workspace_rgb_timestamps", video_ts, expected_video_dt_ns)
    _check_dt("joint_action_lowdim_timestamps", lowdim_action_ts, expected_lowdim_dt_ns)
    _check_dt("joint_state_lowdim_timestamps", lowdim_state_ts, expected_lowdim_dt_ns)
    print("   ✓")

    # ---- C. shape consistency ----
    print("\nC. shape consistency (sample count agrees with timestamps)")
    workspace_rgb = root["workspace_rgb"]
    joint_state = root["joint_state_lowdim"]
    joint_action = root["joint_action_lowdim"]
    pairs = [
        ("workspace_rgb", workspace_rgb.shape[0], len(video_ts)),
        ("joint_state_lowdim", joint_state.shape[0], len(lowdim_state_ts)),
        ("joint_action_lowdim", joint_action.shape[0], len(lowdim_action_ts)),
    ]
    for name, data_len, ts_len in pairs:
        print(f"   {name}: data={data_len}, timestamps={ts_len}")
        if data_len != ts_len:
            raise SystemExit(f"FAIL: {name} has {data_len} samples but {ts_len} timestamps")
    if n_mp4_frames != workspace_rgb.shape[0]:
        raise SystemExit(
            f"FAIL: mp4 has {n_mp4_frames} frames but zarr workspace_rgb has {workspace_rgb.shape[0]}"
        )
    print("   ✓")

    # ---- D. cross-stream length ratio ----
    print("\nD. lowdim / video length ratio")
    expected_ratio = source_fps / SO101_FPS
    actual_ratio = len(lowdim_action_ts) / len(video_ts)
    print(f"   actual {actual_ratio:.3f}  vs expected {expected_ratio:.3f} (= source_fps {source_fps} / SO101_FPS {SO101_FPS})")
    if abs(actual_ratio - expected_ratio) > 0.05:
        raise SystemExit(
            f"FAIL: lowdim/video ratio {actual_ratio:.3f} differs from expected {expected_ratio:.3f} by > 0.05"
        )
    print("   ✓")

    # ---- E. data preview: timestamps + raw values for eyeballing ----
    print("\nE. data preview (first samples; helpful for eyeballing alignment & raw scale)")
    head_n = 6
    step = max(1, int(round(source_fps / SO101_FPS)))

    print(f"\n   video timestamps (s): {(video_ts[:head_n] / 1e9).tolist()}")
    print(f"   first {head_n} mp4 frame indices, expected: {list(range(head_n))}")

    print(f"\n   lowdim timestamps (s): {(lowdim_action_ts[:head_n * step] / 1e9).tolist()}")
    print(f"   ↑ at source rate {source_fps} Hz; every {step}-th sample should match a video timestamp")

    aligned = np.allclose(
        lowdim_action_ts[::step][: len(video_ts)],
        video_ts,
        atol=args.tolerance_ns,
    )
    print(f"   lowdim[::{step}] == video_ts: {aligned}")
    if not aligned:
        raise SystemExit("FAIL: every Nth lowdim timestamp should match a video timestamp")

    np.set_printoptions(precision=4, suppress=True, linewidth=140)
    print(f"\n   joint_state_lowdim[:{head_n}] (raw, units = source dataset's units):")
    for i, row in enumerate(joint_state[:head_n]):
        print(f"     t={lowdim_state_ts[i] / 1e9:6.4f}s  state = {np.asarray(row)}")

    print(f"\n   joint_action_lowdim[:{head_n}] (raw, units = source dataset's units):")
    for i, row in enumerate(joint_action[:head_n]):
        print(f"     t={lowdim_action_ts[i] / 1e9:6.4f}s  action = {np.asarray(row)}")

    state_min, state_max = float(np.asarray(joint_state[:]).min()), float(np.asarray(joint_state[:]).max())
    action_min, action_max = float(np.asarray(joint_action[:]).min()), float(np.asarray(joint_action[:]).max())
    print(f"\n   joint_state range : [{state_min:8.4f}, {state_max:8.4f}]")
    print(f"   joint_action range: [{action_min:8.4f}, {action_max:8.4f}]")
    print(
        "   ↑ raw, NOT normalized. Normalization is applied at MimicDataset load time via the\n"
        "     `.statistics_cache/<stats_id>.json` file in `<dataset_dir>/action_decoder/`,\n"
        "     not pre-baked into the zarrs. So values you see here are in the source dataset's\n"
        "     native units (LeRobot SO101 convention: absolute joint positions, deg or rad,\n"
        "     matching `info.json.features.action.names`)."
    )

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
