#!/usr/bin/env python3
"""Preprocessing check on real HF data — no T5, no training.

Runs the full preprocessing pipeline (download, convert, validate) then
writes visualizations for each episode so you can inspect the results.

Run from repo root:
    model/.venv/bin/python scripts/test_preprocessing.py [--out-dir /tmp/viz]
"""
import argparse
import shutil
import sys
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.so101_pipeline_for_vision_training import (
    compute_trim_bounds,
    convert_dataset,
    download_repo_files,
    dump_yaml,
    get_force,
    load_yaml,
    normalize_preprocessing,
    read_episodes,
    require_mapping,
    resolve_commit_sha,
    resolve_model_commit_sha,
    select_revision,
    validate_conversion,
)

# Dataset sampling constants (must match data_video.py and preprocessing target_fps)
DATA_FPS = 10.0       # stored FPS after preprocessing (target_fps: 10)
EFFECTIVE_FPS = 10.0  # effective training FPS (data_video.py effective_fps)
STEP = int(DATA_FPS / EFFECTIVE_FPS)  # 1 — every frame
NUM_FRAMES = 25
OBS_HISTORY = 5


def run_preprocessing(config: dict, n_episodes: int | None = None, seed: int = 42) -> Path:
    raw = require_mapping(config, "preprocessing")
    source = require_mapping(raw, "source")
    te = raw.get("text_encoder") or {}
    repo_id = source["repo_id"]
    revision = source.get("revision", "main")

    print(f"Resolving HF commit for {repo_id}@{revision} ...")
    commit_sha = resolve_commit_sha(repo_id, revision)
    te_sha = resolve_model_commit_sha(
        te.get("model_id", "google-t5/t5-large"), te.get("revision", "main")
    )
    preprocessing = normalize_preprocessing(
        config, commit_sha=commit_sha, text_encoder_commit_sha=te_sha
    )

    if n_episodes is not None:
        dataset_root = download_repo_files(repo_id, revision)
        all_eps = read_episodes(dataset_root)["episode_index"].tolist()
        already_excluded = set(preprocessing.get("exclude_episodes") or [])
        available = [e for e in all_eps if e not in already_excluded]
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(available, size=min(n_episodes, len(available)), replace=False).tolist())
        preprocessing["exclude_episodes"] = sorted(set(all_eps) - keep)
        print(f"Test mode: keeping {len(keep)} episodes {sorted(keep)}")

    revision_id, revision_dir, should_preprocess = select_revision(
        preprocessing, force=get_force(config)
    )
    print(f"Revision dir: {revision_dir}")

    if should_preprocess:
        if revision_dir.exists():
            shutil.rmtree(revision_dir)
        revision_dir.mkdir(parents=True)
        convert_dataset(revision_dir, preprocessing)
        validate_conversion(revision_dir)
        dump_yaml(revision_dir / "preprocessing.yaml", preprocessing)
        print("Preprocessing complete (T5 skipped).")
    else:
        validate_conversion(revision_dir)
        print("Cache hit — reusing existing preprocessed data.")

    return revision_dir


def load_episode(revision_dir: Path, episode_idx: int):
    stem = f"episode_{episode_idx:06d}"
    video_path = revision_dir / "video_finetune" / "video" / f"{stem}.mp4"
    zarr_path = revision_dir / "action_decoder" / f"{stem}.zarr"

    frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    frames = np.stack(frames)  # [T, H, W, 3]

    root = zarr.open(str(zarr_path), mode="r")
    joint_action = root["joint_action_lowdim"][...]   # [T, 6] at 30fps
    joint_state = root["joint_state_lowdim"][...]     # [T, 6] at 30fps
    return frames, joint_action, joint_state


def sample_clip(frames: np.ndarray, seed: int = 42) -> tuple[np.ndarray, int]:
    n = len(frames)
    rng = np.random.default_rng(seed)
    i = int(rng.integers(0, n + STEP * (OBS_HISTORY - 1)))
    k = np.arange(NUM_FRAMES, dtype=np.float64) - (OBS_HISTORY - 1)
    idx = np.clip(np.rint(i + k * STEP).astype(np.int64), 0, n - 1)
    return frames[idx], i


def write_mp4(frames: np.ndarray, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for arr in frames:
            f = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for pkt in stream.encode(f):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)


def save_trim_visualization(
    joint_state: np.ndarray,
    path: Path,
    trim_cfg: dict | None = None,
) -> None:
    """Two-panel plot: joint state velocity + baseline distance with trim boundary."""
    # joint_state is at 30fps (source FPS), DATA_FPS is the stored video FPS
    src_fps = 30.0
    n = len(joint_state)
    t = np.arange(n) / src_fps

    trim_cfg = trim_cfg or {}
    threshold = float(trim_cfg.get("threshold", 0.5))
    grace = int(trim_cfg.get("grace_frames", 15))
    ref = int(trim_cfg.get("ref_frames", 15))
    trim_enabled = bool(trim_cfg.get("enabled", False))
    do_trim_start = bool(trim_cfg.get("trim_start", True))
    do_trim_end = bool(trim_cfg.get("trim_end", False))

    ref_clamped = min(ref, max(1, n // 4))
    rest_start = joint_state[:ref_clamped].mean(axis=0)
    dist_start = np.linalg.norm(joint_state - rest_start, axis=1)

    vel = np.linalg.norm(np.diff(joint_state, axis=0), axis=1)
    t_vel = np.arange(len(vel)) / src_fps

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # --- top: joint state velocity ---
    ax = axes[0]
    ax.plot(t_vel, vel, color="steelblue", linewidth=0.8, label="state Δ L2 norm")
    ax.set_ylabel("joint state velocity")
    ax.set_title(f"Joint state velocity  ({n} action frames at 30fps)")
    ax.legend(loc="upper right", fontsize=8)

    # --- bottom: distance from start baseline ---
    ax = axes[1]
    ax.plot(t, dist_start, color="dodgerblue", linewidth=0.8, label="dist from start baseline (obs.state)")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.0, label=f"threshold={threshold}")

    noise_window = dist_start[ref_clamped : 2 * ref_clamped]
    noise_floor = float(noise_window.mean()) if len(noise_window) > 0 else 0.0
    noise_p95 = float(np.percentile(noise_window, 95)) if len(noise_window) > 0 else 0.0
    ax.axhline(
        noise_floor,
        color="gray",
        linestyle=":",
        linewidth=0.8,
        label=f"noise floor mean={noise_floor:.3f}  p95={noise_p95:.3f}",
    )

    if trim_enabled:
        trim_start_idx, trim_end_idx = compute_trim_bounds(
            joint_state, threshold, grace, ref,
            trim_start=do_trim_start, trim_end=do_trim_end,
        )
        kept = trim_end_idx - trim_start_idx
        removed = n - kept

        t_start = trim_start_idx / src_fps
        t_end = (trim_end_idx - 1) / src_fps

        if do_trim_start:
            ax.axvline(t_start, color="green", linewidth=1.2, label=f"trim start (frame {trim_start_idx})")
            ax.axvspan(0, t_start, alpha=0.12, color="red", label=f"trimmed start ({trim_start_idx} frames)")
        if do_trim_end:
            ax.axvline(t_end, color="green", linewidth=1.2, linestyle="--", label=f"trim end (frame {trim_end_idx - 1})")
            ax.axvspan(t_end, t[-1], alpha=0.12, color="orange", label=f"trimmed end ({n - trim_end_idx} frames)")

        ax.set_title(
            f"Start-baseline distance  |  threshold={threshold}  grace={grace}  "
            f"kept {kept}/{n} action frames  (removed {removed})"
        )
    else:
        ax.set_title(f"Start-baseline distance  (trim disabled — threshold={threshold} for reference)")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("L2 distance from start rest baseline")
    ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def visualize_episode(revision_dir: Path, episode_idx: int, out_dir: Path) -> None:
    out = out_dir / f"episode_{episode_idx:06d}"
    out.mkdir(parents=True, exist_ok=True)

    # load trim config from stored preprocessing.yaml if present
    trim_cfg: dict = {}
    stored_yaml = revision_dir / "preprocessing.yaml"
    if stored_yaml.exists():
        stored = load_yaml(stored_yaml)
        trim_cfg = stored.get("trim") or {}

    frames, joint_action, joint_state = load_episode(revision_dir, episode_idx)
    n_video = len(frames)
    n_action = len(joint_action)
    duration = n_video / DATA_FPS
    print(f"  episode {episode_idx}: {n_video} video frames = {duration:.1f}s  |  {n_action} action frames at 30fps")

    if trim_cfg.get("enabled", False):
        threshold = float(trim_cfg["threshold"])
        grace = int(trim_cfg["grace_frames"])
        ref = int(trim_cfg["ref_frames"])
        do_trim_start = bool(trim_cfg.get("trim_start", True))
        do_trim_end = bool(trim_cfg.get("trim_end", False))
        ref_clamped = min(ref, max(1, n_action // 4))

        rest_start = joint_state[:ref_clamped].mean(axis=0)
        noise_window = np.linalg.norm(joint_state[ref_clamped : 2 * ref_clamped] - rest_start, axis=1)
        noise_floor = float(noise_window.mean()) if len(noise_window) > 0 else 0.0
        noise_p95 = float(np.percentile(noise_window, 95)) if len(noise_window) > 0 else 0.0

        trim_start_idx, trim_end_idx = compute_trim_bounds(
            joint_state, threshold, grace, ref,
            trim_start=do_trim_start, trim_end=do_trim_end,
        )
        kept = trim_end_idx - trim_start_idx
        removed = n_action - kept
        print(f"    noise floor: mean={noise_floor:.3f}  p95={noise_p95:.3f}  threshold={threshold}")
        if removed > 0:
            print(f"    WARNING: stored data still has {removed} trimmable action frames — recheck threshold")
        else:
            print(f"    trim check OK: no further trimming needed (boundaries at [{trim_start_idx}:{trim_end_idx}])")

    # full episode at stored fps
    write_mp4(frames, out / "full_10fps.mp4", fps=DATA_FPS)

    # one dataloader clip
    clip, start_i = sample_clip(frames)
    src_start = max(0, start_i - (OBS_HISTORY - 1) * STEP)
    src_end = min(n_video - 1, start_i + (NUM_FRAMES - OBS_HISTORY) * STEP)
    print(f"    clip: source [{src_start/DATA_FPS:.2f}s..{src_end/DATA_FPS:.2f}s]")
    write_mp4(clip, out / "dataloader_clip_10fps.mp4", fps=EFFECTIVE_FPS)

    # trim visualization plot (uses joint_state at 30fps)
    save_trim_visualization(joint_state, out / "trim_visualization.png", trim_cfg=trim_cfg)

    print(f"    -> {out}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/so101/video_finetune.yaml"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/viz"))
    parser.add_argument(
        "--episodes", type=int, nargs="+", default=None,
        help="Episode indices to visualize (default: 5 random)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-episodes", type=int, default=3,
        help="Number of random episodes to preprocess (default: 5, 0 = all)"
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    n_ep = args.n_episodes if args.n_episodes > 0 else None
    revision_dir = run_preprocessing(config, n_episodes=n_ep, seed=args.seed)

    zarr_paths = sorted((revision_dir / "action_decoder").glob("*.zarr"))
    all_episodes = [int(p.stem.split("_")[1]) for p in zarr_paths]
    if args.episodes is not None:
        to_viz = args.episodes
    else:
        rng = np.random.default_rng(args.seed)
        to_viz = rng.choice(all_episodes, size=min(5, len(all_episodes)), replace=False).tolist()

    print(f"\nVisualizing {len(to_viz)} episode(s) → {args.out_dir}")
    for ep in to_viz:
        visualize_episode(revision_dir, ep, args.out_dir)

    print("\nOutputs per episode:")
    print("  full_30fps.mp4            — what preprocessing stored (post-trim)")
    print("  full_10fps.mp4            — at model's effective sampling rate")
    print("  dataloader_clip_10fps.mp4 — one 61-frame clip as model sees it (6.1s)")
    print("  trim_visualization.png    — baseline distances, threshold, trim boundaries")


if __name__ == "__main__":
    main()
