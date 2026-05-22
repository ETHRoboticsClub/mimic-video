#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
import zarr
from huggingface_hub import HfApi, hf_hub_download
from numcodecs import Blosc
from PIL import Image, ImageDraw
from tqdm import tqdm

NS_PER_SEC = 1_000_000_000
PIPELINE_VERSION = 3


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def model_python() -> Path:
    return repo_root() / "model" / ".venv" / "bin" / "python"


def model_torchrun() -> Path:
    return repo_root() / "model" / ".venv" / "bin" / "torchrun"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=True)


def require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing required mapping: {key}")
    return value


def normalize_preprocessing(
    config: dict[str, Any],
    *,
    commit_sha: str,
    text_encoder_commit_sha: str,
) -> dict[str, Any]:
    preprocessing = require_mapping(config, "preprocessing")
    source = require_mapping(preprocessing, "source")
    image = require_mapping(preprocessing, "image")
    contact_sheet = preprocessing.get("contact_sheet") or {}
    if not isinstance(contact_sheet, dict):
        raise ValueError("preprocessing.contact_sheet must be a mapping.")
    text_encoder = preprocessing.get("text_encoder") or {}
    if not isinstance(text_encoder, dict):
        raise ValueError("preprocessing.text_encoder must be a mapping.")

    repo_id = source.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError("preprocessing.source.repo_id must be a non-empty string.")

    revision = source.get("revision", "main")
    if not isinstance(revision, str) or not revision:
        raise ValueError("preprocessing.source.revision must be a non-empty string.")

    instruction = preprocessing.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("preprocessing.instruction must be a non-empty string.")

    video_feature = preprocessing.get("video_feature")
    if not isinstance(video_feature, str) or not video_feature:
        raise ValueError("preprocessing.video_feature must be a non-empty string.")

    exclude_episodes = preprocessing.get("exclude_episodes", [])
    if exclude_episodes is None:
        exclude_episodes = []
    if not isinstance(exclude_episodes, list) or not all(isinstance(x, int) for x in exclude_episodes):
        raise ValueError("preprocessing.exclude_episodes must be a list of integer episode ids.")

    max_episodes_raw = preprocessing.get("max_episodes")
    if max_episodes_raw is not None:
        if not isinstance(max_episodes_raw, int) or max_episodes_raw <= 0:
            raise ValueError("preprocessing.max_episodes must be a positive integer.")
    max_episodes: int | None = max_episodes_raw

    crop_xywh = image.get("crop_xywh")
    if not isinstance(crop_xywh, list) or len(crop_xywh) != 4 or not all(isinstance(x, int) for x in crop_xywh):
        raise ValueError("preprocessing.image.crop_xywh must be four integers: [x, y, width, height].")
    if crop_xywh[2] <= 0 or crop_xywh[3] <= 0:
        raise ValueError("preprocessing.image.crop_xywh width and height must be positive.")

    output_size_hw = image.get("output_size_hw", [480, 640])
    if (
        not isinstance(output_size_hw, list)
        or len(output_size_hw) != 2
        or not all(isinstance(x, int) for x in output_size_hw)
    ):
        raise ValueError("preprocessing.image.output_size_hw must be two integers: [height, width].")
    if output_size_hw[0] <= 0 or output_size_hw[1] <= 0:
        raise ValueError("preprocessing.image.output_size_hw values must be positive.")

    text_encoder_model_id = text_encoder.get("model_id", "google-t5/t5-large")
    if not isinstance(text_encoder_model_id, str) or not text_encoder_model_id:
        raise ValueError("preprocessing.text_encoder.model_id must be a non-empty string.")
    text_encoder_revision = text_encoder.get("revision", "main")
    if not isinstance(text_encoder_revision, str) or not text_encoder_revision:
        raise ValueError("preprocessing.text_encoder.revision must be a non-empty string.")

    video_section = preprocessing.get("video") or {}
    if not isinstance(video_section, dict):
        raise ValueError("preprocessing.video must be a mapping.")
    target_fps_raw = video_section.get("target_fps")
    if target_fps_raw is not None:
        if not isinstance(target_fps_raw, int) or target_fps_raw <= 0:
            raise ValueError("preprocessing.video.target_fps must be a positive integer.")
    target_fps: int | None = target_fps_raw

    mask_regions_raw = image.get("mask_regions") or []
    if not isinstance(mask_regions_raw, list):
        raise ValueError("preprocessing.image.mask_regions must be a list.")
    for region in mask_regions_raw:
        if not isinstance(region, list) or len(region) != 4 or not all(isinstance(x, int) for x in region):
            raise ValueError("Each mask region must be four integers: [x, y, width, height].")
        if region[2] <= 0 or region[3] <= 0:
            raise ValueError("Mask region width and height must be positive.")
    mask_regions: list[list[int]] = [list(r) for r in mask_regions_raw]

    letterbox: bool = bool(image.get("letterbox", False))

    trim_section = preprocessing.get("trim") or {}
    if not isinstance(trim_section, dict):
        raise ValueError("preprocessing.trim must be a mapping.")
    trim_enabled = bool(trim_section.get("enabled", False))
    trim_threshold = float(trim_section.get("threshold", 0.5))
    trim_grace_frames = int(trim_section.get("grace_frames", 15))
    trim_ref_frames = int(trim_section.get("ref_frames", 15))
    trim_min_frames = int(trim_section.get("min_frames", 20))
    trim_start_enabled = bool(trim_section.get("trim_start", True))
    trim_end_enabled = bool(trim_section.get("trim_end", False))
    if trim_threshold <= 0:
        raise ValueError("preprocessing.trim.threshold must be positive.")
    if trim_grace_frames < 0:
        raise ValueError("preprocessing.trim.grace_frames must be non-negative.")
    if trim_ref_frames < 1:
        raise ValueError("preprocessing.trim.ref_frames must be at least 1.")
    if trim_min_frames < 1:
        raise ValueError("preprocessing.trim.min_frames must be at least 1.")

    return {
        "pipeline": {"name": "so101_pipeline", "version": PIPELINE_VERSION},
        "source": {"repo_id": repo_id, "revision": revision, "commit_sha": commit_sha},
        "text_encoder": {
            "model_id": text_encoder_model_id,
            "revision": text_encoder_revision,
            "commit_sha": text_encoder_commit_sha,
        },
        "instruction": instruction.strip(),
        "video_feature": video_feature,
        "exclude_episodes": sorted(set(exclude_episodes)),
        "max_episodes": max_episodes,
        "image": {"crop_xywh": crop_xywh, "output_size_hw": output_size_hw, "mask_regions": mask_regions, "letterbox": letterbox},
        "video": {"target_fps": target_fps},
        "contact_sheet": {"enabled": bool(contact_sheet.get("enabled", False))},
        "trim": {
            "enabled": trim_enabled,
            "threshold": trim_threshold,
            "grace_frames": trim_grace_frames,
            "ref_frames": trim_ref_frames,
            "min_frames": trim_min_frames,
            "trim_start": trim_start_enabled,
            "trim_end": trim_end_enabled,
        },
    }


def get_force(config: dict[str, Any]) -> bool:
    return bool(require_mapping(config, "preprocessing").get("force", False))


def compute_trim_bounds(
    joint_state: np.ndarray,
    threshold: float,
    grace_frames: int,
    ref_frames: int,
    trim_start: bool = True,
    trim_end: bool = False,
) -> tuple[int, int]:
    """Return (start, end) slice indices to keep after trimming idle frames.

    Uses joint_state (observation.state) for detection — smoother signal than action.
    Computes rest baseline from the first ref_frames, finds the first frame exceeding
    threshold L2 distance from that baseline, then pads by grace_frames.
    """
    n = len(joint_state)
    ref = min(ref_frames, max(1, n // 4))

    start = 0
    end = n

    if trim_start:
        rest_start = joint_state[:ref].mean(axis=0)
        dist_from_start = np.linalg.norm(joint_state - rest_start, axis=1)
        active_from_start = np.where(dist_from_start >= threshold)[0]
        first_active = int(active_from_start[0]) if len(active_from_start) > 0 else n
        start = max(0, first_active - grace_frames)

    if trim_end:
        rest_end = joint_state[-ref:].mean(axis=0)
        dist_from_end = np.linalg.norm(joint_state - rest_end, axis=1)
        active_from_end = np.where(dist_from_end >= threshold)[0]
        last_active = int(active_from_end[-1]) if len(active_from_end) > 0 else 0
        end = min(n, last_active + grace_frames + 1)

    return start, end


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def resolve_commit_sha(repo_id: str, revision: str) -> str:
    api = HfApi(token=hf_token())
    info = api.repo_info(repo_id=repo_id, repo_type="dataset", revision=revision)
    if not info.sha:
        raise ValueError(f"Could not resolve a commit SHA for {repo_id}@{revision}.")
    return info.sha


def resolve_model_commit_sha(model_id: str, revision: str) -> str:
    api = HfApi(token=hf_token())
    info = api.repo_info(repo_id=model_id, repo_type="model", revision=revision)
    if not info.sha:
        raise ValueError(f"Could not resolve a commit SHA for {model_id}@{revision}.")
    return info.sha


def revision_base(preprocessing: dict[str, Any]) -> Path:
    return repo_root() / "data" / preprocessing["source"]["repo_id"]


def load_cached_preprocessing(path: Path) -> dict[str, Any] | None:
    cache_path = path / "preprocessing.yaml"
    if not cache_path.exists():
        return None
    return load_yaml(cache_path)


def cached_commit_sha(repo_id: str, revision: str) -> str | None:
    base = repo_root() / "data" / repo_id
    if not base.exists():
        return None
    for path in sorted((p for p in base.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
        cached = load_cached_preprocessing(path)
        if not cached:
            continue
        source = cached.get("source", {})
        if source.get("repo_id") == repo_id and source.get("revision") == revision:
            commit_sha = source.get("commit_sha")
            if isinstance(commit_sha, str) and commit_sha:
                return commit_sha
    return None


def select_revision(preprocessing: dict[str, Any], *, force: bool) -> tuple[int, Path, bool]:
    base = revision_base(preprocessing)
    base.mkdir(parents=True, exist_ok=True)

    existing = sorted(int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit())
    for revision in existing:
        path = base / str(revision)
        if load_cached_preprocessing(path) == preprocessing:
            return revision, path, force

    next_revision = existing[-1] + 1 if existing else 1
    return next_revision, base / str(next_revision), True


def download_repo_files(repo_id: str, revision: str) -> Path:
    token = hf_token()
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    wanted = [
        f
        for f in files
        if f == "README.md"
        or f == "meta/info.json"
        or f == "meta/stats.json"
        or f == "meta/tasks.parquet"
        or f.startswith("meta/episodes/")
        or f.startswith("data/")
        or f.startswith("videos/")
    ]
    root = None
    for filename in wanted:
        path = Path(hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision, token=token))
        root = path.parents[len(Path(filename).parts) - 1]
    if root is None:
        raise ValueError(f"No usable files found in {repo_id}@{revision}.")
    return root


def read_info(dataset_root: Path, video_feature: str) -> dict[str, Any]:
    with (dataset_root / "meta" / "info.json").open() as f:
        info = json.load(f)
    features = info.get("features", {})
    if "action" not in features or "observation.state" not in features:
        raise ValueError("Expected LeRobot features 'action' and 'observation.state'.")
    if features["action"].get("shape") != [6] or features["observation.state"].get("shape") != [6]:
        raise ValueError("SO101 pipeline expects 6D action and 6D observation.state.")
    video_keys = [key for key, spec in features.items() if spec.get("dtype") == "video"]
    if video_feature not in video_keys:
        raise ValueError(f"preprocessing.video_feature must be one of {video_keys}; got {video_feature!r}.")
    return info | {"_video_key": video_feature}


def read_episodes(dataset_root: Path) -> pd.DataFrame:
    paths = sorted((dataset_root / "meta" / "episodes").glob("**/*.parquet"))
    if not paths:
        raise ValueError("No episode metadata parquet files found.")
    return pd.concat([pq.read_table(path).to_pandas() for path in paths], ignore_index=True).sort_values(
        "episode_index"
    )


def read_tasks(dataset_root: Path) -> dict[int, str]:
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return {}
    df = pq.read_table(tasks_path).to_pandas().reset_index()
    return {int(row["task_index"]): str(row["task"]) for _, row in df.iterrows()}


def read_data_file(dataset_root: Path, file_index: int) -> pd.DataFrame:
    path = dataset_root / "data" / "chunk-000" / f"file-{file_index:03d}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pandas()


def validate_crop(crop_xywh: list[int], source_hw: tuple[int, int]) -> None:
    x, y, w, h = crop_xywh
    source_h, source_w = source_hw
    if x < 0 or y < 0 or x + w > source_w or y + h > source_h:
        raise ValueError(f"Crop {crop_xywh} is outside source image size {(source_h, source_w)}.")


def process_frame(
    frame: av.VideoFrame,
    crop_xywh: list[int],
    output_size_hw: list[int],
    mask_regions: list[list[int]] | None = None,
    letterbox: bool = False,
) -> np.ndarray:
    x, y, w, h = crop_xywh
    out_h, out_w = output_size_hw
    img = frame.to_image().convert("RGB").crop((x, y, x + w, y + h))
    if letterbox:
        scale = min(out_w / w, out_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (out_w, out_h), (0, 0, 0))
        canvas.paste(img, ((out_w - new_w) // 2, (out_h - new_h) // 2))
        img = canvas
    else:
        img = img.resize((out_w, out_h), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8).copy()
    if mask_regions:
        for mx, my, mw, mh in mask_regions:
            arr[my : my + mh, mx : mx + mw] = 0
    return arr


def read_episode_frames(
    video_path: Path,
    *,
    start_frame: int,
    local_frame_indices: np.ndarray,
    crop_xywh: list[int],
    output_size_hw: list[int],
    mask_regions: list[list[int]] | None = None,
    letterbox: bool = False,
) -> np.ndarray:
    requested = {int(start_frame + idx): i for i, idx in enumerate(local_frame_indices)}
    if not requested:
        raise ValueError("No requested frames.")
    max_requested = max(requested)
    frames: np.ndarray | None = None
    found = np.zeros(len(requested), dtype=bool)

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        validate_crop(crop_xywh, (stream.height, stream.width))
        for frame_idx, frame in enumerate(container.decode(video=0)):
            if frame_idx in requested:
                if frames is None:
                    out_h, out_w = output_size_hw
                    frames = np.empty((len(requested), out_h, out_w, 3), dtype=np.uint8)
                frames[requested[frame_idx]] = process_frame(frame, crop_xywh, output_size_hw, mask_regions, letterbox)
                found[requested[frame_idx]] = True
            if frame_idx >= max_requested:
                break

    if frames is None or not found.all():
        raise ValueError(f"Failed to decode all requested frames from {video_path}.")
    return frames


def write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    codecs = ("libx264", "mpeg4")
    last_error: Exception | None = None
    for codec in codecs:
        try:
            with av.open(str(path), mode="w") as container:
                stream = container.add_stream(codec, rate=fps)
                stream.width = int(frames.shape[2])
                stream.height = int(frames.shape[1])
                stream.pix_fmt = "yuv420p"
                for arr in frames:
                    frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            return
        except Exception as exc:  # pragma: no cover - depends on system codecs
            last_error = exc
            if path.exists():
                path.unlink()
    raise RuntimeError(f"Could not write {path} with available codecs.") from last_error


def write_zarr_episode(
    path: Path,
    *,
    frames: np.ndarray,
    video_timestamps_ns: np.ndarray,
    joint_state: np.ndarray,
    joint_action: np.ndarray,
    action_timestamps_ns: np.ndarray,
    instruction: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    comp = Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)
    t_img = min(65, len(video_timestamps_ns))
    t_ld = min(1024, len(action_timestamps_ns))
    root = zarr.open(str(path), mode="w")

    root.create_dataset("workspace_rgb", shape=frames.shape, dtype=np.uint8, chunks=(t_img, *frames.shape[1:]), compressor=comp)[
        ...
    ] = frames
    root.create_dataset("workspace_rgb_timestamps", shape=video_timestamps_ns.shape, dtype="uint64", chunks=video_timestamps_ns.shape)[
        ...
    ] = video_timestamps_ns

    root.create_dataset(
        "joint_state_lowdim",
        shape=joint_state.shape,
        dtype=np.float32,
        chunks=(t_ld, *joint_state.shape[1:]),
        compressor=comp,
    )[...] = joint_state
    root.create_dataset("joint_state_lowdim_timestamps", shape=action_timestamps_ns.shape, dtype="uint64", chunks=action_timestamps_ns.shape)[
        ...
    ] = action_timestamps_ns

    root.create_dataset(
        "joint_action_lowdim",
        shape=joint_action.shape,
        dtype=np.float32,
        chunks=(t_ld, *joint_action.shape[1:]),
        compressor=comp,
    )[...] = joint_action
    root.create_dataset("joint_action_lowdim_timestamps", shape=action_timestamps_ns.shape, dtype="uint64", chunks=action_timestamps_ns.shape)[
        ...
    ] = action_timestamps_ns

    root.create_dataset("language_instruction", shape=(1,), dtype=bytes, chunks=(1,), compressor=comp)[...] = np.array(
        [instruction.encode("utf-8")]
    )
    root.create_dataset("language_instruction_timestamps", shape=(1,), dtype="uint64", chunks=(1,), compressor=comp)[
        ...
    ] = np.array([0], dtype=np.uint64)


def make_contact_sheet(
    path: Path,
    dataset_root: Path,
    episodes: pd.DataFrame,
    *,
    video_feature: str,
    crop_xywh: list[int],
    output_size_hw: list[int],
    fps: int,
    mask_regions: list[list[int]] | None = None,
    letterbox: bool = False,
) -> None:
    thumbs: list[Image.Image] = []
    for _, episode in tqdm(
        episodes.iterrows(),
        total=len(episodes),
        desc="Building contact sheet",
        unit="episode",
    ):
        video_file_idx = int(episode[f"videos/{video_feature}/file_index"])
        start_frame = round(float(episode[f"videos/{video_feature}/from_timestamp"]) * fps)
        length = int(episode["length"])
        for frac in (0.0, 0.5, 0.95):
            local_idx = int(round((length - 1) * frac))
            video_path = dataset_root / "videos" / video_feature / "chunk-000" / f"file-{video_file_idx:03d}.mp4"
            frame_arr = read_episode_frames(
                video_path,
                start_frame=start_frame,
                local_frame_indices=np.array([local_idx], dtype=np.int64),
                crop_xywh=crop_xywh,
                output_size_hw=output_size_hw,
                mask_regions=mask_regions,
                letterbox=letterbox,
            )[0]
            img = Image.fromarray(frame_arr).resize((320, 240))
            draw = ImageDraw.Draw(img)
            label = f"ep{int(episode['episode_index'])} {frac:.2f}"
            draw.rectangle((0, 0, 120, 22), fill=(255, 255, 255))
            draw.text((4, 4), label, fill=(0, 0, 0))
            thumbs.append(img)

    if not thumbs:
        return
    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 320, rows * 240), (255, 255, 255))
    for idx, img in enumerate(thumbs):
        sheet.paste(img, ((idx % cols) * 320, (idx // cols) * 240))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def convert_dataset(revision_dir: Path, preprocessing: dict[str, Any]) -> None:
    source = preprocessing["source"]
    print(
        f"Preparing SO101 cache from {source['repo_id']}@{source['revision']} "
        f"({source['commit_sha']}).",
        flush=True,
    )
    dataset_root = download_repo_files(source["repo_id"], source["revision"])
    print(f"HF dataset files are available under {dataset_root}.", flush=True)
    video_feature = preprocessing["video_feature"]
    info = read_info(dataset_root, video_feature)
    episodes = read_episodes(dataset_root)
    read_tasks(dataset_root)

    excluded = set(preprocessing["exclude_episodes"])
    selected = episodes[~episodes["episode_index"].isin(excluded)].copy()
    if selected.empty:
        raise ValueError("All episodes were excluded.")
    max_episodes: int | None = preprocessing.get("max_episodes")
    if max_episodes is not None:
        selected = selected.head(max_episodes)
    print(
        f"Converting {len(selected)} episodes with video feature {video_feature!r}; "
        f"excluded episodes: {sorted(excluded)}"
        + (f"; capped at {max_episodes}" if max_episodes is not None else "") + ".",
        flush=True,
    )

    fps = int(info["fps"])
    crop_xywh = preprocessing["image"]["crop_xywh"]
    output_size_hw = preprocessing["image"]["output_size_hw"]
    mask_regions = preprocessing["image"].get("mask_regions") or []
    letterbox = bool(preprocessing["image"].get("letterbox", False))
    if output_size_hw != [480, 640]:
        raise ValueError("The current Cosmos training path requires preprocessing.image.output_size_hw: [480, 640].")

    target_fps: int | None = preprocessing.get("video", {}).get("target_fps")
    if target_fps is not None and target_fps > fps:
        raise ValueError(f"preprocessing.video.target_fps ({target_fps}) exceeds source FPS ({fps}).")
    out_fps = target_fps if target_fps is not None else fps

    if preprocessing["contact_sheet"]["enabled"]:
        print("Writing contact sheet.", flush=True)
        make_contact_sheet(
            revision_dir / "contact_sheet.jpg",
            dataset_root,
            selected,
            video_feature=video_feature,
            crop_xywh=crop_xywh,
            output_size_hw=output_size_hw,
            fps=fps,
            mask_regions=mask_regions,
            letterbox=letterbox,
        )

    data_cache: dict[int, pd.DataFrame] = {}
    video_dir = revision_dir / "video_finetune"
    action_dir = revision_dir / "action_decoder"
    (video_dir / "video").mkdir(parents=True, exist_ok=True)
    (video_dir / "metas").mkdir(parents=True, exist_ok=True)
    action_dir.mkdir(parents=True, exist_ok=True)

    for _, episode in tqdm(
        selected.iterrows(),
        total=len(selected),
        desc="Converting episodes",
        unit="episode",
    ):
        episode_idx = int(episode["episode_index"])
        data_file_idx = int(episode["data/file_index"])
        video_file_idx = int(episode[f"videos/{video_feature}/file_index"])
        data_file = data_cache.setdefault(data_file_idx, read_data_file(dataset_root, data_file_idx))
        rows = data_file[data_file["episode_index"] == episode_idx].sort_values("frame_index")
        if rows.empty:
            raise ValueError(f"No rows for episode {episode_idx}.")

        # Step 1: trim on full source FPS (30fps) — before any stride
        trim_cfg = preprocessing.get("trim", {})
        if trim_cfg.get("enabled", False):
            joint_state_for_trim = np.stack(rows["observation.state"].to_numpy()).astype(np.float32)
            trim_start, trim_end = compute_trim_bounds(
                joint_state_for_trim,
                threshold=float(trim_cfg["threshold"]),
                grace_frames=int(trim_cfg["grace_frames"]),
                ref_frames=int(trim_cfg["ref_frames"]),
                trim_start=bool(trim_cfg.get("trim_start", True)),
                trim_end=bool(trim_cfg.get("trim_end", False)),
            )
            original_len = len(rows)
            rows = rows.iloc[trim_start:trim_end].reset_index(drop=True)
            trimmed = original_len - len(rows)
            if trimmed > 0:
                print(
                    f"  episode {episode_idx}: trimmed {trimmed} idle frames "
                    f"({original_len} → {len(rows)}, kept [{trim_start}:{trim_end}])",
                    flush=True,
                )
            if len(rows) < int(trim_cfg.get("min_frames", 20)):
                print(
                    f"  episode {episode_idx}: skipped — only {len(rows)} frames after trim "
                    f"(min_frames={trim_cfg.get('min_frames', 20)})",
                    flush=True,
                )
                continue

        # Step 2: actions at full source FPS (trimmed, no stride)
        action_timestamps_ns = np.rint(rows["timestamp"].to_numpy(dtype=np.float64) * NS_PER_SEC).astype(np.uint64)
        joint_state = np.stack(rows["observation.state"].to_numpy()).astype(np.float32)
        joint_action = np.stack(rows["action"].to_numpy()).astype(np.float32)

        # Step 3: video at target_fps — stride the trimmed rows for frame selection only
        if target_fps is not None and target_fps < fps:
            stride = round(fps / target_fps)
            rows_video = rows.iloc[::stride].reset_index(drop=True)
        else:
            rows_video = rows
        video_timestamps_ns = np.rint(rows_video["timestamp"].to_numpy(dtype=np.float64) * NS_PER_SEC).astype(np.uint64)

        local_frame_indices = rows_video["frame_index"].to_numpy(dtype=np.int64)
        start_frame = round(float(episode[f"videos/{video_feature}/from_timestamp"]) * fps)
        video_path = dataset_root / "videos" / video_feature / "chunk-000" / f"file-{video_file_idx:03d}.mp4"
        frames = read_episode_frames(
            video_path,
            start_frame=start_frame,
            local_frame_indices=local_frame_indices,
            crop_xywh=crop_xywh,
            output_size_hw=output_size_hw,
            mask_regions=mask_regions,
            letterbox=letterbox,
        )

        stem = f"episode_{episode_idx:06d}"
        write_video(video_dir / "video" / f"{stem}.mp4", frames, fps=out_fps)
        (video_dir / "metas" / f"{stem}.txt").write_text(preprocessing["instruction"] + "\n")
        write_zarr_episode(
            action_dir / f"{stem}.zarr",
            frames=frames,
            video_timestamps_ns=video_timestamps_ns,
            joint_state=joint_state,
            joint_action=joint_action,
            action_timestamps_ns=action_timestamps_ns,
            instruction=preprocessing["instruction"],
        )
    print(f"Finished writing converted data to {revision_dir}.", flush=True)


def validate_conversion(revision_dir: Path) -> None:
    print(f"Validating converted cache at {revision_dir}.", flush=True)
    action_paths = sorted((revision_dir / "action_decoder").glob("*.zarr"))
    if not action_paths:
        raise ValueError("No zarr episodes were written.")
    for path in action_paths:
        root = zarr.open(str(path), mode="r")
        n_video = len(root["workspace_rgb"])
        n_action = len(root["joint_action_lowdim"])
        if n_video == 0:
            raise ValueError(f"No video frames in {path}.")
        if n_action == 0:
            raise ValueError(f"No action frames in {path}.")
        if len(root["workspace_rgb_timestamps"]) != n_video:
            raise ValueError(f"workspace_rgb/timestamps length mismatch in {path}.")
        if len(root["joint_state_lowdim"]) != n_action:
            raise ValueError(f"joint_state_lowdim length {len(root['joint_state_lowdim'])} != action length {n_action} in {path}.")
        if len(root["joint_state_lowdim_timestamps"]) != n_action:
            raise ValueError(f"joint_state_lowdim/timestamps length mismatch in {path}.")
        if len(root["joint_action_lowdim_timestamps"]) != n_action:
            raise ValueError(f"joint_action_lowdim/timestamps length mismatch in {path}.")
        if root["workspace_rgb"].shape[1:] != (480, 640, 3):
            raise ValueError(f"Unexpected image shape in {path}: {root['workspace_rgb'].shape}")
        if root["joint_state_lowdim"].shape[1:] != (6,) or root["joint_action_lowdim"].shape[1:] != (6,):
            raise ValueError(f"Unexpected lowdim shape in {path}.")
        video_ts = root["workspace_rgb_timestamps"][...]
        if video_ts.dtype != np.uint64 or np.any(np.diff(video_ts) <= 0):
            raise ValueError(f"Video timestamps not strictly increasing uint64 in {path}.")
        action_ts = root["joint_action_lowdim_timestamps"][...]
        if action_ts.dtype != np.uint64 or np.any(np.diff(action_ts) <= 0):
            raise ValueError(f"Action timestamps not strictly increasing uint64 in {path}.")
        if action_ts[0] != video_ts[0]:
            raise ValueError(f"Video and action timestamps have different start time in {path}.")

    metas = sorted((revision_dir / "video_finetune" / "metas").glob("*.txt"))
    videos = sorted((revision_dir / "video_finetune" / "video").glob("*.mp4"))
    if len(metas) != len(videos) or len(metas) != len(action_paths):
        raise ValueError("Video finetune and action decoder outputs do not have matching episode counts.")
    print(f"Validated {len(action_paths)} action episodes and matching video files.", flush=True)


def run_command(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def python_env() -> dict[str, str]:
    env = os.environ.copy()
    model_path = str(repo_root() / "model")
    env["PYTHONPATH"] = model_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def load_t5_components(text_encoder_config: dict[str, Any]):
    import torch

    model_path = str(repo_root() / "model")
    if model_path not in sys.path:
        sys.path.insert(0, model_path)

    from transformers import T5EncoderModel, T5TokenizerFast

    if not torch.cuda.is_available():
        raise RuntimeError("SO101 prompt preprocessing expects an NVIDIA GPU for the T5 encoder.")

    model_id = text_encoder_config["model_id"]
    revision = text_encoder_config["revision"]
    cache_dir = repo_root() / "model" / "checkpoints" / "text_encoder" / "hf-cache"
    offload_dir = Path("/tmp/mimic-video-t5-offload")
    offload_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading T5 tokenizer from {model_id}@{revision}.", flush=True)
    tokenizer = T5TokenizerFast.from_pretrained(model_id, revision=revision, cache_dir=str(cache_dir))
    print(f"Loading T5 encoder from {model_id}@{revision}.", flush=True)
    text_encoder = T5EncoderModel.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=str(cache_dir),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map={"": 0},
        max_memory={0: "44GiB", "cpu": "20GiB"},
        offload_folder=str(offload_dir),
        offload_state_dict=True,
        use_safetensors=True,
    )
    text_encoder.eval()
    return tokenizer, text_encoder


def encode_instruction(
    instruction: str,
    *,
    text_encoder_config: dict[str, Any],
    max_length: int = 512,
) -> tuple[list[np.ndarray], np.ndarray]:
    import gc
    import torch

    print("Encoding instruction text with T5.", flush=True)
    tokenizer, text_encoder = load_t5_components(text_encoder_config)
    try:
        print("Tokenizing instruction.", flush=True)
        batch_encoding = tokenizer.batch_encode_plus(
            [instruction],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_length=True,
            return_offsets_mapping=False,
        )
        input_ids = batch_encoding.input_ids.to("cuda:0")
        attn_mask = batch_encoding.attention_mask.to("cuda:0")

        print("Running T5 encoder forward pass.", flush=True)
        with torch.inference_mode():
            encoded_text = text_encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
            lengths = attn_mask.sum(dim=1).cpu().numpy()
            for batch_id in range(encoded_text.shape[0]):
                encoded_text[batch_id][int(lengths[batch_id]) :] = 0
            padded = encoded_text.cpu().numpy().astype(np.float16)
        trimmed = [padded[batch_id][: int(lengths[batch_id])] for batch_id in range(padded.shape[0])]
        if padded.shape != (1, max_length, 1024):
            raise ValueError(f"Unexpected T5 embedding shape: {padded.shape}")
        print(f"Encoded instruction to padded T5 embedding with shape {padded.shape}.", flush=True)
        return trimmed, padded
    finally:
        print("Releasing T5 encoder resources.", flush=True)
        del text_encoder
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def t5_embeddings_missing(revision_dir: Path) -> bool:
    video_dir = revision_dir / "video_finetune"
    metas = sorted((video_dir / "metas").glob("*.txt"))
    video_t5 = sorted((video_dir / "t5_xxl").glob("*.pickle"))
    if len(video_t5) != len(metas):
        print(
            f"T5 cache incomplete for video metas: found {len(video_t5)} embeddings for {len(metas)} metas.",
            flush=True,
        )
        return True

    zarr_paths = sorted((revision_dir / "action_decoder").glob("*.zarr"))
    for path in zarr_paths:
        root = zarr.open(str(path), mode="r")
        if "language_embedding" not in root or root["language_embedding"].shape != (1, 512, 1024):
            print(f"T5 cache incomplete for {path.name}; language_embedding missing or wrong shape.", flush=True)
            return True
    print(f"T5 cache complete for {len(metas)} video metas and {len(zarr_paths)} action episodes.", flush=True)
    return False


def write_t5_embeddings(revision_dir: Path, trimmed: list[np.ndarray], padded: np.ndarray) -> None:
    comp = Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)

    t5_dir = revision_dir / "video_finetune" / "t5_xxl"
    t5_dir.mkdir(parents=True, exist_ok=True)
    meta_paths = sorted((revision_dir / "video_finetune" / "metas").glob("*.txt"))
    zarr_paths = sorted((revision_dir / "action_decoder").glob("*.zarr"))
    print(f"Writing T5 embeddings to {len(meta_paths)} video metas and {len(zarr_paths)} zarr episodes.", flush=True)
    for meta_path in tqdm(meta_paths, desc="Writing video T5 pickles", unit="meta"):
        with (t5_dir / meta_path.with_suffix(".pickle").name).open("wb") as f:
            pickle.dump(trimmed, f)

    for path in tqdm(zarr_paths, desc="Writing zarr T5 embeddings", unit="episode"):
        root = zarr.open(str(path), mode="r+")
        root.create_dataset(
            "language_embedding",
            shape=padded.shape,
            dtype="float16",
            chunks=padded.shape,
            compressor=comp,
            overwrite=True,
        )[...] = padded
        root.create_dataset(
            "language_embedding_timestamps",
            shape=(1,),
            dtype="uint64",
            chunks=(1,),
            compressor=comp,
            overwrite=True,
        )[...] = np.array([0], dtype=np.uint64)
    print("Finished writing T5 embeddings.", flush=True)


def ensure_t5_embeddings(revision_dir: Path, preprocessing: dict[str, Any]) -> None:
    print("Checking T5 embedding cache.", flush=True)
    if not t5_embeddings_missing(revision_dir):
        return

    print("Encoding YAML instruction with T5 for video and action caches.", flush=True)
    trimmed, padded = encode_instruction(
        preprocessing["instruction"],
        text_encoder_config=preprocessing["text_encoder"],
    )
    write_t5_embeddings(revision_dir, trimmed, padded)


def vae_latents_missing(revision_dir: Path) -> bool:
    video_dir = revision_dir / "video_finetune" / "video"
    latent_dir = revision_dir / "video_finetune" / "vae_latents"
    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        return False
    if not latent_dir.exists():
        return True
    existing = {p.stem for p in latent_dir.glob("*.pt")}
    return any(v.stem not in existing for v in videos)


def ensure_vae_latents(revision_dir: Path) -> None:
    if not vae_latents_missing(revision_dir):
        print("VAE latents already computed, skipping.", flush=True)
        return

    import torch
    from decord import VideoReader
    from decord import cpu as decord_cpu
    from cosmos_predict2.configs.config_video2world import get_cosmos_predict2_video2world_pipeline
    from imaginaire.lazy_config import instantiate

    print("Precomputing VAE latents (runs once, saved to disk).", flush=True)

    config = get_cosmos_predict2_video2world_pipeline(model_size="2B", resolution="480", fps=10)
    tokenizer = instantiate(config.tokenizer)
    tokenizer.model.model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigma_data = float(config.sigma_data)

    video_dir = revision_dir / "video_finetune" / "video"
    latent_dir = revision_dir / "video_finetune" / "vae_latents"
    latent_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob("*.mp4"))
    with torch.no_grad():
        for video_path in tqdm(videos, desc="VAE latents"):
            latent_path = latent_dir / (video_path.stem + ".pt")
            if latent_path.exists():
                continue
            vr = VideoReader(str(video_path), ctx=decord_cpu(0), num_threads=0)
            # Cosmos VAE requires 4k+1 frames (temporal_window=4); a tail of 1
            # frame breaks the (3,1,1) temporal conv. Trim to the largest valid length.
            T = ((len(vr) - 1) // 4) * 4 + 1
            frames = vr.get_batch(list(range(T))).asnumpy()  # [T, H, W, C] uint8
            del vr
            # [T,H,W,C] -> [1,C,T,H,W], normalize to [-1,1], cast to bfloat16
            frames = (
                torch.from_numpy(frames)
                .permute(3, 0, 1, 2)
                .unsqueeze(0)
                .to(device=device, dtype=torch.bfloat16)
                / 127.5
                - 1.0
            )
            latents = tokenizer.encode(frames) * sigma_data  # [1, 16, T_lat, 60, 80]
            torch.save(latents[0].cpu(), latent_path)  # [16, T_lat, 60, 80] bfloat16

    print(f"VAE latents saved to {latent_dir}.", flush=True)


def resolve_video_experiment(name: str) -> str:
    if name.startswith("v2w_"):
        return name
    return f"v2w_{name}_lora_rank256_lr1.778e-04_bsz32"


def resolve_action_experiment(name: str) -> str:
    if name.startswith("w2a_"):
        return name
    return f"w2a_{name}_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1"


def train_env(output_root: str | None = None) -> dict[str, str]:
    env = python_env()
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    env.setdefault("NVTE_FUSED_ATTN", "0")
    if output_root is not None:
        env["IMAGINAIRE_OUTPUT_ROOT"] = str(Path(output_root).expanduser().resolve())
    return env


def hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{hydra_value(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(hydra_value(item) for item in value) + "]"
    if isinstance(value, str) and "," in value:
        return "'" + value.replace("'", "\\'") + "'"
    return str(value)


def add_override(overrides: list[str], key: str, value: Any, *, include_none: bool = False) -> None:
    if value is not None or include_none:
        overrides.append(f"{key}={hydra_value(value)}")


def optional_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return value


def add_mapped_overrides(
    overrides: list[str],
    section: dict[str, Any],
    *,
    prefix: str,
    keys: list[str],
    include_none: bool = True,
) -> None:
    for key in keys:
        if key in section:
            add_override(overrides, f"{prefix}.{key}", section.get(key), include_none=include_none)


def add_common_training_overrides(
    overrides: list[str],
    *,
    trainer: dict[str, Any],
    checkpoint: dict[str, Any],
    dataloader_train: dict[str, Any],
    dataloader_val: dict[str, Any],
    optimizer: dict[str, Any],
    lr_scheduler: dict[str, Any],
) -> None:
    if "max_iter" not in trainer:
        raise ValueError("Training configs must set trainer.max_iter.")

    add_mapped_overrides(
        overrides,
        trainer,
        prefix="trainer",
        keys=[
            "max_iter",
            "logging_iter",
            "validation_iter",
            "validate_at_epoch_end",
            "run_validation",
            "grad_accum_iter",
            "max_val_iter",
            "timeout_period",
        ],
    )
    add_mapped_overrides(
        overrides,
        checkpoint,
        prefix="checkpoint",
        keys=[
            "save_iter",
            "save_at_epoch_end",
            "keep_latest",
            "load_path",
            "load_training_state",
            "only_load_scheduler_state",
            "strict_resume",
            "verbose",
            "load_ema_to_reg",
        ],
    )
    add_mapped_overrides(
        overrides,
        dataloader_train,
        prefix="dataloader_train",
        keys=[
            "batch_size",
            "num_workers",
            "persistent_workers",
            "prefetch_factor",
            "pin_memory",
            "drop_last",
            "in_order",
        ],
    )
    add_mapped_overrides(
        overrides,
        dataloader_val,
        prefix="dataloader_val",
        keys=[
            "batch_size",
            "num_workers",
            "persistent_workers",
            "prefetch_factor",
            "pin_memory",
            "drop_last",
            "in_order",
        ],
    )
    add_mapped_overrides(
        overrides,
        optimizer,
        prefix="optimizer",
        keys=["lr", "weight_decay", "betas", "eps", "master_weights", "capturable"],
    )
    _lambda_keys = ["warm_up_steps", "cycle_lengths", "f_start", "f_max", "f_min"]
    if any(k in lr_scheduler for k in _lambda_keys):
        overrides.append("scheduler=lambdalinear")
    add_mapped_overrides(
        overrides,
        lr_scheduler,
        prefix="scheduler",
        keys=_lambda_keys,
    )


def add_job_overrides(overrides: list[str], stage_cfg: dict[str, Any]) -> None:
    job = optional_mapping(stage_cfg, "job")
    wandb = optional_mapping(stage_cfg, "wandb")

    add_override(overrides, "job.project", job.get("project", stage_cfg.get("wandb_project")))
    add_override(overrides, "job.group", job.get("group"))
    add_override(overrides, "job.name", job.get("name"))
    add_override(overrides, "trainer.callbacks.wandb.enabled", wandb.get("enabled"))


def maybe_run_dryrun(overrides: list[str], launch: dict[str, Any]) -> None:
    if launch.get("dryrun", True):
        run_train_command(
            overrides,
            dryrun=True,
            nproc_per_node=int(launch.get("nproc_per_node", 1)),
            output_root=launch.get("output_root"),
        )


def video_finetune_overrides(video_cfg: dict[str, Any], exp: str, revision_dir: Path, instruction: str = "") -> list[str]:
    trainer = optional_mapping(video_cfg, "trainer")
    dataset = optional_mapping(video_cfg, "dataset")
    video_model = optional_mapping(video_cfg, "video_model")
    eval_video = optional_mapping(video_cfg, "eval_video")
    overrides = [
        f"experiment={exp}",
        f"video_dataset_train.dataset_dir={revision_dir / 'video_finetune'}",
        f"video_dataset_val.dataset_dir={revision_dir / 'video_finetune'}",
    ]

    val_ratio = dataset.get("val_ratio", 0.0)
    add_mapped_overrides(overrides, dataset, prefix="video_dataset_train", keys=["val_ratio"])
    add_mapped_overrides(overrides, dataset, prefix="video_dataset_val", keys=["val_ratio"])
    add_mapped_overrides(overrides, video_model, prefix="video_dataset_train", keys=["num_frames", "obs_history"])
    add_mapped_overrides(overrides, video_model, prefix="video_dataset_val", keys=["num_frames", "obs_history"])
    add_override(overrides, "model.config.pipe_config.state_t", video_model.get("state_t"))

    eval_video_dir = str(revision_dir / "video_finetune")
    add_override(overrides, "trainer.callbacks.video_eval.eval_video_dir", eval_video_dir)
    add_override(overrides, "trainer.callbacks.video_eval.prompt", instruction)
    add_override(overrides, "trainer.callbacks.video_eval.val_ratio", val_ratio)
    hf_repo_id = optional_mapping(video_cfg, "upload").get("hf_repo_id")
    add_override(overrides, "trainer.callbacks.video_eval.hf_repo_id", hf_repo_id)
    add_mapped_overrides(
        overrides,
        eval_video,
        prefix="trainer.callbacks.video_eval",
        keys=[
            "enabled",
            "episode_stem",
            "episode_index",
            "seed_start_frame",
            "num_seed_frames",
            "seed",
            "guidance",
            "num_sampling_step",
        ],
    )

    add_common_training_overrides(
        overrides,
        trainer=trainer,
        checkpoint=optional_mapping(video_cfg, "checkpoint"),
        dataloader_train=optional_mapping(video_cfg, "dataloader_train"),
        dataloader_val=optional_mapping(video_cfg, "dataloader_val"),
        optimizer=optional_mapping(video_cfg, "optimizer"),
        lr_scheduler=optional_mapping(video_cfg, "lr_scheduler"),
    )
    add_override(overrides, "trainer.callbacks.grad_clip.clip_norm", trainer.get("grad_clip_norm"))
    add_job_overrides(overrides, video_cfg)
    maybe_run_dryrun(overrides, optional_mapping(video_cfg, "launch"))
    return overrides


def action_decoder_overrides(action_cfg: dict[str, Any], exp: str, revision_dir: Path) -> list[str]:
    trainer = optional_mapping(action_cfg, "trainer")
    dataset = optional_mapping(action_cfg, "dataset")
    model = optional_mapping(action_cfg, "model")
    action_scheduler = optional_mapping(action_cfg, "action_scheduler")
    network = optional_mapping(action_cfg, "network")
    ema = optional_mapping(action_cfg, "ema")

    overrides = [
        f"experiment={exp}",
        f"data_config.data_dir_override={revision_dir / 'action_decoder'}",
    ]
    add_mapped_overrides(
        overrides,
        dataset,
        prefix="data_config",
        keys=["num_val_episodes", "val_episode_ranges"],
    )

    add_common_training_overrides(
        overrides,
        trainer=trainer,
        checkpoint=optional_mapping(action_cfg, "checkpoint"),
        dataloader_train=optional_mapping(action_cfg, "dataloader_train"),
        dataloader_val=optional_mapping(action_cfg, "dataloader_val"),
        optimizer=optional_mapping(action_cfg, "optimizer"),
        lr_scheduler=optional_mapping(action_cfg, "lr_scheduler"),
    )
    add_mapped_overrides(
        overrides,
        model,
        prefix="model.config",
        keys=[
            "train_architecture",
            "lora_rank",
            "lora_alpha",
            "lora_target_modules",
            "init_lora_weights",
            "precision",
            "loss_reduce",
            "loss_scale",
            "validation_mode",
            "fsdp_shard_size",
        ],
    )
    add_override(overrides, "model.config.action_dit_path", model.get("action_dit_path"))
    add_override(overrides, "model.config.video_dit_path", model.get("video_dit_path"))
    add_override(overrides, "model.config.pipe_config.xattn_layer_idx", model.get("xattn_layer_idx"))
    add_override(
        overrides,
        "model.config.video_pipe_config.guardrail_config.enabled",
        model.get("video_guardrail_enabled"),
    )
    add_mapped_overrides(
        overrides,
        action_scheduler,
        prefix="world2action_pipe.scheduler",
        keys=["alpha", "beta", "num_denoising_steps"],
    )
    add_mapped_overrides(overrides, ema, prefix="world2action_pipe.ema", keys=["enabled"])
    add_mapped_overrides(
        overrides,
        network,
        prefix="world2action_pipe.net",
        keys=[
            "max_horizon",
            "in_channels",
            "out_channels",
            "model_channels",
            "num_blocks",
            "num_heads",
            "mlp_ratio",
            "atten_backend",
            "crossattn_emb_channels",
            "use_adaln_lora",
            "adaln_lora_dim",
            "pair_timestep_feature_rank",
        ],
    )
    add_override(overrides, "world2action_pipe.net.sac_config.mode", network.get("sac_mode"))
    add_override(overrides, "world2action_pipe.net.sac_config.every_n_blocks", network.get("sac_every_n_blocks"))

    add_override(overrides, "trainer.callbacks.grad_clip.clip_norm", trainer.get("grad_clip_norm"))
    add_job_overrides(overrides, action_cfg)
    maybe_run_dryrun(overrides, optional_mapping(action_cfg, "launch"))
    return overrides


def run_training(config: dict[str, Any], revision_dir: Path) -> None:
    training = require_mapping(config, "training")
    video_cfg = training.get("video_finetune") or {}
    action_cfg = training.get("action_decoder") or {}
    if not isinstance(video_cfg, dict) or not isinstance(action_cfg, dict):
        raise ValueError("training.video_finetune and training.action_decoder must be mappings.")

    if video_cfg.get("enabled", False):
        exp = resolve_video_experiment(str(video_cfg.get("experiment", "so101")))
        launch = optional_mapping(video_cfg, "launch")
        instruction = str((config.get("preprocessing") or {}).get("instruction", ""))
        overrides = video_finetune_overrides(video_cfg, exp, revision_dir, instruction=instruction)
        run_train_command(
            overrides,
            dryrun=False,
            nproc_per_node=int(launch.get("nproc_per_node", 1)),
            output_root=launch.get("output_root"),
        )

    if action_cfg.get("enabled", False):
        exp = resolve_action_experiment(str(action_cfg.get("experiment", "so101")))
        launch = optional_mapping(action_cfg, "launch")
        overrides = action_decoder_overrides(action_cfg, exp, revision_dir)
        run_train_command(
            overrides,
            dryrun=False,
            nproc_per_node=int(launch.get("nproc_per_node", 1)),
            output_root=launch.get("output_root"),
        )


def run_train_command(
    overrides: list[str],
    *,
    dryrun: bool,
    nproc_per_node: int = 1,
    output_root: str | None = None,
) -> None:
    if dryrun:
        cmd = [
            str(model_python()),
            "-m",
            "scripts.train",
            "--config=cosmos_predict2/configs/config.py",
            "--dryrun",
            "--",
            *overrides,
        ]
    else:
        cmd = [
            str(model_torchrun()),
            f"--nproc_per_node={nproc_per_node}",
            "-m",
            "scripts.train",
            "--config=cosmos_predict2/configs/config.py",
            "--",
            *overrides,
        ]
    run_command(cmd, cwd=repo_root() / "model", env=train_env(output_root))


def preprocess_if_needed(config: dict[str, Any]) -> Path:
    raw_preprocessing = require_mapping(config, "preprocessing")
    source = require_mapping(raw_preprocessing, "source")
    text_encoder = raw_preprocessing.get("text_encoder") or {}
    if not isinstance(text_encoder, dict):
        raise ValueError("preprocessing.text_encoder must be a mapping.")

    repo_id = str(source["repo_id"])
    source_revision = str(source.get("revision", "main"))
    print(f"Resolving HF dataset commit for {repo_id}@{source_revision}.", flush=True)
    try:
        commit_sha = resolve_commit_sha(repo_id, source_revision)
        print(f"Resolved HF dataset commit: {commit_sha}.", flush=True)
    except Exception:
        commit_sha = cached_commit_sha(repo_id, source_revision)
        if commit_sha is None:
            raise
        print(f"Using cached HF dataset commit for {repo_id}@{source_revision}: {commit_sha}", flush=True)
    text_encoder_model_id = str(text_encoder.get("model_id", "google-t5/t5-large"))
    text_encoder_revision = str(text_encoder.get("revision", "main"))
    print(f"Resolving T5 model commit for {text_encoder_model_id}@{text_encoder_revision}.", flush=True)
    text_encoder_commit_sha = resolve_model_commit_sha(text_encoder_model_id, text_encoder_revision)
    print(f"Resolved T5 model commit: {text_encoder_commit_sha}.", flush=True)
    preprocessing = normalize_preprocessing(
        config,
        commit_sha=commit_sha,
        text_encoder_commit_sha=text_encoder_commit_sha,
    )
    revision, revision_dir, should_preprocess = select_revision(preprocessing, force=get_force(config))

    print(f"Using data revision {revision}: {revision_dir}", flush=True)
    if should_preprocess:
        print("Preprocessing cache miss or force=true; rebuilding this data revision.", flush=True)
        if revision_dir.exists():
            print(f"Removing incomplete/stale revision directory {revision_dir}.", flush=True)
            shutil.rmtree(revision_dir)
        revision_dir.mkdir(parents=True)
        convert_dataset(revision_dir, preprocessing)
        validate_conversion(revision_dir)
        dump_yaml(revision_dir / "preprocessing.yaml", preprocessing)
        print(f"Wrote preprocessing cache key to {revision_dir / 'preprocessing.yaml'}.", flush=True)
    else:
        print("Preprocessing cache hit; reusing converted data revision.", flush=True)
        validate_conversion(revision_dir)
    ensure_t5_embeddings(revision_dir, preprocessing)
    training = config.get("training") or {}
    video_cfg = training.get("video_finetune") or {}
    dataset_cfg = video_cfg.get("dataset") or {}
    if dataset_cfg.get("precompute_vae_latents", True):
        ensure_vae_latents(revision_dir)
    print(f"SO101 preprocessing is ready: {revision_dir}.", flush=True)
    return revision_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    revision_dir = preprocess_if_needed(config)
    run_training(config, revision_dir)


if __name__ == "__main__":
    main()
