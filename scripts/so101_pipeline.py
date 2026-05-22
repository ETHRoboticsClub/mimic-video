#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import shutil
import string
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
SO101_FPS = 10  # target rate the cosmos backbone is pretrained on (constants.py: fps == 10)
TIMESTAMP_FRAME_TOLERANCE = 1e-3
PIPELINE_VERSION = 6  # bumped: video at SO101_FPS, lowdim at source rate (per-stream timestamps)
MIXED_SOURCE_CACHE_DIR = "_so101_mixes"


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


def validate_exclude_episodes(value: Any, *, key: str) -> list[int]:
    if value is None:
        value = []
    if not isinstance(value, list) or not all(isinstance(x, int) for x in value):
        raise ValueError(f"{key} must be a list of integer episode ids.")
    return sorted(set(value))


def preprocessing_sources(preprocessing: dict[str, Any]) -> list[dict[str, Any]]:
    sources = preprocessing.get("sources")
    if sources is None:
        source = require_mapping(preprocessing, "source")
        instruction = preprocessing.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("preprocessing.instruction must be a non-empty string.")
        video_feature = preprocessing.get("video_feature")
        if not isinstance(video_feature, str) or not video_feature:
            raise ValueError("preprocessing.video_feature must be a non-empty string.")
        return [
            {
                "repo_id": source.get("repo_id"),
                "revision": source.get("revision", "main"),
                "instruction": instruction.strip(),
                "video_feature": video_feature,
                "exclude_episodes": validate_exclude_episodes(
                    preprocessing.get("exclude_episodes", []),
                    key="preprocessing.exclude_episodes",
                ),
            }
        ]

    if not isinstance(sources, list) or not sources:
        raise ValueError("preprocessing.sources must be a non-empty list.")

    normalized_sources = []
    for idx, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"preprocessing.sources[{idx}] must be a mapping.")
        instruction = source.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"preprocessing.sources[{idx}].instruction must be a non-empty string.")
        video_feature = source.get("video_feature")
        if not isinstance(video_feature, str) or not video_feature:
            raise ValueError(f"preprocessing.sources[{idx}].video_feature must be a non-empty string.")
        normalized_sources.append(
            {
                "repo_id": source.get("repo_id"),
                "revision": source.get("revision", "main"),
                "instruction": instruction.strip(),
                "video_feature": video_feature,
                "exclude_episodes": validate_exclude_episodes(
                    source.get("exclude_episodes", []),
                    key=f"preprocessing.sources[{idx}].exclude_episodes",
                ),
            }
        )
    return normalized_sources


def validate_preprocessing_source(source: dict[str, Any], *, key: str) -> None:
    repo_id = source.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError(f"{key}.repo_id must be a non-empty string.")
    revision = source.get("revision", "main")
    if not isinstance(revision, str) or not revision:
        raise ValueError(f"{key}.revision must be a non-empty string.")


def normalize_preprocessing(
    config: dict[str, Any],
    *,
    commit_shas: list[str] | None = None,
    commit_sha: str | None = None,
    text_encoder_commit_sha: str,
) -> dict[str, Any]:
    preprocessing = require_mapping(config, "preprocessing")
    image = require_mapping(preprocessing, "image")
    contact_sheet = preprocessing.get("contact_sheet") or {}
    if not isinstance(contact_sheet, dict):
        raise ValueError("preprocessing.contact_sheet must be a mapping.")
    text_encoder = preprocessing.get("text_encoder") or {}
    if not isinstance(text_encoder, dict):
        raise ValueError("preprocessing.text_encoder must be a mapping.")

    sources = preprocessing_sources(preprocessing)
    if commit_shas is None:
        if commit_sha is None:
            raise ValueError("normalize_preprocessing requires commit_shas or commit_sha.")
        commit_shas = [commit_sha]
    if len(commit_shas) != len(sources):
        raise ValueError(f"Expected {len(sources)} dataset commit SHAs, got {len(commit_shas)}.")
    for idx, source in enumerate(sources):
        validate_preprocessing_source(
            source,
            key="preprocessing.source" if "sources" not in preprocessing else f"preprocessing.sources[{idx}]",
        )

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

    common = {
        "pipeline": {"name": "so101_pipeline", "version": PIPELINE_VERSION},
        "text_encoder": {
            "model_id": text_encoder_model_id,
            "revision": text_encoder_revision,
            "commit_sha": text_encoder_commit_sha,
        },
        "image": {"crop_xywh": crop_xywh, "output_size_hw": output_size_hw},
        "contact_sheet": {"enabled": bool(contact_sheet.get("enabled", False))},
    }
    if len(sources) == 1 and "sources" not in preprocessing:
        source = sources[0]
        return common | {
            "source": {
                "repo_id": source["repo_id"],
                "revision": source["revision"],
                "commit_sha": commit_shas[0],
            },
            "instruction": source["instruction"],
            "video_feature": source["video_feature"],
            "exclude_episodes": source["exclude_episodes"],
        }

    return common | {
        "sources": [
            {
                "repo_id": source["repo_id"],
                "revision": source["revision"],
                "commit_sha": commit_shas[idx],
                "instruction": source["instruction"],
                "video_feature": source["video_feature"],
                "exclude_episodes": source["exclude_episodes"],
                "stem_prefix": f"source{idx:03d}",
            }
            for idx, source in enumerate(sources)
        ]
    }


def get_force(config: dict[str, Any]) -> bool:
    return bool(require_mapping(config, "preprocessing").get("force", False))


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
    if "sources" in preprocessing:
        return repo_root() / "data" / MIXED_SOURCE_CACHE_DIR
    return repo_root() / "data" / preprocessing["source"]["repo_id"]


def load_cached_preprocessing(path: Path) -> dict[str, Any] | None:
    cache_path = path / "preprocessing.yaml"
    if not cache_path.exists():
        return None
    return load_yaml(cache_path)


def cached_commit_sha(repo_id: str, revision: str) -> str | None:
    bases = [repo_root() / "data" / repo_id, repo_root() / "data" / MIXED_SOURCE_CACHE_DIR]
    for base in bases:
        if not base.exists():
            continue
        for path in sorted((p for p in base.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
            cached = load_cached_preprocessing(path)
            if not cached:
                continue
            cached_sources = cached.get("sources")
            if isinstance(cached_sources, list):
                sources = cached_sources
            else:
                sources = [cached.get("source", {})]
            for source in sources:
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
        or f == "meta/tasks.jsonl"
        or f == "meta/episodes.jsonl"
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
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open() as f:
        info = json.load(f)
    features = info.get("features", {})
    if "action" not in features or "observation.state" not in features:
        raise ValueError(f"{info_path} must define LeRobot features 'action' and 'observation.state'.")
    if features["action"].get("shape") != [6] or features["observation.state"].get("shape") != [6]:
        raise ValueError(
            f"SO101 pipeline expects 6D action and 6D observation.state in {info_path}; "
            f"got action shape {features['action'].get('shape')} and observation.state shape "
            f"{features['observation.state'].get('shape')}."
        )
    fps = info.get("fps")
    if not isinstance(fps, int) or isinstance(fps, bool):
        raise ValueError(f"{info_path} must contain integer fps; got {fps!r}.")
    if fps < SO101_FPS or fps % SO101_FPS != 0:
        raise ValueError(
            f"SO101 preprocessing targets {SO101_FPS}fps (cosmos backbone). Source fps must be "
            f"an integer multiple of {SO101_FPS} (e.g. 10, 20, 30, 60). {info_path} declares fps={fps}."
        )
    subsample_step = fps // SO101_FPS  # 1 for 10fps source, 3 for 30fps, etc.
    video_keys = [key for key, spec in features.items() if spec.get("dtype") == "video"]
    if video_feature not in video_keys:
        raise ValueError(f"preprocessing.video_feature must be one of {video_keys}; got {video_feature!r}.")

    # LeRobot v2.1 declares per-dataset path templates. Two layouts coexist in
    # the wild: bundled (multiple episodes per parquet/mp4, keyed by file_index)
    # and per-episode (one parquet + one mp4 per episode, keyed by
    # episode_index). Default to the bundled template the older code assumed
    # for datasets that don't declare templates.
    data_path_template = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/file-{file_index:03d}.parquet"
    )
    video_path_template = info.get(
        "video_path", "videos/{video_key}/chunk-{episode_chunk:03d}/file-{video_file_index:03d}.mp4"
    )
    chunks_size = int(info.get("chunks_size", 1000))
    layout = "per_episode" if "episode_index" in _template_fields(data_path_template) else "bundled"

    return info | {
        "_video_key": video_feature,
        "_data_path_template": data_path_template,
        "_video_path_template": video_path_template,
        "_chunks_size": chunks_size,
        "_layout": layout,
        "_subsample_step": subsample_step,
    }


def _template_fields(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


def _format_path_template(template: str, **kwargs: Any) -> str:
    fields = _template_fields(template)
    missing = sorted(fields - kwargs.keys())
    if missing:
        raise KeyError(f"Path template {template!r} requires fields {missing} that were not provided.")
    return template.format(**{k: v for k, v in kwargs.items() if k in fields})


def resolve_episode_paths(
    dataset_root: Path,
    info: dict[str, Any],
    episode_row: pd.Series,
) -> tuple[Path, Path, int]:
    """Return (data_path, video_path, start_frame) for one episode.

    start_frame is 0 for per-episode layouts (each mp4 holds exactly one episode)
    and the bundled video's frame offset for bundled layouts.

    Data and video templates are formatted in their own namespaces so that bare
    field names like {chunk_index} / {file_index} resolve to the right episode
    metadata column (data/chunk_index vs videos/<key>/chunk_index, etc.).
    """
    video_key = info["_video_key"]
    chunks_size = info["_chunks_size"]
    episode_idx = int(episode_row["episode_index"])

    def _kwargs_for(scope_prefix: str) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "episode_index": episode_idx,
            "episode_chunk": int(episode_row.get("episode_chunk", episode_idx // chunks_size)),
            "video_key": video_key,
        }
        # Map bare template fields (chunk_index, file_index, ...) to the
        # appropriate scope-prefixed episode row column when present.
        for bare in ("chunk_index", "file_index"):
            col = f"{scope_prefix}/{bare}"
            if col in episode_row.index:
                kw[bare] = int(episode_row[col])
        # Some templates explicitly disambiguate, e.g. video_file_index.
        for full in (f"videos/{video_key}/file_index", f"videos/{video_key}/chunk_index"):
            if full in episode_row.index:
                alias = full.replace(f"videos/{video_key}/", "video_")
                kw[alias] = int(episode_row[full])
        return kw

    data_kwargs = _kwargs_for("data")
    video_kwargs = _kwargs_for(f"videos/{video_key}")
    data_path = dataset_root / _format_path_template(info["_data_path_template"], **data_kwargs)
    video_path = dataset_root / _format_path_template(info["_video_path_template"], **video_kwargs)

    if info["_layout"] == "per_episode":
        start_frame = 0
    else:
        from_ts_col = f"videos/{video_key}/from_timestamp"
        if from_ts_col not in episode_row.index:
            raise ValueError(
                f"Bundled layout requires episode metadata column {from_ts_col!r} for episode "
                f"{episode_idx} but it was not present."
            )
        start_frame = timestamp_to_frame_index(
            episode_row[from_ts_col],
            fps=info["fps"],
            context=f"episode {episode_idx} {from_ts_col}",
        )
    return data_path, video_path, start_frame


def timestamp_to_frame_index(value: Any, *, fps: int, context: str) -> int:
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be numeric seconds; got {value!r}.") from exc
    if not np.isfinite(timestamp):
        raise ValueError(f"{context} must be finite seconds; got {value!r}.")
    if timestamp < 0:
        raise ValueError(f"{context} must be non-negative seconds; got {timestamp}.")
    frame_float = timestamp * fps
    frame_index = int(np.rint(frame_float))
    error = abs(frame_float - frame_index)
    if error > TIMESTAMP_FRAME_TOLERANCE:
        raise ValueError(
            f"{context}={timestamp} does not land on the {fps}fps frame grid: "
            f"timestamp*fps={frame_float:.9f}, nearest frame={frame_index}, error={error:.9f}."
        )
    return frame_index


def derive_episode_frame_timing(
    rows: pd.DataFrame,
    *,
    episode_idx: int,
    source_fps: int,
    target_fps: int = SO101_FPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate per-row timestamps and produce both decimated-video and full-rate-lowdim timing.

    Returns:
        video_row_mask: bool mask over the input rows; True for rows kept for the
            decimated video stream (every `source_fps/target_fps`-th row).
        video_source_frame_indices: source-rate frame indices to decode for the
            decimated video stream (length == video_row_mask.sum()).
        video_timestamps_ns: ns timestamps on the `target_fps` grid for the
            decimated video stream.
        lowdim_timestamps_ns: ns timestamps on the `source_fps` grid for ALL
            input rows; used for joint_state / joint_action zarr writes so the
            full-bandwidth lowdim signal is preserved.
    """
    if "timestamp" not in rows.columns:
        raise ValueError(f"Episode {episode_idx} data rows are missing required 'timestamp' column.")
    timestamps = rows["timestamp"].to_numpy(dtype=np.float64)
    if len(timestamps) == 0:
        raise ValueError(f"Episode {episode_idx} has no timestamped rows.")
    if not np.all(np.isfinite(timestamps)):
        bad = np.where(~np.isfinite(timestamps))[0][:5].tolist()
        raise ValueError(f"Episode {episode_idx} has non-finite timestamps at row offsets {bad}.")
    if np.any(timestamps < 0):
        bad = np.where(timestamps < 0)[0][:5].tolist()
        raise ValueError(f"Episode {episode_idx} has negative timestamps at row offsets {bad}.")

    if source_fps < target_fps or source_fps % target_fps != 0:
        raise ValueError(
            f"source_fps ({source_fps}) must be an integer multiple of target_fps ({target_fps})."
        )
    step = source_fps // target_fps

    # Validate input is dense at source_fps (every consecutive source-frame present).
    frame_float = timestamps * source_fps
    local_frame_indices = np.rint(frame_float).astype(np.int64)
    errors = np.abs(frame_float - local_frame_indices)
    if np.any(errors > TIMESTAMP_FRAME_TOLERANCE):
        bad = int(np.argmax(errors))
        raise ValueError(
            f"Episode {episode_idx} timestamp {timestamps[bad]} at row offset {bad} is not aligned to {source_fps}fps: "
            f"timestamp*fps={frame_float[bad]:.9f}, nearest frame={local_frame_indices[bad]}, "
            f"error={errors[bad]:.9f}."
        )
    if local_frame_indices[0] != 0:
        raise ValueError(
            f"Episode {episode_idx} timestamps must be episode-relative and start at frame 0 for {source_fps}fps conversion; "
            f"first timestamp {timestamps[0]} maps to frame {local_frame_indices[0]}."
        )
    diffs = np.diff(local_frame_indices)
    if np.any(diffs <= 0):
        bad = int(np.where(diffs <= 0)[0][0])
        raise ValueError(
            f"Episode {episode_idx} timestamps must map to strictly increasing {source_fps}fps frame indices. "
            f"Row offsets {bad} and {bad + 1} map to frames {local_frame_indices[bad]} and "
            f"{local_frame_indices[bad + 1]} from timestamps {timestamps[bad]} and {timestamps[bad + 1]}."
        )
    if np.any(diffs != 1):
        bad = int(np.where(diffs != 1)[0][0])
        raise ValueError(
            f"Episode {episode_idx} timestamps must describe contiguous {source_fps}fps frames. "
            f"Row offsets {bad} and {bad + 1} jump from frame {local_frame_indices[bad]} to "
            f"{local_frame_indices[bad + 1]} using timestamps {timestamps[bad]} and {timestamps[bad + 1]}."
        )

    # Video: subsample every `step`-th source frame to land on the target_fps grid.
    video_row_mask = (local_frame_indices % step) == 0
    video_source_frame_indices = local_frame_indices[video_row_mask]
    video_target_indices = (video_source_frame_indices // step).astype(np.int64)
    target_period_ns = NS_PER_SEC // target_fps
    video_timestamps_ns = (
        video_target_indices.astype(np.uint64) * np.uint64(target_period_ns)
    ).astype(np.uint64)

    # Lowdim: keep ALL rows at source_fps; record their ns timestamps on the source grid.
    source_period_ns = NS_PER_SEC // source_fps
    lowdim_timestamps_ns = (
        local_frame_indices.astype(np.uint64) * np.uint64(source_period_ns)
    ).astype(np.uint64)

    return video_row_mask, video_source_frame_indices, video_timestamps_ns, lowdim_timestamps_ns


def read_episodes(dataset_root: Path, chunks_size: int) -> pd.DataFrame:
    parquet_paths = sorted((dataset_root / "meta" / "episodes").glob("**/*.parquet"))
    if parquet_paths:
        df = pd.concat([pq.read_table(p).to_pandas() for p in parquet_paths], ignore_index=True)
    else:
        jsonl_path = dataset_root / "meta" / "episodes.jsonl"
        if not jsonl_path.exists():
            raise ValueError(
                f"No episode metadata under {dataset_root / 'meta'}: expected either "
                "meta/episodes/**/*.parquet (bundled) or meta/episodes.jsonl (per-episode)."
            )
        df = pd.read_json(jsonl_path, lines=True)
    df = df.sort_values("episode_index").reset_index(drop=True)
    if "episode_chunk" not in df.columns:
        df["episode_chunk"] = df["episode_index"].astype(np.int64) // np.int64(chunks_size)
    return df


def read_tasks(dataset_root: Path) -> dict[int, str]:
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return {}
    df = pq.read_table(tasks_path).to_pandas().reset_index()
    return {int(row["task_index"]): str(row["task"]) for _, row in df.iterrows()}


def validate_crop(crop_xywh: list[int], source_hw: tuple[int, int]) -> None:
    x, y, w, h = crop_xywh
    source_h, source_w = source_hw
    if x < 0 or y < 0 or x + w > source_w or y + h > source_h:
        raise ValueError(f"Crop {crop_xywh} is outside source image size {(source_h, source_w)}.")


def process_frame(frame: av.VideoFrame, crop_xywh: list[int], output_size_hw: list[int]) -> np.ndarray:
    x, y, w, h = crop_xywh
    out_h, out_w = output_size_hw
    img = frame.to_image().convert("RGB").crop((x, y, x + w, y + h))
    img = img.resize((out_w, out_h), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def read_episode_frames(
    video_path: Path,
    *,
    start_frame: int,
    local_frame_indices: np.ndarray,
    crop_xywh: list[int],
    output_size_hw: list[int],
) -> np.ndarray:
    if len(local_frame_indices) == 0:
        raise ValueError(f"No requested frames for {video_path}.")
    if np.any(local_frame_indices < 0):
        bad = np.where(local_frame_indices < 0)[0][:5].tolist()
        raise ValueError(f"Requested negative local frame indices for {video_path} at offsets {bad}.")
    diffs = np.diff(local_frame_indices)
    if np.any(diffs <= 0):
        bad = int(np.where(diffs <= 0)[0][0])
        raise ValueError(
            f"Requested local frame indices for {video_path} must be strictly increasing; "
            f"offsets {bad}/{bad + 1} are {local_frame_indices[bad]} and {local_frame_indices[bad + 1]}."
        )

    source_frame_indices = start_frame + local_frame_indices
    requested = {int(source_idx): i for i, source_idx in enumerate(source_frame_indices)}
    max_requested = max(requested)
    frames: np.ndarray | None = None
    found = np.zeros(len(local_frame_indices), dtype=bool)

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        validate_crop(crop_xywh, (stream.height, stream.width))
        for frame_idx, frame in enumerate(container.decode(video=0)):
            if frame_idx in requested:
                if frames is None:
                    out_h, out_w = output_size_hw
                    frames = np.empty((len(local_frame_indices), out_h, out_w, 3), dtype=np.uint8)
                frames[requested[frame_idx]] = process_frame(frame, crop_xywh, output_size_hw)
                found[requested[frame_idx]] = True
            if frame_idx >= max_requested:
                break

    if frames is None or not found.all():
        missing = np.where(~found)[0][:5].tolist()
        missing_source = [int(source_frame_indices[i]) for i in missing]
        raise ValueError(
            f"Failed to decode requested frames from {video_path}. Missing row offsets {missing} "
            f"with source frame indices {missing_source}; start_frame={start_frame}, "
            f"max_requested={max_requested}."
        )
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
    lowdim_timestamps_ns: np.ndarray,
    joint_state: np.ndarray,
    joint_action: np.ndarray,
    instruction: str,
) -> None:
    """Write the per-episode zarr. Video and lowdim may have different sample rates.

    `video_timestamps_ns` corresponds 1:1 to `frames`. `lowdim_timestamps_ns`
    corresponds 1:1 to `joint_state` and `joint_action`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    comp = Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)
    t_img = min(65, len(video_timestamps_ns))
    t_ld = min(1024, len(lowdim_timestamps_ns))
    root = zarr.open(str(path), mode="w")

    root.create_dataset("workspace_rgb", shape=frames.shape, dtype=np.uint8, chunks=(t_img, *frames.shape[1:]), compressor=comp)[
        ...
    ] = frames
    root.create_dataset(
        "workspace_rgb_timestamps",
        shape=video_timestamps_ns.shape,
        dtype="uint64",
        chunks=video_timestamps_ns.shape,
    )[...] = video_timestamps_ns

    root.create_dataset(
        "joint_state_lowdim",
        shape=joint_state.shape,
        dtype=np.float32,
        chunks=(t_ld, *joint_state.shape[1:]),
        compressor=comp,
    )[...] = joint_state
    root.create_dataset(
        "joint_state_lowdim_timestamps",
        shape=lowdim_timestamps_ns.shape,
        dtype="uint64",
        chunks=lowdim_timestamps_ns.shape,
    )[...] = lowdim_timestamps_ns

    root.create_dataset(
        "joint_action_lowdim",
        shape=joint_action.shape,
        dtype=np.float32,
        chunks=(t_ld, *joint_action.shape[1:]),
        compressor=comp,
    )[...] = joint_action
    root.create_dataset(
        "joint_action_lowdim_timestamps",
        shape=lowdim_timestamps_ns.shape,
        dtype="uint64",
        chunks=lowdim_timestamps_ns.shape,
    )[...] = lowdim_timestamps_ns

    root.create_dataset("language_instruction", shape=(1,), dtype=bytes, chunks=(1,), compressor=comp)[...] = np.array(
        [instruction.encode("utf-8")]
    )
    root.create_dataset("language_instruction_timestamps", shape=(1,), dtype="uint64", chunks=(1,), compressor=comp)[
        ...
    ] = np.array([0], dtype=np.uint64)


def make_contact_sheet(
    path: Path,
    dataset_root: Path,
    info: dict[str, Any],
    episodes: pd.DataFrame,
    *,
    crop_xywh: list[int],
    output_size_hw: list[int],
) -> None:
    thumbs: list[Image.Image] = []
    for _, episode in tqdm(
        episodes.iterrows(),
        total=len(episodes),
        desc="Building contact sheet",
        unit="episode",
    ):
        _, video_path, start_frame = resolve_episode_paths(dataset_root, info, episode)
        length = int(episode["length"])
        for frac in (0.0, 0.5, 0.95):
            local_idx = int(round((length - 1) * frac))
            frame_arr = read_episode_frames(
                video_path,
                start_frame=start_frame,
                local_frame_indices=np.array([local_idx], dtype=np.int64),
                crop_xywh=crop_xywh,
                output_size_hw=output_size_hw,
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


def normalized_preprocessing_sources(preprocessing: dict[str, Any]) -> list[dict[str, Any]]:
    if "sources" in preprocessing:
        return preprocessing["sources"]
    return [
        {
            **preprocessing["source"],
            "instruction": preprocessing["instruction"],
            "video_feature": preprocessing["video_feature"],
            "exclude_episodes": preprocessing["exclude_episodes"],
            "stem_prefix": "",
        }
    ]


def convert_dataset_source(
    revision_dir: Path,
    preprocessing: dict[str, Any],
    source: dict[str, Any],
    *,
    source_idx: int,
    num_sources: int,
) -> None:
    print(
        f"Preparing SO101 cache from {source['repo_id']}@{source['revision']} "
        f"({source['commit_sha']}).",
        flush=True,
    )
    dataset_root = download_repo_files(source["repo_id"], source["revision"])
    print(f"HF dataset files are available under {dataset_root}.", flush=True)
    video_feature = source["video_feature"]
    info = read_info(dataset_root, video_feature)
    episodes = read_episodes(dataset_root, info["_chunks_size"])
    read_tasks(dataset_root)

    excluded = set(source["exclude_episodes"])
    selected = episodes[~episodes["episode_index"].isin(excluded)].copy()
    if selected.empty:
        raise ValueError(f"All episodes were excluded for source {source_idx}.")
    print(
        f"Converting {len(selected)} episodes from source {source_idx} with video feature {video_feature!r}; "
        f"excluded episodes: {sorted(excluded)}.",
        flush=True,
    )

    fps = info["fps"]
    subsample_step = info["_subsample_step"]
    if subsample_step > 1:
        print(
            f"Source fps={fps} → target fps={SO101_FPS}: keeping 1 of every {subsample_step} frames.",
            flush=True,
        )
    crop_xywh = preprocessing["image"]["crop_xywh"]
    output_size_hw = preprocessing["image"]["output_size_hw"]
    if output_size_hw != [480, 640]:
        raise ValueError("The current Cosmos training path requires preprocessing.image.output_size_hw: [480, 640].")

    if preprocessing["contact_sheet"]["enabled"]:
        print("Writing contact sheet.", flush=True)
        contact_sheet_name = "contact_sheet.jpg" if num_sources == 1 else f"contact_sheet_source{source_idx:03d}.jpg"
        make_contact_sheet(
            revision_dir / contact_sheet_name,
            dataset_root,
            info,
            selected,
            crop_xywh=crop_xywh,
            output_size_hw=output_size_hw,
        )

    data_cache: dict[str, pd.DataFrame] = {}
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
        data_path, video_path, start_frame = resolve_episode_paths(dataset_root, info, episode)
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        data_file = data_cache.setdefault(str(data_path), pq.read_table(data_path).to_pandas())
        required_columns = ["episode_index", "timestamp", "observation.state", "action"]
        missing_columns = [column for column in required_columns if column not in data_file.columns]
        if missing_columns:
            raise ValueError(f"Episode data file {data_path} is missing required columns: {missing_columns}.")
        rows = data_file[data_file["episode_index"] == episode_idx].sort_values("timestamp")
        if rows.empty:
            raise ValueError(f"No rows for episode {episode_idx} in {data_path}.")

        (
            video_row_mask,
            video_source_frame_indices,
            video_timestamps_ns,
            lowdim_timestamps_ns,
        ) = derive_episode_frame_timing(
            rows, episode_idx=episode_idx, source_fps=fps, target_fps=SO101_FPS,
        )
        frames = read_episode_frames(
            video_path,
            start_frame=start_frame,
            local_frame_indices=video_source_frame_indices,
            crop_xywh=crop_xywh,
            output_size_hw=output_size_hw,
        )

        # Lowdim joint_state / joint_action are kept at the full source rate so
        # the IDM can train against the native-bandwidth control signal (the
        # video stream is the only thing the cosmos backbone constrains to 10
        # fps). The chunk reader uses per-stream timestamps for resampling.
        joint_state = np.stack(rows["observation.state"].to_numpy()).astype(np.float32)
        joint_action = np.stack(rows["action"].to_numpy()).astype(np.float32)

        stem_prefix = source.get("stem_prefix", "")
        stem = f"{stem_prefix}_episode_{episode_idx:06d}" if stem_prefix else f"episode_{episode_idx:06d}"
        write_video(video_dir / "video" / f"{stem}.mp4", frames, fps=SO101_FPS)
        (video_dir / "metas" / f"{stem}.txt").write_text(source["instruction"] + "\n")
        write_zarr_episode(
            action_dir / f"{stem}.zarr",
            frames=frames,
            video_timestamps_ns=video_timestamps_ns,
            lowdim_timestamps_ns=lowdim_timestamps_ns,
            joint_state=joint_state,
            joint_action=joint_action,
            instruction=source["instruction"],
        )


def convert_dataset(revision_dir: Path, preprocessing: dict[str, Any]) -> None:
    sources = normalized_preprocessing_sources(preprocessing)
    for source_idx, source in enumerate(sources):
        convert_dataset_source(
            revision_dir,
            preprocessing,
            source,
            source_idx=source_idx,
            num_sources=len(sources),
        )
    print(f"Finished writing converted data to {revision_dir}.", flush=True)


def validate_conversion(revision_dir: Path) -> None:
    print(f"Validating converted cache at {revision_dir}.", flush=True)
    action_paths = sorted((revision_dir / "action_decoder").glob("*.zarr"))
    if not action_paths:
        raise ValueError("No zarr episodes were written.")
    for path in action_paths:
        root = zarr.open(str(path), mode="r")
        # Each data stream pairs with its own timestamp array; video runs at
        # SO101_FPS while lowdim is kept at the source rate (see
        # convert_dataset_source). So we check pairwise length, not global
        # equality.
        pair_lengths = {
            "workspace_rgb / workspace_rgb_timestamps": (
                len(root["workspace_rgb"]),
                len(root["workspace_rgb_timestamps"]),
            ),
            "joint_state_lowdim / joint_state_lowdim_timestamps": (
                len(root["joint_state_lowdim"]),
                len(root["joint_state_lowdim_timestamps"]),
            ),
            "joint_action_lowdim / joint_action_lowdim_timestamps": (
                len(root["joint_action_lowdim"]),
                len(root["joint_action_lowdim_timestamps"]),
            ),
        }
        for name, (data_len, ts_len) in pair_lengths.items():
            if data_len != ts_len:
                raise ValueError(f"Mismatched length in {path} for {name}: data={data_len}, ts={ts_len}")
        if len(root["joint_state_lowdim"]) != len(root["joint_action_lowdim"]):
            raise ValueError(
                f"joint_state_lowdim ({len(root['joint_state_lowdim'])}) and joint_action_lowdim "
                f"({len(root['joint_action_lowdim'])}) must have matching lengths in {path}."
            )
        if root["workspace_rgb"].shape[1:] != (480, 640, 3):
            raise ValueError(f"Unexpected image shape in {path}: {root['workspace_rgb'].shape}")
        if root["joint_state_lowdim"].shape[1:] != (6,) or root["joint_action_lowdim"].shape[1:] != (6,):
            raise ValueError(f"Unexpected lowdim shape in {path}.")
        for ts_key in ("workspace_rgb_timestamps", "joint_state_lowdim_timestamps", "joint_action_lowdim_timestamps"):
            ts = root[ts_key][...]
            if ts.dtype != np.uint64 or (len(ts) > 1 and np.any(np.diff(ts) <= 0)):
                raise ValueError(f"{ts_key} is not strictly increasing uint64 in {path}.")

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


def source_for_stem(preprocessing: dict[str, Any], stem: str) -> dict[str, Any]:
    sources = normalized_preprocessing_sources(preprocessing)
    if len(sources) == 1 and not sources[0].get("stem_prefix"):
        return sources[0]
    for source in sources:
        prefix = source.get("stem_prefix")
        if prefix and stem.startswith(f"{prefix}_"):
            return source
    raise ValueError(f"Could not map episode stem {stem!r} to a preprocessing source.")


def write_t5_embeddings(
    revision_dir: Path,
    preprocessing: dict[str, Any],
    embeddings_by_prefix: dict[str, tuple[list[np.ndarray], np.ndarray]],
) -> None:
    comp = Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)

    t5_dir = revision_dir / "video_finetune" / "t5_xxl"
    t5_dir.mkdir(parents=True, exist_ok=True)
    meta_paths = sorted((revision_dir / "video_finetune" / "metas").glob("*.txt"))
    zarr_paths = sorted((revision_dir / "action_decoder").glob("*.zarr"))
    print(f"Writing T5 embeddings to {len(meta_paths)} video metas and {len(zarr_paths)} zarr episodes.", flush=True)
    for meta_path in tqdm(meta_paths, desc="Writing video T5 pickles", unit="meta"):
        source = source_for_stem(preprocessing, meta_path.stem)
        trimmed, _ = embeddings_by_prefix[source.get("stem_prefix", "")]
        with (t5_dir / meta_path.with_suffix(".pickle").name).open("wb") as f:
            pickle.dump(trimmed, f)

    for path in tqdm(zarr_paths, desc="Writing zarr T5 embeddings", unit="episode"):
        source = source_for_stem(preprocessing, path.stem)
        _, padded = embeddings_by_prefix[source.get("stem_prefix", "")]
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

    embeddings_by_prefix = {}
    for source in normalized_preprocessing_sources(preprocessing):
        prefix = source.get("stem_prefix", "")
        print(f"Encoding instruction with T5 for source {prefix or 'single'} caches.", flush=True)
        embeddings_by_prefix[prefix] = encode_instruction(
            source["instruction"],
            text_encoder_config=preprocessing["text_encoder"],
        )
    write_t5_embeddings(revision_dir, preprocessing, embeddings_by_prefix)


def resolve_video_experiment(name: str) -> str:
    if name.startswith("v2w_"):
        return name
    return f"v2w_{name}_lora_rank256_lr1.778e-04_bsz32"


def resolve_action_experiment(name: str) -> str:
    if name.startswith("w2a_"):
        return name
    return f"w2a_{name}_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1"


def cache_video_backbone_into_run_dir(
    action_cfg: dict[str, Any],
    overrides: list[str],
    output_root: str | None,
) -> None:
    """Copy the configured video_dit_path into the IDM run's output dir and
    rewrite the override so training reads the cached copy. Protects against
    the source ckpt getting rotated away by a parallel cosmos run's
    keep_latest window."""
    model = optional_mapping(action_cfg, "model")
    src = model.get("video_dit_path")
    if not src:
        return
    src_path = Path(str(src)).expanduser()
    if not src_path.exists():
        raise FileNotFoundError(f"video_dit_path does not exist: {src_path}")

    root = output_root or os.environ.get("IMAGINAIRE_OUTPUT_ROOT")
    if not root:
        raise ValueError("Need IMAGINAIRE_OUTPUT_ROOT (env or launch.output_root) to cache video_dit_path.")
    job = optional_mapping(action_cfg, "job")
    path_local = (
        Path(root).expanduser()
        / str(job.get("project", ""))
        / str(job.get("group", ""))
        / str(job.get("name", ""))
    )
    dst_dir = path_local / "video_backbone"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src_path.name

    if not dst.exists() or dst.stat().st_size != src_path.stat().st_size:
        print(f"Caching video backbone: {src_path} -> {dst}", flush=True)
        shutil.copy2(src_path, dst)
    else:
        print(f"Video backbone already cached at {dst}, skipping copy.", flush=True)

    # Rewrite the existing override (or append) so the trainer loads the cached copy.
    target = "model.config.video_dit_path"
    new_override = f"{target}={hydra_value(str(dst))}"
    for i, ov in enumerate(overrides):
        if ov.startswith(target + "="):
            overrides[i] = new_override
            return
    overrides.append(new_override)


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
    # Only forward lambdalinear-specific scheduler params when the active scheduler
    # actually has those fields. ConstantScheduler has no attributes, so emitting
    # warm_up_steps/cycle_lengths/etc. against it raises ConfigAttributeError.
    if lr_scheduler.get("type") in {"lambdalinear", "cosine"}:
        add_mapped_overrides(
            overrides,
            lr_scheduler,
            prefix="scheduler",
            keys=["warm_up_steps", "cycle_lengths", "f_start", "f_max", "f_min"],
        )


def add_job_overrides(overrides: list[str], stage_cfg: dict[str, Any]) -> None:
    job = optional_mapping(stage_cfg, "job")
    wandb = optional_mapping(stage_cfg, "wandb")

    add_override(overrides, "job.project", job.get("project", stage_cfg.get("wandb_project")))
    add_override(overrides, "job.group", job.get("group"))
    add_override(overrides, "job.name", job.get("name"))
    add_override(overrides, "trainer.callbacks.wandb.enabled", wandb.get("enabled"))


def maybe_run_dryrun(overrides: list[str], launch: dict[str, Any]) -> None:
    if launch.get("dryrun", False):
        run_train_command(
            overrides,
            dryrun=True,
            nproc_per_node=int(launch.get("nproc_per_node", 1)),
            output_root=launch.get("output_root"),
        )


def video_finetune_overrides(
    video_cfg: dict[str, Any], exp: str, revision_dir: Path, instruction: str = "", seed: int | None = None,
) -> list[str]:
    trainer = optional_mapping(video_cfg, "trainer")
    dataset = optional_mapping(video_cfg, "dataset")
    video_model = optional_mapping(video_cfg, "video_model")
    model = optional_mapping(video_cfg, "model")
    eval_video = optional_mapping(video_cfg, "eval_video")
    lr_scheduler_cfg = optional_mapping(video_cfg, "lr_scheduler")
    overrides = [
        f"experiment={exp}",
        f"video_dataset_train.dataset_dir={revision_dir / 'video_finetune'}",
        f"video_dataset_val.dataset_dir={revision_dir / 'video_finetune'}",
    ]
    scheduler_type = lr_scheduler_cfg.get("type")
    if isinstance(scheduler_type, str) and scheduler_type:
        overrides.append(f"scheduler={scheduler_type}")

    if seed is not None:
        add_override(overrides, "video_dataset_train.seed", seed)
        add_override(overrides, "video_dataset_val.seed", seed)
        add_override(overrides, "dataloader_train.sampler.seed", seed)
        add_override(overrides, "dataloader_val.sampler.seed", seed)

    val_ratio = dataset.get("val_ratio", 0.0)
    add_mapped_overrides(overrides, dataset, prefix="video_dataset_train", keys=["val_ratio"])
    add_mapped_overrides(overrides, dataset, prefix="video_dataset_val", keys=["val_ratio"])
    add_mapped_overrides(
        overrides,
        video_model,
        prefix="video_dataset_train",
        keys=["num_frames", "obs_history", "data_fps", "effective_fps"],
    )
    add_mapped_overrides(
        overrides,
        video_model,
        prefix="video_dataset_val",
        keys=["num_frames", "obs_history", "data_fps", "effective_fps"],
    )
    add_override(overrides, "model.config.pipe_config.state_t", video_model.get("state_t"))
    add_mapped_overrides(
        overrides,
        model,
        prefix="model.config",
        keys=[
            "train_architecture",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_target_modules",
            "init_lora_weights",
        ],
    )
    if "train_architecture" in model:
        add_override(
            overrides,
            "trainer.callbacks.video_eval.fuse_lora",
            model.get("train_architecture") == "lora",
        )
    add_override(overrides, "trainer.callbacks.video_eval.lora_alpha", model.get("lora_alpha"))

    add_override(overrides, "trainer.callbacks.video_eval.eval_video_dir", str(revision_dir / "video_finetune"))
    add_override(overrides, "trainer.callbacks.video_eval.prompt", instruction)
    add_override(overrides, "trainer.callbacks.video_eval.val_ratio", val_ratio)
    add_override(
        overrides,
        "trainer.callbacks.video_eval.hf_repo_id",
        optional_mapping(video_cfg, "upload").get("hf_repo_id"),
    )
    add_mapped_overrides(
        overrides,
        eval_video,
        prefix="trainer.callbacks.video_eval",
        keys=[
            "enabled",
            "episode_stem",
            "episode_index",
            "max_episodes_per_val_cycle",
            "seed_start_frame",
            "num_seed_frames",
            "num_eval_video_frames",
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
    add_override(overrides, "trainer.callbacks.grad_clip.log_wandb", True)
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
        keys=["num_val_episodes", "val_episode_ranges", "val_ratio"],
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
            "obs_dropout",
        ],
    )
    add_override(overrides, "model.config.action_dit_path", model.get("action_dit_path"))
    add_override(overrides, "model.config.video_dit_path", model.get("video_dit_path"))
    add_override(overrides, "model.config.pipe_config.xattn_layer_idx", model.get("xattn_layer_idx"))
    add_override(overrides, "model.config.pipe_config.action_target_mode", model.get("action_target_mode"))
    # The backbone runtime state_t lives on the video pipe config (the
    # action-decoder model.pipe_config is World2ActionPipelineConfig and has
    # no state_t). state_t must satisfy (state_t-1)*4 + 1 ==
    # obs.workspace_rgb.horizon + action.workspace_rgb.horizon. For SO101
    # (5+56=61) that's state_t=16.
    add_override(overrides, "model.config.video_pipe_config.state_t", model.get("video_pipe_state_t"))
    add_override(
        overrides,
        "model.config.video_pipe_config.guardrail_config.enabled",
        model.get("video_guardrail_enabled"),
    )
    add_mapped_overrides(
        overrides,
        action_scheduler,
        prefix="model.config.pipe_config.scheduler",
        keys=["alpha", "beta", "num_denoising_steps"],
    )
    add_mapped_overrides(overrides, ema, prefix="model.config.pipe_config.ema", keys=["enabled"])
    add_mapped_overrides(
        overrides,
        network,
        prefix="model.config.pipe_config.net",
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
    add_override(overrides, "model.config.pipe_config.net.sac_config.mode", network.get("sac_mode"))
    add_override(overrides, "model.config.pipe_config.net.sac_config.every_n_blocks", network.get("sac_every_n_blocks"))

    add_override(overrides, "trainer.callbacks.grad_clip.clip_norm", trainer.get("grad_clip_norm"))
    add_override(overrides, "trainer.callbacks.grad_clip.log_wandb", True)
    add_job_overrides(overrides, action_cfg)
    maybe_run_dryrun(overrides, optional_mapping(action_cfg, "launch"))
    return overrides


def run_training(config: dict[str, Any], revision_dir: Path) -> None:
    preprocessing = require_mapping(config, "preprocessing")
    training = require_mapping(config, "training")
    video_cfg = training.get("video_finetune") or {}
    action_cfg = training.get("action_decoder") or {}
    if not isinstance(video_cfg, dict) or not isinstance(action_cfg, dict):
        raise ValueError("training.video_finetune and training.action_decoder must be mappings.")

    seed_raw = config.get("seed")
    if seed_raw is not None and (not isinstance(seed_raw, int) or isinstance(seed_raw, bool)):
        raise ValueError(f"Top-level 'seed' must be an integer; got {seed_raw!r}.")
    seed: int | None = seed_raw

    if video_cfg.get("enabled", False):
        exp = resolve_video_experiment(str(video_cfg.get("experiment", "so101")))
        launch = optional_mapping(video_cfg, "launch")
        normalized_preprocessing = load_yaml(revision_dir / "preprocessing.yaml")
        eval_cfg = optional_mapping(video_cfg, "eval_video")
        episode_stem = eval_cfg.get("episode_stem")
        if isinstance(episode_stem, str) and episode_stem:
            instruction = source_for_stem(normalized_preprocessing, episode_stem)["instruction"]
        else:
            instruction = normalized_preprocessing_sources(normalized_preprocessing)[0]["instruction"]
        overrides = video_finetune_overrides(
            video_cfg,
            exp,
            revision_dir,
            instruction=instruction,
            seed=seed,
        )
        if not launch.get("dryrun", False):
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
        cache_video_backbone_into_run_dir(action_cfg, overrides, launch.get("output_root"))
        if not launch.get("dryrun", False):
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
        # Allow co-tenanted runs on the same node: torchrun's rendezvous server
        # always binds --master_port (default 29500), so a second launch needs
        # its own port. Read from $MASTER_PORT (no-op when unset).
        master_port = os.environ.get("MASTER_PORT")
        if master_port:
            cmd.insert(2, f"--master_port={master_port}")
    run_command(cmd, cwd=repo_root() / "model", env=train_env(output_root))


def preprocess_if_needed(config: dict[str, Any]) -> Path:
    raw_preprocessing = require_mapping(config, "preprocessing")
    text_encoder = raw_preprocessing.get("text_encoder") or {}
    if not isinstance(text_encoder, dict):
        raise ValueError("preprocessing.text_encoder must be a mapping.")

    sources = preprocessing_sources(raw_preprocessing)
    commit_shas = []
    for idx, source in enumerate(sources):
        validate_preprocessing_source(
            source,
            key="preprocessing.source" if "sources" not in raw_preprocessing else f"preprocessing.sources[{idx}]",
        )
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
        commit_shas.append(commit_sha)
    text_encoder_model_id = str(text_encoder.get("model_id", "google-t5/t5-large"))
    text_encoder_revision = str(text_encoder.get("revision", "main"))
    print(f"Resolving T5 model commit for {text_encoder_model_id}@{text_encoder_revision}.", flush=True)
    text_encoder_commit_sha = resolve_model_commit_sha(text_encoder_model_id, text_encoder_revision)
    print(f"Resolved T5 model commit: {text_encoder_commit_sha}.", flush=True)
    preprocessing = normalize_preprocessing(
        config,
        commit_shas=commit_shas,
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
