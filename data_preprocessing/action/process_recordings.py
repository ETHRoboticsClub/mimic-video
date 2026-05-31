from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from typing import Any

import cv2
import imageio.v3 as iio
import numpy as np
import zarr
from mcap.reader import make_reader
from numcodecs import Blosc
import tqdm


NS_PER_SEC = 1_000_000_000
CONVERTER_VERSION = "recordings_to_yams_zarr_v1"

CAMERA_FILES = {
    "workspace_rgb": ("camera_top", "camera_top-images-rgb.mp4", "camera_top-rgb-timestamp.npy"),
    "wrist_rgb_left": ("camera_left", "camera_left-images-rgb.mp4", "camera_left-rgb-timestamp.npy"),
    "wrist_rgb_right": ("camera_right", "camera_right-images-rgb.mp4", "camera_right-rgb-timestamp.npy"),
}

REQUIRED_FILES = [
    "camera_top-images-rgb.mp4",
    "camera_top-rgb-timestamp.npy",
    "camera_left-images-rgb.mp4",
    "camera_left-rgb-timestamp.npy",
    "camera_right-images-rgb.mp4",
    "camera_right-rgb-timestamp.npy",
    "yam_left.mcap",
    "yam_right.mcap",
    "session_meta.json",
]


@dataclass(frozen=True)
class AlignedEpisode:
    frames: dict[str, np.ndarray]
    timestamps_ns: np.ndarray
    joint_state: np.ndarray
    dropped_frames: int
    max_sync_error_ms: float


@dataclass(frozen=True)
class ConversionTask:
    episode_dir: pathlib.Path
    episode_index: int
    output_dir: pathlib.Path
    max_sync_ms: float
    overwrite: bool
    dry_run: bool


def discover_episodes(input_dir: pathlib.Path) -> list[pathlib.Path]:
    episodes = [
        p
        for p in input_dir.expanduser().rglob("episode_*")
        if p.is_dir() and (p / "session_meta.json").exists()
    ]
    return sorted(episodes)


def filter_episodes(episodes: list[pathlib.Path], selections: list[str] | None) -> list[tuple[int, pathlib.Path]]:
    indexed = list(enumerate(episodes))
    if not selections:
        return indexed

    by_name = {p.name: (idx, p) for idx, p in indexed}
    selected: list[tuple[int, pathlib.Path]] = []
    for item in selections:
        if item.isdigit():
            idx = int(item)
            if idx < 0 or idx >= len(episodes):
                raise ValueError(f"episode index {idx} is out of range 0..{len(episodes) - 1}")
            selected.append((idx, episodes[idx]))
        elif item in by_name:
            selected.append(by_name[item])
        else:
            raise ValueError(f"episode selection {item!r} did not match an index or episode directory name")
    return selected


def validate_required_files(episode_dir: pathlib.Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (episode_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{episode_dir.name}: missing required files: {missing}")


def load_session_instruction(episode_dir: pathlib.Path) -> str:
    meta_path = episode_dir / "session_meta.json"
    with meta_path.open("r", encoding="utf-8") as stream:
        meta = json.load(stream)

    instruction = meta.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{episode_dir.name}: session_meta.json must contain a non-empty 'instruction' string")
    return instruction.strip()


def load_camera_timestamps(episode_dir: pathlib.Path, camera: str) -> np.ndarray:
    path = episode_dir / f"{camera}-rgb-timestamp.npy"
    timestamps = np.load(path).astype(np.float64)
    _validate_timestamps(timestamps, f"{episode_dir.name}/{path.name}")
    return timestamps


def decode_video_frames(video_path: pathlib.Path, target_size: tuple[int, int] = (640, 480)) -> np.ndarray:
    frames = iio.imread(video_path)
    if frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError(f"{video_path}: expected RGB video frames with shape (T,H,W,3), got {frames.shape}")
    frames = frames[..., :3].astype(np.uint8, copy=False)
    return np.stack([_resize_rgb_frame(frame, target_size) for frame in frames], axis=0)


def read_mcap_joint_stream(mcap_path: pathlib.Path, topic: str) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    joints: list[np.ndarray] = []
    seen_topics: set[str] = set()

    with mcap_path.open("rb") as stream:
        reader = make_reader(stream)
        for schema, channel, message in reader.iter_messages():
            seen_topics.add(channel.topic)
            if channel.topic != topic and not channel.topic.endswith(f"/{topic}"):
                continue
            joint = _decode_joint_payload(message.data, schema, channel, mcap_path)
            timestamps.append(message.log_time / NS_PER_SEC)
            joints.append(joint)

    if not joints:
        raise ValueError(f"{mcap_path.name}: no messages found for topic {topic!r}; saw topics {sorted(seen_topics)}")

    ts = np.asarray(timestamps, dtype=np.float64)
    arr = np.stack(joints).astype(np.float32)
    _validate_timestamps(ts, f"{mcap_path.name}:{topic}")
    _validate_joints(arr, f"{mcap_path.name}:{topic}")
    return ts, arr


def align_episode_streams(
    episode_dir: pathlib.Path,
    *,
    max_sync_ms: float,
    joint_topic: str = "joint_state",
) -> AlignedEpisode:
    validate_required_files(episode_dir)
    camera_frames: dict[str, np.ndarray] = {}
    camera_timestamps: dict[str, np.ndarray] = {}

    for out_key, (camera_name, video_name, _ts_name) in CAMERA_FILES.items():
        frames = decode_video_frames(episode_dir / video_name)
        timestamps = load_camera_timestamps(episode_dir, camera_name)
        length = min(len(frames), len(timestamps))
        if length == 0:
            raise ValueError(f"{episode_dir.name}/{camera_name}: empty video or timestamp stream")
        if len(frames) != len(timestamps):
            print(
                f"WARN {episode_dir.name}/{camera_name}: video has {len(frames)} frames but "
                f"timestamps has {len(timestamps)}; truncating to {length}"
            )
        camera_frames[out_key] = frames[:length]
        camera_timestamps[out_key] = timestamps[:length]

    left_ts, left_joints = read_mcap_joint_stream(episode_dir / "yam_left.mcap", joint_topic)
    right_ts, right_joints = read_mcap_joint_stream(episode_dir / "yam_right.mcap", joint_topic)

    anchor_ts = camera_timestamps["workspace_rgb"]
    max_sync_sec = max_sync_ms / 1000.0
    kept_anchor_indices: list[int] = []
    selected_indices: dict[str, list[int]] = {"wrist_rgb_left": [], "wrist_rgb_right": []}
    left_joint_indices: list[int] = []
    right_joint_indices: list[int] = []
    sync_errors: list[float] = []

    for anchor_idx, ts in enumerate(anchor_ts):
        matches = {
            "wrist_rgb_left": _nearest_index(camera_timestamps["wrist_rgb_left"], ts),
            "wrist_rgb_right": _nearest_index(camera_timestamps["wrist_rgb_right"], ts),
            "yam_left": _nearest_index(left_ts, ts),
            "yam_right": _nearest_index(right_ts, ts),
        }
        errors = {
            "wrist_rgb_left": abs(camera_timestamps["wrist_rgb_left"][matches["wrist_rgb_left"]] - ts),
            "wrist_rgb_right": abs(camera_timestamps["wrist_rgb_right"][matches["wrist_rgb_right"]] - ts),
            "yam_left": abs(left_ts[matches["yam_left"]] - ts),
            "yam_right": abs(right_ts[matches["yam_right"]] - ts),
        }
        if any(error > max_sync_sec for error in errors.values()):
            continue
        kept_anchor_indices.append(anchor_idx)
        selected_indices["wrist_rgb_left"].append(matches["wrist_rgb_left"])
        selected_indices["wrist_rgb_right"].append(matches["wrist_rgb_right"])
        left_joint_indices.append(matches["yam_left"])
        right_joint_indices.append(matches["yam_right"])
        sync_errors.extend(errors.values())

    if not kept_anchor_indices:
        raise ValueError(f"{episode_dir.name}: no frames remained after {max_sync_ms:g}ms sync threshold")

    kept = np.asarray(kept_anchor_indices, dtype=np.int64)
    aligned_frames = {
        "workspace_rgb": camera_frames["workspace_rgb"][kept],
        "wrist_rgb_left": camera_frames["wrist_rgb_left"][np.asarray(selected_indices["wrist_rgb_left"], dtype=np.int64)],
        "wrist_rgb_right": camera_frames["wrist_rgb_right"][np.asarray(selected_indices["wrist_rgb_right"], dtype=np.int64)],
    }
    joint_state = np.concatenate(
        [
            left_joints[np.asarray(left_joint_indices, dtype=np.int64)],
            right_joints[np.asarray(right_joint_indices, dtype=np.int64)],
        ],
        axis=1,
    ).astype(np.float32)
    _validate_joints(joint_state, f"{episode_dir.name}:joint_state_lowdim", expected_dim=14)

    selected_anchor_ts = anchor_ts[kept]
    rel_ts = selected_anchor_ts - selected_anchor_ts[0]
    timestamps_ns = np.rint(rel_ts * NS_PER_SEC).astype(np.uint64)
    _validate_timestamps_allow_first_zero(timestamps_ns, f"{episode_dir.name}:aligned timestamps")

    return AlignedEpisode(
        frames=aligned_frames,
        timestamps_ns=timestamps_ns,
        joint_state=joint_state,
        dropped_frames=len(anchor_ts) - len(kept_anchor_indices),
        max_sync_error_ms=(max(sync_errors) * 1000.0 if sync_errors else 0.0),
    )


def write_episode_zarr(
    out_path: pathlib.Path,
    aligned: AlignedEpisode,
    *,
    source_episode: pathlib.Path,
    instruction: str,
    max_sync_ms: float,
    overwrite: bool,
) -> None:
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists; pass --overwrite to rewrite it")

    comp = Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)
    length = len(aligned.timestamps_ns)
    t_img = min(65, length)
    t_ld = min(1024, length)
    root = zarr.open(str(out_path), mode="w")
    root.attrs.update(
        {
            "converter_version": CONVERTER_VERSION,
            "source_episode": str(source_episode),
            "camera_mapping": {
                "workspace_rgb": "camera_top",
                "wrist_rgb_left": "camera_left",
                "wrist_rgb_right": "camera_right",
            },
            "joint_mapping": "yam_left_then_yam_right",
            "sync_method": "nearest_neighbor",
            "instruction": instruction,
            "max_sync_ms": max_sync_ms,
        }
    )

    root.create_dataset(
        "language_instruction",
        shape=(1,),
        dtype=bytes,
        chunks=(1,),
        compressor=comp,
        overwrite=True,
    )[...] = np.array([instruction.encode("utf-8")])
    root.create_dataset(
        "language_instruction_timestamps",
        shape=(1,),
        dtype="uint64",
        chunks=(1,),
        compressor=comp,
        overwrite=True,
    )[...] = np.array([0], dtype=np.uint64)

    for key, frames in aligned.frames.items():
        root.create_dataset(
            key,
            shape=frames.shape,
            dtype=np.uint8,
            chunks=(t_img, *frames.shape[1:]),
            compressor=comp,
            overwrite=True,
        )[...] = frames
        root.create_dataset(
            f"{key}_timestamps",
            shape=(length,),
            dtype="uint64",
            chunks=(length,),
            overwrite=True,
        )[...] = aligned.timestamps_ns

    root.create_dataset(
        "joint_state_lowdim",
        shape=aligned.joint_state.shape,
        dtype=np.float32,
        chunks=(t_ld, aligned.joint_state.shape[1]),
        compressor=comp,
        overwrite=True,
    )[...] = aligned.joint_state
    root.create_dataset(
        "joint_state_lowdim_timestamps",
        shape=(length,),
        dtype="uint64",
        chunks=(length,),
        overwrite=True,
    )[...] = aligned.timestamps_ns


def convert_episode(task: ConversionTask) -> str:
    out_path = task.output_dir / f"episode_{task.episode_index:04d}.zarr"
    if out_path.exists() and not task.overwrite:
        return f"skip (exists): {out_path}"

    validate_required_files(task.episode_dir)
    instruction = load_session_instruction(task.episode_dir)
    if task.dry_run:
        return f"dry-run ok: {task.episode_dir.name} -> {out_path.name}"

    aligned = align_episode_streams(task.episode_dir, max_sync_ms=task.max_sync_ms)
    write_episode_zarr(
        out_path,
        aligned,
        source_episode=task.episode_dir,
        instruction=instruction,
        max_sync_ms=task.max_sync_ms,
        overwrite=task.overwrite,
    )
    duration = aligned.timestamps_ns[-1] / NS_PER_SEC if len(aligned.timestamps_ns) else 0.0
    return (
        f"ok: {task.episode_dir.name} -> {out_path.name} "
        f"(T={len(aligned.timestamps_ns)}, duration={duration:.2f}s, "
        f"dropped={aligned.dropped_frames}, max_sync={aligned.max_sync_error_ms:.1f}ms)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw teleop recordings to lerobot_bi_yams zarr episodes.")
    parser.add_argument("--input-dir", type=pathlib.Path, required=True, help="Root containing date/episode_* folders.")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="Directory to write episode_*.zarr files.")
    parser.add_argument("--max-sync-ms", type=float, default=50.0, help="Maximum nearest-neighbor sync gap in ms.")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--episodes", nargs="*", default=None, help="Optional episode indices or directory names.")
    args = parser.parse_args()

    if args.max_sync_ms <= 0:
        raise ValueError("--max-sync-ms must be positive")

    episodes = discover_episodes(args.input_dir)
    selected = filter_episodes(episodes, args.episodes)
    if not selected:
        raise ValueError(f"no episode_* directories with session_meta.json found under {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        ConversionTask(
            episode_dir=episode_dir,
            episode_index=episode_index,
            output_dir=args.output_dir,
            max_sync_ms=args.max_sync_ms,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        for episode_index, episode_dir in selected
    ]

    if args.num_workers <= 1:
        for task in tqdm.tqdm(tasks, desc="recordings -> zarr"):
            print(convert_episode(task))
    else:
        with Pool(processes=args.num_workers) as pool:
            for msg in tqdm.tqdm(pool.imap_unordered(convert_episode, tasks), total=len(tasks), desc="recordings -> zarr"):
                print(msg)


def _resize_rgb_frame(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = target_size
    h, w = frame.shape[:2]
    if (w, h) != target_size:
        frame = _center_crop_to_aspect(frame, target_w / target_h)
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8, copy=False)


def _center_crop_to_aspect(frame: np.ndarray, target_aspect: float) -> np.ndarray:
    h, w = frame.shape[:2]
    aspect = w / h
    if math.isclose(aspect, target_aspect, rel_tol=1e-3):
        return frame
    if aspect > target_aspect:
        new_w = int(round(h * target_aspect))
        x0 = max((w - new_w) // 2, 0)
        return frame[:, x0 : x0 + new_w]
    new_h = int(round(w / target_aspect))
    y0 = max((h - new_h) // 2, 0)
    return frame[y0 : y0 + new_h, :]


def _nearest_index(values: np.ndarray, target: float) -> int:
    idx = int(np.searchsorted(values, target))
    if idx <= 0:
        return 0
    if idx >= len(values):
        return len(values) - 1
    before = idx - 1
    after = idx
    return before if abs(values[before] - target) <= abs(values[after] - target) else after


def _decode_joint_payload(data: bytes, schema: Any, channel: Any, mcap_path: pathlib.Path) -> np.ndarray:
    for decoder in (_decode_json_joint_payload, _decode_raw_float_payload):
        joint = decoder(data)
        if joint is not None:
            return joint

    schema_name = getattr(schema, "name", None)
    schema_encoding = getattr(schema, "encoding", None)
    message_encoding = getattr(channel, "message_encoding", None)
    raise ValueError(
        f"{mcap_path.name}:{channel.topic}: unsupported MCAP payload "
        f"(schema={schema_name!r}, schema_encoding={schema_encoding!r}, "
        f"message_encoding={message_encoding!r}, bytes={len(data)})"
    )


def _decode_json_joint_payload(data: bytes) -> np.ndarray | None:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    values = _find_numeric_vector(payload)
    if values is None:
        return None
    return np.asarray(values, dtype=np.float32)


def _decode_raw_float_payload(data: bytes) -> np.ndarray | None:
    for dtype in (np.float32, np.float64):
        width = np.dtype(dtype).itemsize
        if len(data) == 7 * width:
            arr = np.frombuffer(data, dtype=dtype).astype(np.float32)
            if np.all(np.isfinite(arr)):
                return arr
    return None


def _find_numeric_vector(value: Any) -> list[float] | None:
    preferred_keys = ("position", "positions", "joint_pos", "joint_position", "joint_positions", "data")
    if isinstance(value, dict):
        if "joint_pos" in value and "gripper_pos" in value:
            arm = value["joint_pos"]
            gripper = value["gripper_pos"]
            if (
                isinstance(arm, (list, tuple))
                and isinstance(gripper, (list, tuple))
                and len(arm) == 6
                and len(gripper) == 1
                and all(isinstance(item, (int, float)) for item in [*arm, *gripper])
            ):
                return [float(item) for item in [*arm, *gripper]]
        for key in preferred_keys:
            if key in value:
                found = _find_numeric_vector(value[key])
                if found is not None:
                    return found
        for nested in value.values():
            found = _find_numeric_vector(nested)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        if len(value) == 7 and all(isinstance(item, (int, float)) for item in value):
            return [float(item) for item in value]
        for nested in value:
            found = _find_numeric_vector(nested)
            if found is not None:
                return found
    return None


def _validate_timestamps(timestamps: np.ndarray, label: str) -> None:
    if timestamps.ndim != 1 or len(timestamps) == 0:
        raise ValueError(f"{label}: expected a non-empty 1D timestamp array, got shape {timestamps.shape}")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError(f"{label}: timestamps contain NaN or inf")
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
        raise ValueError(f"{label}: timestamps must be strictly increasing")


def _validate_timestamps_allow_first_zero(timestamps: np.ndarray, label: str) -> None:
    if timestamps.ndim != 1 or len(timestamps) == 0:
        raise ValueError(f"{label}: expected a non-empty 1D timestamp array, got shape {timestamps.shape}")
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
        raise ValueError(f"{label}: timestamps must be strictly increasing")


def _validate_joints(joints: np.ndarray, label: str, expected_dim: int = 7) -> None:
    if joints.ndim != 2 or joints.shape[1] != expected_dim:
        raise ValueError(f"{label}: expected joint array shape (T,{expected_dim}), got {joints.shape}")
    if not np.all(np.isfinite(joints)):
        raise ValueError(f"{label}: joint array contains NaN or inf")


if __name__ == "__main__":
    main()
