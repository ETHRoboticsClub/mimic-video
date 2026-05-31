from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import imageio.v3 as iio
import numpy as np
import zarr as zarr_pkg
from mcap.writer import Writer


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))


_N = 5
_FPS = 10
_INSTRUCTION = "fold the towel"


def _frames(n: int, *, height: int, width: int, offset: int = 0) -> np.ndarray:
    arr = np.zeros((n, height, width, 3), dtype=np.uint8)
    for i in range(n):
        arr[i, :, :, 0] = (i * 11 + offset) % 255
        arr[i, :, :, 1] = np.arange(width, dtype=np.uint16)[None, :] % 255
        arr[i, :, :, 2] = np.arange(height, dtype=np.uint16)[:, None] % 255
    return arr


def _timestamps(n: int, *, start: float = 100.0, dt: float = 0.1, offset: float = 0.0) -> np.ndarray:
    return start + offset + np.arange(n, dtype=np.float64) * dt


def _joint_values(n: int, *, base: float) -> np.ndarray:
    return (base + np.arange(n * 7, dtype=np.float32).reshape(n, 7) / 100.0).astype(np.float32)


def _write_mp4(path: pathlib.Path, frames: np.ndarray) -> None:
    iio.imwrite(path, frames, fps=_FPS)


def _write_joint_mcap(path: pathlib.Path, timestamps: np.ndarray, joints: np.ndarray) -> None:
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.start()
        schema_id = writer.register_schema(
            name="JointStateJson",
            encoding="jsonschema",
            data=json.dumps({"type": "object", "properties": {"position": {"type": "array"}}}).encode("utf-8"),
        )
        channel_id = writer.register_channel(
            topic="joint_state",
            message_encoding="json",
            schema_id=schema_id,
        )
        for ts, joint in zip(timestamps, joints):
            ts_ns = int(round(ts * 1_000_000_000))
            writer.add_message(
                channel_id=channel_id,
                log_time=ts_ns,
                publish_time=ts_ns,
                data=json.dumps({"joint_pos": joint[:6].tolist(), "gripper_pos": joint[6:].tolist()}).encode("utf-8"),
            )
        writer.finish()


def create_recordings_fixture(
    tmp_path: pathlib.Path,
    *,
    n_top: int = _N,
    n_left: int = _N,
    n_right: int = _N,
    right_offset: float = 0.0,
    include_all_files: bool = True,
) -> dict:
    episode_dir = tmp_path / "recordings" / "20260531" / "episode_test"
    episode_dir.mkdir(parents=True)

    top_ts = _timestamps(n_top)
    left_ts = _timestamps(n_left, offset=0.01)
    right_ts = _timestamps(n_right, offset=right_offset)
    joint_ts = _timestamps(max(n_top, n_left, n_right), offset=0.005)
    left_joints = _joint_values(len(joint_ts), base=1.0)
    right_joints = _joint_values(len(joint_ts), base=2.0)

    _write_mp4(episode_dir / "camera_top-images-rgb.mp4", _frames(n_top, height=720, width=1280, offset=3))
    _write_mp4(episode_dir / "camera_left-images-rgb.mp4", _frames(n_left, height=480, width=640, offset=7))
    _write_mp4(episode_dir / "camera_right-images-rgb.mp4", _frames(n_right, height=480, width=640, offset=13))
    np.save(episode_dir / "camera_top-rgb-timestamp.npy", top_ts)
    np.save(episode_dir / "camera_left-rgb-timestamp.npy", left_ts)
    np.save(episode_dir / "camera_right-rgb-timestamp.npy", right_ts)
    _write_joint_mcap(episode_dir / "yam_left.mcap", joint_ts, left_joints)
    _write_joint_mcap(episode_dir / "yam_right.mcap", joint_ts, right_joints)
    (episode_dir / "session_meta.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {"name": "yam_left", "published_topics": ["joint_state"]},
                    {"name": "yam_right", "published_topics": ["joint_state"]},
                ],
                "record_topic": "gello_left/record",
                "instruction": _INSTRUCTION,
            }
        )
    )

    if not include_all_files:
        (episode_dir / "yam_right.mcap").unlink()

    return {
        "episode_dir": episode_dir,
        "top_ts": top_ts,
        "joint_ts": joint_ts,
        "left_joints": left_joints,
        "right_joints": right_joints,
    }


def test_happy_path_schema_shapes_language_and_attrs(tmp_path: pathlib.Path):
    fixture = create_recordings_fixture(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    from data_preprocessing.action.process_recordings import ConversionTask, convert_episode

    msg = convert_episode(
        ConversionTask(
            episode_dir=fixture["episode_dir"],
            episode_index=0,
            output_dir=out_dir,
            max_sync_ms=50,
            overwrite=True,
            dry_run=False,
        )
    )

    assert msg.startswith("ok:")
    z = zarr_pkg.open(str(out_dir / "episode_0000.zarr"))
    for key in (
        "language_instruction",
        "language_instruction_timestamps",
        "workspace_rgb",
        "workspace_rgb_timestamps",
        "wrist_rgb_left",
        "wrist_rgb_left_timestamps",
        "wrist_rgb_right",
        "wrist_rgb_right_timestamps",
        "joint_state_lowdim",
        "joint_state_lowdim_timestamps",
    ):
        assert key in z

    assert z["workspace_rgb"].shape == (_N, 480, 640, 3)
    assert z["wrist_rgb_left"].shape == (_N, 480, 640, 3)
    assert z["wrist_rgb_right"].shape == (_N, 480, 640, 3)
    assert z["joint_state_lowdim"].shape == (_N, 14)
    assert z["workspace_rgb"].dtype == np.uint8
    assert z["joint_state_lowdim"].dtype == np.float32
    assert z["language_instruction"][0].decode("utf-8") == _INSTRUCTION
    assert z.attrs["joint_mapping"] == "yam_left_then_yam_right"
    assert z.attrs["sync_method"] == "nearest_neighbor"


def test_joint_streams_concatenate_left_then_right(tmp_path: pathlib.Path):
    fixture = create_recordings_fixture(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    from data_preprocessing.action.process_recordings import ConversionTask, convert_episode

    convert_episode(
        ConversionTask(fixture["episode_dir"], 0, out_dir, 50, True, False)
    )
    z = zarr_pkg.open(str(out_dir / "episode_0000.zarr"))
    expected = np.concatenate([fixture["left_joints"][:_N], fixture["right_joints"][:_N]], axis=1)
    np.testing.assert_allclose(z["joint_state_lowdim"][:], expected, rtol=0, atol=1e-6)


def test_frame_count_mismatch_aligns_to_top_timeline(tmp_path: pathlib.Path):
    fixture = create_recordings_fixture(tmp_path, n_top=5, n_left=6, n_right=7)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    from data_preprocessing.action.process_recordings import ConversionTask, convert_episode

    msg = convert_episode(
        ConversionTask(fixture["episode_dir"], 0, out_dir, 50, True, False)
    )
    assert "(T=5" in msg
    z = zarr_pkg.open(str(out_dir / "episode_0000.zarr"))
    for key in ("workspace_rgb", "wrist_rgb_left", "wrist_rgb_right", "joint_state_lowdim"):
        assert z[key].shape[0] == 5


def test_sync_threshold_drops_far_frames(tmp_path: pathlib.Path):
    fixture = create_recordings_fixture(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    from data_preprocessing.action.process_recordings import ConversionTask, convert_episode

    try:
        convert_episode(ConversionTask(fixture["episode_dir"], 0, out_dir, 1, True, False))
    except ValueError as exc:
        assert "no frames remained" in str(exc)
    else:
        raise AssertionError("expected sync threshold failure")


def test_missing_required_files_fail_clearly(tmp_path: pathlib.Path):
    fixture = create_recordings_fixture(tmp_path, include_all_files=False)

    from data_preprocessing.action.process_recordings import validate_required_files

    try:
        validate_required_files(fixture["episode_dir"])
    except FileNotFoundError as exc:
        assert "yam_right.mcap" in str(exc)
    else:
        raise AssertionError("expected missing file error")


def test_missing_session_instruction_fails_clearly(tmp_path: pathlib.Path):
    fixture = create_recordings_fixture(tmp_path)
    meta_path = fixture["episode_dir"] / "session_meta.json"
    meta = json.loads(meta_path.read_text())
    del meta["instruction"]
    meta_path.write_text(json.dumps(meta))

    from data_preprocessing.action.process_recordings import load_session_instruction

    try:
        load_session_instruction(fixture["episode_dir"])
    except ValueError as exc:
        assert "instruction" in str(exc)
    else:
        raise AssertionError("expected missing instruction error")


def test_skip_overwrite_and_dry_run(tmp_path: pathlib.Path):
    fixture = create_recordings_fixture(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    from data_preprocessing.action.process_recordings import ConversionTask, convert_episode

    task = ConversionTask(fixture["episode_dir"], 0, out_dir, 50, True, False)
    assert convert_episode(task).startswith("ok:")
    skip_task = ConversionTask(fixture["episode_dir"], 0, out_dir, 50, False, False)
    assert convert_episode(skip_task).startswith("skip (exists):")

    dry_out = tmp_path / "dry"
    dry_out.mkdir()
    dry_task = ConversionTask(fixture["episode_dir"], 0, dry_out, 50, False, True)
    assert convert_episode(dry_task).startswith("dry-run ok:")
    assert not (dry_out / "episode_0000.zarr").exists()


def test_cli_smoke(tmp_path: pathlib.Path):
    fixture = create_recordings_fixture(tmp_path)
    out_dir = tmp_path / "cli_out"
    script = pathlib.Path(__file__).resolve().parent.parent / "process_recordings.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input-dir",
            str(fixture["episode_dir"].parents[1]),
            "--output-dir",
            str(out_dir),
            "--episodes",
            "0",
            "--overwrite",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.stdout:
        print("[cli stdout]", proc.stdout[:1000])
    if proc.stderr:
        print("[cli stderr]", proc.stderr[:1000])
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "episode_0000.zarr").exists()


def test_dry_run_cli_creates_no_zarr(tmp_path: pathlib.Path):
    fixture = create_recordings_fixture(tmp_path)
    out_dir = tmp_path / "dry_cli_out"
    script = pathlib.Path(__file__).resolve().parent.parent / "process_recordings.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input-dir",
            str(fixture["episode_dir"].parents[1]),
            "--output-dir",
            str(out_dir),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "dry-run ok:" in proc.stdout
    assert not list(out_dir.glob("*.zarr"))
