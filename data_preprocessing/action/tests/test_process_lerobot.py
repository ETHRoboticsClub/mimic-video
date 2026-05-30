"""Helpers and smoke tests for LeRobot-style synthetic dataset fixtures.

All helpers are internal to this file; no separate fixture package is created.
The fixture lives only inside pytest's tmp_path (which is auto-removed).
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zarr as zarr_pkg

# Re­quired: ad­d pro­ject root to sys.path for rel­a­tive im­ports
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))


# ── defaults ──────────────────────────────────────────────────────────
_DEFAULT_N = 5
_DEFAULT_FPS = 10
_DEFAULT_EPISODE_INDEX = 0
_DEFAULT_TASK_TEXT = "pick up the cube"
_DEFAULT_STATE_SHAPE = (_DEFAULT_N, 14)
_DEFAULT_FRAME_SHAPE = (_DEFAULT_N, 480, 640, 3)

CAMERAS = ("observation.images.topdown", "observation.images.left_wrist", "observation.images.right_wrist")


def _deterministic_state(n: int) -> np.ndarray:
    """Return a float32 array of shape (n, 14) with reproducible values."""
    rng = np.random.RandomState(42)
    return rng.randn(n, 14).astype(np.float32)


def _deterministic_frames(n: int) -> np.ndarray:
    """Return uint8 frames of shape (n, 480, 640, 3) with reproducible values."""
    rng = np.random.RandomState(123)
    return rng.randint(0, 256, size=(n, 480, 640, 3), dtype=np.uint8)


def _write_mp4(path: pathlib.Path, frames: np.ndarray):
    """Write frames (T, H, W, 3) uint8 to an mp4 using imageio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(path), frames, fps=_DEFAULT_FPS)


def _write_data_parquet(path: pathlib.Path, episode_index: int, n: int):
    """Write the data/chunk-{c}/file-{f}.parquet that process_lerobot reads at lines 113-129."""
    path.parent.mkdir(parents=True, exist_ok=True)
    states = _deterministic_state(n)
    # observation.state must be a fixed_size_list[14] — pyarrow builds this
    # from a flat Float32Array of size (n * 14).
    flat_values = pa.array(states.ravel(), type=pa.float32())
    state_column = pa.FixedSizeListArray.from_arrays(flat_values, 14)

    table = pa.table({
        "episode_index": pa.array([episode_index] * n, type=pa.int64()),
        "frame_index": pa.array(list(range(n)), type=pa.int64()),
        "observation.state": state_column,
    })
    pq.write_table(table, str(path))


def _write_tasks_parquet(path: pathlib.Path, task_text: str, task_index: int = 0):
    """Write meta/tasks.parquet with task text as index and task_index as column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"task_index": [task_index]}, index=pd.Index([task_text], name="task"))
    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(path))


def _write_episodes_parquet(path: pathlib.Path, *,
                            episode_index: int,
                            task_text: str,
                            n: int,
                            fps: int):
    """Write meta/episodes/file-000.parquet with every column required by _load_episodes_meta (lines 17-41)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "episode_index": [episode_index],
        "tasks": [[task_text]],  # stored as list of task strings per episode
        "length": [n],
        # data location
        "data/chunk_index": [0],
        "data/file_index": [0],
        "dataset_from_index": [0],
        "dataset_to_index": [n - 1],
        # topdown video
        "videos/observation.images.topdown/chunk_index": [0],
        "videos/observation.images.topdown/file_index": [0],
        "videos/observation.images.topdown/from_timestamp": [0.0],
        "videos/observation.images.topdown/to_timestamp": [n / fps],
        # left_wrist video
        "videos/observation.images.left_wrist/chunk_index": [0],
        "videos/observation.images.left_wrist/file_index": [0],
        "videos/observation.images.left_wrist/from_timestamp": [0.0],
        "videos/observation.images.left_wrist/to_timestamp": [n / fps],
        # right_wrist video
        "videos/observation.images.right_wrist/chunk_index": [0],
        "videos/observation.images.right_wrist/file_index": [0],
        "videos/observation.images.right_wrist/from_timestamp": [0.0],
        "videos/observation.images.right_wrist/to_timestamp": [n / fps],
    }

    table = pa.table(row)
    pq.write_table(table, str(path))


# ── public fixture builder ────────────────────────────────────────────

def create_synthetic_fixture(tmp_path: pathlib.Path, N: int = _DEFAULT_N, fps: int = _DEFAULT_FPS
                             ) -> dict:
    """Generate a minimal LeRobot-style dataset under *tmp_path*.

    Creates the following layout exactly as process_lerobot.py expects:

      meta/tasks.parquet
      meta/episodes/file-000.parquet
      data/chunk-000/file-000.parquet
      videos/observation.images.topdown/chunk-000/file-000.mp4
      videos/observation.images.left_wrist/chunk-000/file-000.mp4
      videos/observation.images.right_wrist/chunk-000/file-000.mp4

    Returns a dict with source arrays so downstream tests can assert on them:

      {"states": np.ndarray(N, 14),
       "workspace_rgb": np.ndarray(N, 480, 640, 3),
       "wrist_rgb_left": np.ndarray(N, 480, 640, 3),
       "wrist_rgb_right": np.ndarray(N, 480, 640, 3)}
    """
    ep_idx = _DEFAULT_EPISODE_INDEX
    task_text = _DEFAULT_TASK_TEXT

    # deterministic sources
    states = _deterministic_state(N)
    frames_topdown = _deterministic_frames(N)
    frames_left = _deterministic_frames(N)
    frames_right = _deterministic_frames(N)

    # 1. meta/tasks.parquet
    _write_tasks_parquet(tmp_path / "meta" / "tasks.parquet", task_text, task_index=0)

    # 2. meta/episodes/file-000.parquet
    _write_episodes_parquet(tmp_path / "meta" / "episodes" / "file-000.parquet",
                            episode_index=ep_idx, task_text=task_text, n=N, fps=fps)

    # 3. data/chunk-000/file-000.parquet
    _write_data_parquet(tmp_path / "data" / "chunk-000" / "file-000.parquet",
                        episode_index=ep_idx, n=N)

    # 4. video mp4s (one per camera)
    _write_mp4(tmp_path / "videos" / "observation.images.topdown" / "chunk-000" / "file-000.mp4",
               frames_topdown)
    _write_mp4(tmp_path / "videos" / "observation.images.left_wrist" / "chunk-000" / "file-000.mp4",
               frames_left)
    _write_mp4(tmp_path / "videos" / "observation.images.right_wrist" / "chunk-000" / "file-000.mp4",
               frames_right)

    return {
        "states": states,
        "workspace_rgb": frames_topdown,
        "wrist_rgb_left": frames_left,
        "wrist_rgb_right": frames_right,
    }


# ── tests ─────────────────────────────────────────────────────────────

def test_fixture_smoke(tmp_path: pathlib.Path):
    """Smoke test: assert every required file exists after create_synthetic_fixture."""
    expected_files = [
        "meta/tasks.parquet",
        "meta/episodes/file-000.parquet",
        "data/chunk-000/file-000.parquet",
        "videos/observation.images.topdown/chunk-000/file-000.mp4",
        "videos/observation.images.left_wrist/chunk-000/file-000.mp4",
        "videos/observation.images.right_wrist/chunk-000/file-000.mp4",
    ]

    source_arrays = create_synthetic_fixture(tmp_path)

    for rel in expected_files:
        p = tmp_path / rel
        assert p.exists(), f"missing: {rel}  (looked at {p})"
        assert p.stat().st_size > 0, f"empty file: {rel}"

    # verify returned arrays have expected shapes
    assert source_arrays["states"].shape == (_DEFAULT_N, 14)
    for key in ("workspace_rgb", "wrist_rgb_left", "wrist_rgb_right"):
        assert source_arrays[key].shape == _DEFAULT_FRAME_SHAPE, f"{key} shape mismatch"

    # verify parquet content: data parquet has correct columns
    data_table = pq.read_table(str(tmp_path / "data" / "chunk-000" / "file-000.parquet"))
    assert set(data_table.column_names) >= {"episode_index", "frame_index", "observation.state"}
    data_df = data_table.to_pandas()
    assert len(data_df) == _DEFAULT_N
    assert data_df["episode_index"].iloc[0] == _DEFAULT_EPISODE_INDEX
    assert list(data_df["frame_index"]) == list(range(_DEFAULT_N))

    # verify episode meta parquet has all columns from _load_episodes_meta keep list
    episode_table = pq.read_table(str(tmp_path / "meta" / "episodes" / "file-000.parquet"))
    keep_cols = [
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
    for col in keep_cols:
        assert col in episode_table.column_names, f"missing column in episodes parquet: {col}"


# ── Wave 2B Task 6: skip/overwrite, CLI smoke, truncation edge ───────

def test_skip_overwrite(tmp_path: pathlib.Path):
    """Verify _convert skips when output exists and overwrite=False, then overwrites when True."""
    # build a minimal fixture + output dir
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    create_synthetic_fixture(tmp_path, N=_DEFAULT_N)

    # import _convert from the module under test
    from data_preprocessing.action.process_lerobot import _convert, _load_episodes_meta, _load_tasks

    ep_df = _load_episodes_meta(tmp_path)
    tasks = _load_tasks(tmp_path)
    ep_row = ep_df.iloc[0]

    # first run with overwrite=True → should succeed
    result1 = _convert(ep_row, dataset_dir=tmp_path, out_dir=out_dir,
                       fps=_DEFAULT_FPS, overwrite=True, tasks=tasks)
    assert result1.startswith("ok:"), f"first run should succeed, got: {result1!r}"

    # second run with overwrite=False → should skip
    result2 = _convert(ep_row, dataset_dir=tmp_path, out_dir=out_dir,
                       fps=_DEFAULT_FPS, overwrite=False, tasks=tasks)
    assert result2.startswith("skip (exists):"), f"should skip, got: {result2!r}"

    # third run with overwrite=True → should succeed again
    result3 = _convert(ep_row, dataset_dir=tmp_path, out_dir=out_dir,
                       fps=_DEFAULT_FPS, overwrite=True, tasks=tasks)
    assert result3.startswith("ok:"), f"re-overwrite should succeed, got: {result3!r}"


def test_cli_smoke(tmp_path: pathlib.Path):
    """Run process_lerobot.py via subprocess; assert exit code 0 and episode_0000.zarr created."""
    out_dir = tmp_path / "cli_output"
    create_synthetic_fixture(tmp_path, N=_DEFAULT_N)

    script = pathlib.Path(__file__).resolve().parent.parent / "process_lerobot.py"

    proc = subprocess.run(
        [
            sys.executable, str(script),
            "--dataset-path", str(tmp_path),
            "--output-dir", str(out_dir),
            "--num-workers", "1",
            "--fps", str(_DEFAULT_FPS),
            "--overwrite",
            "--episodes", "0",
        ],
        capture_output=True, text=True, timeout=120,
    )
    # print stdout/stderr for debugging only — never assert on tqdm formatting
    if proc.stdout:
        print("[cli stdout]", proc.stdout[:1000])
    if proc.stderr:
        print("[cli stderr]", proc.stderr[:1000])

    assert proc.returncode == 0, f"CLI failed (rc={proc.returncode}): {proc.stderr}"
    zarr_path = out_dir / "episode_0000.zarr"
    assert zarr_path.exists(), f"expected {zarr_path} after CLI run"


def test_truncation_edge(tmp_path: pathlib.Path):
    """One camera video shorter than metadata length → converted zarr uses the shortest length."""
    N = 5  # metadata & parquet claim 5 frames
    short_n = 3  # one camera MP4 only has 3 frames
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    # create full fixture with N=5
    create_synthetic_fixture(tmp_path, N=N)

    # overwrite left_wrist MP4 with only 3 frames
    short_frames = _deterministic_frames(short_n)
    left_mp4 = tmp_path / "videos" / "observation.images.left_wrist" / "chunk-000" / "file-000.mp4"
    _write_mp4(left_mp4, short_frames)

    # convert
    from data_preprocessing.action.process_lerobot import _convert, _load_episodes_meta, _load_tasks

    ep_df = _load_episodes_meta(tmp_path)
    tasks = _load_tasks(tmp_path)
    ep_row = ep_df.iloc[0]

    result = _convert(ep_row, dataset_dir=tmp_path, out_dir=out_dir,
                      fps=_DEFAULT_FPS, overwrite=True, tasks=tasks)
    # should report T=3 (truncated to min video length)
    assert "T=3" in result, f"expected truncation to T=3 in result: {result!r}"

    # open the zarr and verify all arrays use effective length = 3
    z = zarr_pkg.open(str(out_dir / "episode_0000.zarr"))
    for key in ("workspace_rgb", "wrist_rgb_left", "wrist_rgb_right"):
        assert z[key].shape[0] == short_n, f"{key}: expected shape ({short_n},…), got {z[key].shape}"

    # timestamps and joint_state also truncated
    for ts_key in ("workspace_rgb_timestamps", "wrist_rgb_left_timestamps", "wrist_rgb_right_timestamps"):
        assert z[ts_key].shape[0] == short_n, f"{ts_key}: expected length {short_n}, got {z[ts_key].shape[0]}"

    assert z["joint_state_lowdim"].shape[0] == short_n
    assert z["joint_state_lowdim_timestamps"].shape[0] == short_n


# ── Wave 2B Task 3: happy-path + schema regression ────────────────────

def test_happy_path(tmp_path: pathlib.Path):
    """Full happy-path: _convert on synthetic fixture → ZARR with correct schema/shapes."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    # 1. build synthetic fixture (N=5, fps=10)
    create_synthetic_fixture(tmp_path, N=_DEFAULT_N)

    # 2. load episode meta and tasks
    from data_preprocessing.action.process_lerobot import _convert, _load_episodes_meta, _load_tasks

    ep_df = _load_episodes_meta(tmp_path)
    tasks = _load_tasks(tmp_path)
    ep_row = ep_df.iloc[0]

    # 3. run _convert
    result = _convert(ep_row, dataset_dir=tmp_path, out_dir=out_dir,
                      fps=_DEFAULT_FPS, overwrite=True, tasks=tasks)

    # 4. assert return value contains expected markers
    assert "ok: ep 0000" in result, f"expected 'ok: ep 0000' in {result!r}"
    assert "(T=5)" in result, f"expected '(T=5)' in {result!r}"

    # 5. open ZARR and verify schema keys
    z = zarr_pkg.open(str(out_dir / "episode_0000.zarr"))
    expected_keys = [
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
    ]
    for key in expected_keys:
        assert key in list(z.keys()), f"missing ZARR key: {key}  (keys: {list(z.keys())})"

    # 6. assert RGB arrays have shape (5, 480, 640, 3)
    for cam_key in ("workspace_rgb", "wrist_rgb_left", "wrist_rgb_right"):
        assert z[cam_key].shape == (_DEFAULT_N, 480, 640, 3), \
            f"{cam_key}.shape = {z[cam_key].shape}, expected ({_DEFAULT_N}, 480, 640, 3)"

    # 7. assert joint_state_lowdim shape (5, 14)
    assert z["joint_state_lowdim"].shape == (_DEFAULT_N, 14), \
        f"joint_state_lowdim.shape = {z['joint_state_lowdim'].shape}, expected ({_DEFAULT_N}, 14)"


# ── Wave 2B Tasks 4+5: raw frames, timestamps, state, language ───────

def test_raw_frames_and_timestamps(tmp_path: pathlib.Path):
    """Compare decoded camera frames (index 2) across all three cameras; assert timestamp arrays."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    # 1. build synthetic fixture (N=5, fps=10)
    source_arrays = create_synthetic_fixture(tmp_path, N=_DEFAULT_N)

    # 2. convert
    from data_preprocessing.action.process_lerobot import _convert, _load_episodes_meta, _load_tasks

    ep_df = _load_episodes_meta(tmp_path)
    tasks = _load_tasks(tmp_path)
    ep_row = ep_df.iloc[0]

    result = _convert(ep_row, dataset_dir=tmp_path, out_dir=out_dir,
                      fps=_DEFAULT_FPS, overwrite=True, tasks=tasks)
    assert "ok: ep 0000" in result and "(T=5)" in result

    # 3. open ZARR
    z = zarr_pkg.open(str(out_dir / "episode_0000.zarr"))

    # 4. decode source MP4s directly (same path the decord mock uses) and compare frame index 2
    cam_mp4_map = {
        "workspace_rgb": ("observation.images.topdown", source_arrays["workspace_rgb"]),
        "wrist_rgb_left": ("observation.images.left_wrist", source_arrays["wrist_rgb_left"]),
        "wrist_rgb_right": ("observation.images.right_wrist", source_arrays["wrist_rgb_right"]),
    }
    for zarr_key, (cam_prefix, _src) in cam_mp4_map.items():
        mp4_path = str(tmp_path / "videos" / cam_prefix / "chunk-000" / "file-000.mp4")
        # decode the MP4 via imageio (identical to what the decord mock does internally)
        mp4_frames = iio.imread(mp4_path).astype(np.uint8)
        mp4_frame_2 = mp4_frames[2]  # (480, 640, 3)
        zarr_frame_2 = z[zarr_key][2]  # (480, 640, 3)

        # MP4 round-trip can be slightly lossy — use atol=2 tolerance
        np.testing.assert_allclose(
            zarr_frame_2, mp4_frame_2, atol=2,
            err_msg=f"{zarr_key}[2] differs from MP4-decoded frame[2]",
        )

    # 5. assert timestamp arrays match expected pattern: arange(N) * (NS_PER_SEC / fps)
    ns_per_sec = 1_000_000_000
    expected_ts = (np.arange(_DEFAULT_N, dtype=np.uint64)
                   * np.uint64(ns_per_sec // _DEFAULT_FPS))
    assert list(expected_ts) == [0, 100000000, 200000000, 300000000, 400000000]

    for ts_key in ("workspace_rgb_timestamps", "wrist_rgb_left_timestamps",
                    "wrist_rgb_right_timestamps", "joint_state_lowdim_timestamps"):
        actual_ts = z[ts_key][:]
        np.testing.assert_array_equal(
            actual_ts, expected_ts,
            err_msg=f"{ts_key} mismatch: got {list(actual_ts)}, expected {list(expected_ts)}",
        )


def test_state_and_language(tmp_path: pathlib.Path):
    """Assert joint_state_lowdim exact float match, language instruction, and no action dataset."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    # 1. build synthetic fixture (N=5, fps=10)
    source_arrays = create_synthetic_fixture(tmp_path, N=_DEFAULT_N)

    # 2. convert
    from data_preprocessing.action.process_lerobot import _convert, _load_episodes_meta, _load_tasks

    ep_df = _load_episodes_meta(tmp_path)
    tasks = _load_tasks(tmp_path)
    ep_row = ep_df.iloc[0]

    _convert(ep_row, dataset_dir=tmp_path, out_dir=out_dir,
             fps=_DEFAULT_FPS, overwrite=True, tasks=tasks)

    # 3. open ZARR
    z = zarr_pkg.open(str(out_dir / "episode_0000.zarr"))

    # 4. assert joint_state_lowdim matches source states exactly (float32 -> float32, no conversion loss)
    source_states = source_arrays["states"]  # (5, 14) float32
    zarr_states = z["joint_state_lowdim"][:]  # (5, 14) float32
    np.testing.assert_allclose(
        zarr_states, source_states, rtol=0, atol=0,
        err_msg="joint_state_lowdim should match source observation.state exactly",
    )

    # 5. assert language_instruction decodes to "pick up the cube"
    lang = z["language_instruction"][0].decode("utf-8")
    assert lang == "pick up the cube", f"language_instruction = {lang!r}"

    # 6. assert language_instruction_timestamps
    np.testing.assert_array_equal(
        z["language_instruction_timestamps"][:],
        np.array([0], dtype=np.uint64),
        err_msg="language_instruction_timestamps should be [0]",
    )

    # 7. assert no "action" dataset exists (current converter writes no action)
    root_keys = list(z.keys())
    assert "action" not in root_keys, \
        f"'action' should NOT be in ZARR keys: {root_keys}"
