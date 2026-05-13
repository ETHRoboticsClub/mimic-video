"""Convert S3-format ETHRC libero_cosmos HDF5s to mimic-video zarr format.

The S3 HDF5s at s3://ethrc-ml-data-916780037007/robot-learning/cosmos-predict2-libero/
datasets/libero_cosmos/all_episodes/ are NVIDIA-postprocessed per-episode files —
one HDF5 per episode, flat schema with JPEG-encoded images. They differ from the
mimic-video `regenerate_libero.py` output (which has a `data/demo_*` hierarchy
with raw RGB), so the upstream `process_libero.py` cannot read them.

This script produces the same zarr output as `process_libero.py`. Use it when
training an action decoder on the ETHRC unified backbone, where matched-visuals
to the cosmos posttraining data is the goal.

Schema mapping (S3 → process_libero output):
    actions[:, 0:3]   → eef_pos_ref_delta_lowdim     (xyz delta)
    actions[:, 3:6]   → eef_rot_ref_delta_lowdim     (axisangle → rotation matrix)
    actions[:, 6:]    → gripper_action_lowdim         ({-1, +1})
    proprio[:, 0]     → gripper_lowdim                (gripper_qpos[0])
    proprio[:, 2:5]   → eef_pos_lowdim                (xyz)
    proprio[:, 5:9]   → eef_rot_lowdim                (quaternion wxyz)
    primary_images_jpeg → workspace_rgb               (JPEG-decode, 256x256x3)
    wrist_images_jpeg → wrist_rgb                     (JPEG-decode, 256x256x3)
    attrs.task_description → language_instruction
"""
import argparse
import io
import pathlib
from functools import partial
from multiprocessing import Pool

import h5py
import numpy as np
import zarr
import tqdm
from numcodecs import Blosc
from PIL import Image
from scipy.spatial.transform import Rotation

NS_PER_SEC = 1_000_000_000


def _decode_jpegs(jpeg_array: np.ndarray) -> np.ndarray:
    """Decode an array of JPEG byte buffers to an (N, H, W, 3) uint8 stack."""
    frames = []
    for buf in jpeg_array:
        if isinstance(buf, bytes):
            data = buf
        else:
            data = bytes(buf)
        img = Image.open(io.BytesIO(data))
        frames.append(np.array(img, dtype=np.uint8))
    return np.stack(frames, axis=0)


def _convert(h5_path: pathlib.Path, out_dir: pathlib.Path, fps: int, overwrite: bool, episode_idx: int) -> str:
    out_path = out_dir / f"episode_{episode_idx:04d}.zarr"
    if out_path.exists() and not overwrite:
        return f"skip (exists): {out_path}"

    with h5py.File(h5_path, "r") as f:
        actions = f["actions"][()].astype(np.float32)         # (T, 7)
        proprio = f["proprio"][()]                            # (T, 9)
        primary_jpegs = f["primary_images_jpeg"][()]          # (T,) bytes
        wrist_jpegs = f["wrist_images_jpeg"][()]              # (T,) bytes
        task_description = str(f.attrs.get("task_description", "unknown task"))

        agent = _decode_jpegs(primary_jpegs)                  # (T, H, W, 3) uint8
        eih = _decode_jpegs(wrist_jpegs)                      # (T, H, W, 3) uint8

        ee_pos = proprio[:, 2:5].astype(np.float32)           # (T, 3)
        ee_ori = proprio[:, 5:9].astype(np.float32)           # (T, 4) quaternion (wxyz)
        gripper = proprio[:, 0].astype(np.float32)            # (T,) gripper_qpos[0]

    T = agent.shape[0]
    assert proprio.shape[0] == T == actions.shape[0], (
        f"length mismatch: T_img={T} T_proprio={proprio.shape[0]} T_act={actions.shape[0]}"
    )
    dt_ns = NS_PER_SEC / fps
    timestamps = (np.arange(T) * dt_ns).astype(np.uint64)

    comp = Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)
    t_img = min(65, T)
    t_ld = min(1024, T)

    root = zarr.open(str(out_path), mode="w")

    # language
    root.create_dataset(
        "language_instruction",
        shape=(1,),
        dtype=bytes,
        chunks=(1,),
        compressor=comp,
        overwrite=True,
    )[...] = np.array([(task_description + ".").encode()])
    root.create_dataset(
        "language_instruction_timestamps",
        shape=(1,),
        dtype="uint64",
        chunks=(1,),
        compressor=comp,
        overwrite=True,
    )[...] = np.array([0], dtype=np.uint64)

    # images
    root.create_dataset(
        "workspace_rgb",
        shape=agent.shape,
        dtype=np.uint8,
        chunks=(t_img, *agent.shape[1:]),
        compressor=comp,
    )[...] = agent
    root.create_dataset("workspace_rgb_timestamps", shape=(T,), dtype="uint64", chunks=(T,))[...] = timestamps

    root.create_dataset(
        "wrist_rgb",
        shape=eih.shape,
        dtype=np.uint8,
        chunks=(t_img, *eih.shape[1:]),
        compressor=comp,
    )[...] = eih
    root.create_dataset("wrist_rgb_timestamps", shape=(T,), dtype="uint64", chunks=(T,))[...] = timestamps

    # actions → low-dim
    pos_ref_delta = actions[:, :3]
    root.create_dataset(
        "eef_pos_ref_delta_lowdim",
        shape=pos_ref_delta.shape,
        dtype=np.float32,
        chunks=(t_ld, *pos_ref_delta.shape[1:]),
        compressor=comp,
    )[...] = pos_ref_delta
    root.create_dataset(
        "eef_pos_ref_delta_lowdim_timestamps", shape=(T,), dtype="uint64", chunks=(T,),
    )[...] = timestamps

    rot_ref_delta = Rotation.from_rotvec(actions[:, 3:6]).as_matrix()  # (T, 3, 3)
    root.create_dataset(
        "eef_rot_ref_delta_lowdim",
        shape=rot_ref_delta.shape,
        dtype=np.float32,
        chunks=(t_ld, *rot_ref_delta.shape[1:]),
        compressor=comp,
    )[...] = rot_ref_delta
    root.create_dataset(
        "eef_rot_ref_delta_lowdim_timestamps", shape=(T,), dtype="uint64", chunks=(T,),
    )[...] = timestamps

    grip_action = actions[:, 6:]
    root.create_dataset(
        "gripper_action_lowdim",
        shape=grip_action.shape,
        dtype=np.float32,
        chunks=(t_ld, *grip_action.shape[1:]),
        compressor=comp,
    )[...] = grip_action
    root.create_dataset(
        "gripper_action_lowdim_timestamps", shape=(T,), dtype="uint64", chunks=(T,),
    )[...] = timestamps

    # proprio → low-dim
    root.create_dataset(
        "eef_pos_lowdim",
        shape=ee_pos.shape,
        dtype=np.float32,
        chunks=(t_ld, *ee_pos.shape[1:]),
        compressor=comp,
    )[...] = ee_pos
    root.create_dataset("eef_pos_lowdim_timestamps", shape=(T,), dtype="uint64", chunks=(T,))[...] = timestamps

    root.create_dataset(
        "eef_rot_lowdim",
        shape=ee_ori.shape,
        dtype=np.float32,
        chunks=(t_ld, *ee_ori.shape[1:]),
        compressor=comp,
    )[...] = ee_ori
    root.create_dataset("eef_rot_lowdim_timestamps", shape=(T,), dtype="uint64", chunks=(T,))[...] = timestamps

    root.create_dataset(
        "gripper_lowdim",
        shape=gripper.shape,
        dtype=np.float32,
        chunks=(t_ld, *gripper.shape[1:]),
        compressor=comp,
    )[...] = gripper
    root.create_dataset("gripper_lowdim_timestamps", shape=(T,), dtype="uint64", chunks=(T,))[...] = timestamps

    return f"ok: {out_path.name} ({T} steps)"


def _convert_indexed(args):
    h5_path, episode_idx, out_dir, fps, overwrite = args
    return _convert(h5_path, out_dir, fps, overwrite, episode_idx)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=pathlib.Path, required=True, help="dir with S3-format .hdf5 files")
    ap.add_argument("--output-dir", type=pathlib.Path, required=True, help="dir to write episode_NNNN.zarr files")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    h5_paths = sorted(args.input_dir.glob("**/*.hdf5"))
    assert len(h5_paths) > 0, f"no HDF5 files found in {args.input_dir}"

    job_args = [(p, i, args.output_dir, args.fps, args.overwrite) for i, p in enumerate(h5_paths)]

    with Pool(processes=args.num_workers) as pool:
        for msg in tqdm.tqdm(
            pool.imap_unordered(_convert_indexed, job_args),
            total=len(job_args),
            desc="libero_s3 h5 -> zarr",
        ):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
