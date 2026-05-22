import argparse
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import h5py
import imageio.v2 as iio  # ffmpeg backend
import numpy as np
from tqdm import tqdm


def to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    assert frame.dtype == np.uint8, frame.dtype
    return frame


def write_demo_video(images_ds, out_path, fps, rotate180, crf):
    T, _H, _W, C = images_ds.shape
    assert C == 3, "Expected RGB images with 3 channels"
    os.makedirs(Path(out_path).parent, exist_ok=True)
    writer = iio.get_writer(
        out_path,
        format="ffmpeg",  # ty:ignore[invalid-argument-type]
        fps=fps,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-crf", str(crf)],
    )
    try:
        for t in range(T):
            frame = images_ds[t]
            frame = to_uint8_rgb(frame)
            if rotate180:
                frame = np.flipud(np.fliplr(frame))
            writer.append_data(frame)
    finally:
        writer.close()


def write_all_demo_videos(h5_file_and_idx, cam_keys, out_root, args):
    i, h5_file = h5_file_and_idx
    stem = h5_file.stem
    with h5py.File(h5_file, "r") as f:
        data = f["data"]
        demo_names = sorted(list(data.keys()))
        for demo in tqdm(demo_names, desc=f"{h5_file.name} -> MP4s", position=i):
            for cam_key in cam_keys:
                obs_grp = data[demo]["obs"]
                images_ds = obs_grp[cam_key]  # shape: T,H,W,3
                out_path = out_root / f"{h5_file.parent.name}_{stem}_{demo}_{cam_key}.mp4"
                write_demo_video(
                    images_ds,
                    out_path,
                    fps=args.fps,
                    rotate180=args.rotate180,
                    crf=args.crf,
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_path", help="Path to a single .hdf5 (e.g., task_demo.hdf5)")
    ap.add_argument("--h5_dir", help="Directory containing multiple .hdf5/.h5 files")
    ap.add_argument("--out_dir", required=True, help="Directory to write MP4 files (will be created)")
    ap.add_argument("--fps", type=int, default=10, help="Output FPS")
    ap.add_argument("--rotate180", action="store_true", help="Rotate frames by 180°")
    ap.add_argument("--crf", type=int, default=18, help="x264 CRF (lower=better quality)")
    ap.add_argument("--cam_key", default="", help="Override camera key under obs/ (default: both cams)")
    ap.add_argument("--num_workers", type=int, default=1)
    args = ap.parse_args()

    # Collect input files (single file OR all in a dir)
    h5_paths = []
    if args.h5_dir:
        p = Path(args.h5_dir)
        h5_paths = sorted(list(p.glob("**/*.hdf5")) + list(p.glob("**/*.h5")))
    if args.h5_path:
        h5_paths.append(Path(args.h5_path))
    if not h5_paths:
        raise SystemExit("No HDF5 files found. Provide --h5_path or --h5_dir.")

    # Create output folder (your “album” name)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Which cameras to export
    cam_keys = [args.cam_key] if args.cam_key else ["agentview_rgb", "eye_in_hand_rgb"]

    with Pool(processes=args.num_workers) as pool:
        for _res in pool.imap_unordered(
            partial(write_all_demo_videos, cam_keys=cam_keys, out_root=out_root, args=args), enumerate(h5_paths)
        ):
            pass


if __name__ == "__main__":
    main()
