import argparse
import hashlib
import math
import random
from pathlib import Path

import cv2


def is_val_from_basename(name: str) -> bool:
    return int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % 10_000 < 100


def extract_5_at_5fps_to_mp4(video: Path, out_root: Path, orig_fps: float) -> Path:
    if orig_fps < 5:
        raise RuntimeError(f"{video}: orig_fps={orig_fps} < 5.")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"{video}: invalid frame count")

    duration = total_frames / orig_fps
    span = 4.0 / 5.0  # 0.8 s for 5 frames @ 5 fps
    if duration <= span:
        cap.release()
        raise RuntimeError(f"{video}: too short for 5 frames at 5 fps")

    start = random.uniform(0.0, (duration - span) / 2.0)
    times = [start + k / 5.0 for k in range(5)]
    frame_idxs = [min(total_frames - 1, math.floor(t * orig_fps)) for t in times]

    # Seek to first frame and get size
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idxs[0])
    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        cap.release()
        raise RuntimeError(f"{video}: read failed at frame {frame_idxs[0]}")
    h, w = frame0.shape[:2]

    out_path = out_root / f"{video.stem}__t{round(start * 1000)}ms.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty:ignore[unresolved-attribute]
    writer = cv2.VideoWriter(str(out_path), fourcc, 5.0, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open writer for {out_path}")

    writer.write(frame0)
    for fi in frame_idxs[1:]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            writer.release()
            cap.release()
            raise RuntimeError(f"{video}: read failed at frame {fi}")
        writer.write(frame)

    writer.release()
    cap.release()
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--orig-fps", type=float, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_videos = [
        p
        for p in args.in_dir.rglob("libero_10_*_agentview_rgb.mp4")
        if p.is_file() and p.suffix.lower() == ".mp4" and is_val_from_basename(p.name)
    ]
    random.shuffle(val_videos)
    picks = val_videos[:10]

    for v in picks:
        out_path = extract_5_at_5fps_to_mp4(v, args.out_dir, args.orig_fps)
        print(f"{v} -> {out_path}")


if __name__ == "__main__":
    main()
