"""
Convert a LeRobot v3 dataset to mimic-video format.

LeRobot v3 packs multiple episodes into each video file.
This script reads the parquet index to find per-episode frame
ranges, then extracts all episodes from each source file in a
single ffmpeg pass using filter_complex + trim.

Input:
  <lerobot_dir>/videos/<video_key>/chunk-<N>/file-<M>.mp4  (multi-episode)
  <lerobot_dir>/data/chunk-<N>/file-<M>.parquet
  <lerobot_dir>/meta/tasks.parquet

Output:
  <output_dir>/video/ep_<NNN>.mp4
  <output_dir>/metas/ep_<NNN>.txt

Usage:
  pip install pyarrow tqdm
  python scripts/convert_lerobot_v3_to_video_format.py \
    --lerobot-dir ./data/push-that-thing \
    --output-dir  ./data/push-mimic \
    --ffmpeg      /usr/local/bin/ffmpeg
"""
from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lerobot-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--video-key", default="observation.images.front")
    p.add_argument("--ffmpeg", default="ffmpeg", help="path to ffmpeg binary (must support AV1)")
    return p.parse_args()


def read_task_map(lerobot_dir: Path) -> dict[int, str]:
    table = pq.read_table(lerobot_dir / "meta" / "tasks.parquet")
    str_col = next(
        f.name for f in table.schema
        if pa.types.is_string(f.type) or pa.types.is_large_string(f.type)
    )
    return {int(i): str(s) for i, s in zip(
        table["task_index"].to_pylist(), table[str_col].to_pylist()
    )}


def extract_all_from_file(
    ffmpeg: str,
    src: Path,
    episodes: list[dict],  # sorted by ep_idx, each has: ep_idx, start, count, dst_video
) -> None:
    """Decode src once and write all episodes in a single ffmpeg call."""
    n = len(episodes)

    # [0:v]split=N[v0][v1]...[vN-1]
    split = f"[0:v]split={n}" + "".join(f"[v{i}]" for i in range(n))

    # [vi]trim=start_frame=S:end_frame=E,setpts=PTS-STARTPTS[outi]
    trims = []
    for i, ep in enumerate(episodes):
        s = ep["start"]
        e = s + ep["count"]
        trims.append(f"[v{i}]trim=start_frame={s}:end_frame={e},setpts=PTS-STARTPTS[out{i}]")

    filter_complex = split + ";" + ";".join(trims)

    cmd = [ffmpeg, "-y", "-i", str(src), "-filter_complex", filter_complex]
    for i, ep in enumerate(episodes):
        cmd += ["-map", f"[out{i}]", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(ep["dst_video"])]

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode())


def main() -> None:
    args = parse_args()
    (args.output_dir / "video").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metas").mkdir(parents=True, exist_ok=True)

    task_map = read_task_map(args.lerobot_dir)
    video_root = args.lerobot_dir / "videos" / args.video_key

    # Group episodes by source video file
    by_file: dict[Path, list[dict]] = defaultdict(list)
    for parquet_path in sorted((args.lerobot_dir / "data").rglob("*.parquet")):
        rel = parquet_path.relative_to(args.lerobot_dir / "data")
        video_file = video_root / rel.parent / (rel.stem + ".mp4")

        table = pq.read_table(parquet_path, columns=["episode_index", "index", "task_index"])
        ep_col = table["episode_index"].to_pylist()
        idx_col = table["index"].to_pylist()
        task_col = table["task_index"].to_pylist()

        file_offset = min(idx_col)
        ep_info: dict[int, dict] = {}
        for ep, gidx, task in zip(ep_col, idx_col, task_col):
            local = gidx - file_offset
            if ep not in ep_info:
                ep_info[ep] = {"start": local, "count": 0, "task": task}
            ep_info[ep]["count"] += 1

        for ep_idx, info in sorted(ep_info.items()):
            by_file[video_file].append({
                "ep_idx": ep_idx,
                "start": info["start"],
                "count": info["count"],
                "task": info["task"],
                "dst_video": args.output_dir / "video" / f"ep_{ep_idx:03d}.mp4",
                "dst_meta": args.output_dir / "metas" / f"ep_{ep_idx:03d}.txt",
            })

    total_episodes = sum(len(eps) for eps in by_file.values())

    with tqdm(total=total_episodes, unit="ep") as bar:
        for video_file, episodes in sorted(by_file.items()):
            # Skip files where all episodes are already extracted
            missing = [ep for ep in episodes if not ep["dst_video"].exists()]
            if not missing:
                bar.update(len(episodes))
                continue

            bar.set_description(video_file.name)
            extract_all_from_file(args.ffmpeg, video_file, missing)

            for ep in missing:
                ep["dst_meta"].write_text(task_map[ep["task"]])

            bar.update(len(episodes))

    print(f"Done → {args.output_dir}")


if __name__ == "__main__":
    main()
