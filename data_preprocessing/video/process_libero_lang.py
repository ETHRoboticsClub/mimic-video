import hashlib
import os
import re
import pathlib

import tyro
import tqdm


def unique_dest(dest_dir: pathlib.Path, name: str, src: pathlib.Path) -> pathlib.Path:
    p = dest_dir / name
    if not p.exists():
        return p
    stem, ext = os.path.splitext(name)
    h = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:8]
    q = dest_dir / f"{stem}_{h}{ext}"
    return q if not q.exists() else dest_dir / f"{stem}_{h}_1{ext}"


def extract_name(filename: str) -> str:
    base = os.path.splitext(filename)[0]

    # remove suffix
    base = re.sub(r"_demo_demo_\d+_[a-z_]+_rgb$", "", base)
    # remove suite prefix
    base = (
        base.removeprefix("libero_10_")
        .removeprefix("libero_90_")
        .removeprefix("libero_goal_")
        .removeprefix("libero_object_")
        .removeprefix("libero_spatial_")
    )
    # remove env prefix
    m = re.search(r"[a-z]", base)
    base = base[m.start() :]  # ty:ignore[unresolved-attribute]
    # remove underscores
    base = base.replace("_", " ")
    # rename black bowl to just bowl bc it isn't black
    base = base.replace("black bowl", "bowl")
    return base + "."


def main(video_dir: pathlib.Path, out_meta: pathlib.Path):
    out_meta.mkdir(parents=True, exist_ok=True)
    for mp4 in tqdm.tqdm(video_dir.iterdir()):
        if mp4.suffix != ".mp4":
            print(f"Skipping {mp4}.")
        meta = out_meta / mp4.relative_to(video_dir).with_suffix(".txt")
        meta.write_text(extract_name(meta.stem))

    print(f"Done. Wrote to {out_meta}")


if __name__ == "__main__":
    tyro.cli(main)
