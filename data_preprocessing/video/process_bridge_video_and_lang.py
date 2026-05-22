"""
Convert the raw bridge episodes to stand-alone MP4 files (one per episode) and write a matching
metas/<episode>.txt with a short description.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import re
from collections.abc import Sequence
from functools import partial

import imageio.v2 as iio
from tqdm.auto import tqdm


def _episode_to_name(ep_path: pathlib.Path, root: pathlib.Path) -> str:
    """
    Turn episode path
        rss/toykitchen2/set_table/00/2022-01-01_00-00-00/raw/traj_group0/traj3
    into
        rss__toykitchen2__set_table__00__2022-01-01_00-00-00__traj_group0__traj3.mp4
    """
    rel = ep_path.relative_to(root)
    parts = [p for p in rel.parts if p not in {"raw"}]
    return "__".join(parts)


def _language_label(ep_path: pathlib.Path) -> str:
    """Avoids 3046 undesirable labels.

    Can only filter the most obvious issues.
    Does not filter foreign languages (found french, spanish, portuguese, malagasy).
    Does not do anything that would require knowing english.
    Does not catch most typos.
    Most importantly, does not catch when the label is just wrong :/ (high-variance estimate: 4000 episodes / 8%)
    """
    if not (lang_path := (ep_path / "lang.txt")).exists():
        return ""

    for lang in lang_path.read_text().splitlines():
        if "confidence" in lang:
            continue

        lang = lang.strip()

        # 63 episodes
        if " " not in lang:
            continue

        # 24 episodes
        if any(
            complaint in lang
            for complaint in [
                "no image",
                "did not load",
                "not loaded",
                "images loaded",
                "pictures aren't working",
                "not showing",
            ]
        ):
            continue

        # 1400 episodes
        lang = lang.removesuffix(" cardboard fence").strip()

        # 14 episodes
        lang = lang.replace(",", ", ").replace(" ,", ",")

        # 563 episodes
        lang = re.sub(r"\s+", " ", lang)

        # 284 episodes
        if re.search(r"(?<![A-Za-z])[A-Z]{2,}(?![A-Za-z])", lang):
            lang = lang.lower()

        # 257 episodes
        lang = lang.replace("4fbox", " cardboard box")

        # 201 episodes
        lang = lang.replace("1fbox", " cardboard box")

        # 193 episodes
        lang = re.sub(r"stuffed([a-z]+)", r"stuffed \1", lang)

        # 9 episodes
        lang = lang.replace("clockwise90", "clockwise 90 degrees")

        # 1 episode
        lang = lang.replace("bpttle", "bottle")

        # 1 episode
        lang = lang.replace("thng", "thing")

        # 1 episode
        lang = lang.replace("upawrds", "upwards")

        # 23 episodes
        ALLOWED = r"(?:gh|ng|th|ch|sh|ph|wh|ck|qu|bb|cc|dd|ff|gg|ll|mm|nn|pp|rr|ss|tt|zz)"
        if re.search(rf"(?i)(?:(?!(?:{ALLOWED}))[b-df-hj-np-tv-xz\.0-9]){{4,}}", lang):
            continue

        # 1 episode
        lang = lang.removesuffix(" (does not mention the locations of the objects)")

        # 7 episodes (excluding gibberish)
        lang = lang.removeprefix('"').removesuffix('"')

        # 4 further episodes
        if any(mangle in lang for mangle in '"_^~@#$%*+=|?ëěẽēėęćčċ'):
            continue

        return lang

    return ""


def _write_episode(ep_path: pathlib.Path, in_dir: pathlib.Path, out_dir: pathlib.Path, overwrite: bool) -> None:
    if "lmdb" in str(ep_path):
        return

    out_name = _episode_to_name(ep_path, root=in_dir)
    video_out_file_base = out_dir / "video" / out_name
    video_out_file_base.parent.mkdir(exist_ok=True, parents=True)

    for img_dir in ep_path.glob("images*"):
        video_out_file = video_out_file_base.with_stem(video_out_file_base.stem + img_dir.stem[-1]).with_suffix(".mp4")
        if video_out_file.exists() and not overwrite:
            continue

        tlen = sum(1 for _ in img_dir.glob("im_*.jpg"))
        imgs = [img_dir / f"im_{i}.jpg" for i in range(tlen - 1)]
        if len(imgs) < 5:
            continue

        try:
            with iio.get_writer(video_out_file, codec="libx264", fps=5, quality=8) as writer:
                for img_path in imgs:
                    frame = iio.imread(img_path)
                    writer.append_data(frame)  # ty:ignore[unresolved-attribute]
        except Exception as exc:
            raise RuntimeError(f"Failed on {ep_path}: {exc}") from exc

    text_out_file = (out_dir / "metas" / out_name).with_suffix(".txt")
    text_out_file.parent.mkdir(parents=True, exist_ok=True)

    if text_out_file.exists() and not overwrite:
        return

    lang = _language_label(ep_path)
    text_out_file.write_text(lang, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Flatten BridgeData v2 episodes into MP4s and write meta descriptions."
    )
    parser.add_argument("--input-dir", type=pathlib.Path, required=True, help="Root of raw bridge.")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="cosmos dataset dir.")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing mp4s")
    args = parser.parse_args(argv)

    episodes = args.input_dir.glob("**/raw/traj_group*/traj*")

    print(f"Writing dataset to {args.output_dir}.")

    with mp.Pool(args.num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                partial(
                    _write_episode,
                    in_dir=args.input_dir,
                    out_dir=args.output_dir,
                    overwrite=args.overwrite,
                ),
                episodes,
            )
        ):
            pass

    print("Finished.")


if __name__ == "__main__":
    main()
