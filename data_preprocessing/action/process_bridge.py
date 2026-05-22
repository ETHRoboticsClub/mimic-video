# original: https://github.com/rail-berkeley/bridge_data_v2/blob/bc60a35b701a12021c8c95e9d8601274d3acd928/data_processing/bridgedata_raw_to_numpy.py
"""# Bridge action data extraction.

## lowdims

There are states in cartesian and joint coordinates as well as actions in cartesian coordinates in `obs_dict.pkl`.

There is the end effector pose under the `eef_transform` key. Curiously, the eef orientation part (which is RPY, ie
specifically scipy.spatial.transform.Rotation needs seq="XYZ") of the state in `full_state` is rotated by
[[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]] (from the right) relative to that `eef_transform`. If you made
Simpler give you proprio, then the eef pose is like `eef_transform` and not like `full_state`.

If you want the (mostly) binary gripper action instead of computing some questionable quantity from the continous qpos
or modeling the continuous value and not know when to make the gripper press, you can find it in `desired_state[6]`.

Episodes always seem to start with an idle step so we remove it.

## Language instruction

Bad quality. These are the main groups of incorrect labels we found so far:
- quite a substantial part has no language labels, around 13%.
- a non-negligible part has incorrect labels, e.g. the instruction says something like put eggplant in bowl but the arm
  actually picks up a pepper. When we looked at 50 episodes once, this affected 4.
  Total number of incorrect labels could thus be as high as 4000.
- whopping 1768 episodes deal with either a pot or a pan but don't know which (the infamous "pot or pan")
- lots of french, some spanish, some portuguese, even a few in malagasi (eg mametraka lamba iray eo aminy latabatra)
  which throws gemma3:4b into an infinite loop.
- inconsistent sentence types, most are imperative but a large part is declarative.
- large part has the objects written as adjectiveobject as though some part of some pipeline couldn't handle whitespaces
- other weird identifiers that suggest it was sometimes more regarded as an internal language *identifier* than a
  language *description* or *command*
- good part is screaming at you in ALL CAPS
- 1400 episodes where the objects are in a cardboard-fenced area look like f"{command} cardboard fence"
- carboard boxes are called 1fbox and 4fbox? is that an american thing?
- a handful of episodes have language labels that just tell us that the labeler wasn't able to see the video of that
  episode
- a handful of episodes have language labels that tell us that nothing actually happened in that episode. we didn't
  check if those are true yet.
- keyboard smashings
- even a YouTube link is the language instruction twice. This is the video: https://www.youtube.com/watch?v=JWA5hJl4Dv0
"""

import argparse
import logging
import pathlib
import pickle
import re
from typing import Literal

import numpy as np
import tqdm
import zarr
from numcodecs import Blosc
from PIL import Image

S_TO_NS = 1_000_000_000


def read_image(path) -> np.ndarray:
    return np.asarray(Image.open(path)).astype(np.uint8)


def process_images(path: pathlib.Path) -> np.ndarray:
    image_path = path / "images0"
    tlen = len(list((image_path).glob("im_*.jpg")))
    return np.array([read_image(image_path / f"im_{i}.jpg") for i in range(1, tlen)])


def process_eef_state(path: pathlib.Path) -> np.ndarray:
    fp = path / "obs_dict.pkl"
    with fp.open("rb") as f:
        x = pickle.load(f)
    return x["eef_transform"][1:]


def process_gripper_state(path: pathlib.Path) -> np.ndarray:
    fp = path / "obs_dict.pkl"
    with fp.open("rb") as f:
        x = pickle.load(f)
    return x["full_state"][1:, 6].clip(0, 1)


def process_gripper_action(path: pathlib.Path) -> np.ndarray:
    fp = path / "obs_dict.pkl"
    with fp.open("rb") as f:
        x = pickle.load(f)
    return x["desired_state"][1:, 6].clip(0, 1)


def process_time(path: pathlib.Path) -> np.ndarray:
    fp = path / "obs_dict.pkl"
    with fp.open("rb") as f:
        x = np.array(pickle.load(f)["time_stamp"])
    return x[1:] - x[1]


def process_language(path: pathlib.Path) -> str:
    """Avoids 3046 undesirable labels.

    Can only filter the most obvious issues.
    Does not filter foreign languages (found french, spanish, portuguese, malagasy).
    Does not do anything that would require knowing english.
    Does not catch most typos.
    Most importantly, does not catch when the label is just wrong :/ (high-variance estimate: 4000 episodes / 8%)
    """
    if not (lang_path := (path / "lang.txt")).exists():
        return ""

    for lang in lang_path.read_text().splitlines():
        if "confidence" in lang:
            continue

        lang = lang.strip()

        # 63 episodes, all bad
        if " " not in lang:
            continue

        # 24 episodes
        if any(
            complaint in lang
            for complaint in [
                "no image",
                "did not load",
                "not loaded",
                "not downloading",
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


def make_zarr(path: pathlib.Path, in_dir: pathlib.Path, out_dir: pathlib.Path, default_lang: str) -> Literal[0, 1]:
    if "lmdb" in str(path):
        logging.warning(f"Skipping {path} because uhhhh lmdb?")
        return 0

    if not (path / "obs_dict.pkl").exists():
        logging.error(f"{path} missing obs_dict.pkl")
        return 0

    if not (path / "policy_out.pkl").exists():
        logging.error(f"{path} missing policy_out.pkl")
        return 0

    images = process_images(path)
    eef_pose = process_eef_state(path)
    gripper_pos = process_gripper_state(path)
    gripper_action = process_gripper_action(path)
    time_stamps = process_time(path) * S_TO_NS

    if not (
        0
        < images.shape[0]
        == eef_pose.shape[0]
        == gripper_pos.shape[0]
        == gripper_action.shape[0]
        == time_stamps.shape[0]
    ):
        print("missing data!")
        print(
            path,
            images.shape[0],
            eef_pose.shape[0],
            gripper_pos.shape[0],
            gripper_action.shape[0],
            time_stamps.shape[0],
        )
        return 0

    lang = process_language(path)
    res = 0

    if lang == "":
        lang = default_lang
        res = 1

    out_path = (out_dir / path.relative_to(in_dir)).with_suffix(".zarr").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_img = min(65, time_stamps.shape[0])
    t_ld = min(1024, time_stamps.shape[0])

    root: zarr.Group
    with zarr.open(str(out_path), "w") as root:
        root.create_dataset(
            "workspace_rgb",
            shape=images.shape,
            dtype=np.uint8,
            chunks=(t_img, *images.shape[1:]),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["workspace_rgb"][...] = images
        root.create_dataset(
            "workspace_rgb_timestamps",
            shape=(len(time_stamps),),
            dtype="uint64",
            chunks=(len(time_stamps),),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["workspace_rgb_timestamps"][...] = time_stamps.copy()

        root.create_dataset(
            "eef_pose_lowdim",
            shape=eef_pose.shape,
            dtype=np.float32,
            chunks=(t_ld, *eef_pose.shape[1:]),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["eef_pose_lowdim"][...] = eef_pose
        root.create_dataset(
            "eef_pose_lowdim_timestamps",
            shape=(len(time_stamps),),
            dtype="uint64",
            chunks=(len(time_stamps),),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["eef_pose_lowdim_timestamps"][...] = time_stamps.copy()

        root.create_dataset(
            "gripper_state_lowdim",
            shape=gripper_pos.shape,
            dtype=np.float32,
            chunks=(t_ld, *gripper_pos.shape[1:]),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["gripper_state_lowdim"][...] = gripper_pos
        root.create_dataset(
            "gripper_state_lowdim_timestamps",
            shape=(len(time_stamps),),
            dtype="uint64",
            chunks=(len(time_stamps),),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["gripper_state_lowdim_timestamps"][...] = time_stamps.copy()

        root.create_dataset(
            "gripper_action_lowdim",
            shape=gripper_action.shape,
            dtype=np.float32,
            chunks=(t_ld, *gripper_action.shape[1:]),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["gripper_action_lowdim"][...] = gripper_action
        root.create_dataset(
            "gripper_action_lowdim_timestamps",
            shape=(len(time_stamps),),
            dtype="uint64",
            chunks=(len(time_stamps),),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["gripper_action_lowdim_timestamps"][...] = time_stamps.copy()

        root.create_dataset(
            "language_instruction",
            shape=(1,),
            dtype=bytes,
            chunks=(1,),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["language_instruction"][...] = np.array([lang.encode()])
        root.create_dataset(
            "language_instruction_timestamps",
            shape=(1,),
            dtype="uint64",
            chunks=(1,),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["language_instruction_timestamps"][...] = np.array([0], dtype=np.uint64)

        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-dir", required=True, type=pathlib.Path, help="Root directory with raw bridge data (jpgs and pkls)."
    )
    ap.add_argument("--output-dir", required=True, type=pathlib.Path, help="Directory to write per-demo .zarr groups.")
    ap.add_argument(
        "--default-lang",
        type=str,
        default="",
        help="Default language instruction when label is empty or bad.",
    )
    # all sorts of race conditions if you stress the file system too much so no parallelism
    args = ap.parse_args()
    paths = pathlib.Path(args.raw_dir).glob("**/raw/traj_group*/traj*")
    num_replaced = sum(
        make_zarr(path, in_dir=args.raw_dir, out_dir=args.output_dir, default_lang=args.default_lang)
        for path in tqdm.tqdm(paths)
    )

    logging.info(f"Replaced {num_replaced} language labels.")


if __name__ == "__main__":
    main()
