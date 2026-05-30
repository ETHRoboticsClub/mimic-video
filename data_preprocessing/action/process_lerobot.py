import argparse
import pathlib

import numpy as np
import zarr
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from numcodecs import Blosc
from torch.utils.data import DataLoader, Subset
import tqdm

NS_PER_SEC = 1_000_000_000


def _assert_zarr_consistent(path: pathlib.Path) -> None:
    keys = [
        "joint_action_lowdim",
        "joint_action_lowdim_timestamps",
        "joint_state_lowdim",
        "joint_state_lowdim_timestamps",
        "workspace_rgb",
        "workspace_rgb_timestamps",
        "wrist_rgb_left",
        "wrist_rgb_left_timestamps",
        "wrist_rgb_right",
        "wrist_rgb_right_timestamps",
    ]
    with zarr.open(str(path), "r") as root:
        target_dim = root["joint_action_lowdim"].shape[0]
        for k in keys:
            assert root[k].shape[0] == target_dim, f"got bad dims for k={k} in {path}"


def _convert(
    ep_idx: int,
    out_ep_idx: int,
    *,
    dataset: LeRobotDataset,
    out_dir: pathlib.Path,
    overwrite: bool,
    task_override: str | None = None,
) -> str:
    out_path = out_dir / f"episode_{out_ep_idx:04d}.zarr"
    if out_path.exists() and not overwrite:
        return f"skip: {out_path.name}"

    ep = dataset.meta.episodes[ep_idx]
    start, end = ep["dataset_from_index"], ep["dataset_to_index"]
    length = end - start
    fps = dataset.fps

    ep_data = dataset.hf_dataset[start:end]
    state  = np.stack(ep_data["observation.state"]).astype(np.float32)  # (T, 14)
    action = np.stack(ep_data["action"]).astype(np.float32)             # (T, 14)
    task_text = task_override if task_override is not None else ep["tasks"][0]

    cams = {
        "workspace_rgb":  "observation.images.topdown",
        "wrist_rgb_left": "observation.images.left_wrist",
        "wrist_rgb_right":"observation.images.right_wrist",
    }

    loader = DataLoader(
        Subset(dataset, list(range(start, end))),
        batch_size=64,
        num_workers=4,
        shuffle=False,
    )

    decoded = {k: [] for k in cams}
    for batch in tqdm.tqdm(loader, desc=f"ep {out_ep_idx:04d} frames", leave=False):
        for out_key, cam_key in cams.items():
            imgs = batch[cam_key]  # (B, C, H, W) float32 [0, 1]
            imgs_np = (imgs.permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)
            decoded[out_key].append(imgs_np)

    decoded = {k: np.concatenate(v) for k, v in decoded.items()}  # (T, H, W, 3)

    timestamps = np.arange(length, dtype=np.uint64) * np.uint64(NS_PER_SEC / fps)
    comp  = Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)
    t_img = min(65, length)
    t_ld  = min(1024, length)

    root = zarr.open(str(out_path), mode="w")

    root.create_dataset("language_instruction", shape=(1,), dtype=bytes, chunks=(1,), compressor=comp)[...] = np.array([task_text.encode()])
    root.create_dataset("language_instruction_timestamps", shape=(1,), dtype="uint64", chunks=(1,), compressor=comp)[...] = np.array([0], dtype=np.uint64)

    for out_key, arr in decoded.items():
        root.create_dataset(out_key, shape=arr.shape, dtype=np.uint8, chunks=(t_img, *arr.shape[1:]), compressor=comp)[...] = arr
        root.create_dataset(f"{out_key}_timestamps", shape=(length,), dtype="uint64", chunks=(length,))[...] = timestamps

    for key, arr in [("joint_state_lowdim", state), ("joint_action_lowdim", action)]:
        root.create_dataset(key, shape=arr.shape, dtype=np.float32, chunks=(t_ld, arr.shape[1]), compressor=comp)[...] = arr
        root.create_dataset(f"{key}_timestamps", shape=(length,), dtype="uint64", chunks=(length,))[...] = timestamps

    _assert_zarr_consistent(out_path)
    return f"ok: ep {out_ep_idx:04d} (src {ep_idx}) -> {out_path.name} (T={length})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", nargs=2, action="append", required=True,
        metavar=("REPO_ID", "TASK_TEXT"),
        help="dataset tuple: repo_id task_text. Use 'none' to keep the dataset's own text. Repeat for multiple datasets.",
    )
    ap.add_argument("--output-dir", type=pathlib.Path, required=True)
    ap.add_argument("--overwrite",  action="store_true")
    ap.add_argument("--episodes",   type=int, nargs="*", default=None,
                    help="subset of episode indices to process per dataset (default: all)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_ep_idx = 0
    for repo_id, task_text in args.dataset:
        task_override = None if task_text.lower() == "none" else task_text
        print(f"\n=== loading {repo_id} ===")
        try:
            dataset = LeRobotDataset(repo_id)
        except Exception as e:
            print(f"FAIL loading {repo_id}: {type(e).__name__}: {str(e)[:200]}")
            continue

        print(f"    {dataset.num_episodes} episodes, fps={dataset.fps}")
        ep_indices = args.episodes if args.episodes is not None else range(dataset.num_episodes)

        for ep_idx in tqdm.tqdm(ep_indices, desc=repo_id):
            try:
                msg = _convert(
                    ep_idx, out_ep_idx,
                    dataset=dataset,
                    out_dir=args.output_dir,
                    overwrite=args.overwrite,
                    task_override=task_override,
                )
            except Exception as e:
                msg = f"FAIL ep {ep_idx} (out {out_ep_idx}): {type(e).__name__}: {str(e)[:200]}"
            print(msg)
            out_ep_idx += 1


if __name__ == "__main__":
    main()
