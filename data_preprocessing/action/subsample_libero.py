import argparse
import os
import pathlib
import random
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Create a directory with random sample symlinks of Zarr episodes.")
    parser.add_argument("source_dir", type=pathlib.Path, help="Path to the directory containing original Zarr episodes")
    parser.add_argument("dest_dir", type=pathlib.Path, help="Path to create the new directory with symlinks")
    parser.add_argument("--ratio", type=float, help="Fraction of episodes to sample per task.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without creating files")

    args = parser.parse_args()

    all_episodes = list(args.source_dir.glob("*.zarr"))

    if not all_episodes:
        print("No .zarr items found in source directory.")
        return

    task_groups = defaultdict(list)
    for ep in all_episodes:
        task_name = ep.stem.split("_demo__demo")[0]
        task_groups[task_name].append(ep)

    print(f"Found {len(all_episodes)} episodes across {len(task_groups)} tasks.")

    if not args.dry_run:
        args.dest_dir.mkdir(parents=True, exist_ok=True)

    total_linked = 0

    random.seed(args.seed)
    for task, episodes in task_groups.items():
        random.shuffle(episodes)

        sample_size = max(round(len(episodes) * args.ratio), 1)

        selected_episodes = episodes[:sample_size]

        print(f"Task '{task}': Linking {len(selected_episodes)}/{len(episodes)} episodes")

        for ep in selected_episodes:
            link_name = args.dest_dir / ep.name
            target = ep

            if args.dry_run:
                print(f"  [Dry Run] Would link {link_name} -> {target}")
            else:
                os.symlink(target, link_name)
                total_linked += 1

    if args.dry_run:
        print("\nDry run complete. No files created.")
    else:
        print(f"\nDone. Created {total_linked} symlinks in '{args.dest_dir}'.")


if __name__ == "__main__":
    main()
