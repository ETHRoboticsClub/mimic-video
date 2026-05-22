import argparse
import os
import re
import sys


def main():
    parser = argparse.ArgumentParser(description="Cleanup checkpoint files, keeping 5k intervals AND the latest 2.")
    parser.add_argument("folder", help="Path to the folder containing checkpoints")
    parser.add_argument("--interval", type=int, default=5000, help="The iteration interval to keep (default: 5000)")
    parser.add_argument("--delete", action="store_true", help="Actually perform deletion. If not set, runs a DRY RUN.")

    args = parser.parse_args()

    folder_path = args.folder
    interval = args.interval

    if not os.path.isdir(folder_path):
        print(f"Error: Directory '{folder_path}' does not exist.")
        sys.exit(1)

    # 1. Parse all files in the directory
    pattern = re.compile(r"iter_(\d+)\.pt$")
    checkpoints = []

    print(f"Scanning '{folder_path}'...")

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            iteration = int(match.group(1))
            full_path = os.path.join(folder_path, filename)
            checkpoints.append((iteration, filename, full_path))

    if not checkpoints:
        print("No files matching 'iter_dddd.pt' found.")
        sys.exit(0)

    print(f"Found {len(checkpoints)} checkpoint files.")

    # 2. Determine which files to keep
    files_to_keep = set()

    # A. Keep closest to interval (0, 5000, 10000...)
    max_iter = max(c[0] for c in checkpoints)
    targets = range(0, max_iter + interval, interval)

    for target in targets:
        closest = min(checkpoints, key=lambda c: abs(c[0] - target))
        files_to_keep.add(closest)

    # B. Keep the absolute latest 2 checkpoints
    # Sort by iteration (highest first)
    checkpoints_sorted = sorted(checkpoints, key=lambda x: x[0], reverse=True)
    latest_two = checkpoints_sorted[:2]

    # Add them to the set (set handles duplicates if the latest are also 5k intervals)
    for cp in latest_two:
        files_to_keep.add(cp)

    # 3. Separate into lists for reporting
    to_delete = []
    to_keep = []

    for cp in checkpoints:
        if cp in files_to_keep:
            to_keep.append(cp)
        else:
            to_delete.append(cp)

    to_delete.sort(key=lambda x: x[0])
    to_keep.sort(key=lambda x: x[0])

    # 4. Execute (or Dry Run)
    print("---")
    print("Analysis complete.")
    print(f"Files to KEEP: {len(to_keep)}")
    print(f"Files to DELETE: {len(to_delete)}")

    # Print specific info about the latest two
    print("Latest 2 files (Always Kept):")
    for cp in latest_two:
        print(f"  - {cp[1]}")
    print("---")

    if not args.delete:
        print("!! DRY RUN MODE (No files will be deleted) !!")
        print("To actually delete files, run this script again with the --delete flag.")

        print("\nFiles that would be KEPT (Sample):")
        for _, name, _ in to_keep[:10]:
            print(f"  [KEEP] {name}")
        if len(to_keep) > 10:
            print("  ...")

        print("\nFiles that would be DELETED (Sample):")
        for _, name, _ in to_delete[:10]:
            print(f"  [DEL] {name}")
        if len(to_delete) > 10:
            print("  ...")

    else:
        print("Deleting files...")
        count = 0
        for _, name, path in to_delete:
            try:
                os.remove(path)
                count += 1
            except OSError as e:
                print(f"Error deleting {name}: {e}")

        print(f"\nSuccessfully deleted {count} files.")
        print(f"Retained {len(to_keep)} files.")


if __name__ == "__main__":
    main()
