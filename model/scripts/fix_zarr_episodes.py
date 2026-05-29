"""
Scan a zarr dataset directory for episodes with internal array length mismatches
and rename them to .zarr.bak so the dataloader skips them.

Usage:
    python scripts/fix_zarr_episodes.py --data_dir /path/to/zarr_root
    python scripts/fix_zarr_episodes.py --data_dir /path/to/zarr_root --dry_run
"""

import argparse
import pathlib

import zarr


def check_episode(path: pathlib.Path) -> list[str]:
    """Return list of problems found in this episode, empty if clean."""
    problems = []
    try:
        with zarr.open(str(path), "r") as root:
            # Collect all arrays and their lengths
            arrays = {}
            def _collect(name, obj):
                if isinstance(obj, zarr.Array):
                    arrays[name] = len(obj)
            root.visititems(_collect)
    except Exception as e:
        return [f"cannot open (empty or corrupted): {e}"]

    if not arrays:
        return ["empty zarr store"]

    # For each *_timestamps array, check the matching data array has >= length
    for name, length in arrays.items():
        if name.endswith("_timestamps"):
            data_key = name[: -len("_timestamps")]
            if data_key in arrays:
                data_len = arrays[data_key]
                if data_len < length:
                    problems.append(
                        f"{data_key}: timestamps has {length} entries but data has {data_len} (off by {length - data_len})"
                    )

    # Also flag any array with size 0
    for name, length in arrays.items():
        if length == 0:
            problems.append(f"{name}: array is empty")

    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--dry_run", action="store_true", help="Report only, don't rename")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    episodes = sorted(data_dir.glob("**/*.zarr"))
    print(f"Scanning {len(episodes)} episodes in {data_dir}\n")

    bad, ok = 0, 0
    for ep in episodes:
        problems = check_episode(ep)
        if problems:
            bad += 1
            print(f"BAD  {ep.name}")
            for p in problems:
                print(f"     {p}")
            if not args.dry_run:
                bak = ep.with_suffix(".zarr.bak")
                ep.rename(bak)
                print(f"     → renamed to {bak.name}")
        else:
            ok += 1

    print(f"\n{ok} clean, {bad} bad" + (" (dry run — nothing renamed)" if args.dry_run else " (bad ones renamed to .zarr.bak)"))
    if bad > 0 and not args.dry_run:
        print("Delete paths.pkl and re-run training.")


if __name__ == "__main__":
    main()
