#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-data/teleop_raw}"
OUTPUT_DIR="${OUTPUT_DIR:-data/teleop_converted}"
MAX_SYNC_MS="${MAX_SYNC_MS:-50}"
NUM_WORKERS="${NUM_WORKERS:-1}"

PYTHONPATH=. uv run --project model python data_preprocessing/action/process_recordings.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-sync-ms "${MAX_SYNC_MS}" \
  --num-workers "${NUM_WORKERS}" \
  --overwrite \
  "$@"
