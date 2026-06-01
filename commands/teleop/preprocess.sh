#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-data/teleop_raw}"
OUTPUT_DIR="${OUTPUT_DIR:-data/teleop_converted}"
MAX_SYNC_MS="${MAX_SYNC_MS:-50}"
NUM_WORKERS="${NUM_WORKERS:-1}"
UV_RUN_ARGS="${UV_RUN_ARGS:---extra cu128}"
OVERWRITE="${OVERWRITE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

OVERWRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" || "${OVERWRITE}" == "TRUE" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi

PYTHONPATH=. uv run --project model ${UV_RUN_ARGS} python data_preprocessing/action/process_recordings.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-sync-ms "${MAX_SYNC_MS}" \
  --num-workers "${NUM_WORKERS}" \
  "${OVERWRITE_ARGS[@]}" \
  "$@"
