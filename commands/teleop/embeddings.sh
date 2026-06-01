#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${DATASET_PATH:-data/teleop_converted}"
if [[ $# -gt 0 && "${1:0:2}" != "--" ]]; then
  DATASET_PATH="$1"
  shift
fi

CACHE_PATH="${CACHE_PATH:-${DATASET_PATH}/t5_instruction_cache.pkl}"
UV_RUN_ARGS="${UV_RUN_ARGS:---extra cu128}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

PYTHONPATH=model:. uv run --project model ${UV_RUN_ARGS} python data_preprocessing/action/precompute_t5.py \
  --dataset-path "${DATASET_PATH}" \
  --instruction-source attrs \
  --cache-path "${CACHE_PATH}" \
  "$@"
