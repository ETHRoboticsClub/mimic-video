#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${DATASET_PATH:-data/teleop_converted}"
if [[ $# -gt 0 && "${1:0:2}" != "--" ]]; then
  DATASET_PATH="$1"
  shift
fi

CACHE_PATH="${CACHE_PATH:-${DATASET_PATH}/t5_instruction_cache.pkl}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NVME="${NVME:-/nvme}"
if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
  if [[ -d "/workspace/.venv" ]]; then
    export UV_PROJECT_ENVIRONMENT="/workspace/.venv"
  else
    export UV_PROJECT_ENVIRONMENT="${NVME}/mimic-video-venv"
  fi
fi
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"

cd "${REPO_ROOT}"

PYTHONPATH=model:. uv run --project model python data_preprocessing/action/precompute_t5.py \
  --dataset-path "${DATASET_PATH}" \
  --instruction-source attrs \
  --cache-path "${CACHE_PATH}" \
  "$@"
