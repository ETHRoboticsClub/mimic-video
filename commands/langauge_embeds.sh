#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/zarr_yams-carton-box-closing-fri-tom-mat-varing-fan-position}"

cd "$REPO_ROOT"
source "$REPO_ROOT/model/.venv/bin/activate"
PYTHONPATH=model python data_preprocessing/action/precompute_t5.py --dataset-path "$DATA_DIR"
