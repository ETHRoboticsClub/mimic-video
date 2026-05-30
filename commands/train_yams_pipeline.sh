#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET:-$REPO_ROOT/data/hf_datasets/yams-carton-box-closing-fri-tom-mat-varing-fan-position}"
ZARR="${ZARR:-$REPO_ROOT/data/zarr_yams-carton-box-closing-fri-tom-mat-varing-fan-position}"
VIDEO_CKPT="${VIDEO_CKPT:-$REPO_ROOT/model/checkpoints/video_backbone/cosmos-predict2_v2w_480p_10fps.pt}"
NUM_WORKERS="${NUM_WORKERS:-1}"

cd "$REPO_ROOT"
source "$REPO_ROOT/model/.venv/bin/activate"

echo "=== $(date) YAMS training pipeline ==="
echo "repo: $REPO_ROOT"
echo "dataset: $DATASET"
echo "zarr: $ZARR"
echo "video checkpoint: $VIDEO_CKPT"

if [[ ! -d "$DATASET" ]]; then
  echo "ERROR: missing LeRobot dataset at $DATASET" >&2
  echo "Run: bash commands/download_hf_data.sh" >&2
  exit 1
fi

echo "=== preprocessing LeRobot to Zarr ==="
python data_preprocessing/action/process_lerobot.py \
  --dataset-path "$DATASET" \
  --output-dir "$ZARR" \
  --num-workers "$NUM_WORKERS" \
  --overwrite

echo "=== precomputing language embeddings ==="
DATA_DIR="$ZARR" bash commands/langauge_embeds.sh

echo "=== validating converted Zarr ==="
python data_preprocessing/action/check_lerobot_zarr_quality.py \
  --dataset-path "$DATASET" \
  --output-dir "$ZARR" \
  --frame-samples 5

echo "=== starting training ==="
DATA_DIR="$ZARR" VIDEO_CKPT="$VIDEO_CKPT" bash commands/train.sh
