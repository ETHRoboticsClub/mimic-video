#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET:-$REPO_ROOT/data/hf_datasets/yams-carton-box-closing-fri-tom-mat-varing-fan-position}"
ZARR="${ZARR:-$REPO_ROOT/data/zarr_yams-carton-box-closing-fri-tom-mat-varing-fan-position}"
T5_DIR="${T5_DIR:-$REPO_ROOT/model/checkpoints/text_encoder/t5-11b}"
VIDEO_CKPT="${VIDEO_CKPT:-$REPO_ROOT/model/checkpoints/video_backbone/cosmos-predict2_v2w_480p_10fps.pt}"

cd "$REPO_ROOT"

echo "=== $(date) starting train_yams pipeline ==="
echo "host: $(hostname)"
echo "dataset: $DATASET"
echo "zarr: $ZARR"

nvidia-smi || true

source "$REPO_ROOT/model/.venv/bin/activate"

if [[ ! -d "$ZARR" ]] || ! find "$ZARR" -maxdepth 1 -name "episode_*.zarr" | grep -q .; then
  echo "=== preprocessing LeRobot to Zarr ==="
  python data_preprocessing/action/process_lerobot.py \
    --dataset-path "$DATASET" \
    --output-dir "$ZARR" \
    --num-workers 1
else
  echo "=== Zarr already exists, skipping preprocessing ==="
fi

if [[ ! -d "$T5_DIR" ]]; then
  echo "ERROR: missing T5 encoder at $T5_DIR"
  echo "Cannot precompute language embeddings or train until this checkpoint is present."
  exit 2
fi

echo "=== precomputing language embeddings ==="
python data_preprocessing/action/precompute_t5.py --dataset-path "$ZARR"

if [[ ! -f "$VIDEO_CKPT" ]]; then
  echo "ERROR: missing video checkpoint at $VIDEO_CKPT"
  exit 3
fi

echo "=== starting training ==="
DATA_DIR="$ZARR" bash commands/train.sh
