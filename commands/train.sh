#!/usr/bin/env bash
set -euo pipefail

# Script variables
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# User variables
# ----------
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/zarr_yams-carton-box-closing-fri-tom-mat-varing-fan-position}"
EXPERIMENT="${EXPERIMENT:-w2a_yams_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1}"
VIDEO_CKPT="${VIDEO_CKPT:-checkpoints/video_backbone/cosmos-predict2_v2w_480p_10fps.pt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-12341}"
# ----------

cd "$REPO_ROOT"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

cd "$REPO_ROOT/model"
if [[ ! -f ".venv/bin/activate" ]]; then
  echo "ERROR: model/.venv does not exist. Create it first with: cd model && uv sync --extra cu128" >&2
  exit 1
fi
source .venv/bin/activate

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. This instance does not look GPU-ready." >&2
  exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: DATA_DIR does not exist: $DATA_DIR" >&2
  exit 1
fi

TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}" \
CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}" \
NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}" \
torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" -m scripts.train \
  --config=cosmos_predict2/configs/config.py \
  -- experiment="$EXPERIMENT" \
  data_config.dataset.dataset.data_dir="$DATA_DIR" \
  model.config.video_dit_path="$VIDEO_CKPT"
