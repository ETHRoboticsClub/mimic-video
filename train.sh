#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/path/to/your/zarr_root}"
EXPERIMENT="${EXPERIMENT:-w2a_bridge_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz128}"
VIDEO_CKPT="${VIDEO_CKPT:-checkpoints/video_backbone/cosmos-predict2_v2w_480p_10fps.pt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-12341}"
MAX_ITER="${MAX_ITER:-7000}"

cd "$(dirname "$0")/model"

TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}" \
CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}" \
NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}" \
NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-0}" \
torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" -m scripts.train \
  --config=cosmos_predict2/configs/config.py \
  -- experiment="${EXPERIMENT}" \
  model.config.video_dit_path="${VIDEO_CKPT}" \
  trainer.max_iter="${MAX_ITER}"
