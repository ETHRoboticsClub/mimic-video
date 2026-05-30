#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/path/to/your/zarr_root}"
EXPERIMENT="${EXPERIMENT:-w2a_bridge_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz128}"
VIDEO_CKPT="${VIDEO_CKPT:-checkpoints/video_backbone/cosmos-predict2_v2w_480p_10fps.pt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-12341}"
ACTION_ATTN_BACKEND="${ACTION_ATTN_BACKEND:-torch}"
MAX_VAL_ITER="${MAX_VAL_ITER:-3}"

cd "$(dirname "$0")/model"

TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}" \
CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}" \
NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}" \
torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" -m scripts.train \
  --config=cosmos_predict2/configs/config.py \
  -- experiment="${EXPERIMENT}" \
  data_config.data_dir="${DATA_DIR}" \
  world2action_pipe.net.atten_backend="${ACTION_ATTN_BACKEND}" \
  model.config.pipe_config.net.atten_backend="${ACTION_ATTN_BACKEND}" \
  trainer.max_val_iter="${MAX_VAL_ITER}" \
  model.config.video_dit_path="${VIDEO_CKPT}"
