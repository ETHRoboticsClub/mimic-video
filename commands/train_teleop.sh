#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${EXPERIMENT:-w2a_yams_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz128}"
VIDEO_CKPT="${VIDEO_CKPT:-checkpoints/video_backbone/cosmos-predict2_v2w_480p_10fps.pt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-12341}"
NUM_VAL_EPISODES="${NUM_VAL_EPISODES:-0}"
RUN_VALIDATION="${RUN_VALIDATION:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/teleop_converted}"
TORCHRUN="${TORCHRUN:-${REPO_ROOT}/model/.venv/bin/torchrun}"

if [[ ! -x "${TORCHRUN}" ]]; then
  echo "torchrun not found at ${TORCHRUN}. Run 'uv sync --extra cu128' from ${REPO_ROOT}/model first." >&2
  exit 1
fi

cd "${REPO_ROOT}/model"

TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}" \
CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}" \
NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}" \
PYTHONPATH="${REPO_ROOT}/model:${PYTHONPATH:-}" \
"${TORCHRUN}" --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" -m scripts.train \
  --config=cosmos_predict2/configs/config.py \
  -- experiment="${EXPERIMENT}" \
  data_config.dataset.dataset.data_dir="${DATA_DIR}" \
  data_config.dataset.dataset.num_val_episodes="${NUM_VAL_EPISODES}" \
  trainer.run_validation="${RUN_VALIDATION}" \
  model.config.video_dit_path="${VIDEO_CKPT}" \
  "$@"
