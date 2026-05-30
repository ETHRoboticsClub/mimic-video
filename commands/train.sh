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
OPTIMIZER="${OPTIMIZER:-fusedadamw}"
ACTION_ATTN_BACKEND="${ACTION_ATTN_BACKEND:-torch}"
MAX_VAL_ITER="${MAX_VAL_ITER:-3}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
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

if [[ ! "$EXPERIMENT" == w2a_yams_* ]]; then
  echo "ERROR: EXPERIMENT must use the YAMS data_config, got: $EXPERIMENT" >&2
  exit 1
fi

if [[ ! -f "$VIDEO_CKPT" ]]; then
  echo "ERROR: VIDEO_CKPT does not exist: $VIDEO_CKPT" >&2
  exit 1
fi

if [[ "$OPTIMIZER" == "fusedadamw" ]]; then
  if ! python - <<'PY' >/dev/null 2>&1
from apex.multi_tensor_apply import multi_tensor_applier  # noqa: F401
PY
  then
    cat >&2 <<'EOF'
ERROR: OPTIMIZER=fusedadamw requires NVIDIA Apex, but apex.multi_tensor_apply is not importable.

Install the CUDA 12.8 extras into this environment:
  cd model && uv sync --extra cu128

Or use the slower PyTorch optimizer for this run:
  OPTIMIZER=adamw ./commands/train.sh
EOF
    exit 1
  fi
fi

DATA_DIR="$DATA_DIR" python - <<'PY'
import os
from pathlib import Path

import zarr

data_dir = Path(os.environ["DATA_DIR"]).expanduser()
episodes = sorted(data_dir.glob("episode_*.zarr"))
if len(episodes) < 3:
    raise SystemExit(f"ERROR: need at least 3 episodes for 2 validation episodes + training, found {len(episodes)}")

required = {
    "workspace_rgb",
    "workspace_rgb_timestamps",
    "joint_state_lowdim",
    "joint_state_lowdim_timestamps",
    "joint_action_lowdim",
    "joint_action_lowdim_timestamps",
    "language_embedding",
    "language_embedding_timestamps",
}

for episode in episodes:
    root = zarr.open(str(episode), mode="r")
    missing = sorted(required.difference(root.array_keys()))
    if missing:
        hint = ""
        if "language_embedding" in missing:
            hint = f'\nRun: DATA_DIR="{data_dir}" bash commands/langauge_embeds.sh'
        raise SystemExit(f"ERROR: {episode} missing arrays: {missing}{hint}")

    length = root["workspace_rgb"].shape[0]
    checks = {
        "workspace_rgb_timestamps": (length,),
        "joint_state_lowdim": (length, 14),
        "joint_state_lowdim_timestamps": (length,),
        "joint_action_lowdim": (length, 14),
        "joint_action_lowdim_timestamps": (length,),
        "language_embedding": (1, 512, 1024),
        "language_embedding_timestamps": (1,),
    }
    for key, expected in checks.items():
        actual = root[key].shape
        if actual != expected:
            raise SystemExit(f"ERROR: {episode}/{key} shape {actual} != expected {expected}")
PY

if [[ $? -ne 0 ]]; then
  exit 1
fi

TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}" \
CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}" \
NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}" \
torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" -m scripts.train \
  --config=cosmos_predict2/configs/config.py \
  -- experiment="$EXPERIMENT" \
  optimizer="$OPTIMIZER" \
  data_config.data_dir="$DATA_DIR" \
  world2action_pipe.net.atten_backend="$ACTION_ATTN_BACKEND" \
  model.config.pipe_config.net.atten_backend="$ACTION_ATTN_BACKEND" \
  trainer.max_val_iter="$MAX_VAL_ITER" \
  dataloader_train.batch_size.global_bsz="$GLOBAL_BATCH_SIZE" \
  model.config.video_dit_path="$VIDEO_CKPT"
