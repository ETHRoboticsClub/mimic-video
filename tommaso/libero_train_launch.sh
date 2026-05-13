#!/usr/bin/env bash
# libero_object action-decoder training launch script.
# AWS 8x A100 40GB (the cheaper instance), training ETHRC libero-finetuned cosmos backbone.
#
# Backbone: s3://ethrc-ml-data-916780037007/robot-learning/cosmos-predict2-libero/.../2b_libero_cosmos/iter_000007000.pt
# Data:    libero_object_full (HF/S3 LIBERO-Object zarrs)
# Eval:    `eval/libero/run.py` (libero-sim, 10 rollouts)
#
# NOTE re: WandB:
#   The currently-running libero training was launched BEFORE the WandbLogger
#   callback was added (2026-05-11). Live libero run therefore has NO wandb
#   metrics streaming — only the train.log on disk.
#   For future libero relaunches, rsync these two files to the libero AWS box
#   before running this script:
#     model/cosmos_predict2/callbacks/wandb_logger.py
#     model/cosmos_predict2/configs/defaults/callbacks.py
#   then the env vars below will take effect.
set -euo pipefail

cd /home/ubuntu/workspace/mimic-video/model
source .venv/bin/activate

# --- env: cuda + nccl + dataloader + wandb ---
ulimit -n 1048576
NVIDIA_LIBS=$(find $PWD/.venv/lib/python3.10/site-packages/nvidia -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH=${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}
export CUDA_HOME=$PWD/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc
export PATH="$HOME/.local/bin:$PATH"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=vam
export WANDB_ENTITY=eth-robotics-club
export WANDB_TAGS=libero,libero_object,ethrc_unified_backbone,aws_8xa100_40gb

EXP=w2a_libero_object_full_v2w_libero_cosmos_unified_iter_000007000_fused_lr1.000e-04_layer20_bsz128
export WANDB_RUN_NAME=$EXP
LOGDIR=checkpoints/vam/libero/$EXP
mkdir -p $LOGDIR
# rm -rf $LOGDIR/checkpoints   # uncomment for a fully-clean start; otherwise resumes

# Sanity check dataset path
DATA_DIR=$(grep "data_dir:" cosmos_predict2/configs/dataloading/libero_object_full.yaml | awk '{print $2}')
[ -d "$DATA_DIR" ] || { echo "ERROR: data_dir does not exist: $DATA_DIR"; exit 1; }
echo "data_dir: $DATA_DIR ($(ls $DATA_DIR | wc -l) zarrs)"

# --- training ---
#  8 GPUs * micro=4 = bsz=32 global, grad_accum=4 -> effective bsz=128
#  micro=16 would overflow on 40 GB cards
#  save_iter=5000 -> 10 saves over 50K iters
torchrun --nproc_per_node=8 --master_port=12341 \
  -m scripts.train --config=cosmos_predict2/configs/config.py \
  -- experiment=$EXP \
  trainer.max_iter=50000 \
  trainer.run_validation=False \
  trainer.logging_iter=100 \
  trainer.grad_accum_iter=4 \
  checkpoint.save_iter=5000 \
  dataloader_train.batch_size=4 \
  dataloader_train.num_workers=12 \
  dataloader_train.prefetch_factor=4 \
  optimizer.lr=1.0e-04 2>&1 | tee $LOGDIR/train.log
