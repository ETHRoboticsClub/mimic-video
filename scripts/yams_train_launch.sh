#!/usr/bin/env bash
# bi_yams action-decoder training launch script
# AWS p4de.24xlarge (8x A100 80 GB), training ETHRC YAMS-finetuned cosmos
# Launched 2026-05-11 by Tommaso.
#
# This is the EXACT command currently running in tmux on the AWS box.
# tmux session: yams_train_20260511_155253
# instance:     ec2-user@54.89.85.197
#
# To resume / relaunch (in tmux):
#   tmux new -s yams_train
#   cd /workspace/mimic-video/model
#   bash /path/to/this/script
set -euo pipefail

cd model
source .venv/bin/activate

# --- env: cuda + nccl + dataloader + wandb ---
ulimit -n 1048576       # avoid dataloader bus errors at high parallel-worker count
NVIDIA_ROOT=$PWD/.venv/lib/python3.10/site-packages/nvidia
NVIDIA_LIBS=$(find $NVIDIA_ROOT -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH=$NVIDIA_ROOT/cu13/lib:${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}
export CUDA_HOME=$PWD/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=idm
export WANDB_ENTITY=eth-robotics-club
export WANDB_TAGS=bi_yams,real_robot,ethrc_yams_backbone,aws_8xa100_80gb,fs26_both_tasks,micro8_ga2

CONFIG_EXP=w2a_bi_yams_v2w_yams_unified_iter_000007000_fused_lr1.000e-04_layer20_bsz128
RUN_NAME=16may_run
export WANDB_RUN_NAME=$RUN_NAME
LOGDIR=/checkpoints/vam/bi_yams/$RUN_NAME
mkdir -p $LOGDIR
# rm -rf $LOGDIR/checkpoints   # uncomment for a fully-clean start; otherwise resumes

# --- training ---
#  8 GPUs * micro=8 = bsz=64 global, grad_accum=2 -> effective bsz=128
#  micro=16 would overflow on 80 GB cards (~80 GB used)
#  save_iter=200 -> first usable checkpoint at iter 200 (~90 min after launch)
torchrun --nproc_per_node=4 --master_port=12360 \
  -m scripts.train --config=cosmos_predict2/configs/config.py \
  -- experiment=$CONFIG_EXP \
  job.name=$RUN_NAME \
  trainer.max_iter=10000 \
  trainer.run_validation=True \
  trainer.run_initial_validation=False \
  trainer.validation_iter=2000 \
  trainer.max_val_iter=12 \
  trainer.logging_iter=100 \
  trainer.grad_accum_iter=2 \
  checkpoint.save_iter=200 \
  optimizer=adamw \
  dataloader_train.batch_size=8 \
  dataloader_train.num_workers=4 \
  dataloader_train.prefetch_factor=2 \
  optimizer.lr=1.0e-04 2>&1 | tee $LOGDIR/train.log
