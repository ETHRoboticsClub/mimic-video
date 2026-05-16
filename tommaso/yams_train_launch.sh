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

cd /workspace/mimic-video/model
source .venv/bin/activate

# --- env: cuda + nccl + dataloader + wandb ---
ulimit -n 1048576       # avoid dataloader bus errors at high parallel-worker count
NVIDIA_LIBS=$(find $PWD/.venv/lib/python3.10/site-packages/nvidia -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH=${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}
export CUDA_HOME=$PWD/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=vam
export WANDB_ENTITY=eth-robotics-club
export WANDB_TAGS=bi_yams,real_robot,ethrc_yams_backbone,aws_8xa100_80gb,fs26_both_tasks,micro8_ga2

EXP=w2a_bi_yams_v2w_yams_unified_iter_000007000_fused_lr1.000e-04_layer20_bsz128
export WANDB_RUN_NAME=$EXP
LOGDIR=checkpoints/vam/bi_yams/$EXP
mkdir -p $LOGDIR
# rm -rf $LOGDIR/checkpoints   # uncomment for a fully-clean start; otherwise resumes

# --- training ---
#  8 GPUs * micro=8 = bsz=64 global, grad_accum=2 -> effective bsz=128
#  micro=16 would overflow on 80 GB cards (~80 GB used)
#  save_iter=200 -> first usable checkpoint at iter 200 (~90 min after launch)
torchrun --nproc_per_node=8 --master_port=12360 \
  -m scripts.train --config=cosmos_predict2/configs/config.py \
  -- experiment=$EXP \
  trainer.max_iter=10000 \
  trainer.run_validation=True \
  trainer.logging_iter=100 \
  trainer.grad_accum_iter=2 \
  checkpoint.save_iter=200 \
  dataloader_train.batch_size=8 \
  dataloader_train.num_workers=4 \
  dataloader_train.prefetch_factor=2 \
  optimizer.lr=1.0e-04 2>&1 | tee $LOGDIR/train.log
