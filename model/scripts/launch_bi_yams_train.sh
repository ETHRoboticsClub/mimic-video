#!/usr/bin/env bash
set -euo pipefail

# Full training launcher for the bi_yams action decoder, tuned for a single
# RTX 5090 (sm_120, 32GB VRAM) where apex's FusedAdam/FusedRMSNorm CUDA kernels
# are not compiled.
#
# Overrides vs the auto-generated experiment defaults:
#   * optimizer=adamw          plain torch.optim.AdamW (apex FusedAdam fails on sm_120)
#   * trainer.max_iter=40000   ~21h on 5090 at ~1.9s/iter with bsz=2, grad_accum=1
#   * dataloader_train.batch_size=2  fuse the two micro-steps into one batched
#                              forward (~1.7x faster: kernel-launch amortization
#                              + better SM occupancy + no DDP grad-sync per accum)
#   * trainer.grad_accum_iter=1  no accumulation needed at effective batch 2
#   * checkpoint.save_iter=10000  4 saves x ~3GB each ≈ 12GB total
#   * trainer.run_validation=False  9.5K val anchor frames @ ~108s each = infeasible at iter 0;
#                              do offline val from checkpoints instead
#   * trainer.logging_iter=50  log every 50 optim steps

cd /home/ethrc/Desktop/mimic-video/model
. scripts/env_setup.sh

EXP=w2a_bi_yams_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz1

exec torchrun --nproc_per_node=1 --master_port=12341 \
  -m scripts.train --config=cosmos_predict2/configs/config.py \
  -- experiment="$EXP" \
  optimizer=adamw \
  optimizer.lr=1.0e-04 \
  trainer.max_iter=40000 \
  trainer.grad_accum_iter=1 \
  trainer.run_validation=False \
  trainer.logging_iter=50 \
  checkpoint.save_iter=10000 \
  dataloader_train.batch_size=2 \
  dataloader_train.num_workers=4 \
  dataloader_train.prefetch_factor=2
