#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/path/to/your/zarr_root}"
EXPERIMENT="${EXPERIMENT:-w2a_bridge_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1}"

cd "$(dirname "$0")/model"

torchrun --nproc_per_node=1 -m scripts.train \
  --config=cosmos_predict2/configs/config.py \
  --dryrun \
  -- experiment="${EXPERIMENT}" \
  data_config.dataset.dataset.data_dir="${DATA_DIR}"
