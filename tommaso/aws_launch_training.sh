#!/usr/bin/env bash
# Launch the libero_object action decoder training on AWS, in a tmux session
# so it survives any SSH disconnect.
#
# Run AFTER the smoke test (200 iters with save_iter=100) has produced
# at least 2 checkpoints under checkpoints/vam/libero/<EXP>/.
set -euo pipefail

cd /home/ubuntu/workspace/mimic-video/model
source .venv/bin/activate

# Make sure CUDA env is set for transformer_engine
export PATH="$HOME/.local/bin:$PATH"
NVIDIA_LIBS=$(find "$PWD/.venv/lib/python3.10/site-packages/nvidia" -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"
export CUDA_HOME="$PWD/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc"

# Distributed training env
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export CUDA_DEVICE_MAX_CONNECTIONS=1

# WandB (assumes ~/.netrc has the API key OR WANDB_API_KEY env var is set)
export WANDB_PROJECT=vam
export WANDB_TAGS=libero,libero_object,ethrc_unified_backbone,aws_8xa100_40gb

EXP=w2a_libero_object_full_v2w_libero_cosmos_unified_iter_000007000_fused_lr1.000e-04_layer20_bsz128

# Verify the data_dir is set
DATA_DIR=$(grep "data_dir:" cosmos_predict2/configs/dataloading/libero_object_full.yaml | awk '{print $2}')
echo "data_dir from config: $DATA_DIR"
[ -d "$DATA_DIR" ] || { echo "ERROR: data_dir does not exist on disk"; exit 1; }
echo "zarrs in data_dir: $(ls $DATA_DIR | wc -l)"

# Start tmux session named libero_object_train_<date>
SESSION=libero_object_train_$(date +%Y%m%d_%H%M%S)
echo "Launching tmux session: $SESSION"

tmux new-session -d -s "$SESSION" -c /home/ubuntu/workspace/mimic-video/model "
source .venv/bin/activate
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export CUDA_DEVICE_MAX_CONNECTIONS=1
NVIDIA_LIBS=\$(find \$PWD/.venv/lib/python3.10/site-packages/nvidia -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH=\${NVIDIA_LIBS}\${LD_LIBRARY_PATH:-}
export CUDA_HOME=\$PWD/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc
export WANDB_PROJECT=vam
export WANDB_TAGS=libero,libero_object,ethrc_unified_backbone,aws_8xa100_40gb

torchrun --nproc_per_node=8 --master_port=12341 \\
  -m scripts.train --config=cosmos_predict2/configs/config.py \\
  -- experiment=$EXP \\
  trainer.max_iter=50000 \\
  trainer.run_validation=False \\
  trainer.logging_iter=100 \\
  checkpoint.save_iter=5000 \\
  dataloader_train.num_workers=12 \\
  dataloader_train.prefetch_factor=4 \\
  2>&1 | tee /home/ubuntu/workspace/mimic-video/model/checkpoints/vam/libero/${EXP}/train.log
"

echo
echo "Training launched in tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Detach (within tmux): Ctrl+b then d"
echo "WandB: project=vam, group=libero, name=${EXP}"
echo
echo "Save cadence: every 5000 iters → 10 saves over 50K iters"
echo "Expected wall-clock: ~10-13 hours on 8x A100 40GB"
