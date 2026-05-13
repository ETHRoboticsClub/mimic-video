#!/usr/bin/env bash
# AWS-side setup: clone repo, download + fuse ETHRC backbone, register, smoke-test save_iter.
# Run AFTER the pem-key SSH in is established and zarrs have been rsynced to /workspace/data/.
set -euo pipefail

REPO_REMOTE=https://github.com/ETHRoboticsClub/mimic-video.git    # adjust if private/different
WORKSPACE=/home/ubuntu/workspace
ETHRC_BACKBONE_S3=s3://ethrc-ml-data-916780037007/robot-learning/cosmos-predict2-libero/outputs/posttraining/video2world_lora/2b_libero_cosmos/checkpoints/model/iter_000007000.pt
ETHRC_CONFIG_S3=s3://ethrc-ml-data-916780037007/robot-learning/cosmos-predict2-libero/outputs/posttraining/video2world_lora/2b_libero_cosmos/config.yaml
NEW_BACKBONE_NAME=v2w_libero_cosmos_unified_iter_000007000_fused

# ---------- 0. AMI prep ----------
sudo apt-get update -y
sudo apt-get install -y rsync
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

# ---------- 1. Clone (or rsync from 5090) ----------
# If repo isn't on github, comment this and rsync from 5090 instead.
if [ ! -d mimic-video ]; then
  git clone "$REPO_REMOTE" mimic-video || {
    echo "Git clone failed. rsync the 5090 mimic-video tree to $WORKSPACE/mimic-video/ instead."
    exit 1
  }
fi
cd mimic-video/model

# ---------- 2. uv + venv ----------
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source ~/.bashrc 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --extra cu126
source .venv/bin/activate

# Sanity: cosmos imports
python -c "import cosmos_predict2; print('cosmos_predict2 ok')"

# ---------- 3. Download + verify + fuse ETHRC backbone ----------
mkdir -p /tmp/cosmos_dl
aws s3 cp "$ETHRC_CONFIG_S3"   /tmp/cosmos_dl/ethrc_unified_config.yaml
aws s3 cp "$ETHRC_BACKBONE_S3" /tmp/cosmos_dl/iter_000007000.pt

LORA_ALPHA=$(grep -E "^\s*lora_alpha:" /tmp/cosmos_dl/ethrc_unified_config.yaml | head -1 | awk '{print $2}')
LORA_RANK=$(grep -E "^\s*lora_rank:"  /tmp/cosmos_dl/ethrc_unified_config.yaml | head -1 | awk '{print $2}')
echo "ETHRC LoRA config: alpha=$LORA_ALPHA rank=$LORA_RANK"
[[ "$LORA_ALPHA" == "16" ]] || { echo "WARN: expected alpha=16, got $LORA_ALPHA — verify before fusing"; exit 1; }

python scripts/fuse_lora_ckpt.py /tmp/cosmos_dl/iter_000007000.pt --alpha "$LORA_ALPHA"
mv /tmp/cosmos_dl/iter_000007000_fused.pt checkpoints/video_backbone/${NEW_BACKBONE_NAME}.pt
ls -lh checkpoints/video_backbone/${NEW_BACKBONE_NAME}.pt
rm /tmp/cosmos_dl/iter_000007000.pt

# ---------- 4. Code edits already in the 5090 working tree (rsync brings them) ----------
# - world2action_model.py:VIDEO_MODEL_CKPT_NAMES has the new entry registered
# - world2action.py:104-106 has save_iter hardcode removed
# - process_libero_s3.py exists for completeness
grep -q "$NEW_BACKBONE_NAME" cosmos_predict2/configs/defaults/world2action_model.py \
  || { echo "ERROR: $NEW_BACKBONE_NAME not registered in world2action_model.py"; exit 1; }

# ---------- 5. Sanity: experiment + DATA_CONFIGS ----------
python -c "
from cosmos_predict2.configs.defaults.data_action import DATA_CONFIGS
assert 'libero_object_full' in DATA_CONFIGS, f'libero_object_full not auto-discovered: {sorted(DATA_CONFIGS)}'
print('DATA_CONFIGS OK')
"

EXP=w2a_libero_object_full_${NEW_BACKBONE_NAME}_lr1.000e-04_layer20_bsz128
python -m scripts.train --config=cosmos_predict2/configs/config.py \
  -- experiment="$EXP" \
  trainer.max_iter=2 trainer.run_validation=False \
  dataloader_train.batch_size=1 \
  --print-config 2>&1 | grep -E "video_dit_path|save_iter" | head -5

# ---------- 6. save_iter smoke test (200 iters, save_iter=100) ----------
# Requires zarrs present at /workspace/data/libero_object_full/. Skip if absent.
if [ ! -d /workspace/data/libero_object_full ] && [ ! -d "$WORKSPACE/data/libero_object_full" ]; then
  echo "zarrs not yet present at /workspace/data/libero_object_full or $WORKSPACE/data/libero_object_full"
  echo "rsync them from the 5090, then re-run the smoke step manually:"
  cat <<EOF

  EXP=$EXP
  torchrun --nproc_per_node=1 -m scripts.train \\
    --config=cosmos_predict2/configs/config.py \\
    -- experiment="\$EXP" \\
    trainer.max_iter=200 \\
    trainer.run_validation=False \\
    trainer.logging_iter=50 \\
    checkpoint.save_iter=100 \\
    dataloader_train.batch_size=1 \\
    dataloader_train.num_workers=2

EOF
  exit 0
fi

# Wire the data_dir into libero_object_full.yaml
DATA_DIR=$WORKSPACE/data/libero_object_full
[ -d /workspace/data/libero_object_full ] && DATA_DIR=/workspace/data/libero_object_full
sed -i "s|data_dir:.*|data_dir: $DATA_DIR|" cosmos_predict2/configs/dataloading/libero_object_full.yaml
echo "Wired data_dir=$DATA_DIR"

torchrun --nproc_per_node=1 -m scripts.train \
  --config=cosmos_predict2/configs/config.py \
  -- experiment="$EXP" \
  trainer.max_iter=200 \
  trainer.run_validation=False \
  trainer.logging_iter=50 \
  checkpoint.save_iter=100 \
  dataloader_train.batch_size=1 \
  dataloader_train.num_workers=2

# Verify saves
SAVE_DIR=checkpoints/vam/libero/${EXP}
ls "$SAVE_DIR" 2>/dev/null
N_SAVES=$(find "$SAVE_DIR" -name "iter_*" -type d 2>/dev/null | wc -l)
echo "Saved checkpoints: $N_SAVES"
[[ "$N_SAVES" -ge 2 ]] || { echo "ERROR: expected >= 2 saves, got $N_SAVES"; exit 1; }

echo
echo "=========================================="
echo "AWS setup complete. Ready to launch full 50K training in tmux."
echo "EXP=$EXP"
echo "=========================================="
