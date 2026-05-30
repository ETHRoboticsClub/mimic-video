cd ~/mimic-video/model
source .venv/bin/activate

python scripts/download_checkpoints.py \
  --models pretrained_cosmos_bridge \
  --checkpoint-dir checkpoints