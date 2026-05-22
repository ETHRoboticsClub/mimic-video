set -euo pipefail

export UV_CACHE_DIR=/tmp/uv-cache-$USER
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- CUDA libnvrtc ---
sudo apt-get update -q
sudo apt-get install -y cuda-nvrtc-12-6
echo /usr/local/cuda-12.6/targets/x86_64-linux/lib | sudo tee /etc/ld.so.conf.d/cuda-12-6.conf
sudo ldconfig
ldconfig -p | grep libnvrtc

# --- uv ---
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# --- Python env ---
cd "$REPO_ROOT/model"
uv sync --extra cu126
source .venv/bin/activate

# --- credentials ---
read -rsp "HuggingFace token (hf_...): " HF_TOKEN && echo
read -rsp "Weights & Biases API key: " WANDB_API_KEY && echo

huggingface-cli login --token "$HF_TOKEN"
wandb login "$WANDB_API_KEY"

# --- checkpoints (video backbone only) ---
python scripts/download_checkpoints.py \
    --models pretrained_cosmos_bridge \
    --checkpoint-dir "$REPO_ROOT/model/checkpoints"

echo ""
echo "Setup complete. Venv active. Run training with:"
echo "  bash $REPO_ROOT/brev/run_video_finetune.sh"
