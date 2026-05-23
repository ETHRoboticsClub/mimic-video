curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cd model
CUDA_EXTRA=${1:-cu130}
uv sync --extra "$CUDA_EXTRA"
source .venv/bin/activate
