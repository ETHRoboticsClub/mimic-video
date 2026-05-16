curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cd model
uv sync --extra cu126
source .venv/bin/activate