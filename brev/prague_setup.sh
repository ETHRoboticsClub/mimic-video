set -euo pipefail

export UV_CACHE_DIR=/tmp/uv-cache-$USER
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# sudo apt-get update
# sudo apt-get install cuda-nvrtc-12-6
# sudo ldconfig

# install and set up libnvrtc
# echo /usr/local/cuda-12.6/targets/x86_64-linux/lib | sudo tee /etc/ld.so.conf.d/cuda-12-6.conf
# sudo ldconfig
# ldconfig -p | grep libnvrtc

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

cd "$REPO_ROOT/model"
uv sync --extra cu126
source .venv/bin/activate
python scripts/download_checkpoints.py --checkpoint-dir "$REPO_ROOT/model/checkpoints"

# cd "$REPO_ROOT/eval/libero"
# uv pip install -r LIBERO/requirements.txt
# uv pip install -e LIBERO

# Create robosuite macros file after robosuite has been installed.
# python -c "import pathlib, robosuite, runpy; runpy.run_path(str(pathlib.Path(robosuite.__file__).parent / 'scripts' / 'setup_macros.py'), run_name='__main__')"
