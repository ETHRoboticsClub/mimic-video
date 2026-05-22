set -euo pipefail

export UV_CACHE_DIR=/tmp/uv-cache-$USER
source "$HOME/.local/bin/env"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT"
source "$REPO_ROOT/model/.venv/bin/activate"

cd eval/libero

uv pip install -r LIBERO/requirements.txt
uv pip install -e LIBERO

echo "MAKE SURE TO SET eval.sh TO THE RIGHT CHECKPOINT!"

export PYTHONWARNINGS="ignore::DeprecationWarning"
bash eval.sh
