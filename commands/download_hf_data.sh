# User variables

# Script variables
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Setup
cd "$REPO_ROOT"
source ".env"
cd "$REPO_ROOT/model/"
uv sync
source .venv/bin/activate
cd "$REPO_ROOT"