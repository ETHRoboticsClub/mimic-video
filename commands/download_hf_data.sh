# Script variables
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# User variables
# ----------
HF_ORG="ETHRC"
HF_REPO="robot-learning-fs26"
DOWNLOAD_ROOT="$REPO_ROOT/data/hf_datasets/$HF_REPO"
# ----------

# Script variables
HF_PATH="$HF_ORG/$HF_REPO"

# Setup
cd "$REPO_ROOT"
source ".env"
cd "$REPO_ROOT/model/"
uv sync
source .venv/bin/activate
cd "$REPO_ROOT"

echo "You are signed in on Hugging Face as:"
hf auth whoami

hf download "$HF_PATH" --repo-type dataset --local-dir "$DOWNLOAD_ROOT"
