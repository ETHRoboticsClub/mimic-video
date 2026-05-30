REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT/model"
source .venv/bin/activate

hf download jonpai/mimic-video \
  --include "text_encoder/t5-11b/*" \
  --local-dir checkpoints
