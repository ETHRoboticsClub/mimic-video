set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$HOME/workspace/mimic-video-outputs}"
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"

cd "$REPO_ROOT/model"
uv run python "$REPO_ROOT/scripts/so101_pipeline_for_vision_training.py" \
    --config "$REPO_ROOT/configs/so101/smoke.yaml"
