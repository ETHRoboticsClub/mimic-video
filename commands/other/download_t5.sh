#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT_DIR="$REPO_ROOT/model/checkpoints"
T5_DIR="$CHECKPOINT_DIR/text_encoder/t5-11b"
NVME="${NVME:-/nvme}"
if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
  if [[ -d "/workspace/.venv" ]]; then
    export UV_PROJECT_ENVIRONMENT="/workspace/.venv"
  else
    export UV_PROJECT_ENVIRONMENT="${NVME}/mimic-video-venv"
  fi
fi
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"

cd "$REPO_ROOT/model"
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"

echo "Downloading T5-11B text encoder to $T5_DIR"
if [[ -f "$T5_DIR/pytorch_model.bin" ]]; then
  echo "Found existing pytorch_model.bin; fetching only the remaining T5 files."
  hf download jonpai/mimic-video \
    --include "text_encoder/t5-11b/*" \
    --exclude "text_encoder/t5-11b/pytorch_model.bin" \
    --local-dir "$CHECKPOINT_DIR"
else
  hf download jonpai/mimic-video \
    --include "text_encoder/t5-11b/*" \
    --local-dir "$CHECKPOINT_DIR"
fi

if [[ ! -f "$T5_DIR/pytorch_model.bin" ]]; then
  echo "Expected $T5_DIR/pytorch_model.bin, but it was not created." >&2
  echo "The Hugging Face cache may contain an unfinished .incomplete staging file." >&2
  exit 1
fi

echo "T5 download complete."
