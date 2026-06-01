#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${FROZEN_CHECKPOINT_DIR:-$REPO_ROOT/model/checkpoints}}"
S3_URI="${S3_URI:-s3://ethrc-ml-data-916780037007/robot-learning/checkpoints/cosmos/iter_000007000_fused.pt}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CHECKPOINT_DIR/video_backbone/cosmos-predict2_v2w_480p_10fps.pt}"

if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: aws CLI is not installed or not on PATH." >&2
  exit 1
fi

mkdir -p "$(dirname "$CHECKPOINT_PATH")"

if [[ -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint already exists: $CHECKPOINT_PATH"
  echo "Set OVERWRITE=1 to download it again."
  if [[ "${OVERWRITE:-0}" != "1" ]]; then
    exit 0
  fi
fi

echo "Downloading Cosmos checkpoint:"
echo "  from: $S3_URI"
echo "  to:   $CHECKPOINT_PATH"
aws s3 cp "$S3_URI" "$CHECKPOINT_PATH"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "ERROR: checkpoint was not created: $CHECKPOINT_PATH" >&2
  exit 1
fi

echo "Cosmos checkpoint download complete."
