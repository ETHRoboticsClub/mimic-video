#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="$REPO_ROOT/model/checkpoints"
T5_DIR="${T5_DIR:-$CHECKPOINT_DIR/text_encoder/t5-11b}"
S3_URI="${S3_URI:-s3://ethrc-ml-data-916780037007/robot-learning/frozen_models/t5/t5-11b/}"

cd "$REPO_ROOT"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: aws CLI is not installed or not on PATH." >&2
  exit 1
fi

mkdir -p "$T5_DIR"

echo "Downloading T5-11B text encoder:"
echo "  from: $S3_URI"
echo "  to:   $T5_DIR"
# `aws s3 sync` only transfers missing/changed files, so re-runs resume safely.
aws s3 sync "$S3_URI" "$T5_DIR"

if [[ ! -f "$T5_DIR/pytorch_model.bin" ]]; then
  echo "Expected $T5_DIR/pytorch_model.bin, but it was not created." >&2
  echo "Check that $S3_URI contains the t5-11b files." >&2
  exit 1
fi

echo "T5 download complete."
