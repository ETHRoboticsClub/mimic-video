#!/usr/bin/env bash

set -euo pipefail

HF_REPO="task1_2_subsampled_true_10fps_r64-2400"
HF_ORG="rl26-world-models"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT"


source model/.venv/bin/activate
source .env

hf download "$HF_ORG/$HF_REPO" --repo-type model --local-dir "checkpoints/$HF_REPO"