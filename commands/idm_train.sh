#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
set -a; source .env; set +a

pwd

source model/.venv/bin/activate

python scripts/so101_pipeline.py --config configs/so101/idm_train.yaml
