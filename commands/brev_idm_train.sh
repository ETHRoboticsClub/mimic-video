#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
set -a; source .env; set +a

pwd

source model/.venv/bin/activate

if [[ $# -lt 1 || -z "${1:-}" ]]; then
    echo "Usage: $0 <config_path>" >&2
    exit 2
fi

CONFIG=configs/so101/"$1"

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: config not found at ${REPO_ROOT}/configs/so101/${CONFIG}" >&2
    exit 2
fi

python scripts/so101_pipeline.py --config "${CONFIG}"
