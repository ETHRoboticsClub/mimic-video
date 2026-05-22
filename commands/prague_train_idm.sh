#!/usr/bin/env bash
# Launch IDM training inside the mimic-video Apptainer container.
# Usage: ./commands/prague_train_idm.sh <config_name>
#   where <config_name> is a yaml stem under configs/so101/ (no path, no .yaml).
#   e.g. ./commands/prague_train_idm.sh idm_train_subsampled_dual_task_low_rank
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="/mnt/personal/korcadav/apptainer/mimic-video-cuda126"
CONFIG_DIR="configs/so101"
IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-/mnt/personal/korcadav/mimic-video-outputs/cosmos-lora}"

if [[ $# -lt 1 || -z "${1:-}" ]]; then
    echo "ERROR: missing config name." >&2
    echo "Usage: $0 <config_name>" >&2
    echo "Available configs under ${CONFIG_DIR}/:" >&2
    ls "${REPO_ROOT}/${CONFIG_DIR}" 2>/dev/null | sed 's/\.yaml$//' | sed 's/^/  /' >&2
    exit 2
fi

CONFIG_NAME="${1%.yaml}"
CONFIG="${CONFIG_DIR}/${CONFIG_NAME}.yaml"

if [[ ! -f "${REPO_ROOT}/${CONFIG}" ]]; then
    echo "ERROR: config not found at ${REPO_ROOT}/${CONFIG}" >&2
    exit 2
fi

if [[ ! -d "${CONTAINER}" ]]; then
    echo "ERROR: Apptainer sandbox not found at ${CONTAINER}" >&2
    exit 1
fi

exec apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind /mnt/personal:/mnt/personal \
    "${CONTAINER}" \
    bash -c "
        set -euo pipefail
        export REPO_ROOT='${REPO_ROOT}'
        export IMAGINAIRE_OUTPUT_ROOT='${IMAGINAIRE_OUTPUT_ROOT}'
        cd '${REPO_ROOT}'
        set -a; source .env; set +a
        source model/.venv/bin/activate

        # Recommended NCCL/TE settings for this training stack
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        export NVTE_FUSED_ATTN=0
        export TOKENIZERS_PARALLELISM=false
        export OMP_NUM_THREADS=8

        # Set DEBUG_CUDA=1 to make CUDA kernels synchronous so a failure
        # produces a Python frame pointing at the culprit op.  ~2x slower per
        # iter; only flip on while diagnosing.
        if [[ \"\${DEBUG_CUDA:-0}\" == \"1\" ]]; then
            export CUDA_LAUNCH_BLOCKING=1
            export TORCH_USE_CUDA_DSA=1
            export NCCL_DEBUG=INFO
            echo \"[prague_train_idm] DEBUG_CUDA=1: kernels are synchronous, NCCL verbose.\" >&2
        fi

        exec python scripts/so101_pipeline.py \
            --config '${CONFIG}'
    "
