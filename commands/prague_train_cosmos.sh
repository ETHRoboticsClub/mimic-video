#!/usr/bin/env bash
# Launch Cosmos backbone LoRA training inside the mimic-video Apptainer container.
# Usage: ./commands/prague_train_cosmos.sh <config_name>
#   where <config_name> is a yaml stem under configs/so101/ (no path, no .yaml).
#   e.g. ./commands/prague_train_cosmos.sh video_finetune_subsampled_dual_task
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="/mnt/personal/korcadav/apptainer/mimic-video-cuda126"
CONFIG_DIR="configs/so101"

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

IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-/mnt/personal/korcadav/mimic-video-outputs/cosmos-lora}"

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
        # Reduce CUDA allocator fragmentation overhead — recovers reserved-but-
        # unallocated memory under bursty activation spikes (e.g. DiT MLP up-proj).
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        exec python scripts/so101_pipeline.py \
            --config '${CONFIG}'
    "
