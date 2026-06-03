#!/usr/bin/env bash
set -euo pipefail

source .env.secrets
source .env.paths

image_tag="cosmos-predict2.5:nightly"

mkdir -p "${NVME}/hf-home" "${NVME}/container-root-cache" outputs

docker run -it --runtime=nvidia --ipc=host --rm \
  -v .:/workspace \
  -v /workspace/.venv \
  -v "${NVME}/container-root-cache:/root/.cache" \
  -v "${NVME}:/nvme" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_HOME="/nvme/hf-home" \
  -e NVME="/nvme" \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  -e IMAGINAIRE_OUTPUT_ROOT="$IMAGINAIRE_OUTPUT_ROOT" \
  "$image_tag"
