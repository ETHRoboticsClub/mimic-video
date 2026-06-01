#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

log() {
  echo "[teleop-entrypoint] $*" >&2
}

run_step() {
  local name="$1"
  shift
  log "starting: ${name}"
  "$@"
  log "finished: ${name}"
}

detect_gpu_count() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d '[:space:]'
    return
  fi
  echo "0"
}

cd "${REPO_ROOT}"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-$(detect_gpu_count)}"
if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  log "no GPUs detected; cannot start training"
  exit 1
fi
log "detected ${NPROC_PER_NODE} GPU(s); training will use NPROC_PER_NODE=${NPROC_PER_NODE}"

run_step "setup aws/python environment" "${REPO_ROOT}/commands/setup_aws.sh"
run_step "download Cosmos checkpoint" "${REPO_ROOT}/commands/download_cosmos_checkpoint.sh"
run_step "download T5 checkpoint" "${REPO_ROOT}/commands/download_t5.sh"
run_step "download teleop recordings" "${REPO_ROOT}/commands/teleop/download.sh"
run_step "preprocess teleop recordings" "${REPO_ROOT}/commands/teleop/preprocess.sh"
run_step "precompute teleop language embeddings" "${REPO_ROOT}/commands/teleop/embeddings.sh"
run_step "train teleop model" "${REPO_ROOT}/commands/teleop/train.sh" "$@"
