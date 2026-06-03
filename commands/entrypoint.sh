#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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

detect_cpu_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  getconf _NPROCESSORS_ONLN 2>/dev/null || echo "1"
}

detect_mem_gib() {
  if [[ -r /proc/meminfo ]]; then
    awk '/MemTotal:/ { print int($2 / 1024 / 1024) }' /proc/meminfo
    return
  fi
  echo "0"
}

recommend_preprocess_workers() {
  local cpu_count="$1"
  local mem_gib="$2"
  local by_cpu=$((cpu_count / 6))
  local by_mem

  if [[ "${by_cpu}" -lt 1 ]]; then
    by_cpu=1
  fi

  if [[ "${mem_gib}" -gt 0 ]]; then
    by_mem=$((mem_gib / 32))
    if [[ "${by_mem}" -lt 1 ]]; then
      by_mem=1
    fi
  else
    by_mem="${by_cpu}"
  fi

  local workers="${by_cpu}"
  if [[ "${by_mem}" -lt "${workers}" ]]; then
    workers="${by_mem}"
  fi
  if [[ "${workers}" -gt 32 ]]; then
    workers=32
  fi
  echo "${workers}"
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

CPU_COUNT="$(detect_cpu_count)"
MEM_GIB="$(detect_mem_gib)"
export NUM_WORKERS="${NUM_WORKERS:-$(recommend_preprocess_workers "${CPU_COUNT}" "${MEM_GIB}")}"
log "preprocessing will use NUM_WORKERS=${NUM_WORKERS} (detected cpu=${CPU_COUNT}, mem=${MEM_GIB}GiB)"

run_step "setup aws/python environment" "${REPO_ROOT}/commands/other/setup_aws.sh"
run_step "download Cosmos checkpoint" "${REPO_ROOT}/commands/other/download_cosmos_checkpoint.sh"
run_step "download T5 checkpoint" "${REPO_ROOT}/commands/other/download_t5.sh"
run_step "download teleop recordings" "${REPO_ROOT}/commands/download.sh"
run_step "preprocess teleop recordings" "${REPO_ROOT}/commands/preprocess.sh"
run_step "precompute teleop language embeddings" "${REPO_ROOT}/commands/embeddings.sh"
run_step "train teleop model" "${REPO_ROOT}/commands/train.sh" "$@"
