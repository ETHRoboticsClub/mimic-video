#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

NGPU=${NGPU:-4}
START_BATCH=${START_BATCH:-1}
MAX_BATCH=${MAX_BATCH:-64}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-12400}
LOG_ROOT=${LOG_ROOT:-/tmp/yams_batch_search}
mkdir -p "$LOG_ROOT"

batch=$START_BATCH
last_good=0
attempt=0

while [ "$batch" -le "$MAX_BATCH" ]; do
  attempt=$((attempt + 1))
  port=$((MASTER_PORT_BASE + attempt))
  run="bs_probe_ngpu${NGPU}_b${batch}"
  log="$LOG_ROOT/${run}.log"
  echo "==> probing local batch_size=$batch on NGPU=$NGPU"
  set +e
  WANDB_MODE=disabled RUN="$run" BATCH="$batch" NGPU="$NGPU" MASTER_PORT="$port" MAX_ITER=1 \
    bash scripts/yams_train_launch.sh \
      2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e

  if [ "$status" -eq 0 ]; then
    echo "OK batch_size=$batch"
    last_good=$batch
    batch=$((batch * 2))
  else
    echo "FAILED batch_size=$batch (see $log)"
    break
  fi
done

echo "largest passing probed local batch_size=$last_good"
if [ "$last_good" -gt 0 ]; then
  echo "effective batch with grad_accum=2 and NGPU=$NGPU: $((last_good * NGPU * 2))"
fi
