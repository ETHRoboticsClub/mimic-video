#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UV_SYNC_ARGS="${UV_SYNC_ARGS:---extra cu128}"
NVME="${NVME:-/nvme}"
if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
  if [[ -d "/workspace/.venv" ]]; then
    export UV_PROJECT_ENVIRONMENT="/workspace/.venv"
  else
    export UV_PROJECT_ENVIRONMENT="${NVME}/mimic-video-venv"
  fi
fi
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [[ -f "$HOME/.local/bin/env" ]]; then
  source "$HOME/.local/bin/env"
fi

cd "${REPO_ROOT}/model"
uv sync ${UV_SYNC_ARGS}
