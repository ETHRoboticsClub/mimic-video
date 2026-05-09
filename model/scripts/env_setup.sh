#!/usr/bin/env bash
# Source this to set CUDA-related env for the mimic-video venv.
# Usage: . scripts/env_setup.sh
VENV=/home/ethrc/Desktop/mimic-video/model/.venv

NVIDIA_LIBS=$(find "$VENV/lib/python3.10/site-packages/nvidia" -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"
export CUDA_HOME="$VENV/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
export PATH="$VENV/bin:$PATH"

# transformer_engine flags carried over from the upstream launch hint
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
