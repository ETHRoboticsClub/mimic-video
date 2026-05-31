#!/usr/bin/env bash
set -euo pipefail

S3_URI="${S3_URI:-s3://ethrc-ml-data-916780037007/robot-learning/teleop/}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/Downloads/recordings}"

aws s3 sync "${S3_URI}" "${OUTPUT_DIR}" "$@"
