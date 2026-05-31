#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

S3_URI="${S3_URI:-s3://ethrc-ml-data-916780037007/robot-learning/teleop/}"
OUTPUT_DIR="${OUTPUT_DIR:-data/teleop_raw}"

cd "${REPO_ROOT}"
source ".env"

aws s3 sync "${S3_URI}" "${OUTPUT_DIR}" "$@"
