#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
export PYTHONPATH="${ROOT_DIR}/src"
export TABPFN_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CONFIG_PATH="${1:-configs/v6.yaml}"

"${PYTHON_BIN}" -m geroprotector.cli --root "${ROOT_DIR}" v6-checkpoints \
  --config "${CONFIG_PATH}"
