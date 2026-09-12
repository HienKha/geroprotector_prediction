#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 outputs/RUN_ID" >&2
  exit 2
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
export PYTHONPATH="${ROOT_DIR}/src"

"${PYTHON_BIN}" -m geroprotector.cli --root "${ROOT_DIR}" report "$1"
