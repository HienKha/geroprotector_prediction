#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 v6_RUN_ID [core|full]" >&2
  exit 2
fi
RUN_ID="$1"
SUITE="${2:-full}"
[[ "${RUN_ID}" =~ ^v6_[a-z0-9_.-]+$ ]] || {
  echo "RUN_ID must start with v6_ and use lowercase safe characters" >&2
  exit 2
}
[[ "${SUITE}" == "core" || "${SUITE}" == "full" ]] || exit 2

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
export PYTHONPATH="${ROOT_DIR}/src"
export TABPFN_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"${PYTHON_BIN}" -m geroprotector.cli --root "${ROOT_DIR}" run plan \
  --config configs/v6.yaml --suite "${SUITE}"
"${PYTHON_BIN}" -m geroprotector.cli --root "${ROOT_DIR}" run nested-cv \
  --config configs/v6.yaml --suite "${SUITE}" --run-id "${RUN_ID}"
