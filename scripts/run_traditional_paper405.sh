#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 traditional405_RUN_ID [positive.tsv negative.csv]" >&2
  exit 2
fi

RUN_ID="$1"
[[ "${RUN_ID}" =~ ^traditional405_[a-z0-9_.-]+$ ]] || {
  echo "RUN_ID must start with traditional405_ and use lowercase safe characters" >&2
  exit 2
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
POSITIVE="${2:-${ROOT_DIR}/artifacts/data/raw/upstream/geroprotectors_reported.tsv}"
NEGATIVE="${3:-${ROOT_DIR}/artifacts/data/raw/upstream/no_geroprotectors_and_toxicos.csv}"
export PYTHONPATH="${ROOT_DIR}/src"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/freeze_package_manifest.py" --check
"${PYTHON_BIN}" -m geroprotector.cli --root "${ROOT_DIR}" traditional-paper405 \
  --config "${ROOT_DIR}/configs/traditional_paper405.yaml" \
  --positive "${POSITIVE}" \
  --negative "${NEGATIVE}" \
  --run-id "${RUN_ID}"
