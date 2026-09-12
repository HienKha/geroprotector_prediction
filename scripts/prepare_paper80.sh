#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 /abs/prior_ml.csv /abs/reported.tsv /abs/weak_refs.csv" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
export PYTHONPATH="${ROOT_DIR}/src"

"${PYTHON_BIN}" -m geroprotector.cli --root "${ROOT_DIR}" data acquire \
  --local-source "prior_ml_unlabeled=$1" \
  --local-source "reported_positive=$2" \
  --local-source "weak_chembl_reference=$3"
"${PYTHON_BIN}" -m geroprotector.cli --root "${ROOT_DIR}" data curate
"${PYTHON_BIN}" -m geroprotector.cli --root "${ROOT_DIR}" paper-splits build
"${PYTHON_BIN}" -m geroprotector.cli --root "${ROOT_DIR}" paper-splits audit
