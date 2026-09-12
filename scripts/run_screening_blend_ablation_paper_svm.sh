#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
RUN_ID="${1:-screeningblend_ablation_papersvm_20260818}"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Unsafe run id: $RUN_ID" >&2
  exit 2
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m geroprotector.screening_blend_ablation \
  --root "$ROOT" \
  --config "$ROOT/configs/screening_blend_ablation_paper_svm_protocol.yaml" \
  --run-id "$RUN_ID"
