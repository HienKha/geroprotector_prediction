#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
STAMP="${1:-20260825}"
RUN_ID="repeated_rank_stability_${STAMP}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
if [[ -d "outputs/$RUN_ID" ]]; then
  "$PYTHON_BIN" scripts/verify_completed_run.py "outputs/$RUN_ID"
  echo "verified and skipping: outputs/$RUN_ID"; exit 0
fi
# ~10 repeats x 5 folds x (8 base models); seed 42 gates seeds 43..51 on parity.
"$PYTHON_BIN" -m geroprotector.repeated_rank_stability \
  --root "$ROOT" --config "$ROOT/configs/repeated_rank_stability_protocol.yaml" \
  --run-id "$RUN_ID"
