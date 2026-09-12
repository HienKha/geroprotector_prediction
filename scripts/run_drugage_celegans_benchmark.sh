#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
STAMP="${1:-20260825}"
RUN_ID="drugage_celegans_benchmark_${STAMP}"
POSITIVE="$ROOT/artifacts/data/raw/upstream/geroprotectors_reported.tsv"
NEGATIVE="$ROOT/artifacts/data/raw/upstream/no_geroprotectors_and_toxicos.csv"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
if [[ -d "outputs/$RUN_ID" ]]; then
  "$PYTHON_BIN" scripts/verify_completed_run.py "outputs/$RUN_ID"
  echo "verified and skipping: outputs/$RUN_ID"; exit 0
fi
"$PYTHON_BIN" -m geroprotector.drugage_celegans_benchmark \
  --root "$ROOT" --config "$ROOT/configs/drugage_celegans_benchmark_protocol.yaml" \
  --positive "$POSITIVE" --negative "$NEGATIVE" --run-id "$RUN_ID"
