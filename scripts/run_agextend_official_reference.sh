#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${AGEXTEND_PYTHON:-${GERO_PYTHON:-python3}}"
STAMP="${1:-20260826}"
RUN_ID="agextend_official_reference_${STAMP}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "outputs/$RUN_ID" ]]; then
  "$PYTHON_BIN" scripts/verify_completed_run.py "outputs/$RUN_ID"
  echo "verified and skipping: outputs/$RUN_ID"
  exit 0
fi
EXTRA=()
if [[ "${AGEXTEND_REQUIRE_PARITY:-1}" == "1" ]]; then
  EXTRA+=(--require-parity)
fi
"$PYTHON_BIN" -m geroprotector.agextend_official_reference \
  --root "$ROOT" \
  --config "$ROOT/configs/agextend_official_reference_protocol.yaml" \
  --run-id "$RUN_ID" "${EXTRA[@]}"
