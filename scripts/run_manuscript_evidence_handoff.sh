#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
STAMP="${1:-20260826}"
RUN_ID="manuscript_evidence_handoff_${STAMP}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "outputs/$RUN_ID" ]]; then
  "$PYTHON_BIN" scripts/verify_completed_run.py "outputs/$RUN_ID"
  echo "verified and skipping: outputs/$RUN_ID"
  exit 0
fi
"$PYTHON_BIN" -m geroprotector.manuscript_evidence_handoff \
  --root "$ROOT" \
  --config "$ROOT/configs/manuscript_evidence_handoff_protocol.yaml" \
  --run-id "$RUN_ID"
