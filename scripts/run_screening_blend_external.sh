#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
RUN_ID="${1:-screeningblend_external_20260817}"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TABPFN_DISABLE_TELEMETRY=true

exec "$PYTHON_BIN" -m geroprotector.screening_blend_external \
  --root "$ROOT" \
  --config "$ROOT/configs/screening_blend_external_protocol.yaml" \
  --run-id "$RUN_ID"
