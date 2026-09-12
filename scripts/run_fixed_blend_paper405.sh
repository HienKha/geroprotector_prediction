#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
RUN_ID="${1:-fixedblend405_20260817}"
POSITIVE="${2:-$ROOT/artifacts/data/raw/upstream/geroprotectors_reported.tsv}"
NEGATIVE="${3:-$ROOT/artifacts/data/raw/upstream/no_geroprotectors_and_toxicos.csv}"

if [[ ! "$RUN_ID" =~ ^fixedblend405_[a-z0-9_.-]+$ ]]; then
  echo "RUN_ID must start with fixedblend405_" >&2
  exit 2
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TABPFN_DISABLE_TELEMETRY=true
exec "$PYTHON_BIN" -m geroprotector.cli --root "$ROOT" fixed-blend-paper405 \
  --config "$ROOT/configs/fixed_blend_paper405.yaml" \
  --positive "$POSITIVE" \
  --negative "$NEGATIVE" \
  --run-id "$RUN_ID"
