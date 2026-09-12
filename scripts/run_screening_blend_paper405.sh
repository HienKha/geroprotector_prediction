#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
RUN_ID="${1:-screeningblend405_20260817}"
WEIGHTED_RUN="${2:-$ROOT/outputs/weightedblend405_20260817}"
TRADITIONAL_RUN="${3:-$ROOT/outputs/traditional405_20260817}"
POSITIVE="${4:-$ROOT/artifacts/data/raw/upstream/geroprotectors_reported.tsv}"
NEGATIVE="${5:-$ROOT/artifacts/data/raw/upstream/no_geroprotectors_and_toxicos.csv}"

if [[ ! "$RUN_ID" =~ ^screeningblend405_[a-z0-9_.-]+$ ]]; then
  echo "RUN_ID must start with screeningblend405_" >&2
  exit 2
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TABPFN_DISABLE_TELEMETRY=true
exec "$PYTHON_BIN" -m geroprotector.cli --root "$ROOT" screening-blend-paper405 \
  --config "$ROOT/configs/screening_blend_paper405.yaml" \
  --weighted-run "$WEIGHTED_RUN" \
  --traditional-run "$TRADITIONAL_RUN" \
  --positive "$POSITIVE" \
  --negative "$NEGATIVE" \
  --run-id "$RUN_ID"
