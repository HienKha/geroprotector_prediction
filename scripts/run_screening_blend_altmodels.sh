#!/usr/bin/env bash
set -euo pipefail

# Equal-weight (1/3, 1/3, 1/3) screening blends in which the tabular-foundation-model
# slot is replaced by BiSHop, TabM or TabNet.  This run only reads the sealed
# 2026-08-17/18 outputs; it writes a new directory and never touches them.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
RUN_ID="${1:-screeningblend_altmodels_20260819}"
POSITIVE="${2:-$ROOT/artifacts/data/raw/upstream/geroprotectors_reported.tsv}"
NEGATIVE="${3:-$ROOT/artifacts/data/raw/upstream/no_geroprotectors_and_toxicos.csv}"

if [[ ! "$RUN_ID" =~ ^screeningblend_altmodels_[a-z0-9_.-]+$ ]]; then
  echo "RUN_ID must start with screeningblend_altmodels_" >&2
  exit 2
fi

if [[ -e "$ROOT/outputs/$RUN_ID" ]]; then
  echo "Run directory already exists: $ROOT/outputs/$RUN_ID" >&2
  exit 2
fi

# pytorch-tabnet is the only package this experiment adds to the environment.
if ! "$PYTHON_BIN" -c "import pytorch_tabnet" >/dev/null 2>&1; then
  echo "pytorch-tabnet is missing.  Install it with:" >&2
  echo "  $PYTHON_BIN -m pip install pytorch-tabnet==4.1.0" >&2
  exit 3
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON_BIN" -m geroprotector.screening_blend_altmodels \
  --root "$ROOT" \
  --config "$ROOT/configs/screening_blend_altmodels_protocol.yaml" \
  --positive "$POSITIVE" \
  --negative "$NEGATIVE" \
  --run-id "$RUN_ID"
