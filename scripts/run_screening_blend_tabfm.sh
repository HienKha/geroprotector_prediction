#!/usr/bin/env bash
set -euo pipefail

# Equal-thirds SVM / Tanimoto / TabFM blend (google-research/tabfm v1.0.1).
# Reads the sealed runs read-only; writes one new output directory.
#
# NOTE: TabFM pretrained weights are licensed `tabfm-non-commercial-v1.0`
# (non-commercial, non-production use only). Academic evaluation is in scope.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
RUN_ID="${1:-screeningblend_tabfm_20260821}"
POSITIVE="${2:-$ROOT/artifacts/data/raw/upstream/geroprotectors_reported.tsv}"
NEGATIVE="${3:-$ROOT/artifacts/data/raw/upstream/no_geroprotectors_and_toxicos.csv}"

if [[ ! "$RUN_ID" =~ ^screeningblend_tabfm_[a-z0-9_.-]+$ ]]; then
  echo "RUN_ID must start with screeningblend_tabfm_" >&2
  exit 2
fi
if [[ -e "$ROOT/outputs/$RUN_ID" ]]; then
  echo "Run directory already exists: $ROOT/outputs/$RUN_ID" >&2
  exit 2
fi
if ! "$PYTHON_BIN" -c "import tabfm" >/dev/null 2>&1; then
  echo "tabfm is missing. Install a locally staged upstream copy with:" >&2
  echo "  cd $ROOT/third_party/tabfm && $PYTHON_BIN -m pip install -e '.[pytorch]'" >&2
  exit 3
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

exec "$PYTHON_BIN" -m geroprotector.screening_blend_tabfm \
  --root "$ROOT" \
  --config "$ROOT/configs/screening_blend_tabfm_protocol.yaml" \
  --positive "$POSITIVE" \
  --negative "$NEGATIVE" \
  --run-id "$RUN_ID"
