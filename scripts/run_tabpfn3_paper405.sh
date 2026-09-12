#!/usr/bin/env bash
set -euo pipefail

# Fold-safe TabPFN-3 selection and evaluation under the locked paper-405 protocol.
# Reads the sealed 2026-08-17/18 runs read-only and writes one new output directory.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
RUN_ID="${1:-tabpfn3_paper405_20260819}"
POSITIVE="${2:-$ROOT/artifacts/data/raw/upstream/geroprotectors_reported.tsv}"
NEGATIVE="${3:-$ROOT/artifacts/data/raw/upstream/no_geroprotectors_and_toxicos.csv}"

if [[ ! "$RUN_ID" =~ ^tabpfn3_paper405_[a-z0-9_.-]+$ ]]; then
  echo "RUN_ID must start with tabpfn3_paper405_" >&2
  exit 2
fi
if [[ -e "$ROOT/outputs/$RUN_ID" ]]; then
  echo "Run directory already exists: $ROOT/outputs/$RUN_ID" >&2
  exit 2
fi

echo "== dependency check =="
bash "$ROOT/scripts/check_tabpfn3_env.sh"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TABPFN_DISABLE_TELEMETRY=true
export HF_HUB_OFFLINE=1          # the checkpoint is already staged; no network at run time
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

exec "$PYTHON_BIN" -m geroprotector.tabpfn3_paper405 \
  --root "$ROOT" \
  --config "$ROOT/configs/tabpfn3_paper405_protocol.yaml" \
  --positive "$POSITIVE" \
  --negative "$NEGATIVE" \
  --run-id "$RUN_ID"
