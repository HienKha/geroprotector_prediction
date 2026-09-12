#!/usr/bin/env bash
set -euo pipefail

# Fold-safe Optuna search over TabPFNv2 hyperparameters (and, in `blend` mode,
# Tanimoto's C) inside the equal-thirds SVM/Tanimoto/TabPFNv2 blend.  Reads sealed
# runs read-only; writes one new output directory.
#
# Usage:
#   bash scripts/run_blend_svm_tani_tabpfnv2_optuna.sh standalone [run_id]
#   bash scripts/run_blend_svm_tani_tabpfnv2_optuna.sh blend      [run_id]
#
# Recommended order: run `standalone` first (cheaper, answers "should the blend
# reuse the same TabPFNv2 params" directly).  Run `blend` afterwards only if you
# want to see whether jointly tuning Tanimoto's C changes the picture.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
OBJECTIVE="${1:?Usage: $0 standalone-or-blend [run_id]}"
RUN_ID="${2:-blend_svm_tani_tabpfnv2_optuna_${OBJECTIVE}_20260819}"
POSITIVE="${3:-$ROOT/artifacts/data/raw/upstream/geroprotectors_reported.tsv}"
NEGATIVE="${4:-$ROOT/artifacts/data/raw/upstream/no_geroprotectors_and_toxicos.csv}"

if [[ "$OBJECTIVE" != "standalone" && "$OBJECTIVE" != "blend" ]]; then
  echo "First argument must be 'standalone' or 'blend'" >&2
  exit 2
fi
if [[ ! "$RUN_ID" =~ ^blend_svm_tani_tabpfnv2_optuna_[a-z0-9_.-]+$ ]]; then
  echo "RUN_ID must start with blend_svm_tani_tabpfnv2_optuna_" >&2
  exit 2
fi
if [[ -e "$ROOT/outputs/$RUN_ID" ]]; then
  echo "Run directory already exists: $ROOT/outputs/$RUN_ID" >&2
  exit 2
fi

echo "== dependency check =="
bash "$ROOT/scripts/check_optuna_env.sh"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TABPFN_DISABLE_TELEMETRY=true
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

exec "$PYTHON_BIN" -m geroprotector.blend_svm_tani_tabpfnv2_optuna \
  --root "$ROOT" \
  --config "$ROOT/configs/blend_svm_tani_tabpfnv2_optuna_protocol.yaml" \
  --positive "$POSITIVE" \
  --negative "$NEGATIVE" \
  --objective "$OBJECTIVE" \
  --run-id "$RUN_ID"
