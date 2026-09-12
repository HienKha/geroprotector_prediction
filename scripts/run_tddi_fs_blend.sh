#!/usr/bin/env bash
set -euo pipefail

# T-DDI-style feature selection for the DL components, then an equal-thirds
# blend of the published paper SVM / TabPFN-v2 / TabFM.
#
#   FS pipeline  : T-DDI Fulltext.pdf Methods p.17 (audit -> median impute ->
#                  ANOVA F rank -> Pearson |r| > 0.995 prune -> RFE at K ->
#                  K chosen by retraining under the same 5-fold CV)
#   FS scope     : DL components only.  The SVM slot is read from the sealed
#                  runs, so it stays on the paper's 7 DataWarrior descriptors
#                  with the paper's exact hyperparameters and is never refitted.
#   Tanimoto     : untouched, and not a component of this blend.
#
# Reads every existing run read-only; writes exactly one new output directory
# and refuses to start if that directory already exists.
#
# Usage:
#   scripts/run_tddi_fs_blend.sh                      # full 12-point K ladder
#   scripts/run_tddi_fs_blend.sh tddi_fs_blend_quick --quick   # 3-point ladder
#
# Runtime note: the full ladder trains TabPFN-v2 + TabFM 12 K x 5 folds = 120
# fits, plus 5 nested-FS folds and one locked fit.  Budget roughly 1.5-3 h on
# this GPU.  --quick collapses the ladder to {7, 30, full} (~35 min) if you want
# to smoke-test the plumbing before spending the full budget.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
RUN_ID="${1:-tddi_fs_blend_20260821}"
shift || true
EXTRA=("$@")
POSITIVE="${GERO_POSITIVE:-$ROOT/artifacts/data/raw/upstream/geroprotectors_reported.tsv}"
NEGATIVE="${GERO_NEGATIVE:-$ROOT/artifacts/data/raw/upstream/no_geroprotectors_and_toxicos.csv}"

if [[ ! "$RUN_ID" =~ ^tddi_fs_blend_[a-z0-9_.-]+$ ]]; then
  echo "RUN_ID must start with tddi_fs_blend_" >&2
  exit 2
fi
if [[ -e "$ROOT/outputs/$RUN_ID" ]]; then
  echo "Run directory already exists, refusing to touch it: $ROOT/outputs/$RUN_ID" >&2
  exit 2
fi
if ! "$PYTHON_BIN" -c "import tabfm" >/dev/null 2>&1; then
  echo "tabfm is missing from this interpreter. Install a locally staged upstream copy with:" >&2
  echo "  cd $ROOT/third_party/tabfm && $PYTHON_BIN -m pip install -e '.[pytorch]'" >&2
  exit 3
fi
if ! "$PYTHON_BIN" -c "import tabpfn" >/dev/null 2>&1; then
  echo "tabpfn is missing from this interpreter." >&2
  exit 3
fi
TABPFN_V2_CHECKPOINT="${TABPFN_V2_CHECKPOINT:-$ROOT/checkpoints/tabpfn-v2-classifier.ckpt}"
if [[ ! -f "$TABPFN_V2_CHECKPOINT" ]]; then
  echo "TabPFN-v2 checkpoint is missing: $TABPFN_V2_CHECKPOINT" >&2
  exit 3
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TABPFN_DISABLE_TELEMETRY=true

exec "$PYTHON_BIN" -m geroprotector.tddi_fs_blend_paper405 \
  --root "$ROOT" \
  --config "$ROOT/configs/tddi_fs_blend_paper405_protocol.yaml" \
  --positive "$POSITIVE" \
  --negative "$NEGATIVE" \
  --run-id "$RUN_ID" \
  "${EXTRA[@]}"
