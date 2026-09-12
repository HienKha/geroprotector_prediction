#!/usr/bin/env bash
set -euo pipefail
# ---------------------------------------------------------------------------
# Ablation grids for six candidate fifth components, on the held-out D1 test
# partition and 5-fold CV over D1 train.
#
# STEP 1 computes the only thing that does not already exist: cross-fitted
#        train-OOF predictions for CatBoost, LightGBM and XGBoost on the full
#        RDKit2D panel.  Everything else is read from sealed runs.
# STEP 2 runs the 31-subset ablation for each candidate.  BiSHop, TabM and
#        TabNet reuse the alt-model streams; the three GBMs use step 1.
#
# Nothing is refitted in step 2 and no existing run directory is written to.
# Total runtime is dominated by step 1 (a few minutes; CPU-bound, no GPU needed
# despite the environment having CUDA available).
#
# Usage:   scripts/run_candidate_ablations.sh
#          scripts/run_candidate_ablations.sh 20260824      # custom date suffix
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
STAMP="${1:-20260823}"
POSITIVE="$ROOT/artifacts/data/raw/upstream/geroprotectors_reported.tsv"
NEGATIVE="$ROOT/artifacts/data/raw/upstream/no_geroprotectors_and_toxicos.csv"
GBM_CV="nineml_cv5_full_${STAMP}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

echo "=============================================================="
echo "STEP 1/2  full-panel 5-fold CV for CatBoost, LightGBM, XGBoost"
echo "=============================================================="
if [[ -d "outputs/$GBM_CV" ]]; then
  echo "  already present, skipping: outputs/$GBM_CV"
else
  "$PYTHON_BIN" -m geroprotector.nine_ml_cv5_fullfeat \
    --root "$ROOT" --config "$ROOT/configs/traditional_paper405.yaml" \
    --positive "$POSITIVE" --negative "$NEGATIVE" --run-id "$GBM_CV"
fi

echo
echo "=============================================================="
echo "STEP 2/2  31-subset ablation per candidate"
echo "=============================================================="
for CAND in catboost lightgbm xgboost bishop tabm tabnet; do
  RUN="ablation_${CAND}_${STAMP}"
  if [[ -d "outputs/$RUN" ]]; then
    echo "--- $CAND: already present, skipping outputs/$RUN"
    continue
  fi
  echo "--- $CAND -> outputs/$RUN"
  "$PYTHON_BIN" -m geroprotector.ablation_candidate \
    --root "$ROOT" --candidate "$CAND" --run-id "$RUN" --gbm-cv-run "$GBM_CV"
done

echo
echo "DONE. Results in outputs/ablation_<candidate>_${STAMP}/"
echo "  ablation_grid.csv            31 subsets x 2 cohorts x 9 metrics, with 95% CIs"
echo "  candidate_paired_effect.csv  marginal effect of the candidate, 15 base subsets"
echo "  parity_checks.csv            shared-stream agreement across source runs"
