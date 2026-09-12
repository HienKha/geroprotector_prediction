#!/usr/bin/env bash
# Master runner for the insight-analysis suite (prespecified insight-analysis protocol).
#
# Usage:
#   bash scripts/run_insight_suite.sh --stamp 20260825 --resume
#
# Runs, in order: A0 (external GBM parity/scoring), A (modelwide drug-likeness
# bias), B (repeated rank stability),
# C (chemistry-aware grouped CV), D (AgeXtend endpoint benchmark), E (DrugAge
# C. elegans endpoint benchmark), then the integrated report. Each stage is
# skipped if its output directory already exists (the --resume contract: a
# completed run is only skipped after its own COMPLETED.json is present, which
# every stage's Python module already enforces on its own before writing
# anything). GPU stages (A pulls sealed streams only; B, C, D, E fit TabPFN-v2,
# TabFM and TabM) are NOT run concurrently with each other by this script, since
# they would compete for the same GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="20260825"
RESUME=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) STAMP="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

cd "$ROOT"
export GERO_PYTHON="${GERO_PYTHON:-python3}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
set -o pipefail

run_stage () {
  local name="$1" script="$2"
  local run_dir="outputs/${name}_${STAMP}"
  echo "=============================================================="
  echo "STAGE: $name -> $run_dir"
  echo "=============================================================="
  if [[ -d "$run_dir" ]]; then
    if [[ -f "$run_dir/COMPLETED.json" ]]; then
      "$GERO_PYTHON" scripts/verify_completed_run.py "$run_dir"
      echo "  hash-verified COMPLETE, skipping"
      return 0
    else
      echo "  ERROR: $run_dir exists but has no valid COMPLETED.json. Refusing " \
           "to continue. Inspect it manually; --resume never bypasses a partial " \
           "final directory." >&2
      exit 1
    fi
  fi
  bash "scripts/$script" "$STAMP"
}

run_stage "gbm_external_raw"           "run_gbm_external_predictions.sh"
run_stage "modelwide_druglikeness_bias" "run_modelwide_druglikeness_bias.sh"
run_stage "repeated_rank_stability"     "run_repeated_rank_stability.sh"
run_stage "chemical_space_cv"           "run_chemical_space_cv.sh"
run_stage "agextend_endpoint_benchmark" "run_agextend_endpoint_benchmark.sh"
run_stage "drugage_celegans_benchmark"  "run_drugage_celegans_benchmark.sh"

echo "=============================================================="
echo "STAGE: insight_suite_report -> outputs/insight_suite_report_${STAMP}"
echo "=============================================================="
if [[ -d "outputs/insight_suite_report_${STAMP}" ]]; then
  "$GERO_PYTHON" scripts/verify_completed_run.py "outputs/insight_suite_report_${STAMP}"
  echo "  hash-verified COMPLETE, skipping"
else
  for required in modelwide_druglikeness_bias repeated_rank_stability chemical_space_cv \
                  agextend_endpoint_benchmark drugage_celegans_benchmark; do
    "$GERO_PYTHON" scripts/verify_completed_run.py "outputs/${required}_${STAMP}"
  done
  "$GERO_PYTHON" -m geroprotector.insight_suite_report \
    --root "$ROOT" \
    --config "$ROOT/configs/insight_suite_report_protocol.yaml" \
    --run-id "insight_suite_report_${STAMP}" --stamp "$STAMP"
fi

echo
echo "DONE. See outputs/insight_suite_report_${STAMP}/recommended_main_text_results.md"
