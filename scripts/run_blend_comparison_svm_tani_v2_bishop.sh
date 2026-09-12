#!/usr/bin/env bash
set -euo pipefail

# Seven-model comparison table (SVM/Tanimoto/TabPFNv2 blends + BiSHop blends, at
# threshold 0.5 and OOF-MCC).  Fits nothing; reads sealed prediction files only.
# No GPU is required for this specific script.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
RUN_ID="${1:-blend_comparison_svm_tani_v2_bishop_20260819}"
POSITIVE="${2:-$ROOT/artifacts/data/raw/upstream/geroprotectors_reported.tsv}"
NEGATIVE="${3:-$ROOT/artifacts/data/raw/upstream/no_geroprotectors_and_toxicos.csv}"

if [[ ! "$RUN_ID" =~ ^blend_comparison_svm_tani_v2_bishop_[a-z0-9_.-]+$ ]]; then
  echo "RUN_ID must start with blend_comparison_svm_tani_v2_bishop_" >&2
  exit 2
fi
if [[ -e "$ROOT/outputs/$RUN_ID" ]]; then
  echo "Run directory already exists: $ROOT/outputs/$RUN_ID" >&2
  exit 2
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" -m geroprotector.blend_comparison_svm_tani_v2_bishop \
  --root "$ROOT" \
  --config "$ROOT/configs/blend_comparison_svm_tani_v2_bishop_protocol.yaml" \
  --positive "$POSITIVE" \
  --negative "$NEGATIVE" \
  --run-id "$RUN_ID"
