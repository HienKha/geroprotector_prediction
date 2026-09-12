#!/usr/bin/env bash
set -euo pipefail
# Post-lock xAI for the equal-thirds SVM/Tanimoto/TabFM blend, run alongside the
# equal-thirds SVM/Tanimoto/TabPFN-v2 blend for a weight-matched comparison.
# Reads every existing run read-only; writes one new output directory. CPU only.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
RUN_ID="${1:-blend_xai_v3_20260821}"
POSITIVE="${2:-$ROOT/artifacts/data/raw/upstream/geroprotectors_reported.tsv}"
NEGATIVE="${3:-$ROOT/artifacts/data/raw/upstream/no_geroprotectors_and_toxicos.csv}"
[[ "$RUN_ID" =~ ^blend_xai_v3_[a-z0-9_.-]+$ ]] || { echo "RUN_ID must start with blend_xai_v3_" >&2; exit 2; }
[[ -e "$ROOT/outputs/$RUN_ID" ]] && { echo "Run directory already exists: $ROOT/outputs/$RUN_ID" >&2; exit 2; }
cd "$ROOT"; export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m geroprotector.blend_xai_v3 --root "$ROOT" \
  --config "$ROOT/configs/blend_xai_v3_protocol.yaml" \
  --positive "$POSITIVE" --negative "$NEGATIVE" --run-id "$RUN_ID"
