#!/usr/bin/env bash
set -euo pipefail
# Four-component blend (paper SVM + Tanimoto + TabPFN-v2 + TabFM), threshold pinned at 0.5.
# Re-weights existing fold-aligned streams only: no model is refitted, no GPU is used.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
RUN_ID="${1:-quad_blend_20260822}"
[[ "$RUN_ID" =~ ^quad_blend_[a-z0-9_.-]+$ ]] || { echo "RUN_ID must start with quad_blend_" >&2; exit 2; }
[[ -e "$ROOT/outputs/$RUN_ID" ]] && { echo "Run directory already exists: $ROOT/outputs/$RUN_ID" >&2; exit 2; }
cd "$ROOT"; export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m geroprotector.quad_blend_paper405 \
  --root "$ROOT" --config "$ROOT/configs/quad_blend_paper405_protocol.yaml" --run-id "$RUN_ID"
