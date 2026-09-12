#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
FAMILY="${1:?usage: run_resource_telemetry_family.sh FAMILY [STAMP]}"
STAMP="${2:-telemetry_20260906}"
TELEMETRY_ROOT="${GERO_TELEMETRY_ROOT:-$ROOT/resource_telemetry}"
LOCK_FILE="/tmp/geroprotector_resource_telemetry_gpu.lock"

case "$FAMILY" in
  repeated_rank_stability)
    WRAPPER="scripts/run_repeated_rank_stability.sh"
    RUN_ID="repeated_rank_stability_${STAMP}"
    ;;
  chemical_space_cv)
    WRAPPER="scripts/run_chemical_space_cv.sh"
    RUN_ID="chemical_space_cv_${STAMP}"
    ;;
  agextend_endpoint_benchmark)
    WRAPPER="scripts/run_agextend_endpoint_benchmark.sh"
    RUN_ID="agextend_endpoint_benchmark_${STAMP}"
    ;;
  drugage_celegans_benchmark)
    WRAPPER="scripts/run_drugage_celegans_benchmark.sh"
    RUN_ID="drugage_celegans_benchmark_${STAMP}"
    ;;
  kapsiani_historical_benchmark)
    WRAPPER="scripts/run_kapsiani_historical_benchmark.sh"
    RUN_ID="kapsiani_historical_benchmark_${STAMP}"
    ;;
  *)
    echo "Unknown telemetry family: $FAMILY" >&2
    exit 2
    ;;
esac

TELEMETRY_DIR="$TELEMETRY_ROOT/$FAMILY"
EXPECTED_OUTPUT="$ROOT/outputs/$RUN_ID"
if [[ -e "$TELEMETRY_DIR" ]]; then
  echo "Refusing to overwrite telemetry directory: $TELEMETRY_DIR" >&2
  exit 3
fi
if [[ -e "$EXPECTED_OUTPUT" || -e "$ROOT/outputs/.$RUN_ID.work" ]]; then
  echo "Refusing to overwrite an output or work directory for $RUN_ID" >&2
  exit 4
fi

mkdir -p "$TELEMETRY_ROOT"
cd "$ROOT"
export GERO_PYTHON="$PYTHON_BIN"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

echo "[$(date -u +%FT%TZ)] $FAMILY is waiting for the exclusive GPU telemetry lock"
exec flock -x "$LOCK_FILE" \
  "$PYTHON_BIN" scripts/run_with_resource_telemetry.py \
    --family "$FAMILY" \
    --telemetry-dir "$TELEMETRY_DIR" \
    --expected-output "$EXPECTED_OUTPUT" \
    -- bash "$WRAPPER" "$STAMP"
