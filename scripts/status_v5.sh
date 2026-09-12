#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 v5_RUN_ID" >&2
  exit 2
fi
RUN_ID="$1"
[[ "${RUN_ID}" =~ ^v5_[a-z0-9_.-]+$ ]] || {
  echo "RUN_ID must start with v5_ and use lowercase safe characters" >&2
  exit 2
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
WORK_DIR="${ROOT_DIR}/outputs/.${RUN_ID}.work"
FINAL_DIR="${ROOT_DIR}/outputs/${RUN_ID}"

if [[ -d "${FINAL_DIR}" ]]; then
  TARGET="${FINAL_DIR}"
  STATUS="SEALED"
elif [[ -d "${WORK_DIR}" ]]; then
  TARGET="${WORK_DIR}"
  STATUS="RUNNING_OR_INTERRUPTED"
else
  echo "No work or sealed directory exists for ${RUN_ID}" >&2
  exit 1
fi

export V5_STATUS_TARGET="${TARGET}"
export V5_STATUS_RUN_ID="${RUN_ID}"
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

import yaml

target = Path(os.environ["V5_STATUS_TARGET"])
run_id = os.environ["V5_STATUS_RUN_ID"]
running = target / "RUNNING.json"
config_path = target / "resolved_config.yaml"
model_count = None
if running.is_file():
    model_count = len(json.loads(running.read_text())["model_ids"])
if config_path.is_file():
    config = yaml.safe_load(config_path.read_text())
    partitions = int(config["primary_split"]["outer_repeats"]) * int(
        config["primary_split"]["outer_folds"]
    )
else:
    partitions = None
expected = None if model_count is None or partitions is None else model_count * partitions

def count(pattern: str) -> int:
    return sum(1 for _ in target.glob(pattern))

print(f"run_id={run_id}")
print(f"target={target}")
print(f"scientific_outer_partitions={partitions}")
print(f"expected_model_bundles={expected}")
print(f"completed_model_bundles={count('jobs/**/JOB_COMPLETED.json')}")
print(f"stage_a_dependencies={count('shared/v5_preparation/**/stage_a/*/manifest.json')}")
print(f"stage_b_complete_states={count('shared/v5_preparation/**/stage_b/*/manifest.json')}")
print(
    "stage_b_subfold_vector_checkpoints="
    f"{count('shared/v5_preparation/**/stage_b_vectors/*/manifest.json')}"
)
events = target / "logs" / "events.jsonl"
if events.is_file():
    lines = events.read_text(encoding="utf-8").splitlines()
    if lines:
        print(f"last_event={lines[-1]}")
PY

echo "status=${STATUS}"
pgrep -af "run-id ${RUN_ID}" || true
