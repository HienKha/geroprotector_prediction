#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_ID" >&2
  exit 2
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
RUN_ID="$1"
[[ "${RUN_ID}" =~ ^v(5|6)bis_[a-z0-9_.-]+$ ]] || {
  echo "RUN_ID must start with v5bis_ or v6bis_ and use lowercase safe characters" >&2
  exit 2
}
FINAL="${ROOT_DIR}/outputs/${RUN_ID}"
WORK="${ROOT_DIR}/outputs/.${RUN_ID}.work"
TARGET="${FINAL}"
[[ -d "${TARGET}" ]] || TARGET="${WORK}"
echo "run_id=${RUN_ID}"
echo "target=${TARGET}"
if [[ ! -d "${TARGET}" ]]; then
  echo "status=NOT_STARTED"
  exit 0
fi
echo "completed_model_bundles=$(find "${TARGET}/jobs" -type f -name JOB_COMPLETED.json 2>/dev/null | wc -l)"
if [[ -f "${TARGET}/COMPLETED.json" ]]; then
  echo "status=COMPLETE"
elif [[ -f "${TARGET}/logs/events.jsonl" ]]; then
  echo "last_event=$(tail -n 1 "${TARGET}/logs/events.jsonl")"
  echo "status=RUNNING_OR_INTERRUPTED"
else
  echo "status=INITIALIZING"
fi
pgrep -af "run-id ${RUN_ID}" || true
