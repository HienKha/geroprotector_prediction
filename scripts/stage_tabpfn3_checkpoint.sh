#!/usr/bin/env bash
set -euo pipefail

# One-time TabPFN-3 checkpoint staging.
#
# This is the ONLY step that touches the network.  It writes nothing inside
# outputs/ and trains nothing.
#
# The TabPFN-3 weights are LICENSE-GATED by Prior Labs.  Before this can succeed
# you must, once:
#   1. open https://ux.priorlabs.ai and log in / register
#   2. accept the licence on the Licenses tab
#   3. copy your API key from https://ux.priorlabs.ai/account
#   4. export TABPFN_TOKEN="<your-api-key>"
# Run this script from an interactive terminal (your tmux pane) and it will also
# accept the licence interactively if TABPFN_TOKEN is not set.
#
# On success it prints the four values to paste into
# configs/tabpfn3_paper405_protocol.yaml under `model:`.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TABPFN_DISABLE_TELEMETRY=true

if [[ -z "${TABPFN_TOKEN:-}" ]]; then
  echo "note: TABPFN_TOKEN is not set; the licence prompt will be interactive." >&2
  echo "      If this is not a real terminal the download will fail with" >&2
  echo "      TabPFNLicenseError.  See the header of this script." >&2
  echo >&2
fi

"$PYTHON_BIN" - <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

from tabpfn.constants import ModelVersion
from tabpfn.model_loading import ModelSource, download_model, get_cache_dir

source = ModelSource.get_classifier_v3()
target = Path(get_cache_dir()) / source.default_filename
print(f"repo_id          : {source.repo_id}")
print(f"default_filename : {source.default_filename}")
print(f"target           : {target}")
print()

if target.is_file():
    print("checkpoint already staged; re-hashing only")
else:
    result = download_model(target, version=ModelVersion.V3, which="classifier")
    if result != "ok":
        print("DOWNLOAD FAILED:", file=sys.stderr)
        for error in result:
            print(f"  {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)

if not target.is_file():
    raise SystemExit(f"TabPFN did not materialise a checkpoint at {target}")

digest = hashlib.sha256()
with target.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)

print()
print("Paste these into configs/tabpfn3_paper405_protocol.yaml under `model:`")
print("-" * 72)
print(f"  checkpoint_path: {target}")
print(f"  checkpoint_sha256: {digest.hexdigest()}")
print(f"  checkpoint_source: huggingface://{source.repo_id}/{source.default_filename}")
print(f'  access_date_utc: "{datetime.now(timezone.utc).date().isoformat()}"')
print("-" * 72)
print(json.dumps({"size_bytes": target.stat().st_size}, indent=2))
PY
