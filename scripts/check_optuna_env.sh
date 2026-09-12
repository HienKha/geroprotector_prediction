#!/usr/bin/env bash
set -euo pipefail

# Read-only dependency and artifact check for the Optuna-tuned SVM/Tanimoto/TabPFNv2
# blend.  Installs nothing, downloads nothing, trains nothing.
#
# If optuna is missing, install it first:
#   python3 -m pip install optuna
# then paste the version this script prints into
# configs/blend_svm_tani_tabpfnv2_optuna_protocol.yaml under `optuna:`.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TABPFN_DISABLE_TELEMETRY=true

"$PYTHON_BIN" - "$ROOT" <<'PY'
import importlib.metadata, json, sys
from pathlib import Path

root = Path(sys.argv[1])
report = {"ok": True, "checks": []}


def check(name, ok, detail):
    report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
    if not ok:
        report["ok"] = False


try:
    optuna_version = importlib.metadata.version("optuna")
    check("optuna installed", True, optuna_version)
except Exception as error:                                    # noqa: BLE001
    check("optuna installed", False, repr(error))
    optuna_version = None

try:
    tabpfn_version = importlib.metadata.version("tabpfn")
    check("tabpfn installed", True, tabpfn_version)
except Exception as error:                                    # noqa: BLE001
    check("tabpfn installed", False, repr(error))

protocol_path = root / "configs" / "blend_svm_tani_tabpfnv2_optuna_protocol.yaml"
try:
    import yaml

    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    required = str(protocol["optuna"]["required_optuna_version"])
    check(
        "protocol optuna version resolved",
        required != "REQUIRED_AT_RUNTIME",
        required,
    )
    if required != "REQUIRED_AT_RUNTIME" and optuna_version is not None:
        check(
            "protocol optuna version matches interpreter",
            required == optuna_version,
            {"protocol": required, "installed": optuna_version},
        )
    from geroprotector.hashing import sha256_file

    model = protocol["model"]
    checkpoint = Path(model["checkpoint_path"])
    exists = checkpoint.is_file() and not checkpoint.is_symlink()
    check("TabPFN v2 checkpoint present", exists, str(checkpoint))
    if exists:
        observed = sha256_file(checkpoint)
        check(
            "TabPFN v2 checkpoint sha256 matches protocol",
            observed == model["checkpoint_sha256"],
            {"observed": observed, "protocol": model["checkpoint_sha256"]},
        )
    for name, record in protocol["sealed_inputs"].items():
        expected = str(record.get("sha256"))
        path = root / record["path"]
        present = path.is_file() and not path.is_symlink()
        resolved = expected != "REQUIRED_AT_RUNTIME"
        check(
            f"sealed input {name}",
            resolved and present and sha256_file(path) == expected,
            record["path"],
        )
except Exception as error:                                    # noqa: BLE001
    check("protocol readable", False, repr(error))

try:
    import torch

    check("torch cuda available", torch.cuda.is_available(), torch.__version__)
except Exception as error:                                    # noqa: BLE001
    check("torch import", False, repr(error))

print(json.dumps(report, indent=2, default=str))
sys.exit(0 if report["ok"] else 1)
PY
