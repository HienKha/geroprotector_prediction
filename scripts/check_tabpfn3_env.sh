#!/usr/bin/env bash
set -euo pipefail

# Read-only dependency and artifact check for the TabPFN-3 experiment.
# Downloads nothing, writes nothing, trains nothing.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GERO_PYTHON:-python3}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TABPFN_DISABLE_TELEMETRY=true

"$PYTHON_BIN" - "$ROOT" <<'PY'
import importlib.metadata, inspect, json, sys
from pathlib import Path

root = Path(sys.argv[1])
report = {"ok": True, "checks": []}


def check(name, ok, detail):
    report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
    if not ok:
        report["ok"] = False


try:
    version = importlib.metadata.version("tabpfn")
    check("tabpfn installed", True, version)
except Exception as error:                                    # noqa: BLE001
    check("tabpfn installed", False, repr(error))
    version = None

try:
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    members = [m.name for m in ModelVersion]
    check("ModelVersion.V3 available", "V3" in members, members)
    params = [p for p in inspect.signature(TabPFNClassifier.__init__).parameters if p != "self"]
    for required in ("n_estimators", "softmax_temperature", "model_path", "device", "random_state"):
        check(f"TabPFNClassifier accepts {required}", required in params, params[:8])
    from tabpfn.model_loading import ModelSource

    source = ModelSource.get_classifier_v3()
    check(
        "v3 checkpoint identity",
        True,
        {"repo_id": source.repo_id, "default_filename": source.default_filename},
    )
except Exception as error:                                    # noqa: BLE001
    check("tabpfn API import", False, repr(error))

protocol_path = root / "configs" / "tabpfn3_paper405_protocol.yaml"
try:
    import yaml

    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    model = protocol["model"]
    check(
        "protocol package version matches interpreter",
        str(model["required_package_version"]) == str(version),
        {"protocol": model["required_package_version"], "installed": version},
    )
    unresolved = [
        key
        for key in ("checkpoint_path", "checkpoint_sha256", "checkpoint_source", "access_date_utc")
        if str(model.get(key)) == "REQUIRED_AT_RUNTIME"
    ]
    check(
        "checkpoint placeholders resolved",
        not unresolved,
        unresolved or "all resolved — run scripts/stage_tabpfn3_checkpoint.sh if not",
    )
    if not unresolved:
        from geroprotector.hashing import sha256_file

        checkpoint = Path(model["checkpoint_path"])
        exists = checkpoint.is_file() and not checkpoint.is_symlink()
        check("checkpoint file present", exists, str(checkpoint))
        if exists:
            observed = sha256_file(checkpoint)
            check(
                "checkpoint sha256 matches protocol",
                observed == model["checkpoint_sha256"],
                {"observed": observed, "protocol": model["checkpoint_sha256"]},
            )
    for name, record in protocol["sealed_inputs"].items():
        from geroprotector.hashing import sha256_file

        path = root / record["path"]
        present = path.is_file() and not path.is_symlink()
        check(
            f"sealed input {name}",
            present and sha256_file(path) == record["sha256"],
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
