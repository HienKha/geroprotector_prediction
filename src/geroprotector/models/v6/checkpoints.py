"""Offline-only V6 package/checkpoint/license staging."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...hashing import atomic_write_json, canonical_sha256, sha256_file
from ...logging import utc_now


class CheckpointError(RuntimeError):
    pass


_SHA = re.compile(r"^[0-9a-f]{64}$")


def _torch_environment() -> dict[str, Any]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    properties = torch.cuda.get_device_properties(0) if cuda_available else None
    driver_version = None
    get_driver = getattr(torch._C, "_cuda_getDriverVersion", None)
    if cuda_available and callable(get_driver):
        driver_version = str(get_driver())
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": importlib.metadata.version("torch"),
        "cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_device_name": device_name,
        "gpu_compute_capability": (
            None if properties is None else f"{properties.major}.{properties.minor}"
        ),
        "cuda_driver_version": driver_version,
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def _project_file(root: Path, configured: object, *, role: str) -> Path:
    value = str(configured).strip()
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise CheckpointError(f"{role} must be a project-relative path without '..': {value}")
    candidate = root / path
    cursor = root
    for part in path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise CheckpointError(f"{role} path contains a symlink: {cursor}")
    resolved = candidate.resolve()
    if root.resolve() not in resolved.parents:
        raise CheckpointError(f"{role} escapes project root: {value}")
    if resolved.is_symlink() or not resolved.is_file():
        raise CheckpointError(f"{role} must be a regular non-symlink file: {resolved}")
    return resolved


def stage_checkpoints(
    config: Mapping[str, Any],
    *,
    root: str | Path,
    output_path: str | Path,
    resolved_config_sha256: str,
) -> dict[str, Any]:
    project = Path(root).resolve()
    if bool(config["inference"].get("telemetry_disabled")) is not True:
        raise CheckpointError("V6 telemetry must remain disabled")
    if os.environ.get("TABPFN_DISABLE_TELEMETRY") != "1":
        raise CheckpointError("Set TABPFN_DISABLE_TELEMETRY=1 before staging V6")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise CheckpointError("Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
    records: list[dict[str, Any]] = []
    for model_id, model in config["models"].items():
        if not isinstance(model, Mapping) or model.get("enabled") is not True:
            continue
        required = {
            "package",
            "package_version",
            "explicit_model_version",
            "checkpoint_path",
            "checkpoint_sha256",
            "license_path",
            "license_sha256",
            "checkpoint_source",
            "access_date_utc",
        }
        missing = required - set(model)
        if missing:
            raise CheckpointError(f"{model_id} staging fields missing: {sorted(missing)}")
        package = str(model["package"])
        expected_version = str(model["package_version"])
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise CheckpointError(f"Required package is not installed: {package}") from exc
        if installed != expected_version:
            raise CheckpointError(
                f"{model_id} package version mismatch: {installed} != {expected_version}"
            )
        checkpoint = _project_file(
            project, model["checkpoint_path"], role=f"{model_id} checkpoint"
        )
        license_path = _project_file(project, model["license_path"], role=f"{model_id} license")
        expected_checkpoint = str(model["checkpoint_sha256"]).strip().lower()
        expected_license = str(model["license_sha256"]).strip().lower()
        if not _SHA.fullmatch(expected_checkpoint) or not _SHA.fullmatch(expected_license):
            raise CheckpointError(f"{model_id} requires literal lowercase SHA-256 values")
        actual_checkpoint = sha256_file(checkpoint)
        actual_license = sha256_file(license_path)
        if actual_checkpoint != expected_checkpoint:
            raise CheckpointError(f"{model_id} checkpoint SHA-256 mismatch")
        if actual_license != expected_license:
            raise CheckpointError(f"{model_id} license SHA-256 mismatch")
        records.append(
            {
                "model_id": model_id,
                "package": package,
                "package_version": installed,
                "explicit_model_version": str(model["explicit_model_version"]),
                "checkpoint_path": checkpoint.relative_to(project).as_posix(),
                "checkpoint_sha256": actual_checkpoint,
                "license_path": license_path.relative_to(project).as_posix(),
                "license_sha256": actual_license,
                "checkpoint_source": str(model["checkpoint_source"]),
                "access_date_utc": str(model["access_date_utc"]),
            }
        )
    if not records:
        raise CheckpointError("No enabled V6 checkpoint family was staged")
    checkpoint_paths = [str(record["checkpoint_path"]) for record in records]
    checkpoint_hashes = [str(record["checkpoint_sha256"]) for record in records]
    if len(checkpoint_paths) != len(set(checkpoint_paths)):
        raise CheckpointError("Distinct enabled V6 model IDs cannot share a checkpoint path")
    if len(checkpoint_hashes) != len(set(checkpoint_hashes)):
        raise CheckpointError("Distinct enabled V6 model IDs cannot share checkpoint bytes")
    environment = _torch_environment()
    requested_device = str(config.get("execution", {}).get("device", "auto"))
    if requested_device.startswith("cuda") and environment["cuda_available"] is not True:
        raise CheckpointError("V6 config requires CUDA but no CUDA device is available")
    ledger = {
        "schema_version": "geroprotector.v6_checkpoint_ledger.v1",
        "created_utc": utc_now(),
        "implicit_downloads_allowed": False,
        "telemetry_disabled": bool(config["inference"]["telemetry_disabled"]),
        "offline_environment": {
            "TABPFN_DISABLE_TELEMETRY": os.environ.get("TABPFN_DISABLE_TELEMETRY"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
        "resolved_config_sha256": str(resolved_config_sha256),
        "environment": environment,
        "models": records,
    }
    ledger["canonical_sha256"] = canonical_sha256(ledger)
    atomic_write_json(output_path, ledger)
    return ledger


def load_checkpoint_ledger(
    path: str | Path,
    *,
    config: Mapping[str, Any],
    root: str | Path,
    resolved_config_sha256: str,
) -> dict[str, Any]:
    import json

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise CheckpointError(f"Checkpoint ledger must be regular/non-symlink: {source}")
    ledger = json.loads(source.read_text(encoding="utf-8"))
    payload = dict(ledger)
    claimed = payload.pop("canonical_sha256", None)
    actual = canonical_sha256(payload)
    if claimed != actual or ledger.get("implicit_downloads_allowed") is not False:
        raise CheckpointError("Checkpoint ledger integrity/implicit-download contract failed")
    if ledger.get("resolved_config_sha256") != str(resolved_config_sha256):
        raise CheckpointError("Checkpoint ledger was staged for a different resolved config")
    if ledger.get("telemetry_disabled") is not True or ledger.get("offline_environment") != {
        "TABPFN_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }:
        raise CheckpointError("Checkpoint ledger lacks the locked offline environment")
    if (
        os.environ.get("TABPFN_DISABLE_TELEMETRY") != "1"
        or os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
    ):
        raise CheckpointError("V6 runtime offline/telemetry environment is not enforced")
    current_environment = _torch_environment()
    if ledger.get("environment") != current_environment:
        raise CheckpointError(
            "Python/PyTorch/CUDA environment changed after checkpoint staging"
        )
    expected_models = {
        model_id
        for model_id, settings in config["models"].items()
        if isinstance(settings, Mapping) and settings.get("enabled") is True
    }
    observed_models = {str(record.get("model_id")) for record in ledger.get("models", [])}
    if observed_models != expected_models:
        raise CheckpointError("Checkpoint ledger enabled-model set changed")
    checkpoint_paths = [str(record.get("checkpoint_path")) for record in ledger["models"]]
    checkpoint_hashes = [str(record.get("checkpoint_sha256")) for record in ledger["models"]]
    if len(checkpoint_paths) != len(set(checkpoint_paths)) or len(checkpoint_hashes) != len(
        set(checkpoint_hashes)
    ):
        raise CheckpointError("Distinct enabled V6 model IDs share checkpoint bytes/path")
    project = Path(root).resolve()
    for record in ledger["models"]:
        model_id = str(record["model_id"])
        settings = config["models"][model_id]
        expected_record = {
            "model_id": model_id,
            "package": str(settings["package"]),
            "package_version": str(settings["package_version"]),
            "explicit_model_version": str(settings["explicit_model_version"]),
            "checkpoint_path": str(settings["checkpoint_path"]),
            "checkpoint_sha256": str(settings["checkpoint_sha256"]).lower(),
            "license_path": str(settings["license_path"]),
            "license_sha256": str(settings["license_sha256"]).lower(),
            "checkpoint_source": str(settings["checkpoint_source"]),
            "access_date_utc": str(settings["access_date_utc"]),
        }
        if {key: record.get(key) for key in expected_record} != expected_record:
            raise CheckpointError(f"Checkpoint ledger/config binding changed: {model_id}")
        if importlib.metadata.version(record["package"]) != record["package_version"]:
            raise CheckpointError(f"Installed package changed after staging: {model_id}")
        if record["package_version"] != str(settings["package_version"]):
            raise CheckpointError(f"Configured package version changed: {model_id}")
        checkpoint = _project_file(
            project, record["checkpoint_path"], role=f"{model_id} checkpoint"
        )
        license_path = _project_file(
            project, record["license_path"], role=f"{model_id} license"
        )
        if sha256_file(checkpoint) != record["checkpoint_sha256"]:
            raise CheckpointError(f"Checkpoint changed after staging: {model_id}")
        if sha256_file(license_path) != record["license_sha256"]:
            raise CheckpointError(f"License changed after staging: {model_id}")
    return ledger


def checkpoint_record(ledger: Mapping[str, Any], model_id: str) -> dict[str, Any]:
    matches = [item for item in ledger["models"] if item["model_id"] == model_id]
    if len(matches) != 1:
        raise CheckpointError(f"Checkpoint ledger has {len(matches)} records for {model_id}")
    return dict(matches[0])
