"""Runtime and source-tree bindings for immutable benchmark artifacts."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import re
import sys
from pathlib import Path
from typing import Any

from .hashing import canonical_sha256, sha256_file


class RuntimeLockError(RuntimeError):
    """Raised when the active interpreter differs from the executable lock."""


_CORE_RUNTIME_PACKAGES = (
    "joblib",
    "jsonschema",
    "numpy",
    "pandas",
    "pyarrow",
    "PyYAML",
    "rdkit",
    "scikit-learn",
    "scipy",
    "xgboost",
)

_MODULE_IMPORTS = {
    "joblib": "joblib",
    "jsonschema": "jsonschema",
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "PyYAML": "yaml",
    "rdkit": "rdkit",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "xgboost": "xgboost",
}


def _canonical_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _distribution_versions(package: str) -> tuple[str, ...]:
    target = _canonical_distribution_name(package)
    versions = [
        str(distribution.version)
        for distribution in importlib.metadata.distributions()
        if _canonical_distribution_name(str(distribution.metadata.get("Name", ""))) == target
    ]
    return tuple(sorted(versions))


def _module_version(package: str) -> str | None:
    try:
        module = importlib.import_module(_MODULE_IMPORTS[package])
    except ImportError:
        return None
    if package == "rdkit":
        from rdkit import rdBase

        return str(rdBase.rdkitVersion)
    if package == "jsonschema":
        # jsonschema exposes __version__ only through a deprecated warning-producing
        # attribute.  Importing above proves the module is present; the separately
        # duplicate-checked distribution record supplies its release version.
        versions = _distribution_versions(package)
        return versions[0] if len(versions) == 1 else None
    value = getattr(module, "__version__", None)
    return None if value is None else str(value)


def _installed_version(package: str) -> str | None:
    return _module_version(package)


def _version_key(value: str) -> tuple[int, ...] | str:
    """Normalize numeric release spelling (for example 2025.03 == 2025.3)."""

    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", value):
        return tuple(int(part) for part in value.split("."))
    return value


def validate_core_runtime(lock_path: str | Path) -> dict[str, str]:
    """Require exact executable dependency pins before data or model operations."""

    path = Path(lock_path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeLockError(f"Runtime lock must be regular/non-symlink: {path}")
    expected: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise RuntimeLockError(
                f"Runtime lock line {line_number} is not an exact package==version pin"
            )
        package, version = (part.strip() for part in line.split("==", maxsplit=1))
        if (
            not package
            or not version
            or package.lower() in {value.lower() for value in expected}
        ):
            raise RuntimeLockError(f"Runtime lock has an invalid/duplicate pin: {line}")
        expected[package] = version
    missing_pins = {package.lower() for package in _CORE_RUNTIME_PACKAGES} - {
        package.lower() for package in expected
    }
    if missing_pins:
        raise RuntimeLockError(f"Runtime lock lacks core pins: {sorted(missing_pins)}")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeLockError(
            f"Python 3.12 is required; found {sys.version_info.major}.{sys.version_info.minor}"
        )
    observed: dict[str, str] = {}
    for package in _CORE_RUNTIME_PACKAGES:
        expected_name = next(name for name in expected if name.lower() == package.lower())
        installed = _module_version(package)
        if installed is None:
            raise RuntimeLockError(f"Required runtime package is absent: {package}")
        if _version_key(installed) != _version_key(expected[expected_name]):
            raise RuntimeLockError(
                f"Imported runtime package mismatch for {package}: {installed} != "
                f"{expected[expected_name]}"
            )
        distribution_versions = _distribution_versions(package)
        allowed_distribution_counts = {0, 1} if package == "rdkit" else {1}
        if len(distribution_versions) not in allowed_distribution_counts:
            raise RuntimeLockError(
                f"Runtime package metadata is ambiguous for {package}: "
                f"{list(distribution_versions)}; expected one installed distribution"
            )
        unexpected_distributions = {
            value
            for value in distribution_versions
            if _version_key(value) != _version_key(expected[expected_name])
        }
        if unexpected_distributions:
            raise RuntimeLockError(
                f"Runtime package metadata is ambiguous for {package}: "
                f"{list(distribution_versions)}; expected only {expected[expected_name]}"
            )
        observed[package] = installed
    return observed


_TOP_LEVEL_FILES = {
    "CODEX_HANDOFF.md",
    "CONTENTS.txt",
    "DECISION_MATRIX.csv",
    "IMPLEMENTATION_STATUS.md",
    "LEAKAGE_CONTRACT.md",
    "PACKAGE_INDEX.md",
    "PAPER80_BIS_PROTOCOL.md",
    "README.md",
    "REFERENCES.md",
    "RUN_ORDER.md",
    "VALIDATION_REPORT.txt",
    "pyproject.toml",
    "requirements-lock.txt",
    "requirements-traditional-lock.txt",
}
_SOURCE_DIRECTORIES = (
    "00_SHARED_PROTOCOL",
    "01_LITERATURE_REVIEW",
    "V5_ELIXIRFP_REBUILT",
    "V6_TABULAR_FOUNDATION_MODELS",
    "V5BIS_PAPER80",
    "V6BIS_PAPER80",
    "configs",
    "schemas",
    "scripts",
    "src",
    "tests",
)


def source_tree_files(root: str | Path) -> tuple[Path, ...]:
    """Return the complete internal code/protocol surface, rejecting symlinks."""

    project = Path(root).resolve()
    selected: set[Path] = set()
    for name in _TOP_LEVEL_FILES:
        path = project / name
        if path.exists():
            selected.add(path)
    data_readme = project / "data" / "README.md"
    if data_readme.exists():
        selected.add(data_readme)
    for directory_name in _SOURCE_DIRECTORIES:
        directory = project / directory_name
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Hashed source directory is unsafe: {directory}")
        for path in directory.rglob("*"):
            if path.is_dir():
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Hashed source file is unsafe: {path}")
            relative = path.relative_to(project)
            if "__pycache__" in relative.parts or ".ruff_cache" in relative.parts:
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            selected.add(path)
    for path in selected:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Hashed source file is unsafe: {path}")
    return tuple(sorted(selected, key=lambda item: item.relative_to(project).as_posix()))


def source_tree_inventory(root: str | Path) -> list[dict[str, Any]]:
    project = Path(root).resolve()
    return [
        {
            "path": path.relative_to(project).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in source_tree_files(project)
    ]


def source_tree_sha256(root: str | Path) -> str:
    return canonical_sha256(
        {
            "schema": "geroprotector.source_tree.v1",
            "files": source_tree_inventory(root),
        }
    )


def runtime_environment() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    distribution_versions: dict[str, list[str]] = {}
    for package in _CORE_RUNTIME_PACKAGES:
        versions[package] = _module_version(package)
        distribution_versions[package] = list(_distribution_versions(package))
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "package_distribution_versions": distribution_versions,
    }
