"""Fail-closed recursive YAML configuration resolution."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .hashing import canonical_sha256


class ConfigError(ValueError):
    """Raised when a configuration is ambiguous or unsafe."""


_PLACEHOLDERS = {"REQUIRED_AT_RUNTIME", "REQUIRED_IF_ENABLED"}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ConfigError(f"Config must be a regular non-symlink file: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    return value


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ConfigError(f"Config path contains a symlink: {cursor}")


def resolve_config(path: str | Path) -> dict[str, Any]:
    """Resolve ``extends`` left-to-right and return a canonical plain mapping."""

    lexical_root = Path(path).absolute()
    _reject_symlink_components(lexical_root)
    root = lexical_root.resolve()

    def visit(current: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
        _reject_symlink_components(current)
        current = current.resolve()
        if current in stack:
            cycle = " -> ".join(map(str, (*stack, current)))
            raise ConfigError(f"Config extends cycle: {cycle}")
        raw = _load_yaml(current)
        extends = raw.get("extends", [])
        if isinstance(extends, str):
            extends = [extends]
        if not isinstance(extends, list) or not all(isinstance(x, str) for x in extends):
            raise ConfigError(f"extends must be a string list: {current}")
        merged: dict[str, Any] = {}
        for parent in extends:
            lexical_parent = current.parent / parent
            _reject_symlink_components(lexical_parent)
            parent_path = lexical_parent.resolve()
            if parent_path.parent != current.parent.resolve():
                raise ConfigError(f"Config parent must remain in {current.parent}: {parent}")
            merged = _deep_merge(merged, visit(parent_path, (*stack, current)))
        return _deep_merge(merged, raw)

    resolved = visit(root, ())
    validate_resolved_config(resolved, allow_runtime_placeholders=True)
    return resolved


def unresolved_placeholders(value: Any, *, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        if value.get("enabled") is False:
            return found
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            found.extend(unresolved_placeholders(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(unresolved_placeholders(child, path=f"{path}[{index}]"))
    elif isinstance(value, str) and value in _PLACEHOLDERS:
        found.append(path)
    return found


def validate_resolved_config(
    config: Mapping[str, Any], *, allow_runtime_placeholders: bool = False
) -> None:
    pipeline = config.get("pipeline")
    if pipeline is not None and pipeline not in {
        "reference",
        "V5_ELIXIRFP_REBUILT",
        "V6_TABULAR_FOUNDATION_MODELS",
        "V5BIS_PAPER80",
        "V6BIS_PAPER80",
    }:
        raise ConfigError(f"Unsupported implementation pipeline: {pipeline!r}")
    if pipeline is not None:
        required_roots = {"project", "upstream", "roles", "curation", "primary_split"}
        missing = required_roots - set(config)
        if missing:
            raise ConfigError(f"Resolved pipeline config is missing roots: {sorted(missing)}")
    if pipeline in {"V6_TABULAR_FOUNDATION_MODELS", "V6BIS_PAPER80"}:
        expected_anchor = {
            "model_id": "tabpfn3",
            "panel_family": "chemistry_32",
            "ensemble_size": 4,
        }
        if dict(config.get("inference", {}).get("controlled_ablation_anchor", {})) != (
            expected_anchor
        ):
            raise ConfigError("V6 controlled-ablation anchor differs from the locked contract")
    if pipeline in {"V5BIS_PAPER80", "V6BIS_PAPER80"}:
        design = config.get("evaluation_design", {})
        expected = {
            "name": "paper_random_80_20",
            "role": "retrospective_contextual_only",
            "identity_resolution_before_split": True,
            "source": "v3_1_identity_safe_paper_split",
            "source_paper_rows": 405,
            "curated_rows": 382,
            "train_size": 0.8,
            "test_size": 0.2,
            "random_state": 42,
            "shuffle": True,
            "stratify": False,
            "expected_train_rows": 305,
            "expected_test_rows": 77,
            "expected_test_class_counts": {0: 29, 1: 48},
            "test_identity_file": "configs/paper80_v3_1_test_identities.txt",
            "test_identity_file_sha256": (
                "354c8ffafeb9d404d672077937fae60ae26de3034a6909a5fea98263261b08f4"
            ),
            "inner_selection": {
                "folds": 3,
                "group_unit": "primary_similarity_component",
                "assignment_seed": 142,
            },
            "outer_test_used_for_selection": False,
            "outer_test_used_for_calibration": False,
            "headline_eligible": False,
            "replaces_grouped_primary_validation": False,
        }
        if dict(design) != expected:
            raise ConfigError("Paper80 evaluation design differs from its locked contract")
        expected_base = {
            "V5BIS_PAPER80": "V5_ELIXIRFP_REBUILT",
            "V6BIS_PAPER80": "V6_TABULAR_FOUNDATION_MODELS",
        }[str(pipeline)]
        if config.get("base_pipeline") != expected_base:
            raise ConfigError("Paper80 base-pipeline binding changed")
    data_roots = {"project", "upstream", "roles", "curation"}
    present_data_roots = data_roots & set(config)
    if present_data_roots and present_data_roots != data_roots:
        missing = data_roots - set(config)
        raise ConfigError(f"Incomplete data config roots: {sorted(missing)}")
    if (
        config.get("roles", {}).get("prior_ml_candidates") != "unlabeled_only"
        and "roles" in config
    ):
        raise ConfigError("The 1,488 prior candidates must remain unlabeled_only")
    firewall = config.get("external_firewall")
    if firewall is not None and firewall.get("hagr_development_access") is not False:
        raise ConfigError("HAGR development access must be false")
    if not allow_runtime_placeholders:
        placeholders = unresolved_placeholders(config)
        if placeholders:
            raise ConfigError(
                "Replace runtime placeholders before execution: " + ", ".join(placeholders)
            )


def resolved_config_sha256(config: Mapping[str, Any]) -> str:
    return canonical_sha256(config)


_V6_RUNTIME_FIELDS = {
    "checkpoint_path",
    "checkpoint_sha256",
    "license_path",
    "license_sha256",
    "checkpoint_source",
    "access_date_utc",
}


def protocol_contract_sha256(config: Mapping[str, Any]) -> str:
    """Hash the full protocol while redacting only approved V6 deployment values."""

    value = copy.deepcopy(dict(config))
    if value.get("pipeline") in {"V6_TABULAR_FOUNDATION_MODELS", "V6BIS_PAPER80"}:
        for settings in value.get("models", {}).values():
            if not isinstance(settings, dict) or settings.get("enabled") is not True:
                continue
            for key in _V6_RUNTIME_FIELDS:
                if key in settings:
                    settings[key] = f"<RUNTIME:{key}>"
    return canonical_sha256(value)


def validate_protocol_lock(
    *, root: str | Path, contract_name: str, config: Mapping[str, Any]
) -> None:
    path = Path(root).resolve() / "configs" / "protocol_lock.json"
    _reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise ConfigError(f"Protocol lock is missing or unsafe: {path}")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != "geroprotector.protocol_lock.v1":
        raise ConfigError("Protocol lock schema changed")
    expected = lock.get("contracts", {}).get(contract_name)
    observed = protocol_contract_sha256(config)
    if expected != observed:
        raise ConfigError(
            f"Protocol contract changed for {contract_name}: {observed} != {expected}"
        )


def validate_locked_shared_contract(config: Mapping[str, Any]) -> None:
    """Reject configuration prose that does not match the implemented Stage-0 contract."""

    validate_resolved_config(config, allow_runtime_placeholders=True)
    expected_roles = {
        "reported_positive": "supervised_positive",
        "weak_chembl_reference": "weak_negative_or_unlabeled",
        "prior_ml_candidates": "unlabeled_only",
        "hagr_drugage": "historical_external_stress_test_only",
    }
    if dict(config["roles"]) != expected_roles:
        raise ConfigError("Data roles differ from the locked implementation contract")
    if config["upstream"].get("commit") != "c8f458925f5ea2beeba87c7c2dda62eefacf618c":
        raise ConfigError("Upstream commit differs from the locked implementation")
    curation = config["curation"]
    expected_curation = {
        "rdkit_version_pin": "2025.03.6",
        "parent_policy": "largest_organic_fragment_versioned",
        "normalize_functional_groups": True,
        "reionize": True,
        "tautomer_canonicalization": "diagnostic_only",
        "identity_key": "connectivity_inchikey",
        "preserve_stereochemistry": True,
        "duplicate_policy": "collapse_consistent",
        "conflict_policy": "exclude_entire_identity_group",
        "raw_universe": "direct_pinned_upstream_1893_rows",
        "source_attribution_precedes_standardization": True,
    }
    changed = {
        key: curation.get(key)
        for key, expected in expected_curation.items()
        if curation.get(key) != expected
    }
    if changed:
        raise ConfigError(f"Curation contract changed: {changed}")
    if dict(curation.get("expected_counts", {})) != {
        "total": 382,
        "positive": 202,
        "weak_reference": 180,
    }:
        raise ConfigError("Expected curated counts differ from the locked implementation")
    split = config["primary_split"]
    expected_split = {
        "family": "morgan_bit",
        "radius": 2,
        "n_bits": 2048,
        "use_chirality": False,
    }
    if (
        dict(split.get("fingerprint", {})) != expected_split
        or float(split.get("edge_threshold", -1)) != 0.40
        or int(split.get("outer_repeats", -1)) != 5
        or int(split.get("outer_folds", -1)) != 5
        or list(split.get("outer_seeds", [])) != [41, 42, 43, 44, 45]
    ):
        raise ConfigError("Primary split differs from the locked implementation")
    if config.get("inner_cv", {}).get("folds") != 3 or config.get("inner_cv", {}).get(
        "seeds"
    ) != [101, 102, 103]:
        raise ConfigError("Inner split differs from the locked shared registry")
    firewall = config.get("external_firewall", {})
    expected_firewall = {
        "hagr_role": "reused_historical_stress_test",
        "hagr_development_access": False,
        "external2_score_once": True,
        "adaptation_requires_second_untouched_set": True,
    }
    if dict(firewall) != expected_firewall:
        raise ConfigError("External firewall keys or values differ from the locked contract")
