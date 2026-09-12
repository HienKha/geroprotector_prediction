"""Physical and logical development-data firewalls."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class LeakageError(RuntimeError):
    pass


_EXTERNAL_PATTERN = re.compile(r"hagr|drug[\s_.-]*age|external", re.IGNORECASE)

_LOCKED_FIREWALL = {
    "hagr_role": "reused_historical_stress_test",
    "hagr_development_access": False,
    "external2_score_once": True,
    "adaptation_requires_second_untouched_set": True,
}


def assert_internal_config_safe(config: Mapping[str, Any]) -> None:
    # Policy declarations are allowed. No internal execution value may name an external source.
    if dict(config.get("external_firewall", {})) != _LOCKED_FIREWALL:
        raise LeakageError("External firewall keys/values differ from the locked policy")
    scrubbed = dict(config)
    scrubbed.pop("external_firewall", None)
    roles = dict(scrubbed.get("roles", {}))
    roles.pop("hagr_drugage", None)
    if roles:
        scrubbed["roles"] = roles

    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if _EXTERNAL_PATTERN.search(str(key)):
                    if child is False and str(key) in {
                        "hagr_metric_consulted",
                        "hagr_used_for_model_selection",
                        "hagr_labels_loaded_during_development",
                    }:
                        continue
                    if (
                        child is True
                        and str(key) == "thresholds_selected_without_external_outcomes"
                    ):
                        continue
                    raise LeakageError(
                        f"Internal configuration contains an external key: {child_path}"
                    )
                walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, str) and _EXTERNAL_PATTERN.search(value):
            raise LeakageError(
                f"Internal configuration references HAGR/DrugAge/external data: {path}"
            )

    walk(scrubbed)
    roles = config.get("roles", {})
    if roles.get("prior_ml_candidates") != "unlabeled_only":
        raise LeakageError("Prior-predicted candidates are not locked as unlabeled_only")
    if config.get("external_firewall", {}).get("hagr_development_access") is not False:
        raise LeakageError("HAGR development access is not disabled")


def assert_fit_scope(
    *,
    fit_ids: Iterable[object],
    transform_ids: Iterable[object],
    forbidden_ids: Iterable[object] = (),
) -> None:
    fit_values = tuple(str(value) for value in fit_ids)
    transform_values = tuple(str(value) for value in transform_ids)
    forbidden_values = tuple(str(value) for value in forbidden_ids)
    fit = set(fit_values)
    transformed = set(transform_values)
    forbidden = set(forbidden_values)
    if not fit or len(fit) != len(fit_values):
        raise LeakageError("Fit IDs must be non-empty and unique")
    if fit & forbidden:
        raise LeakageError("A forbidden held-out ID entered transformer/model fit")
    if transformed & forbidden and fit & transformed:
        raise LeakageError("Fit and held-out transform scopes overlap")


def assert_no_hagr_paths(paths: Iterable[str | Path]) -> None:
    bad = [str(path) for path in paths if _EXTERNAL_PATTERN.search(str(path))]
    if bad:
        raise LeakageError(f"Internal command received forbidden external paths: {bad}")


def assert_label_blind(records: Iterable[Mapping[str, Any]]) -> None:
    for index, record in enumerate(records):
        if (
            record.get("evaluation_role") == "external_label_blind"
            and record.get("y_true") is not None
        ):
            raise LeakageError(f"External label-blind record {index} contains y_true")
