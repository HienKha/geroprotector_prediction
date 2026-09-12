"""Guarded secondary-phase entry points; primary V6 is always zero-shot first."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def assert_primary_zero_shot(config: Mapping[str, Any]) -> None:
    inference = config.get("inference", {})
    expected_inference = {
        "preprocessing_mode": "training_median_drop_constant",
        "canonical_mode": "one_query_per_call",
        "batch_composition_audit": True,
        "row_order_audit": True,
        "feature_order_audit": True,
        "repeated_call_audit": True,
        "telemetry_disabled": True,
    }
    changed = {
        key: inference.get(key)
        for key, expected in expected_inference.items()
        if inference.get(key) != expected
    }
    if changed:
        raise ValueError(f"Primary V6 inference contract changed: {changed}")
    if config.get("execution", {}).get("implicit_network_access") is not False:
        raise ValueError("Primary V6 must prohibit implicit network access")
    adaptation = config["adaptation"]
    if adaptation.get("execution_phase") != "zero_shot_only":
        raise ValueError("Primary V6 execution must be zero_shot_only")
    active = [
        name
        for name in ("beta_style_encoder", "context_only_ttt", "full_finetuning")
        if adaptation.get(name, {}).get("enabled") is True
    ]
    if active:
        raise ValueError(
            "Adaptation branches require a separate config/run after zero-shot lock: "
            + ", ".join(active)
        )


def run_secondary_adaptation(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError(
        "V6 adaptation is intentionally not part of the primary zero-shot run. "
        "Implement against a pinned checkpoint API in a separately locked protocol."
    )
