"""Paired chemical-component bootstrap."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def paired_component_bootstrap(
    frame: pd.DataFrame,
    *,
    candidate_column: str,
    reference_column: str,
    label_column: str = "label",
    component_column: str = "component_id",
    n_resamples: int = 10_000,
    seed: int = 20260408,
    metric: Callable[[np.ndarray, np.ndarray], float] = average_precision_score,
) -> dict:
    required = {candidate_column, reference_column, label_column, component_column}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Bootstrap columns missing: {sorted(missing)}")
    table = frame.reset_index(drop=True).copy()
    units = sorted(table[component_column].astype(str).unique())
    if len(units) < 2:
        raise ValueError("At least two bootstrap components are required")
    by_unit = {
        unit: table.index[table[component_column].astype(str).eq(unit)].to_numpy()
        for unit in units
    }
    y = table[label_column].to_numpy(dtype=int)
    candidate = table[candidate_column].to_numpy(dtype=float)
    reference = table[reference_column].to_numpy(dtype=float)
    observed = float(metric(y, candidate) - metric(y, reference))
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(int(n_resamples)):
        sampled = rng.choice(units, size=len(units), replace=True)
        indices = np.concatenate([by_unit[str(unit)] for unit in sampled])
        if len(np.unique(y[indices])) != 2:
            continue
        differences.append(
            float(
                metric(y[indices], candidate[indices]) - metric(y[indices], reference[indices])
            )
        )
    if not differences:
        raise ValueError("No valid two-class bootstrap samples")
    values = np.asarray(differences, dtype=float)
    centered = values - observed
    p_value = float(
        (np.count_nonzero(np.abs(centered) >= abs(observed)) + 1) / (len(values) + 1)
    )
    return {
        "metric": "ap_positive",
        "delta_candidate_minus_reference": observed,
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "p_two_sided": p_value,
        "superiority_probability": float(np.mean(values > 0)),
        "bootstrap_unit": "primary_component",
        "bootstrap_units": len(units),
        "n_resamples": int(n_resamples),
        "n_valid_resamples": len(values),
        "seed": int(seed),
        "multiple_testing_adjustment": "none",
    }
