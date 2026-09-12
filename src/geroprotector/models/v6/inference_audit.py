"""Label-free audit of nonstandard foundation-model inference semantics."""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def release_accelerator_memory() -> None:
    """Best-effort release between independently reconstructed audit adapters."""

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _finite_rank_correlation(left: np.ndarray, right: np.ndarray) -> tuple[float | None, str]:
    if np.array_equal(left, right):
        return 1.0, "identical_vectors"
    value = float(spearmanr(left, right).statistic)
    if np.isfinite(value):
        return value, "spearman"
    return None, "undefined_constant_or_tied_vector"


def audit_inference_semantics(
    adapter_factory: Callable[[], Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_query: pd.DataFrame,
    *,
    atol: float,
    seed: int,
    random_compositions_per_query: int = 10,
    hard_fail_on_context_row_order: bool = True,
    hard_fail_on_feature_order: bool = False,
) -> dict[str, Any]:
    if len(X_train) < 2 or len(X_query) < 1:
        raise ValueError(
            "Inference audit requires nonempty query and at least two context rows"
        )
    if set(X_train.index.astype(str)) & set(X_query.index.astype(str)):
        raise ValueError("Inference audit context/query IDs overlap")
    rng = np.random.default_rng(seed)
    random_batch_differences: list[float] = []
    random_batch_means = np.zeros(len(X_query), dtype=float)
    adapter = None
    try:
        adapter = adapter_factory().fit_context(X_train.copy(), np.asarray(y_train).copy())
        canonical = adapter.predict_proba(X_query.copy(), canonical_mode=True)[:, 1]
        repeated = adapter.predict_proba(X_query.copy(), canonical_mode=True)[:, 1]
        whole_batch = adapter.predict_proba(X_query.copy(), canonical_mode=False)[:, 1]
        reversed_prediction = adapter.predict_proba(
            X_query.iloc[::-1].copy(), canonical_mode=True
        )[::-1, 1]
        if len(X_query) > 1:
            for query_index in range(len(X_query)):
                other = np.delete(np.arange(len(X_query)), query_index)
                values: list[float] = []
                for _ in range(int(random_compositions_per_query)):
                    sample = rng.choice(other, size=min(9, len(other)), replace=False)
                    batch_indices = np.concatenate(([query_index], sample))
                    batch_indices = batch_indices[rng.permutation(len(batch_indices))]
                    batch = X_query.iloc[batch_indices].copy()
                    query_position = int(np.flatnonzero(batch_indices == query_index)[0])
                    value = float(
                        adapter.predict_proba(batch, canonical_mode=False)[query_position, 1]
                    )
                    values.append(value)
                    random_batch_differences.append(abs(value - float(canonical[query_index])))
                random_batch_means[query_index] = float(np.mean(values))
        else:
            random_batch_means[:] = canonical
    finally:
        if adapter is not None:
            del adapter
        release_accelerator_memory()
    row_reordered = None
    try:
        row_reordered = adapter_factory().fit_context(
            X_train.iloc[::-1].copy(), np.asarray(y_train[::-1]).copy()
        )
        row_order_probability = row_reordered.predict_proba(
            X_query.copy(), canonical_mode=True
        )[:, 1]
    finally:
        if row_reordered is not None:
            del row_reordered
        release_accelerator_memory()
    permutation = rng.permutation(X_train.shape[1])
    feature_reordered = None
    try:
        feature_reordered = adapter_factory().fit_context(
            X_train.iloc[:, permutation].copy(), np.asarray(y_train).copy()
        )
        feature_order_probability = feature_reordered.predict_proba(
            X_query.iloc[:, permutation].copy(), canonical_mode=True
        )[:, 1]
    finally:
        if feature_reordered is not None:
            del feature_reordered
        release_accelerator_memory()
    differences = {
        "repeated_call": float(np.max(np.abs(canonical - repeated))),
        "query_row_order": float(np.max(np.abs(canonical - reversed_prediction))),
        "whole_batch_vs_singleton": float(np.max(np.abs(canonical - whole_batch))),
        "random_composition_vs_singleton": (
            float(max(random_batch_differences)) if random_batch_differences else 0.0
        ),
        "context_row_order": float(np.max(np.abs(canonical - row_order_probability))),
        "matched_feature_permutation": float(
            np.max(np.abs(canonical - feature_order_probability))
        ),
    }
    hard_checks = {
        "repeated_call": differences["repeated_call"] <= atol,
        "query_row_order": differences["query_row_order"] <= atol,
        "context_row_order": (
            differences["context_row_order"] <= atol if hard_fail_on_context_row_order else True
        ),
        "matched_feature_permutation": (
            differences["matched_feature_permutation"] <= atol
            if hard_fail_on_feature_order
            else True
        ),
    }
    whole_rank, whole_rank_reason = _finite_rank_correlation(canonical, whole_batch)
    random_rank, random_rank_reason = _finite_rank_correlation(canonical, random_batch_means)
    canonical_passed = bool(all(hard_checks.values()))
    return {
        "schema_version": "geroprotector.v6_inference_audit.v1",
        "n_context": len(X_train),
        "n_query": len(X_query),
        "atol": float(atol),
        "canonical_mode": "one_query_per_call",
        "canonical_repeat_and_query_order_passed": canonical_passed,
        "hard_checks": hard_checks,
        "hard_fail_on_context_row_order": bool(hard_fail_on_context_row_order),
        "hard_fail_on_feature_order": bool(hard_fail_on_feature_order),
        "raw_batch_dependence_is_diagnostic_only": True,
        "max_abs_difference": differences,
        "median_abs_difference": {
            "whole_batch_vs_singleton": float(np.median(np.abs(canonical - whole_batch))),
            "random_composition_vs_singleton": (
                float(np.median(random_batch_differences)) if random_batch_differences else 0.0
            ),
        },
        "rank_correlation": {
            "whole_batch_vs_singleton": whole_rank,
            "whole_batch_reason": whole_rank_reason,
            "random_composition_mean_vs_singleton": random_rank,
            "random_composition_reason": random_rank_reason,
        },
        "random_compositions_per_query": int(random_compositions_per_query),
        "passed": canonical_passed,
    }
