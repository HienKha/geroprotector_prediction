"""Stage-A fingerprint resolution search confined to the active training partition."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score

from ...hashing import canonical_sha256
from ...validation.split_registry import grouped_inner_folds
from ..resources import abort_on_resource_exhaustion
from .fingerprint_bank import V5FeatureBank, semantic_family


def _training_diagnostics(matrix: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    """Compact training-only collision/support diagnostics for one fingerprint block."""

    X = np.asarray(matrix, dtype=float)
    y = np.asarray(labels, dtype=int)
    occupied = X != 0
    support = occupied.sum(axis=0)
    column_hashes = {
        hashlib.sha256(np.asarray(X[:, index], dtype="<f4").tobytes()).digest()
        for index in range(X.shape[1])
    }
    unique_columns = len(column_hashes)
    top = np.argsort(support, kind="stable")[-min(64, X.shape[1]) :]
    top_values = X[:, top]
    varying = np.ptp(top_values, axis=0) > 0
    correlation_values: np.ndarray
    if int(varying.sum()) >= 2:
        correlation = np.corrcoef(top_values[:, varying], rowvar=False)
        correlation_values = np.abs(correlation[np.triu_indices_from(correlation, k=1)])
        correlation_values = correlation_values[np.isfinite(correlation_values)]
    else:
        correlation_values = np.asarray([], dtype=float)
    centered = X - X.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    eigenvalues = np.clip(np.linalg.eigvalsh(gram), 0.0, None)
    total = float(eigenvalues.sum())
    if total > 0:
        probabilities = eigenvalues[eigenvalues > 0] / total
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    else:
        effective_rank = 0.0
    positive = occupied[y == 1].mean(axis=0)
    negative = occupied[y == 0].mean(axis=0)
    class_difference = np.abs(positive - negative)
    occupied_values = X[occupied]
    return {
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "all_zero_fraction": float(np.mean(support == 0)),
        "exact_duplicate_column_count": int(X.shape[1] - unique_columns),
        "occupied_count_gt_one_fraction": (
            float(np.mean(occupied_values > 1)) if len(occupied_values) else 0.0
        ),
        "class_conditional_support_difference_mean": float(class_difference.mean()),
        "class_conditional_support_difference_max": float(class_difference.max()),
        "high_support_abs_correlation_median": (
            float(np.median(correlation_values)) if len(correlation_values) else 0.0
        ),
        "centered_effective_rank": effective_rank,
    }


def select_resolutions(
    bank: V5FeatureBank,
    y: pd.Series,
    groups: pd.Series,
    *,
    config: Mapping[str, Any],
    seed: int,
    folds: Sequence[tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
) -> tuple[tuple[str, ...], pd.DataFrame]:
    if set(bank.ids) != set(y.index.astype(str)) or set(bank.ids) != set(
        groups.index.astype(str)
    ):
        raise ValueError("Stage-A bank, labels and groups must cover identical IDs")
    active_folds = (
        list(folds)
        if folds is not None
        else grouped_inner_folds(y, groups, n_splits=3, seed=seed)
    )
    settings = config["stage_a"]
    traces: list[dict[str, Any]] = []
    by_family: dict[str, list[str]] = defaultdict(list)
    for spec_id, spec in bank.specs.items():
        by_family[semantic_family(spec)].append(spec_id)
    if sorted(by_family) != sorted(map(str, settings["semantic_families"])):
        raise ValueError("Stage-A semantic families differ from the locked contract")
    selected: list[str] = []
    for family in sorted(by_family):
        candidates = sorted(by_family[family], key=lambda key: bank.specs[key].n_bits)
        family_rows: list[dict[str, Any]] = []
        for spec_id in candidates:
            candidate_spec = {"stage": "A_resolution", "family": family, "spec_id": spec_id}
            scores: list[float] = []
            fold_lineage: list[dict[str, Any]] = []
            try:
                for fold_index, (train_ids, validation_ids) in enumerate(active_folds):
                    X_train = bank.matrix(spec_id, train_ids)
                    y_train = y.loc[list(train_ids)].to_numpy(dtype=int)
                    estimator = ExtraTreesClassifier(
                        n_estimators=int(settings["n_estimators"]),
                        min_samples_leaf=int(settings["min_samples_leaf"]),
                        max_features=settings["max_features"],
                        class_weight=settings["class_weight"],
                        n_jobs=-1,
                        random_state=int(seed + fold_index),
                    )
                    estimator.fit(X_train, y_train)
                    classes = tuple(map(int, estimator.classes_))
                    probability_matrix = np.asarray(
                        estimator.predict_proba(bank.matrix(spec_id, validation_ids)),
                        dtype=float,
                    )
                    if (
                        classes != (0, 1)
                        or probability_matrix.shape != (len(validation_ids), 2)
                        or not np.isfinite(probability_matrix).all()
                        or np.any((probability_matrix < 0) | (probability_matrix > 1))
                        or not np.allclose(probability_matrix.sum(axis=1), 1.0, atol=1e-6)
                    ):
                        raise ValueError("Stage-A estimator returned invalid probabilities")
                    scores.append(
                        float(
                            average_precision_score(
                                y.loc[list(validation_ids)], probability_matrix[:, 1]
                            )
                        )
                    )
                    fold_lineage.append(
                        {
                            "fold": fold_index,
                            "train_ids_sha256": hashlib.sha256(
                                "\n".join(sorted(train_ids)).encode()
                            ).hexdigest(),
                            "validation_ids_sha256": hashlib.sha256(
                                "\n".join(sorted(validation_ids)).encode()
                            ).hexdigest(),
                            "fit_validation_overlap": 0,
                            "training_only_diagnostics": _training_diagnostics(
                                X_train, y_train
                            ),
                        }
                    )
                mean = float(np.mean(scores))
                sd = float(np.std(scores, ddof=0))
                status = "success"
                failure_type = None
                failure_message = None
            except Exception as exc:
                abort_on_resource_exhaustion(exc, stage="V5 Stage A")
                mean = float("nan")
                sd = float("nan")
                status = "failed"
                failure_type = type(exc).__name__
                failure_message = str(exc)[:1000]
            row = {
                "stage": "A_resolution",
                "family": family,
                "spec_id": spec_id,
                "candidate_id": canonical_sha256(candidate_spec),
                "candidate_spec": candidate_spec,
                "status": status,
                "failure_type": failure_type,
                "failure_message": failure_message,
                "n_bits": int(bank.specs[spec_id].n_bits),
                "mean_inner_ap": mean,
                "sd_inner_ap": sd,
                "selection_objective": (
                    mean - float(settings["objective_sd_penalty"]) * sd
                    if status == "success"
                    else float("nan")
                ),
                "active_fit_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(bank.ids)).encode()
                ).hexdigest(),
                "fold_lineage": fold_lineage,
                "outer_test_metric_consulted": False,
                "hagr_metric_consulted": False,
            }
            traces.append(row)
            if status == "success":
                family_rows.append(row)
        if not family_rows:
            raise RuntimeError(f"Every Stage-A resolution candidate failed for {family}")
        winner = max(
            family_rows,
            key=lambda row: (row["selection_objective"], -row["n_bits"], row["spec_id"]),
        )
        selected.append(str(winner["spec_id"]))
    trace = pd.DataFrame(traces)
    trace["selected"] = trace["status"].eq("success") & trace["spec_id"].isin(selected)
    return tuple(selected), trace
