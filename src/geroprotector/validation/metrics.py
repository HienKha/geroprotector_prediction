"""Compound-level metrics; repeated OOF rows are never treated as independent."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def aggregate_repeated_oof(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "compound_id",
        "label",
        "probability_raw",
        "probability_calibrated",
        "repeat",
    }
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Prediction columns missing: {sorted(missing)}")
    duplicate = predictions.duplicated(["compound_id", "repeat"])
    if duplicate.any():
        raise ValueError("A compound has multiple outer-test predictions in one repeat")
    label_counts = predictions.groupby("compound_id")["label"].nunique()
    if (label_counts != 1).any():
        raise ValueError("Compound labels drift across repeats")
    repeat_counts = predictions.groupby("compound_id")["repeat"].nunique()
    if repeat_counts.nunique() != 1:
        raise ValueError("OOF repeat coverage differs across compounds")
    aggregations = {
        "label": ("label", "first"),
        "probability_calibrated": ("probability_calibrated", "mean"),
        "probability_raw": ("probability_raw", "mean"),
    }
    if "component_id" in predictions:
        component_counts = predictions.groupby("compound_id")["component_id"].nunique()
        if (component_counts != 1).any():
            raise ValueError("Component membership drifts across repeats")
        aggregations["component_id"] = ("component_id", "first")
    result = predictions.groupby("compound_id", as_index=False).agg(**aggregations)
    return result.sort_values("compound_id", kind="stable").reset_index(drop=True)


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    value = 0.0
    for index in range(bins):
        mask = indices == index
        if mask.any():
            value += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(value)


def _adaptive_ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    order = np.argsort(p, kind="stable")
    value = 0.0
    for positions in np.array_split(order, min(int(bins), len(order))):
        if len(positions):
            value += (len(positions) / len(order)) * abs(
                float(y[positions].mean()) - float(p[positions].mean())
            )
    return float(value)


def metric_bundle(
    y: np.ndarray, probability: np.ndarray, *, threshold: float | None = None
) -> dict[str, float | None]:
    labels = np.asarray(y, dtype=int)
    scores = np.asarray(probability, dtype=float)
    if len(labels) != len(scores) or set(labels) != {0, 1}:
        raise ValueError("Metrics require aligned two-class predictions")
    prevalence = float(labels.mean())
    brier = float(brier_score_loss(labels, scores))
    null_brier = prevalence * (1.0 - prevalence)
    output: dict[str, float | None] = {
        "ap_positive": float(average_precision_score(labels, scores)),
        "ap_negative": float(average_precision_score(1 - labels, 1.0 - scores)),
        "auroc": float(roc_auc_score(labels, scores)),
        "brier": brier,
        "brier_skill": float(1.0 - brier / null_brier) if null_brier > 0 else None,
        "ece_fixed": _ece(labels, scores),
        "ece_adaptive": _adaptive_ece(labels, scores),
    }
    logits = np.log(np.clip(scores, 1e-6, 1 - 1e-6) / np.clip(1 - scores, 1e-6, 1))
    calibration = LogisticRegression(C=1e6, solver="lbfgs").fit(logits.reshape(-1, 1), labels)
    output["calibration_intercept"] = float(calibration.intercept_[0])
    output["calibration_slope"] = float(calibration.coef_[0, 0])
    if threshold is None:
        return output
    decision = (scores >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, decision, labels=[0, 1]).ravel()
    output.update(
        mcc=float(matthews_corrcoef(labels, decision)),
        balanced_accuracy=float(balanced_accuracy_score(labels, decision)),
        macro_f1=float(f1_score(labels, decision, average="macro", zero_division=0)),
        sensitivity=float(tp / (tp + fn)) if tp + fn else None,
        specificity=float(tn / (tn + fp)) if tn + fp else None,
        precision=float(tp / (tp + fp)) if tp + fp else None,
        npv=float(tn / (tn + fn)) if tn + fn else None,
    )
    return output
