"""Deterministic, capped downstream V5 candidate portfolio."""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping
from typing import Any

from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier


def _stable_order(value: dict[str, Any], seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{sorted(value.items())}".encode()).digest()


def downstream_candidates(
    config: Mapping[str, Any], *, seed: int
) -> tuple[dict[str, Any], ...]:
    models = config["models"]
    candidates: list[dict[str, Any]] = []
    tree = models["extra_trees"]
    for leaf, max_features, class_weight in itertools.product(
        tree["min_samples_leaf"], tree["max_features"], tree["class_weight"]
    ):
        candidates.append(
            {
                "model": "extra_trees",
                "n_estimators": int(tree["n_estimators"]),
                "min_samples_leaf": int(leaf),
                "max_features": max_features,
                "class_weight": class_weight,
            }
        )
    xgb = models["xgboost"]
    # A deterministic low-discrepancy-like portfolio replaces an uncontrolled Cartesian grid.
    full_xgb = [
        {
            "model": "xgboost",
            "n_estimators": int(values[0]),
            "max_depth": int(values[1]),
            "learning_rate": float(values[2]),
            "min_child_weight": float(values[3]),
            "subsample": float(values[4]),
            "colsample_bytree": float(values[5]),
            "reg_alpha": float(values[6]),
            "reg_lambda": float(values[7]),
        }
        for values in itertools.product(
            xgb["n_estimators"],
            xgb["max_depth"],
            xgb["learning_rate"],
            xgb["min_child_weight"],
            xgb["subsample"],
            xgb["colsample_bytree"],
            xgb["reg_alpha"],
            xgb["reg_lambda"],
        )
    ]
    full_xgb.sort(key=lambda item: _stable_order(item, seed))
    candidates.extend(full_xgb[:24])
    candidates.sort(key=lambda item: (item["model"], _stable_order(item, seed)))
    return tuple(candidates)


def build_estimator(spec: Mapping[str, Any], *, seed: int):
    parameters = dict(spec)
    model = parameters.pop("model")
    if model == "extra_trees":
        return ExtraTreesClassifier(**parameters, n_jobs=-1, random_state=int(seed))
    if model == "xgboost":
        return XGBClassifier(
            **parameters,
            eval_metric="logloss",
            n_jobs=1,
            random_state=int(seed),
        )
    raise ValueError(f"Unsupported V5 downstream estimator: {model}")
