"""Classical controls selected on exactly the same fold-local V6 panels."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ...hashing import atomic_write_json, canonical_sha256, sha256_file
from ...validation.split_registry import validate_selection_folds
from ..resources import abort_on_resource_exhaustion
from .feature_panels import (
    V6FeatureStore,
    build_panel_transformer,
    panel_candidates,
    panel_declared_dimension,
)

_MODEL_IDS = {
    "elastic_net": "v6_baseline_elastic_net",
    "extra_trees": "v6_baseline_extra_trees",
    "xgboost": "v6_baseline_xgboost",
}


def _candidate_complexity(kind: str, candidate: Mapping[str, Any]) -> int:
    dimension = panel_declared_dimension(candidate["panel"])
    estimator = candidate["estimator"]
    if kind == "elastic_net":
        # Larger C means weaker regularisation; larger l1_ratio means more sparsity.
        model_term = round(1_000 * np.log10(float(estimator["C"]) + 1.0))
        model_term += round(100 * (1.0 - float(estimator["l1_ratio"])))
    elif kind == "extra_trees":
        model_term = int(estimator["n_estimators"]) * 100
        model_term += round(1_000 / int(estimator["min_samples_leaf"]))
    else:
        model_term = int(estimator["n_estimators"]) * (2 ** int(estimator["max_depth"]))
    return int(dimension * 1_000_000 + model_term)


def _model_candidates(kind: str, config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    settings = config["baseline_search"][kind]
    if kind == "elastic_net":
        return tuple(
            {"C": float(c), "l1_ratio": float(ratio)}
            for c in settings["C"]
            for ratio in settings["l1_ratio"]
        )
    if kind == "extra_trees":
        return tuple(
            {
                "n_estimators": int(settings["n_estimators"]),
                "min_samples_leaf": int(leaf),
                "max_features": value,
            }
            for leaf in settings["min_samples_leaf"]
            for value in settings["max_features"]
        )
    if kind == "xgboost":
        return tuple(dict(value) for value in settings["locked_candidates"])
    raise ValueError(f"Unknown V6 classical baseline: {kind}")


def _portfolio(
    kind: str,
    config: Mapping[str, Any],
    *,
    seed: int,
    maximum_svd_components: int,
    fixed_panel_family: str | None = None,
) -> tuple[dict, ...]:
    panels = tuple(
        panel
        for panel in panel_candidates(config)
        if panel["panel"] != "morgan_svd_plus_descriptors"
        or int(panel["n_components"]) <= int(maximum_svd_components)
    )
    if not panels:
        raise ValueError("No V6 baseline panel is feasible in every selection fold")
    if fixed_panel_family is not None:
        panels = tuple(panel for panel in panels if panel["panel"] == str(fixed_panel_family))
        if not panels:
            raise ValueError("Fixed V6 baseline panel has no feasible candidate")
    models = _model_candidates(kind, config)
    # Every panel is represented by the same locked anchor. Remaining contrasts are
    # deterministically assigned to compact descriptor panels, without score peeking.
    raw = [{"panel": panel, "estimator": models[0]} for panel in panels]
    compact = [panel for panel in panels if panel["panel"] != "morgan_svd_plus_descriptors"]
    raw.extend(
        {"panel": panel, "estimator": estimator}
        for panel in compact
        for estimator in models[1:]
    )
    unique = {canonical_sha256(item): item for item in raw}
    ordered = sorted(
        unique.values(),
        key=lambda item: hashlib.sha256(f"{seed}\0{item}".encode()).digest(),
    )
    cap = int(config["baseline_search"]["max_candidates_per_active_fit"])
    if len(panels) > cap:
        raise ValueError("V6 baseline cap cannot represent every input panel")
    anchors = [item for item in raw[: len(panels)]]
    anchor_hashes = {canonical_sha256(item) for item in anchors}
    remainder = [item for item in ordered if canonical_sha256(item) not in anchor_hashes]
    return tuple((*anchors, *remainder[: max(0, cap - len(anchors))]))


def _estimator(kind: str, spec: Mapping[str, Any], *, seed: int):
    if kind == "elastic_net":
        return LogisticRegression(
            C=float(spec["C"]),
            l1_ratio=float(spec["l1_ratio"]),
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=10_000,
            random_state=seed,
        )
    if kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(spec["n_estimators"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            max_features=spec["max_features"],
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    if kind == "xgboost":
        return XGBClassifier(**dict(spec), eval_metric="logloss", n_jobs=1, random_state=seed)
    raise ValueError(f"Unknown V6 classical baseline: {kind}")


class V6ClassicalPipeline:
    """One baseline family with panel/config selection confined to active fit IDs."""

    def __init__(
        self,
        kind: str,
        config: Mapping[str, Any],
        *,
        seed: int,
        fixed_panel_family: str | None = None,
    ):
        if kind not in _MODEL_IDS:
            raise ValueError(f"Unsupported V6 baseline: {kind}")
        self.kind = kind
        self.model_id = _MODEL_IDS[kind]
        self.config = dict(config)
        self.seed = int(seed)
        self.fixed_panel_family = fixed_panel_family
        if fixed_panel_family is not None:
            self.model_id = f"{self.model_id}_fixed_{fixed_panel_family}"

    def _fit_candidate(
        self,
        store: V6FeatureStore,
        ids: Sequence[str],
        labels: pd.Series,
        candidate: Mapping[str, Any],
        *,
        seed: int,
    ) -> tuple[Any, Any, StandardScaler | None]:
        panel = build_panel_transformer(candidate["panel"], seed=seed)
        panel.fit(store, fit_ids=ids)
        X = panel.transform(store, ids).to_numpy(dtype=float)
        scaler = None
        if self.kind == "elastic_net":
            scaler = StandardScaler().fit(X)
            X = scaler.transform(X)
        estimator = _estimator(self.kind, candidate["estimator"], seed=seed)
        kwargs: dict[str, Any] = {}
        y = labels.loc[list(ids)].to_numpy(dtype=int)
        if self.kind == "xgboost":
            kwargs["sample_weight"] = compute_sample_weight("balanced", y)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            estimator.fit(X, y, **kwargs)
        if tuple(map(int, estimator.classes_)) != (0, 1):
            raise RuntimeError("V6 baseline estimator class order differs from [0, 1]")
        return panel, estimator, scaler

    @staticmethod
    def _predict(
        panel: Any,
        estimator: Any,
        scaler: StandardScaler | None,
        store: V6FeatureStore,
        ids: Sequence[str],
    ) -> np.ndarray:
        X = panel.transform(store, ids).to_numpy(dtype=float)
        if scaler is not None:
            X = scaler.transform(X)
        probability = np.asarray(estimator.predict_proba(X), dtype=float)
        if (
            probability.shape != (len(ids), 2)
            or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))
            or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6)
        ):
            raise RuntimeError("V6 baseline returned invalid probabilities")
        return probability

    def fit(
        self,
        store: V6FeatureStore,
        y: pd.Series,
        *,
        groups: pd.Series,
        selection_folds: Sequence[tuple[tuple[str, ...], tuple[str, ...]]],
    ) -> V6ClassicalPipeline:
        store.assert_integrity()
        labels = y.copy()
        labels.index = labels.index.astype(str)
        group_values = groups.copy()
        group_values.index = group_values.index.astype(str)
        folds = validate_selection_folds(
            fit_ids=store.ids,
            labels=labels,
            groups=group_values,
            folds=selection_folds,
        )
        trace: list[dict[str, Any]] = []
        maximum_svd_components = min(len(train_ids) - 1 for train_ids, _ in folds)
        candidates = _portfolio(
            self.kind,
            self.config,
            seed=self.seed,
            maximum_svd_components=maximum_svd_components,
            fixed_panel_family=self.fixed_panel_family,
        )
        for candidate in candidates:
            try:
                oof = pd.Series(index=labels.index, dtype=float)
                for fold_index, (train_ids, validation_ids) in enumerate(folds):
                    local_seed = self.seed + fold_index
                    panel, estimator, scaler = self._fit_candidate(
                        store, train_ids, labels, candidate, seed=local_seed
                    )
                    oof.loc[list(validation_ids)] = self._predict(
                        panel, estimator, scaler, store, validation_ids
                    )[:, 1]
                if oof.isna().any():
                    raise RuntimeError("V6 baseline candidate lacks complete inner OOF")
                outcome = {
                    "status": "success",
                    "failure_type": None,
                    "failure_message": None,
                    "ap_positive": float(average_precision_score(labels, oof)),
                    "auroc": float(roc_auc_score(labels, oof)),
                    "brier": float(brier_score_loss(labels, oof)),
                }
            except Exception as exc:
                abort_on_resource_exhaustion(exc, stage=f"V6 {self.kind} baseline selection")
                outcome = {
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc)[:1000],
                    "ap_positive": np.nan,
                    "auroc": np.nan,
                    "brier": np.nan,
                }
            trace.append(
                {
                    "candidate_id": canonical_sha256(candidate),
                    "candidate_spec": dict(candidate),
                    **outcome,
                    "complexity": _candidate_complexity(self.kind, candidate),
                    "outer_test_metric_consulted": False,
                    "hagr_metric_consulted": False,
                }
            )
        table = pd.DataFrame(trace)
        successful = table.loc[table["status"].eq("success")]
        if successful.empty:
            raise RuntimeError(f"Every V6 {self.kind} baseline candidate failed")
        expected_panels = {canonical_sha256(item["panel"]) for item in candidates}
        successful_panels = {
            canonical_sha256(item["panel"]) for item in successful["candidate_spec"]
        }
        if expected_panels != successful_panels:
            raise RuntimeError("A required V6 baseline input panel failed completely")
        best_ap = float(successful["ap_positive"].max())
        eligible = successful.loc[successful["ap_positive"].ge(best_ap - 0.005)]
        best_auc = float(eligible["auroc"].max())
        winner_row = (
            eligible.loc[eligible["auroc"].ge(best_auc - 0.005)]
            .sort_values(["brier", "complexity", "candidate_id"], kind="stable")
            .iloc[0]
        )
        winner = dict(winner_row["candidate_spec"])
        self.panel_, self.estimator_, self.scaler_ = self._fit_candidate(
            store, store.ids, labels, winner, seed=self.seed + 999_983
        )
        self.fit_ids_ = tuple(store.ids)
        self.store_hash_ = store.store_hash
        self.feature_contract_hash_ = store.feature_contract_hash
        self.winner_ = winner
        self.selection_trace_ = table
        self.selection_trace_["selected"] = self.selection_trace_["candidate_id"].eq(
            canonical_sha256(winner)
        )
        return self

    def predict_proba(self, store: V6FeatureStore, ids: Sequence[str]) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("V6 classical baseline is not fitted")
        store.assert_integrity()
        if store.feature_contract_hash != self.feature_contract_hash_:
            raise ValueError("V6 baseline feature-extractor contract differs from fit")
        requested = tuple(map(str, ids))
        if set(requested) & set(self.fit_ids_):
            raise ValueError("V6 baseline held-out prediction IDs overlap fitted IDs")
        return self._predict(self.panel_, self.estimator_, self.scaler_, store, requested)

    def get_selection_trace(self) -> pd.DataFrame:
        return self.selection_trace_.copy()

    def get_manifest(self) -> dict[str, Any]:
        return {
            "pipeline": "V6_TABULAR_FOUNDATION_MODELS",
            "model_id": self.model_id,
            "fixed_panel_family": self.fixed_panel_family,
            "winner": self.winner_,
            "fit_ids": list(self.fit_ids_),
            "fit_feature_content_sha256": self.store_hash_,
            "feature_contract_sha256": self.feature_contract_hash_,
            "panel": self.panel_.get_manifest(),
            "selection_trace_sha256": canonical_sha256(
                self.selection_trace_.fillna("<NA>").to_dict(orient="records")
            ),
            "outer_test_metric_consulted": False,
            "hagr_metric_consulted": False,
        }

    def save(self, path: str | Path) -> dict[str, Any]:
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite V6 baseline: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        manifest = self.get_manifest()
        manifest["model_sha256"] = sha256_file(destination)
        atomic_write_json(
            destination.with_suffix(destination.suffix + ".manifest.json"), manifest
        )
        return manifest
