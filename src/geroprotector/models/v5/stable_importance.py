"""Fold-scoped stable bit importance and shrunk weights."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

from ...hashing import atomic_write_json, canonical_sha256, sha256_file
from ...validation.split_registry import grouped_inner_folds
from .fingerprint_bank import V5FeatureBank


def _normalise_importance(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        return np.full(len(clipped), 1.0 / max(len(clipped), 1), dtype=float)
    return clipped / total


def _load_checkpoint(root: Path, binding: str) -> tuple[np.ndarray, np.ndarray] | None:
    directory = root / binding
    if not directory.exists():
        return None
    artifact = directory / "vectors.joblib"
    manifest_path = directory / "manifest.json"
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or artifact.is_symlink()
        or not artifact.is_file()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
    ):
        raise RuntimeError(f"Unsafe or incomplete Stage-B checkpoint: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = dict(manifest)
    claimed = payload.pop("canonical_sha256", None)
    if (
        claimed != canonical_sha256(payload)
        or manifest.get("schema_version") != "geroprotector.v5_stage_b_vector_checkpoint.v1"
        or manifest.get("binding_sha256") != binding
        or manifest.get("vectors_sha256") != sha256_file(artifact)
    ):
        raise RuntimeError(f"Stage-B checkpoint integrity failed: {directory}")
    value = joblib.load(artifact)
    forest = np.asarray(value["extra_trees_permutation"], dtype=float)
    xgboost = np.asarray(value["xgboost_gain"], dtype=float)
    if (
        forest.ndim != 1
        or xgboost.shape != forest.shape
        or not np.isfinite(forest).all()
        or not np.isfinite(xgboost).all()
    ):
        raise RuntimeError(f"Stage-B checkpoint vectors are invalid: {directory}")
    return forest, xgboost


def _publish_checkpoint(
    root: Path,
    binding: str,
    forest: np.ndarray,
    xgboost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    root.mkdir(parents=True, exist_ok=True)
    destination = root / binding
    staging = Path(tempfile.mkdtemp(prefix=f".{binding}.", dir=root))
    try:
        artifact = staging / "vectors.joblib"
        value = {
            "extra_trees_permutation": np.asarray(forest, dtype=float),
            "xgboost_gain": np.asarray(xgboost, dtype=float),
        }
        joblib.dump(value, artifact)
        with artifact.open("rb") as handle:
            os.fsync(handle.fileno())
        manifest = {
            "schema_version": "geroprotector.v5_stage_b_vector_checkpoint.v1",
            "binding_sha256": binding,
            "vectors_sha256": sha256_file(artifact),
        }
        manifest["canonical_sha256"] = canonical_sha256(manifest)
        atomic_write_json(staging / "manifest.json", manifest)
        try:
            os.rename(staging, destination)
        except FileExistsError:
            cached = _load_checkpoint(root, binding)
            if cached is None:
                raise RuntimeError(
                    f"Concurrent Stage-B checkpoint is incomplete: {destination}"
                ) from None
            return cached
        return value["extra_trees_permutation"], value["xgboost_gain"]
    finally:
        if staging.exists():
            shutil.rmtree(staging)


@dataclass(frozen=True)
class StableImportanceState:
    selected_specs: tuple[str, ...]
    base_importance: Mapping[str, np.ndarray]
    stability_spearman: Mapping[str, float]
    fit_ids: tuple[str, ...]
    subfold_validation_ids: tuple[tuple[str, ...], ...]
    state_hash: str
    stability_diagnostics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def weights(
        self, alpha: float, *, clip: tuple[float, float], epsilon: float
    ) -> dict[str, np.ndarray]:
        if alpha < 0:
            raise ValueError("Importance shrinkage alpha must be non-negative")
        lower, upper = map(float, clip)
        if not 0 < lower <= 1.0 <= upper:
            raise ValueError("Weight bounds must contain one and have a positive lower bound")
        output: dict[str, np.ndarray] = {}
        for spec_id in self.selected_specs:
            importance = np.asarray(self.base_importance[spec_id], dtype=float)
            raw = ((importance + epsilon) / float(np.mean(importance + epsilon))) ** float(
                alpha
            )
            # Project onto the box while preserving an exact mean of one. Dividing after
            # clipping would violate the upper bound for sparse importance vectors.
            low_scale, high_scale = 0.0, max(1.0, upper / float(np.min(raw)))
            for _ in range(100):
                midpoint = (low_scale + high_scale) / 2.0
                if float(np.mean(np.clip(midpoint * raw, lower, upper))) < 1.0:
                    low_scale = midpoint
                else:
                    high_scale = midpoint
            bounded = np.clip(high_scale * raw, lower, upper)
            if not np.isclose(float(np.mean(bounded)), 1.0, atol=1e-10):
                raise RuntimeError("Could not project importance weights to bounded mean one")
            output[spec_id] = bounded
        return output

    def get_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "geroprotector.v5_importance.v1",
            "selected_specs": list(self.selected_specs),
            "stability_spearman": dict(self.stability_spearman),
            "stability_diagnostics": {
                key: dict(value) for key, value in self.stability_diagnostics.items()
            },
            "fit_ids": list(self.fit_ids),
            "subfold_validation_ids": [list(values) for values in self.subfold_validation_ids],
            "state_hash": self.state_hash,
            "validation_labels_outside_fit_scope_used": False,
        }


def fit_stable_importance(
    bank: V5FeatureBank,
    selected_specs: Sequence[str],
    y: pd.Series,
    groups: pd.Series,
    *,
    config: Mapping[str, Any],
    seed: int,
    checkpoint_root: str | Path | None = None,
    parent_binding_sha256: str | None = None,
) -> StableImportanceState:
    """Fit importance using grouped sub-CV wholly inside ``bank.ids``.

    The caller must pass only the active fit partition. Consequently, the validation
    labels of the caller's enclosing fold cannot influence these weights.
    """

    ids = tuple(bank.ids)
    if set(ids) != set(y.index.astype(str)) or set(ids) != set(groups.index.astype(str)):
        raise ValueError("Importance bank/labels/groups are misaligned")
    settings = config["weighting"]
    folds = grouped_inner_folds(
        y,
        groups,
        n_splits=int(settings["grouped_subcv_folds"]),
        seed=int(seed),
    )
    base: dict[str, np.ndarray] = {}
    stability: dict[str, float] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    checkpoint_directory = None if checkpoint_root is None else Path(checkpoint_root)
    if checkpoint_directory is not None and not parent_binding_sha256:
        raise ValueError("Stage-B checkpoints require a parent dependency binding")
    for spec_index, spec_id in enumerate(selected_specs):
        vectors: list[np.ndarray] = []
        for fold_index, (train_ids, validation_ids) in enumerate(folds):
            X_train = bank.matrix(spec_id, train_ids)
            X_valid = bank.matrix(spec_id, validation_ids)
            y_train = y.loc[list(train_ids)].to_numpy(dtype=int)
            y_valid = y.loc[list(validation_ids)].to_numpy(dtype=int)
            local_seed = int(seed + spec_index * 1009 + fold_index * 31)
            vector_binding = canonical_sha256(
                {
                    "schema": "geroprotector.v5_stage_b_vector_dependency.v1",
                    "parent_binding_sha256": parent_binding_sha256,
                    "spec_id": str(spec_id),
                    "spec_index": int(spec_index),
                    "fold_index": int(fold_index),
                    "train_ids": sorted(map(str, train_ids)),
                    "validation_ids": sorted(map(str, validation_ids)),
                    "local_seed": local_seed,
                    "n_features": int(X_train.shape[1]),
                    "permutation_repeats": int(settings["permutation_repeats"]),
                    "permutation_n_jobs": int(settings["permutation_n_jobs"]),
                    "importance_estimator_n_jobs": int(settings["importance_estimator_n_jobs"]),
                    "estimator_contract": "et400_perm_ap_xgb300_gain.v1",
                }
            )
            cached = (
                None
                if checkpoint_directory is None
                else _load_checkpoint(checkpoint_directory, vector_binding)
            )
            if cached is None:
                forest = ExtraTreesClassifier(
                    n_estimators=400,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced",
                    n_jobs=int(settings["importance_estimator_n_jobs"]),
                    random_state=local_seed,
                )
                forest.fit(X_train, y_train)
                permutation = permutation_importance(
                    forest,
                    X_valid,
                    y_valid,
                    scoring="average_precision",
                    n_repeats=int(settings["permutation_repeats"]),
                    random_state=local_seed,
                    n_jobs=int(settings["permutation_n_jobs"]),
                )
                forest_vector = _normalise_importance(permutation.importances_mean)
                tree = XGBClassifier(
                    n_estimators=300,
                    max_depth=2,
                    learning_rate=0.03,
                    min_child_weight=2,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    reg_alpha=0.5,
                    reg_lambda=10.0,
                    eval_metric="logloss",
                    n_jobs=int(settings["importance_estimator_n_jobs"]),
                    random_state=local_seed,
                )
                tree.fit(X_train, y_train)
                xgboost_vector = _normalise_importance(tree.feature_importances_)
                cached = (
                    (forest_vector, xgboost_vector)
                    if checkpoint_directory is None
                    else _publish_checkpoint(
                        checkpoint_directory,
                        vector_binding,
                        forest_vector,
                        xgboost_vector,
                    )
                )
            vectors.extend(cached)
        matrix = np.vstack(vectors)
        median = np.median(matrix, axis=0)
        base[spec_id] = _normalise_importance(median)
        correlations: list[float] = []
        for left in range(len(matrix)):
            for right in range(left):
                value = float(spearmanr(matrix[left], matrix[right]).statistic)
                if np.isfinite(value):
                    correlations.append(value)
        stability[spec_id] = float(np.median(correlations)) if correlations else 0.0
        top_count = max(1, int(np.ceil(matrix.shape[1] * 0.05)))
        top_frequency = np.zeros(matrix.shape[1], dtype=float)
        for vector in matrix:
            top_frequency[np.argsort(vector, kind="stable")[-top_count:]] += 1.0
        top_frequency /= len(matrix)
        rng = np.random.default_rng(seed + spec_index * 7919)
        bootstrap_medians = np.vstack(
            [
                np.median(matrix[rng.integers(0, len(matrix), size=len(matrix))], axis=0)
                for _ in range(200)
            ]
        )
        ci_width = np.quantile(bootstrap_medians, 0.975, axis=0) - np.quantile(
            bootstrap_medians, 0.025, axis=0
        )
        diagnostics[spec_id] = {
            "top_5pct_selection_frequency_mean": float(top_frequency.mean()),
            "top_5pct_selection_frequency_max": float(top_frequency.max()),
            "bootstrap_importance_ci_width_median": float(np.median(ci_width)),
            "n_importance_vectors": float(len(matrix)),
            "bootstrap_resamples": 200.0,
        }
    payload = {
        "selected_specs": list(selected_specs),
        "fit_ids": sorted(ids),
        "importance_sha256": {
            key: hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()
            for key, value in base.items()
        },
        "seed": int(seed),
    }
    state_hash = hashlib.sha256(repr(sorted(payload.items())).encode()).hexdigest()
    return StableImportanceState(
        selected_specs=tuple(selected_specs),
        base_importance=base,
        stability_spearman=stability,
        fit_ids=ids,
        subfold_validation_ids=tuple(tuple(valid) for _, valid in folds),
        state_hash=state_hash,
        stability_diagnostics=diagnostics,
    )
