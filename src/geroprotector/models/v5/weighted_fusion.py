"""Fold-local weighted fingerprint fusion and descriptor preprocessing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from .fingerprint_bank import V5FeatureBank


def weighted_concatenation(
    bank: V5FeatureBank,
    ids: Sequence[str],
    selected_specs: Sequence[str],
    weights: Mapping[str, np.ndarray],
) -> np.ndarray:
    blocks = []
    for spec_id in selected_specs:
        matrix = bank.matrix(spec_id, ids)
        weight = np.asarray(weights[spec_id], dtype=float)
        if (
            matrix.shape[1] != len(weight)
            or np.any(weight <= 0)
            or not np.isfinite(weight).all()
        ):
            raise ValueError(f"Invalid V5 weight vector: {spec_id}")
        blocks.append(matrix * np.sqrt(weight)[None, :])
    return np.concatenate(blocks, axis=1).astype(np.float64, copy=False)


@dataclass
class DescriptorPreprocessor:
    imputer: SimpleImputer | None = None
    scaler: RobustScaler | None = None
    fit_ids: tuple[str, ...] = ()
    keep_columns: np.ndarray | None = None

    def fit(self, matrix: np.ndarray, *, fit_ids: Sequence[str]) -> DescriptorPreprocessor:
        X = np.asarray(matrix, dtype=float)
        if X.shape[0] != len(fit_ids) or len(set(fit_ids)) != len(fit_ids):
            raise ValueError("Descriptor fit IDs are invalid")
        finite_support = np.isfinite(X).any(axis=0)
        if not finite_support.any():
            raise ValueError("All descriptor columns are non-finite")
        self.keep_columns = finite_support
        self.imputer = SimpleImputer(strategy="median").fit(X[:, finite_support])
        imputed = self.imputer.transform(X[:, finite_support])
        self.scaler = RobustScaler(quantile_range=(25.0, 75.0)).fit(imputed)
        self.fit_ids = tuple(map(str, fit_ids))
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if self.imputer is None or self.scaler is None or self.keep_columns is None:
            raise RuntimeError("Descriptor preprocessor is not fitted")
        output = self.scaler.transform(
            self.imputer.transform(np.asarray(matrix)[:, self.keep_columns])
        )
        if not np.isfinite(output).all():
            raise ValueError("Descriptor preprocessing produced non-finite values")
        return np.asarray(output, dtype=np.float64)

    def get_manifest(self) -> dict:
        if self.keep_columns is None:
            raise RuntimeError("Descriptor preprocessor is not fitted")
        return {
            "kind": "median_impute_robust_scale",
            "fit_ids": list(self.fit_ids),
            "kept_columns": np.flatnonzero(self.keep_columns).tolist(),
        }


@dataclass
class WeightedLinearSVD:
    n_components: int
    random_state: int
    estimator: TruncatedSVD | None = None
    fit_ids: tuple[str, ...] = ()

    def fit(self, matrix: np.ndarray, *, fit_ids: Sequence[str]) -> WeightedLinearSVD:
        X = np.asarray(matrix, dtype=np.float64)
        if X.shape[0] != len(fit_ids) or len(set(fit_ids)) != len(fit_ids):
            raise ValueError("SVD fit IDs are invalid")
        effective = min(int(self.n_components), max(1, min(X.shape) - 1))
        self.estimator = TruncatedSVD(n_components=effective, random_state=self.random_state)
        self.estimator.fit(X)
        self.fit_ids = tuple(map(str, fit_ids))
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("SVD transformer is not fitted")
        output = self.estimator.transform(np.asarray(matrix, dtype=np.float64))
        if not np.isfinite(output).all():
            raise ValueError("SVD transform produced non-finite values")
        return output

    def get_manifest(self) -> dict:
        if self.estimator is None:
            raise RuntimeError("SVD transformer is not fitted")
        return {
            "kind": "weighted_linear_truncated_svd",
            "requested_components": int(self.n_components),
            "effective_components": int(self.estimator.n_components),
            "fit_ids": list(self.fit_ids),
        }


@dataclass
class WeightedIdentityTransform:
    """No-reduction ablation; it still records the exact training-only fit scope."""

    n_features_: int | None = None
    fit_ids: tuple[str, ...] = ()

    def fit(self, matrix: np.ndarray, *, fit_ids: Sequence[str]) -> WeightedIdentityTransform:
        X = np.asarray(matrix, dtype=np.float64)
        if X.shape[0] != len(fit_ids) or len(set(fit_ids)) != len(fit_ids):
            raise ValueError("Identity-transform fit IDs are invalid")
        if not np.isfinite(X).all():
            raise ValueError("Identity-transform input is non-finite")
        self.n_features_ = int(X.shape[1])
        self.fit_ids = tuple(map(str, fit_ids))
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if self.n_features_ is None:
            raise RuntimeError("Identity transformer is not fitted")
        X = np.asarray(matrix, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_ or not np.isfinite(X).all():
            raise ValueError("Identity-transform feature contract changed")
        return X

    def get_manifest(self) -> dict:
        if self.n_features_ is None:
            raise RuntimeError("Identity transformer is not fitted")
        return {
            "kind": "weighted_no_dimensionality_reduction_ablation",
            "effective_components": self.n_features_,
            "fit_ids": list(self.fit_ids),
        }
