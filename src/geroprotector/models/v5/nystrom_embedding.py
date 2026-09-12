"""Inductive weighted RBF Nyström map with training-only landmarks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(np.square(left), axis=1)[:, None]
    right_norm = np.sum(np.square(right), axis=1)[None, :]
    return np.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0)


@dataclass
class WeightedRBFNystrom:
    n_components: int
    gamma_multiplier: float
    random_state: int
    landmarks_: np.ndarray | None = None
    normalizer_: np.ndarray | None = None
    gamma_: float | None = None
    landmark_ids_: tuple[str, ...] = ()
    fit_ids_: tuple[str, ...] = ()

    def fit(self, X: np.ndarray, *, fit_ids: Sequence[str]) -> WeightedRBFNystrom:
        matrix = np.asarray(X, dtype=np.float64)
        ids = tuple(map(str, fit_ids))
        if matrix.shape[0] != len(ids) or len(ids) != len(set(ids)):
            raise ValueError("Nyström fit IDs are invalid")
        if not np.isfinite(matrix).all():
            raise ValueError("Nyström requires finite training features")
        distances = _squared_distances(matrix, matrix)
        positive = distances[np.triu_indices(len(matrix), k=1)]
        positive = positive[positive > 0]
        if len(positive) == 0:
            raise ValueError("Nyström cannot derive gamma from all-identical training rows")
        self.gamma_ = float(self.gamma_multiplier) / float(np.median(positive))
        count = min(int(self.n_components), len(matrix))
        rng = np.random.default_rng(int(self.random_state))
        indices = np.sort(rng.choice(len(matrix), size=count, replace=False))
        landmarks = matrix[indices]
        kernel = np.exp(-self.gamma_ * _squared_distances(landmarks, landmarks))
        eigenvalues, eigenvectors = np.linalg.eigh(kernel)
        tolerance = max(float(eigenvalues.max()) * 1e-12, 1e-12)
        inverse_sqrt = np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, tolerance)))
        self.normalizer_ = eigenvectors @ inverse_sqrt @ eigenvectors.T
        self.landmarks_ = landmarks
        self.landmark_ids_ = tuple(ids[int(index)] for index in indices)
        self.fit_ids_ = ids
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.landmarks_ is None or self.normalizer_ is None or self.gamma_ is None:
            raise RuntimeError("Nyström transformer is not fitted")
        matrix = np.asarray(X, dtype=np.float64)
        kernel = np.exp(-self.gamma_ * _squared_distances(matrix, self.landmarks_))
        output = kernel @ self.normalizer_
        if not np.isfinite(output).all():
            raise ValueError("Nyström transform produced non-finite values")
        return output

    def get_manifest(self) -> dict:
        if self.gamma_ is None:
            raise RuntimeError("Nyström transformer is not fitted")
        return {
            "kind": "weighted_rbf_nystrom",
            "n_components": int(self.n_components),
            "effective_components": len(self.landmark_ids_),
            "gamma_multiplier": float(self.gamma_multiplier),
            "gamma": float(self.gamma_),
            "fit_ids": list(self.fit_ids_),
            "landmark_ids": list(self.landmark_ids_),
            "landmarks_subset_of_fit_ids": set(self.landmark_ids_).issubset(self.fit_ids_),
        }
