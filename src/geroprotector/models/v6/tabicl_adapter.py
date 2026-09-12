"""Fail-closed adapter for explicitly staged TabICL checkpoints."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...hashing import sha256_file
from .checkpoints import _project_file


class TabICLAdapter:
    def __init__(
        self,
        record: Mapping[str, Any],
        *,
        project_root: str | Path,
        n_estimators: int,
        random_state: int,
        device: str = "auto",
    ) -> None:
        self.record = dict(record)
        self.project_root = Path(project_root).resolve()
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.device = device

    def _classifier(self):
        try:
            from tabicl import TabICLClassifier
        except Exception as exc:
            raise RuntimeError("Pinned tabicl package/TabICLClassifier is unavailable") from exc
        signature = inspect.signature(TabICLClassifier)
        required_parameters = {
            "model_path",
            "allow_auto_download",
            "random_state",
            "n_estimators",
            "device",
        }
        missing = required_parameters - set(signature.parameters)
        if missing:
            raise RuntimeError(
                "Pinned TabICL API lacks required reproducibility parameters: "
                f"{sorted(missing)}"
            )
        path = _project_file(
            self.project_root,
            self.record["checkpoint_path"],
            role="TabICL checkpoint",
        )
        if sha256_file(path) != self.record["checkpoint_sha256"]:
            raise RuntimeError("TabICL checkpoint integrity failed")
        kwargs: dict[str, Any] = {
            "model_path": str(path),
            "allow_auto_download": False,
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
            "device": self.device,
        }
        estimator = TabICLClassifier(**kwargs)
        if hasattr(estimator, "get_params"):
            effective = estimator.get_params(deep=False)
            for key, expected in kwargs.items():
                if str(effective.get(key)) != str(expected):
                    raise RuntimeError(f"TabICL did not retain requested parameter {key}")
        self.effective_constructor_parameters_ = kwargs
        return estimator

    def fit_context(self, X_train: pd.DataFrame, y_train: np.ndarray) -> TabICLAdapter:
        if not isinstance(X_train, pd.DataFrame) or X_train.index.has_duplicates:
            raise ValueError("TabICL context requires an ID-indexed DataFrame")
        matrix = X_train.to_numpy(dtype=float)
        labels = np.asarray(y_train, dtype=int)
        if len(labels) != len(X_train) or set(labels) != {0, 1}:
            raise ValueError("TabICL context requires aligned two-class labels")
        if not np.isfinite(matrix).all():
            raise ValueError("TabICL context contains non-finite values")
        self.estimator_ = self._classifier()
        self.estimator_.fit(matrix, labels)
        if tuple(map(int, self.estimator_.classes_)) != (0, 1):
            raise ValueError("TabICL class order is not [0, 1]")
        self.feature_names_ = tuple(map(str, X_train.columns))
        self.context_ids_ = tuple(map(str, X_train.index))
        return self

    def predict_proba(self, X_query: pd.DataFrame, *, canonical_mode: bool) -> np.ndarray:
        if tuple(map(str, X_query.columns)) != self.feature_names_:
            raise ValueError("TabICL query feature schema/order differs from context")
        if X_query.index.has_duplicates:
            raise ValueError("TabICL query IDs are duplicated")
        if set(map(str, X_query.index)) & set(self.context_ids_):
            raise ValueError("TabICL context/query IDs overlap")
        matrix = X_query.to_numpy(dtype=float)
        if len(matrix) == 0:
            raise ValueError("TabICL query cannot be empty")
        if not np.isfinite(matrix).all():
            raise ValueError("TabICL query contains non-finite values")
        if canonical_mode:
            row_keys = [
                (np.asarray(matrix[index], dtype="<f8").tobytes(), str(X_query.index[index]))
                for index in range(len(matrix))
            ]
            order = sorted(range(len(matrix)), key=row_keys.__getitem__)
            output = np.empty((len(matrix), 2), dtype=float)
            for index in order:
                output[index] = np.asarray(
                    self.estimator_.predict_proba(matrix[index : index + 1])[0], dtype=float
                )
        else:
            output = np.asarray(self.estimator_.predict_proba(matrix), dtype=float)
        if (
            output.shape != (len(matrix), 2)
            or not np.isfinite(output).all()
            or np.any((output < 0) | (output > 1))
            or not np.allclose(output.sum(axis=1), 1.0, atol=1e-6)
        ):
            raise ValueError("TabICL returned invalid probabilities")
        return output

    def save_manifest(self) -> dict[str, Any]:
        return {
            **self.record,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "device": self.device,
            "effective_constructor_parameters": {
                key: str(value)
                for key, value in getattr(self, "effective_constructor_parameters_", {}).items()
            },
            "context_ids": list(getattr(self, "context_ids_", ())),
            "feature_names": list(getattr(self, "feature_names_", ())),
            "canonical_inference": "one_query_per_call",
            "implicit_downloads_allowed": False,
        }
