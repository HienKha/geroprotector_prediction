"""Explicit-checkpoint TabPFN adapter with canonical singleton inference."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...hashing import sha256_file
from .checkpoints import _project_file


class TabPFNAdapter:
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
        self.device = str(device)

    def _checkpoint(self) -> Path:
        path = _project_file(
            self.project_root,
            self.record["checkpoint_path"],
            role="TabPFN checkpoint",
        )
        if sha256_file(path) != self.record["checkpoint_sha256"]:
            raise ValueError("TabPFN checkpoint changed after staging")
        return path

    def fit_context(self, X_train: pd.DataFrame, y_train: np.ndarray) -> TabPFNAdapter:
        if os.environ.get("TABPFN_DISABLE_TELEMETRY") != "1":
            raise RuntimeError("Set TABPFN_DISABLE_TELEMETRY=1 before any TabPFN import")
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        if not isinstance(X_train, pd.DataFrame) or X_train.index.has_duplicates:
            raise ValueError("TabPFN context requires an ID-indexed DataFrame")
        matrix = X_train.to_numpy(dtype=float)
        labels = np.asarray(y_train, dtype=int)
        if len(labels) != len(X_train) or set(labels) != {0, 1}:
            raise ValueError("TabPFN context requires aligned two-class labels")
        if not np.isfinite(matrix).all():
            raise ValueError("TabPFN context contains non-finite values")
        version_name = str(self.record["explicit_model_version"])
        try:
            version = ModelVersion[version_name]
        except KeyError as exc:
            raise ValueError(f"Installed TabPFN lacks model version {version_name}") from exc
        checkpoint = self._checkpoint()
        self.estimator_ = TabPFNClassifier.create_default_for_version(
            version,
            model_path=str(checkpoint),
            n_estimators=self.n_estimators,
            auto_scale_n_estimators=False,
            random_state=self.random_state,
            device=self.device,
            show_progress_bar=False,
        )
        if hasattr(self.estimator_, "get_params"):
            effective = self.estimator_.get_params(deep=False)
            required = {
                "n_estimators": self.n_estimators,
                "auto_scale_n_estimators": False,
                "random_state": self.random_state,
                "device": self.device,
                "model_path": str(checkpoint),
                "show_progress_bar": False,
            }
            for key, expected in required.items():
                if str(effective.get(key)) != str(expected):
                    raise RuntimeError(f"TabPFN did not retain requested parameter {key}")
            self.effective_constructor_parameters_ = required
        self.estimator_.fit(matrix, labels)
        effective_n_estimators = getattr(self.estimator_, "n_estimators_", None)
        if effective_n_estimators is not None and int(effective_n_estimators) != int(
            self.n_estimators
        ):
            raise RuntimeError("TabPFN changed the locked ensemble size during fit")
        if tuple(map(int, self.estimator_.classes_)) != (0, 1):
            raise ValueError("TabPFN class order is not [0, 1]")
        self.feature_names_ = tuple(map(str, X_train.columns))
        self.context_ids_ = tuple(map(str, X_train.index))
        return self

    def _validate_query(self, X_query: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("TabPFN context is not fitted")
        if tuple(map(str, X_query.columns)) != self.feature_names_:
            raise ValueError("TabPFN query feature schema/order differs from context")
        if X_query.index.has_duplicates:
            raise ValueError("TabPFN query IDs are duplicated")
        if set(map(str, X_query.index)) & set(self.context_ids_):
            raise ValueError("TabPFN context/query IDs overlap")
        matrix = X_query.to_numpy(dtype=float)
        if len(matrix) == 0:
            raise ValueError("TabPFN query cannot be empty")
        if not np.isfinite(matrix).all():
            raise ValueError("TabPFN query contains non-finite values")
        return matrix

    def predict_proba(self, X_query: pd.DataFrame, *, canonical_mode: bool) -> np.ndarray:
        matrix = self._validate_query(X_query)
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
            raise ValueError("TabPFN returned invalid probabilities")
        return output

    def save_manifest(self) -> dict[str, Any]:
        return {
            **self.record,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "device": self.device,
            "auto_scale_n_estimators": False,
            "effective_constructor_parameters": {
                key: str(value)
                for key, value in getattr(self, "effective_constructor_parameters_", {}).items()
            },
            "context_ids": list(getattr(self, "context_ids_", ())),
            "feature_names": list(getattr(self, "feature_names_", ())),
            "canonical_inference": "one_query_per_call",
            "implicit_downloads_allowed": False,
        }
