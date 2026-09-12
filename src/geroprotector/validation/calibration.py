"""Cross-fitted, lineage-checked monotone probability calibration."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from ..hashing import sha256_bytes, sha256_file


class CalibrationLeakageError(ValueError):
    """Raised when purported calibration OOF predictions are not truly held out."""


def _logit(probability: np.ndarray, epsilon: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def validate_oof_lineage(
    *,
    fit_ids: Sequence[str],
    fold_ids: Sequence[str],
    training_ids_by_fold: Mapping[str, Sequence[str]],
) -> None:
    ids = tuple(map(str, fit_ids))
    folds = tuple(map(str, fold_ids))
    if len(ids) != len(folds) or len(ids) != len(set(ids)):
        raise CalibrationLeakageError("Calibration OOF IDs/folds are misaligned or duplicated")
    if len(set(folds)) < 2:
        raise CalibrationLeakageError(
            "Calibration requires OOF predictions from at least 2 folds"
        )
    if set(folds) - set(map(str, training_ids_by_fold)):
        raise CalibrationLeakageError("Calibration fold training lineage is incomplete")
    for fold in sorted(set(folds)):
        predicted = {
            compound for compound, assigned in zip(ids, folds, strict=True) if assigned == fold
        }
        trained = {str(value) for value in training_ids_by_fold[fold]}
        overlap = predicted & trained
        if overlap:
            raise CalibrationLeakageError(
                f"Calibration fold {fold} predicts fitted IDs: {sorted(overlap)[:5]}"
            )
        expected_training = set(ids) - predicted
        if trained != expected_training:
            raise CalibrationLeakageError(
                f"Calibration fold {fold} training IDs are not the exact OOF complement"
            )


@dataclass
class PlattCalibrator:
    """Regularized monotone Platt map with a rank-preserving fallback."""

    epsilon: float = 1e-6
    regularization_c: float = 1.0
    minimum_slope: float = 1e-6
    intercept_: float | None = None
    slope_: float | None = None
    fit_mode_: str | None = None
    fallback_reason_: str | None = None
    fit_ids: tuple[str, ...] = ()
    fold_ids: tuple[str, ...] = ()

    @staticmethod
    def _prevalence_intercept(logits: np.ndarray, prevalence: float) -> float:
        lower = -64.0 - float(np.max(logits))
        upper = 64.0 - float(np.min(logits))

        def error(intercept: float) -> float:
            return float(np.mean(expit(intercept + logits)) - prevalence)

        return float(brentq(error, lower, upper, xtol=1e-12, rtol=1e-12))

    def fit(
        self,
        raw_probability: np.ndarray,
        y: np.ndarray,
        *,
        fit_ids: Sequence[str],
        fold_ids: Sequence[str],
        training_ids_by_fold: Mapping[str, Sequence[str]],
    ) -> PlattCalibrator:
        probability = np.asarray(raw_probability, dtype=float)
        labels = np.asarray(y, dtype=int)
        ids = tuple(map(str, fit_ids))
        folds = tuple(map(str, fold_ids))
        if len(probability) != len(labels) or len(labels) != len(ids):
            raise CalibrationLeakageError("Calibration arrays/IDs are misaligned")
        if set(labels) != {0, 1}:
            raise CalibrationLeakageError("Calibration requires two classes")
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise CalibrationLeakageError("Calibration probabilities must lie in [0, 1]")
        validate_oof_lineage(
            fit_ids=ids,
            fold_ids=folds,
            training_ids_by_fold=training_ids_by_fold,
        )
        logits = _logit(probability, self.epsilon)
        prevalence = float(labels.mean())
        if np.ptp(logits) <= np.finfo(float).eps:
            self.intercept_ = float(np.log(prevalence / (1.0 - prevalence)))
            self.slope_ = 0.0
            self.fit_mode_ = "constant_input_prevalence"
            self.fallback_reason_ = "constant_raw_probabilities"
        else:
            estimator = LogisticRegression(
                C=float(self.regularization_c),
                solver="lbfgs",
                max_iter=2000,
                random_state=0,
            ).fit(logits.reshape(-1, 1), labels)
            intercept = float(estimator.intercept_[0])
            slope = float(estimator.coef_[0, 0])
            if np.isfinite([intercept, slope]).all() and slope >= self.minimum_slope:
                self.intercept_ = intercept
                self.slope_ = slope
                self.fit_mode_ = "regularized_monotone_platt"
                self.fallback_reason_ = None
            else:
                self.intercept_ = self._prevalence_intercept(logits, prevalence)
                self.slope_ = 1.0
                self.fit_mode_ = "rank_preserving_intercept_only_fallback"
                self.fallback_reason_ = (
                    "nonfinite_fit"
                    if not np.isfinite([intercept, slope]).all()
                    else "slope_below_minimum"
                )
        self.fit_ids = ids
        self.fold_ids = folds
        return self

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        if self.intercept_ is None or self.slope_ is None:
            raise RuntimeError("Calibrator has not been fitted")
        probability = np.asarray(raw_probability, dtype=float)
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("Calibration inputs must lie in [0, 1]")
        return expit(self.intercept_ + self.slope_ * _logit(probability, self.epsilon))

    def get_manifest(self) -> dict:
        if self.intercept_ is None or self.slope_ is None:
            raise RuntimeError("Calibrator has not been fitted")
        return {
            "kind": "regularized_monotone_platt_on_full_pipeline_crossfit_oof",
            "epsilon": float(self.epsilon),
            "regularization_c": float(self.regularization_c),
            "minimum_slope": float(self.minimum_slope),
            "intercept": float(self.intercept_),
            "slope": float(self.slope_),
            "fit_mode": self.fit_mode_,
            "fallback_reason": self.fallback_reason_,
            "fit_ids": list(self.fit_ids),
            "fold_ids": list(self.fold_ids),
            "lineage_checked": True,
            "monotone_nondecreasing": True,
        }

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, object]) -> PlattCalibrator:
        if (
            manifest.get("kind") != "regularized_monotone_platt_on_full_pipeline_crossfit_oof"
            or manifest.get("lineage_checked") is not True
            or manifest.get("monotone_nondecreasing") is not True
        ):
            raise ValueError("Platt calibration manifest lacks the locked lineage contract")
        calibrator = cls(
            epsilon=float(manifest["epsilon"]),
            regularization_c=float(manifest["regularization_c"]),
            minimum_slope=float(manifest["minimum_slope"]),
        )
        calibrator.intercept_ = float(manifest["intercept"])
        calibrator.slope_ = float(manifest["slope"])
        calibrator.fit_mode_ = str(manifest["fit_mode"])
        fallback = manifest.get("fallback_reason")
        calibrator.fallback_reason_ = None if fallback is None else str(fallback)
        calibrator.fit_ids = tuple(map(str, manifest["fit_ids"]))
        calibrator.fold_ids = tuple(map(str, manifest["fold_ids"]))
        if (
            not np.isfinite(
                [
                    calibrator.epsilon,
                    calibrator.regularization_c,
                    calibrator.minimum_slope,
                    calibrator.intercept_,
                    calibrator.slope_,
                ]
            ).all()
            or not 0 < calibrator.epsilon < 0.5
            or calibrator.regularization_c <= 0
            or calibrator.minimum_slope < 0
            or calibrator.slope_ < 0
            or len(calibrator.fit_ids) != len(calibrator.fold_ids)
            or len(calibrator.fit_ids) != len(set(calibrator.fit_ids))
            or len(set(calibrator.fold_ids)) < 2
        ):
            raise ValueError("Platt calibration manifest contains invalid fitted state")
        valid_mode = (
            (
                calibrator.fit_mode_ == "constant_input_prevalence"
                and calibrator.slope_ == 0.0
                and calibrator.fallback_reason_ == "constant_raw_probabilities"
            )
            or (
                calibrator.fit_mode_ == "regularized_monotone_platt"
                and calibrator.slope_ >= calibrator.minimum_slope
                and calibrator.fallback_reason_ is None
            )
            or (
                calibrator.fit_mode_ == "rank_preserving_intercept_only_fallback"
                and calibrator.slope_ == 1.0
                and calibrator.fallback_reason_ in {"nonfinite_fit", "slope_below_minimum"}
            )
        )
        if not valid_mode:
            raise ValueError("Platt calibration fit mode is inconsistent with its state")
        return calibrator


@dataclass
class BetaCalibrator:
    """Prespecified monotone beta-calibration sensitivity on full-pipeline OOF."""

    epsilon: float = 1e-6
    l2_penalty: float = 1e-3
    a_: float | None = None
    b_: float | None = None
    intercept_: float | None = None
    fit_ids: tuple[str, ...] = ()
    fold_ids: tuple[str, ...] = ()

    def fit(
        self,
        raw_probability: np.ndarray,
        y: np.ndarray,
        *,
        fit_ids: Sequence[str],
        fold_ids: Sequence[str],
        training_ids_by_fold: Mapping[str, Sequence[str]],
    ) -> BetaCalibrator:
        probability = np.asarray(raw_probability, dtype=float)
        labels = np.asarray(y, dtype=int)
        ids = tuple(map(str, fit_ids))
        folds = tuple(map(str, fold_ids))
        if len(probability) != len(labels) or len(labels) != len(ids):
            raise CalibrationLeakageError("Beta calibration arrays/IDs are misaligned")
        if set(labels) != {0, 1}:
            raise CalibrationLeakageError("Beta calibration requires two classes")
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise CalibrationLeakageError("Beta probabilities must lie in [0, 1]")
        validate_oof_lineage(
            fit_ids=ids,
            fold_ids=folds,
            training_ids_by_fold=training_ids_by_fold,
        )
        clipped = np.clip(probability, self.epsilon, 1.0 - self.epsilon)
        log_p = np.log(clipped)
        log_one_minus = np.log1p(-clipped)

        def objective(parameters: np.ndarray) -> float:
            a, b, intercept = map(float, parameters)
            linear = intercept + a * log_p - b * log_one_minus
            # Stable binary cross entropy plus a small prespecified slope penalty.
            loss = np.logaddexp(0.0, linear) - labels * linear
            return float(np.mean(loss) + self.l2_penalty * (a * a + b * b))

        prevalence = float(labels.mean())
        initial = np.asarray([1.0, 1.0, np.log(prevalence / (1.0 - prevalence))], dtype=float)
        fitted = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=((0.0, None), (0.0, None), (None, None)),
        )
        if not fitted.success or not np.isfinite(fitted.x).all():
            raise RuntimeError(f"Beta calibration optimization failed: {fitted.message}")
        self.a_, self.b_, self.intercept_ = map(float, fitted.x)
        self.fit_ids = ids
        self.fold_ids = folds
        return self

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        if self.a_ is None or self.b_ is None or self.intercept_ is None:
            raise RuntimeError("Beta calibrator has not been fitted")
        probability = np.asarray(raw_probability, dtype=float)
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("Beta calibration inputs must lie in [0, 1]")
        clipped = np.clip(probability, self.epsilon, 1.0 - self.epsilon)
        linear = self.intercept_ + self.a_ * np.log(clipped) - self.b_ * np.log1p(-clipped)
        return expit(linear)

    def get_manifest(self) -> dict:
        if self.a_ is None or self.b_ is None or self.intercept_ is None:
            raise RuntimeError("Beta calibrator has not been fitted")
        return {
            "kind": "monotone_beta_sensitivity_on_full_pipeline_crossfit_oof",
            "epsilon": float(self.epsilon),
            "l2_penalty": float(self.l2_penalty),
            "a": self.a_,
            "b": self.b_,
            "intercept": self.intercept_,
            "fit_ids": list(self.fit_ids),
            "fold_ids": list(self.fold_ids),
            "lineage_checked": True,
            "used_for_selection": False,
        }

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, object]) -> BetaCalibrator:
        if (
            manifest.get("kind") != "monotone_beta_sensitivity_on_full_pipeline_crossfit_oof"
            or manifest.get("lineage_checked") is not True
            or manifest.get("used_for_selection") is not False
        ):
            raise ValueError("Beta calibration manifest lacks the locked sensitivity role")
        calibrator = cls(
            epsilon=float(manifest["epsilon"]),
            l2_penalty=float(manifest["l2_penalty"]),
        )
        calibrator.a_ = float(manifest["a"])
        calibrator.b_ = float(manifest["b"])
        calibrator.intercept_ = float(manifest["intercept"])
        calibrator.fit_ids = tuple(map(str, manifest["fit_ids"]))
        calibrator.fold_ids = tuple(map(str, manifest["fold_ids"]))
        if (
            not np.isfinite(
                [
                    calibrator.epsilon,
                    calibrator.l2_penalty,
                    calibrator.a_,
                    calibrator.b_,
                    calibrator.intercept_,
                ]
            ).all()
            or not 0 < calibrator.epsilon < 0.5
            or calibrator.l2_penalty < 0
            or calibrator.a_ < 0
            or calibrator.b_ < 0
            or len(calibrator.fit_ids) != len(calibrator.fold_ids)
            or len(calibrator.fit_ids) != len(set(calibrator.fit_ids))
            or len(set(calibrator.fold_ids)) < 2
        ):
            raise ValueError("Beta calibration manifest contains invalid fitted state")
        return calibrator


@dataclass(frozen=True)
class LoadedCalibrationBundle:
    platt: PlattCalibrator
    beta_sensitivity: BetaCalibrator
    threshold: float
    manifest: Mapping[str, object]

    def primary_probability(self, raw_probability: np.ndarray) -> np.ndarray:
        return self.platt.predict(raw_probability)

    def beta_probability(self, raw_probability: np.ndarray) -> np.ndarray:
        return self.beta_sensitivity.predict(raw_probability)

    def decision(self, raw_probability: np.ndarray) -> np.ndarray:
        return (self.primary_probability(raw_probability) >= self.threshold).astype(int)


def load_calibration_bundle(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_model_artifact_sha256: str,
    expected_fit_ids: Sequence[str],
    expected_job_binding: Mapping[str, object],
) -> LoadedCalibrationBundle:
    """Load a calibration only when its sealed model and outer-job lineage agree."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Calibration bundle must be regular/non-symlink: {source}")
    if sha256_file(source) != str(expected_sha256):
        raise ValueError("Calibration bundle byte hash differs from sealed job inventory")
    record = json.loads(source.read_text(encoding="utf-8"))
    if record.get("schema_version") != "geroprotector.calibration_bundle.v2":
        raise ValueError("Unsupported calibration bundle schema")
    binding = record.get("binding")
    required_binding_keys = {
        "run_id",
        "pipeline_id",
        "model_id",
        "repeat",
        "outer_fold",
        "fit_ids_sha256",
        "outer_test_ids_sha256",
        "model_artifact_sha256",
    }
    if not isinstance(binding, Mapping) or set(binding) != required_binding_keys:
        raise ValueError("Calibration bundle job binding schema is invalid")
    if binding.get("model_artifact_sha256") != str(expected_model_artifact_sha256):
        raise ValueError("Calibration bundle is bound to a different model artifact")
    required_expected_keys = {
        "run_id",
        "pipeline_id",
        "model_id",
        "repeat",
        "outer_fold",
        "outer_test_ids_sha256",
    }
    if set(expected_job_binding) != required_expected_keys:
        raise ValueError("Expected calibration job binding schema is invalid")
    actual_job_binding = {key: binding[key] for key in required_expected_keys}
    if actual_job_binding != dict(expected_job_binding):
        raise ValueError("Calibration bundle is bound to a different outer job")
    threshold_record = record.get("threshold")
    if not isinstance(threshold_record, Mapping) or (
        threshold_record.get("objective") != "mcc"
        or threshold_record.get("used_for_primary_model_selection") is not False
    ):
        raise ValueError("Calibration threshold contract is invalid")
    threshold = float(threshold_record["value"])
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Calibration threshold lies outside [0, 1]")
    platt = PlattCalibrator.from_manifest(record["platt"])
    beta = BetaCalibrator.from_manifest(record["beta_sensitivity"])
    expected_ids = tuple(map(str, expected_fit_ids))
    expected_ids_sha256 = sha256_bytes(("\n".join(sorted(expected_ids)) + "\n").encode())
    if (
        not expected_ids
        or len(expected_ids) != len(set(expected_ids))
        or binding.get("fit_ids_sha256") != expected_ids_sha256
        or platt.fit_ids != expected_ids
        or beta.fit_ids != expected_ids
        or platt.fold_ids != beta.fold_ids
    ):
        raise ValueError("Calibration fit IDs/folds differ from the loaded model/job")
    return LoadedCalibrationBundle(
        platt=platt,
        beta_sensitivity=beta,
        threshold=threshold,
        manifest=record,
    )
