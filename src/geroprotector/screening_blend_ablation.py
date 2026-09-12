"""Post-lock component ablation and similarity-domain analysis.

The v1 protocol never fits a model and reconstructs every component threshold from
sealed D1-train OOF predictions.  The v2 paper-SVM protocol independently refits the
published linear SVC on the 324 D1 paper-training rows, proves parity with the
already sealed SVM probability stream, and uses the classifier's native decision
boundary.  Neither protocol uses an external outcome to fit or select anything.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from geroprotector.fixed_blend_paper405 import _tanimoto
from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.screening_blend_external import _d1_frame
from geroprotector.screening_blend_external import load_protocol as load_external_protocol
from geroprotector.screening_blend_paper405 import load_locked_bundle
from geroprotector.weighted_blend_paper405 import select_threshold


class ScreeningBlendAblationError(RuntimeError):
    """Raised when a sealed ablation input or analysis invariant differs."""


COMPONENTS = ("paper_svm", "tanimoto_svc", "tabpfn_v2")
MODELS = (*COMPONENTS, "blend_010_060_030")
PROBABILITY_COLUMNS = {
    "paper_svm": "probability_paper_svm",
    "tanimoto_svc": "probability_tanimoto_svc",
    "tabpfn_v2": "probability_tabpfn_v2",
    "blend_010_060_030": "blend_probability",
}
WEIGHTS = np.asarray([0.10, 0.60, 0.30], dtype=np.float64)
BLEND_THRESHOLD = 0.5299579802368826


def _regular_file(path: Path, role: str, expected_sha256: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ScreeningBlendAblationError(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected_sha256 is not None and sha256_file(resolved) != expected_sha256:
        raise ScreeningBlendAblationError(f"{role} SHA256 differs from the lock")
    return resolved


def _read_json(path: Path, role: str = "JSON artifact") -> dict[str, Any]:
    value = json.loads(_regular_file(path, role).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ScreeningBlendAblationError(f"{role} is not a JSON object")
    return value


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite immutable CSV: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    value = yaml.safe_load(_regular_file(path, "ablation protocol").read_text())
    accepted_schemas = {
        "geroprotector.screening_blend_ablation.protocol.v1",
        "geroprotector.screening_blend_ablation.protocol.v2",
    }
    if not isinstance(value, dict) or value.get("schema_version") not in accepted_schemas:
        raise ScreeningBlendAblationError("Unknown ablation protocol schema")
    locked = value.get("locked_model", {})
    if (
        locked.get("components") != list(COMPONENTS)
        or not np.allclose(locked.get("blend_weights"), WEIGHTS, rtol=0.0, atol=1e-15)
        or float(locked.get("blend_oof_mcc_threshold", np.nan)) != BLEND_THRESHOLD
        or float(locked.get("fixed_threshold", np.nan)) != 0.5
    ):
        raise ScreeningBlendAblationError("Model, weights or thresholds differ")
    analysis = value.get("threshold_analysis", {})
    allowed_threshold_sources = {
        "full_324_d1_train_cross_fitted_oof_mcc",
        "paper_svm_native_decision_other_components_d1_train_oof_mcc",
    }
    if analysis.get("component_binary_threshold_source") not in allowed_threshold_sources or (
        analysis.get("external_labels_may_select_or_change_threshold") is not False
    ):
        raise ScreeningBlendAblationError("Threshold leakage contract differs")
    if value["schema_version"].endswith(".v2"):
        paper = value.get("paper_svm_contract", {})
        required = {
            "primary_authority": "official_executable_notebook_and_reported_test_result",
            "features": [
                "Total Molweight",
                "cLogP",
                "H-Acceptors",
                "H-Donors",
                "Total Surface Area",
                "Relative PSA",
                "Rotatable Bonds",
            ],
            "preprocessing": "none_matching_official_notebook_and_reported_accuracy",
            "kernel": "linear",
            "C": 1.0,
            "gamma": 1.0,
            "probability": True,
            "random_state": 42,
            "native_decision_rule": "svc_predict_decision_function_zero",
        }
        for key, expected in required.items():
            if paper.get(key) != expected:
                raise ScreeningBlendAblationError(
                    f"Published paper-SVM contract differs at {key}"
                )
        if paper.get("external_labels_used_for_fit_or_decision_rule") is not False:
            raise ScreeningBlendAblationError("Paper-SVM external leakage contract differs")
    applicability = value.get("applicability", {})
    if (
        float(applicability.get("threshold", np.nan)) != 0.4
        or applicability.get("threshold_selected_without_test_or_external_labels") is not True
        or applicability.get("use_as_exclusion_filter") is not False
    ):
        raise ScreeningBlendAblationError("Applicability contract differs")
    for record in value.get("sealed_inputs", {}).values():
        _regular_file(root / record["path"], "sealed analysis input", record["sha256"])
    return value, canonical_sha256(value)


def _verify_artifact_map(directory: Path, records: dict[str, str]) -> None:
    for relative, expected in records.items():
        path = directory / relative
        _regular_file(path, f"sealed artifact {relative}", expected)


def _verify_inputs(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    records = protocol["sealed_inputs"]
    weighted_completed_path = root / records["weighted_completed"]["path"]
    screening_completed_path = root / records["screening_completed"]["path"]
    external_completed_path = root / records["external_completed"]["path"]
    weighted = _read_json(weighted_completed_path, "weighted completion")
    screening = _read_json(screening_completed_path, "screening completion")
    external = _read_json(external_completed_path, "external completion")
    for value, name in (
        (weighted, "weighted"),
        (screening, "screening"),
        (external, "external"),
    ):
        if value.get("status") != "COMPLETE":
            raise ScreeningBlendAblationError(f"{name} source run is not complete")

    weighted_directory = weighted_completed_path.parent
    weighted_manifest = _read_json(
        weighted_directory / "run_manifest.json", "weighted manifest"
    )
    if sha256_file(weighted_directory / "run_manifest.json") != weighted.get(
        "run_manifest_sha256"
    ):
        raise ScreeningBlendAblationError("Weighted manifest differs from completion")
    _verify_artifact_map(weighted_directory, weighted_manifest["artifact_hashes"])
    if weighted_manifest.get("outer_test_used_for_weight_or_threshold_selection") is not False:
        raise ScreeningBlendAblationError("Weighted source used outer test for selection")

    screening_directory = screening_completed_path.parent
    model_lock_path = screening_directory / "models" / "MODEL_LOCK.json"
    if sha256_file(model_lock_path) != screening.get("model_lock_sha256"):
        raise ScreeningBlendAblationError("Screening model lock differs")
    model_lock = _read_json(model_lock_path, "screening model lock")
    if (
        model_lock.get("component_state_sha256")
        != protocol["locked_model"]["component_state_sha256"]
        or not np.allclose(model_lock.get("weights"), WEIGHTS, rtol=0.0, atol=1e-15)
        or model_lock.get("selection_used_outer_test") is not False
    ):
        raise ScreeningBlendAblationError("Screening model semantics differ")
    bundle_record = next(
        record for record in model_lock["bundles"] if record["bundle_id"].endswith("oof_mcc")
    )
    bundle = load_locked_bundle(
        screening_directory / "models" / bundle_record["path"],
        expected_artifact_sha256=bundle_record["sha256"],
    )

    external_directory = external_completed_path.parent
    for cohort in ("drugage", "agextend"):
        scored = external_directory / cohort / "scored"
        scored_lock = _read_json(scored / "SCORED.json", f"{cohort} scored lock")
        if sha256_file(scored / "SCORED.json") != external[f"{cohort}_scored_sha256"]:
            raise ScreeningBlendAblationError(f"{cohort} scored lock differs")
        _verify_artifact_map(scored, scored_lock["artifact_hashes"])
        predicted = external_directory / cohort / "predicted"
        prediction_lock = _read_json(
            predicted / "PREDICTION_LOCK.json", f"{cohort} prediction lock"
        )
        if (
            prediction_lock.get("model_lock_sha256") != sha256_file(model_lock_path)
            or prediction_lock.get("outcome_endpoint_aggregation_or_metrics_run_before_lock")
            is not False
        ):
            raise ScreeningBlendAblationError(f"{cohort} prediction/model binding differs")
        _regular_file(
            predicted / "predictions.csv",
            f"{cohort} predictions",
            prediction_lock["predictions_sha256"],
        )
        feature_audit = _read_json(predicted / "feature_audit.json")
        if (
            sha256_file(predicted / "feature_audit.json")
            != prediction_lock["feature_audit_sha256"]
            or feature_audit.get("model_component_state_sha256")
            != model_lock["component_state_sha256"]
            or feature_audit.get("external_refit_calibration_or_threshold_selection")
            is not False
        ):
            raise ScreeningBlendAblationError(f"{cohort} component-fit audit differs")
    return {
        "weighted_directory": weighted_directory,
        "screening_directory": screening_directory,
        "external_directory": external_directory,
        "bundle": bundle,
        "model_lock_sha256": sha256_file(model_lock_path),
        "input_completed_sha256": {
            name: sha256_file(root / record["path"])
            for name, record in records.items()
            if name.endswith("completed")
        },
    }


def _add_blend(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    component = output[[PROBABILITY_COLUMNS[name] for name in COMPONENTS]].to_numpy(
        dtype=float
    )
    if not np.isfinite(component).all() or ((component < 0) | (component > 1)).any():
        raise ScreeningBlendAblationError("Component probabilities are invalid")
    calculated = component @ WEIGHTS
    if "blend_probability" in output and not np.allclose(
        output["blend_probability"], calculated, rtol=0.0, atol=1e-12
    ):
        raise ScreeningBlendAblationError("Stored blend differs from locked components")
    output["blend_probability"] = calculated
    return output


def _safe_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) != len(p) or len(y) == 0 or not set(y).issubset({0, 1}):
        raise ScreeningBlendAblationError("Metric labels/probabilities are invalid")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ScreeningBlendAblationError("Metric probability is invalid")
    decision = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, decision, labels=[0, 1]).ravel()
    has_both = set(y) == {0, 1}
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return {
        "n_test": len(y),
        "positive_prevalence": float(y.mean()),
        "class_complete": has_both,
        "auprc_average_precision_positive": (
            float(average_precision_score(y, p)) if y.sum() else np.nan
        ),
        "auroc": float(roc_auc_score(y, p)) if has_both else np.nan,
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "accuracy": float((decision == y).mean()),
        "balanced_accuracy": (
            float(0.5 * (tp / (tp + fn) + tn / (tn + fp))) if has_both else np.nan
        ),
        "mcc": float(matthews_corrcoef(y, decision)) if has_both else np.nan,
        "macro_f1": float(
            f1_score(y, decision, labels=[0, 1], average="macro", zero_division=0)
        ),
        "recall_sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "precision_positive": float(tp / (tp + fp)) if tp + fp else np.nan,
        "npv": float(tn / (tn + fn)) if tn + fn else np.nan,
        "cohen_kappa": float(cohen_kappa_score(y, decision)) if has_both else np.nan,
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _svc_probability_at_native_boundary(model: SVC) -> float:
    """Return the positive probability at decision_function == 0 for binary libsvm."""

    if (
        np.asarray(model.classes_).tolist() != [0, 1]
        or np.asarray(model.probA_).shape != (1,)
        or np.asarray(model.probB_).shape != (1,)
        or float(model.probA_[0]) >= 0
    ):
        raise ScreeningBlendAblationError("Paper SVM has unexpected probability semantics")
    # libsvm's binary positive probability crosses the native decision boundary
    # at sigmoid(probB), not sigmoid(-probB).  This is deliberately verified
    # against SVC.predict below instead of trusting a 0.5 probability cutoff.
    boundary = float(1.0 / (1.0 + np.exp(-float(model.probB_[0]))))
    if not 0 < boundary < 1:
        raise ScreeningBlendAblationError("Paper SVM probability boundary is invalid")
    return boundary


def _refit_and_validate_published_svm(
    d1: pd.DataFrame,
    bundle: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[SVC, float, dict[str, Any]]:
    """Refit the executable-paper SVC and prove parity with the sealed component."""

    contract = protocol["paper_svm_contract"]
    features = list(contract["features"])
    if features != list(bundle["paper_svm_feature_names"]):
        raise ScreeningBlendAblationError("Paper-SVM feature order differs from bundle")
    if set(features) - set(d1):
        raise ScreeningBlendAblationError("D1 lacks a published paper-SVM descriptor")
    by_row = d1.set_index("paper_row_index", verify_integrity=True)
    fit_ids = np.asarray(bundle["fit_paper_indices"], dtype=int)
    fit_labels = np.asarray(bundle["fit_labels"], dtype=int)
    if len(fit_ids) != 324 or len(set(fit_ids)) != 324:
        raise ScreeningBlendAblationError("Paper-SVM fit IDs are not the 324 D1 rows")
    if not np.array_equal(by_row.loc[fit_ids, "label"].to_numpy(dtype=int), fit_labels):
        raise ScreeningBlendAblationError("Paper-SVM fit labels differ from D1")
    all_ids = np.sort(d1.paper_row_index.to_numpy(dtype=int))
    test_ids = np.asarray(sorted(set(all_ids) - set(fit_ids)), dtype=int)
    if len(test_ids) != 81:
        raise ScreeningBlendAblationError("Paper-SVM held-out IDs are not exactly 81")
    X_fit = by_row.loc[fit_ids, features].to_numpy(dtype=np.float64)
    X_all = by_row.loc[all_ids, features].to_numpy(dtype=np.float64)
    X_test = by_row.loc[test_ids, features].to_numpy(dtype=np.float64)
    y_test = by_row.loc[test_ids, "label"].to_numpy(dtype=int)
    if not np.isfinite(X_fit).all() or not np.isfinite(X_all).all():
        raise ScreeningBlendAblationError("Paper-SVM descriptors are not finite")

    parameters = {
        "kernel": "linear",
        "C": 1.0,
        "gamma": 1.0,
        "probability": True,
        "random_state": 42,
    }
    refitted = SVC(**parameters).fit(X_fit, fit_labels)
    upstream = bundle["paper_svm"]
    if not isinstance(upstream, SVC):
        raise ScreeningBlendAblationError("Sealed paper-SVM component is not sklearn SVC")
    for key, expected in parameters.items():
        if upstream.get_params().get(key) != expected:
            raise ScreeningBlendAblationError(f"Sealed paper-SVM differs at {key}")
    refitted_probability = np.asarray(refitted.predict_proba(X_all)[:, 1], dtype=float)
    upstream_probability = np.asarray(upstream.predict_proba(X_all)[:, 1], dtype=float)
    if not np.allclose(refitted_probability, upstream_probability, rtol=0.0, atol=1e-15):
        raise ScreeningBlendAblationError(
            "Refitted paper SVM differs from sealed probabilities"
        )
    if not np.allclose(
        refitted.decision_function(X_all),
        upstream.decision_function(X_all),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ScreeningBlendAblationError("Refitted paper SVM decision scores differ")

    notebook_classifier = SVC(
        kernel="linear", C=1.0, gamma=1.0, probability=False, random_state=42
    ).fit(X_fit, fit_labels)
    native_test_decision = refitted.predict(X_test).astype(int)
    if not np.array_equal(native_test_decision, notebook_classifier.predict(X_test)):
        raise ScreeningBlendAblationError(
            "probability=True changed the official notebook classifier boundary"
        )
    boundary = _svc_probability_at_native_boundary(refitted)
    test_probability = np.asarray(refitted.predict_proba(X_test)[:, 1], dtype=float)
    if not np.array_equal(native_test_decision, (test_probability >= boundary).astype(int)):
        raise ScreeningBlendAblationError(
            "Paper-SVM probability boundary does not reproduce native predict"
        )
    all_probability_decision = (refitted_probability >= boundary).astype(int)
    if not np.array_equal(all_probability_decision, refitted.predict(X_all)):
        raise ScreeningBlendAblationError(
            "Paper-SVM boundary parity failed on the complete D1 frame"
        )
    native_metrics = _safe_metrics(y_test, test_probability, boundary)
    expected = contract["expected_official_test_result"]
    observed_confusion = [
        native_metrics["tn"],
        native_metrics["fp"],
        native_metrics["fn"],
        native_metrics["tp"],
    ]
    if not np.isclose(
        native_metrics["accuracy"], float(expected["accuracy"]), rtol=0.0, atol=1e-15
    ) or observed_confusion != list(expected["confusion_tn_fp_fn_tp"]):
        raise ScreeningBlendAblationError(
            "Refitted SVM does not reproduce the official reported test result"
        )

    scaled = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(**parameters)),
        ]
    ).fit(X_fit, fit_labels)
    scaled_probability = np.asarray(scaled.predict_proba(X_test)[:, 1], dtype=float)
    scaled_svc = scaled.named_steps["svc"]
    scaled_boundary = _svc_probability_at_native_boundary(scaled_svc)
    if not np.array_equal(
        scaled.predict(X_test), (scaled_probability >= scaled_boundary).astype(int)
    ):
        raise ScreeningBlendAblationError("Scaled sensitivity decision parity failed")
    scaled_metrics = _safe_metrics(y_test, scaled_probability, scaled_boundary)
    audit = {
        "schema_version": "geroprotector.paper_svm_source_resolution.v1",
        "official_repository_commit": contract["official_repository_commit"],
        "official_notebook_sha256": contract["official_notebook_sha256"],
        "manuscript_statement": "used_scaled_data",
        "executable_notebook_preprocessing": "none",
        "primary_resolution": (
            "unscaled executable notebook because it exactly reproduces the published "
            "55_of_81 accuracy; scaling is retained only as a sensitivity"
        ),
        "features": features,
        "parameters": parameters,
        "gamma_has_no_effect_for_linear_kernel": True,
        "probability_true_role": "ranking_and_blend_probability_only",
        "native_binary_rule": "sklearn_svc_predict_equivalent_decision_function_ge_zero",
        "native_probability_boundary": boundary,
        "native_d1_test_metrics": native_metrics,
        "scaled_text_sensitivity_d1_test_metrics": scaled_metrics,
        "refitted_probability_equals_existing_sealed_component": True,
        "external_predictions_were_generated_by_same_sealed_component": True,
        "external_outcomes_used_for_fit_probability_or_decision_rule": False,
    }
    return refitted, boundary, audit


def _thresholds(
    d1: pd.DataFrame,
    oof: pd.DataFrame,
    bundle: dict[str, Any],
    paper_svm_boundary: float | None = None,
) -> tuple[dict[str, float], dict[str, str], pd.DataFrame]:
    if oof.paper_row_index.duplicated().any() or len(oof) != 324:
        raise ScreeningBlendAblationError("D1 OOF rows are not exactly 324 unique rows")
    by_row = d1.set_index("paper_row_index")
    oof = oof.copy()
    oof["label"] = by_row.loc[oof.paper_row_index, "label"].to_numpy(dtype=int)
    if set(oof.paper_row_index) != set(np.asarray(bundle["fit_paper_indices"], dtype=int)):
        raise ScreeningBlendAblationError("D1 OOF identities differ from bundle fit IDs")
    if not np.array_equal(
        by_row.loc[np.asarray(bundle["fit_paper_indices"], dtype=int), "label"].to_numpy(
            dtype=int
        ),
        np.asarray(bundle["fit_labels"], dtype=int),
    ):
        raise ScreeningBlendAblationError("Bundle fit labels differ from D1")
    oof = _add_blend(oof)
    thresholds: dict[str, float] = {}
    operating_points: dict[str, str] = {}
    rows = []
    for model in MODELS:
        probability = oof[PROBABILITY_COLUMNS[model]].to_numpy(dtype=float)
        selected = select_threshold(oof.label.to_numpy(dtype=int), probability)
        if selected is None:
            raise ScreeningBlendAblationError(f"No D1 OOF threshold for {model}")
        threshold, oof_mcc = selected
        threshold_source = "full_324_d1_train_cross_fitted_oof_mcc"
        operating_point = "d1_train_oof_mcc"
        if model == "paper_svm" and paper_svm_boundary is not None:
            threshold = float(paper_svm_boundary)
            threshold_source = "published_svc_predict_decision_function_zero"
            operating_point = "published_svc_native_predict"
            oof_mcc = _safe_metrics(
                oof.label.to_numpy(dtype=int), probability, threshold
            )["mcc"]
        thresholds[model] = threshold
        operating_points[model] = operating_point
        metric = _safe_metrics(oof.label.to_numpy(dtype=int), probability, threshold)
        rows.append(
            {
                "model": model,
                "threshold": threshold,
                "operating_point": operating_point,
                "threshold_source": threshold_source,
                "oof_mcc_at_selection": oof_mcc,
                "d1_train_oof_auprc": metric["auprc_average_precision_positive"],
                "d1_train_oof_auroc": metric["auroc"],
                "d1_train_oof_brier": metric["brier"],
                "external_labels_used": False,
            }
        )
    if not np.isclose(thresholds["blend_010_060_030"], BLEND_THRESHOLD, atol=1e-15):
        raise ScreeningBlendAblationError("Reconstructed blend threshold differs from lock")
    return thresholds, operating_points, pd.DataFrame(rows)


def _d1_test_similarity(
    d1: pd.DataFrame, test: pd.DataFrame, bundle: dict[str, Any]
) -> np.ndarray:
    by_row = d1.set_index("paper_row_index")
    if test.paper_row_index.duplicated().any() or len(test) != 81:
        raise ScreeningBlendAblationError("D1 test is not exactly 81 unique rows")
    expected = by_row.loc[test.paper_row_index, "label"].to_numpy(dtype=int)
    if not np.array_equal(expected, test.label.to_numpy(dtype=int)):
        raise ScreeningBlendAblationError("D1 test labels differ from sealed source")
    if set(test.paper_row_index) & set(np.asarray(bundle["fit_paper_indices"], dtype=int)):
        raise ScreeningBlendAblationError("D1 row IDs overlap fit and test")
    contract = bundle["portable_contract"]
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(contract["morgan_radius"]),
        fpSize=int(contract["morgan_bits"]),
        includeChirality=bool(contract["morgan_include_chirality"]),
    )
    bits = []
    for smiles in by_row.loc[test.paper_row_index, "smiles"].astype(str):
        molecule = Chem.MolFromSmiles(smiles.strip())
        if molecule is None:
            raise ScreeningBlendAblationError("D1 test structure became unparsable")
        bits.append(np.asarray(generator.GetFingerprint(molecule), dtype=np.uint8))
    similarity = _tanimoto(
        np.stack(bits), np.asarray(bundle["tanimoto_train_bits"], dtype=np.uint8)
    )
    return similarity.max(axis=1)


def _metric_table(
    frame: pd.DataFrame,
    *,
    cohort: str,
    endpoint: str,
    thresholds: dict[str, float],
    operating_points: dict[str, str],
) -> pd.DataFrame:
    rows = []
    labels = frame.label.to_numpy(dtype=int)
    for model in MODELS:
        probability = frame[PROBABILITY_COLUMNS[model]].to_numpy(dtype=float)
        for operating_point, threshold in (
            (operating_points[model], thresholds[model]),
            ("fixed_0p5", 0.5),
        ):
            rows.append(
                {
                    "cohort": cohort,
                    "endpoint": endpoint,
                    "model": model,
                    "operating_point": operating_point,
                    "is_primary_operating_point": (
                        operating_point == operating_points[model]
                    ),
                    **_safe_metrics(labels, probability, threshold),
                }
            )
    return pd.DataFrame(rows)


def _union_cluster_ids(frame: pd.DataFrame) -> np.ndarray:
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(first: str, second: str) -> None:
        a, b = find(first), find(second)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for row in frame.itertuples(index=False):
        compound = "compound:" + str(row.external_id)
        find(compound)
        for publication in json.loads(row.publication_ids_json):
            union(compound, "publication:" + str(publication))
    return np.asarray([find("compound:" + str(value)) for value in frame.external_id])


def _comparison_values(
    labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    first_threshold: float,
    second_threshold: float,
) -> dict[str, float]:
    first_metric = _safe_metrics(labels, first, first_threshold)
    second_metric = _safe_metrics(labels, second, second_threshold)
    names = (
        "auprc_average_precision_positive",
        "auroc",
        "brier",
        "mcc",
        "macro_f1",
    )
    return {name: float(first_metric[name] - second_metric[name]) for name in names}


def paired_bootstrap(
    *,
    labels: np.ndarray,
    blend_probability: np.ndarray,
    component_probability: np.ndarray,
    blend_threshold: float,
    component_threshold: float,
    resamples: int,
    seed: int,
    confidence_level: float,
    cluster_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Paired compound or union-cluster bootstrap of blend-minus-component deltas."""

    y = np.asarray(labels, dtype=int)
    first = np.asarray(blend_probability, dtype=float)
    second = np.asarray(component_probability, dtype=float)
    if (
        len(y) != len(first)
        or len(y) != len(second)
        or set(y) != {0, 1}
        or resamples < 100
        or not (0 < confidence_level < 1)
    ):
        raise ScreeningBlendAblationError("Paired-bootstrap inputs are invalid")
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y == value) for value in (0, 1)]
    clusters: list[np.ndarray] | None = None
    if cluster_ids is not None:
        group = np.asarray(cluster_ids).astype(str)
        if len(group) != len(y) or (pd.Series(group).str.strip() == "").any():
            raise ScreeningBlendAblationError("Bootstrap cluster IDs are invalid")
        clusters = [np.flatnonzero(group == value) for value in sorted(set(group))]
    point = _comparison_values(
        y, first, second, blend_threshold, component_threshold
    )
    draws = {name: [] for name in point}
    for _ in range(resamples):
        if clusters is None:
            index = np.concatenate(
                [rng.choice(group, size=len(group), replace=True) for group in by_class]
            )
        else:
            chosen = rng.integers(0, len(clusters), size=len(clusters))
            index = np.concatenate([clusters[value] for value in chosen])
            if set(y[index]) != {0, 1}:
                continue
        values = _comparison_values(
            y[index],
            first[index],
            second[index],
            blend_threshold,
            component_threshold,
        )
        for name, value in values.items():
            draws[name].append(value)
    valid = min(len(values) for values in draws.values())
    if valid < int(0.9 * resamples):
        raise ScreeningBlendAblationError("Too few valid paired-bootstrap resamples")
    alpha = (1 - confidence_level) / 2
    rows = []
    for name, values in draws.items():
        array = np.asarray(values, dtype=float)
        p_value = min(
            1.0,
            2
            * min(
                (np.count_nonzero(array <= 0) + 1) / (len(array) + 1),
                (np.count_nonzero(array >= 0) + 1) / (len(array) + 1),
            ),
        )
        rows.append(
            {
                "metric": name,
                "delta_blend_minus_component": point[name],
                "ci_lower": float(np.quantile(array, alpha)),
                "ci_upper": float(np.quantile(array, 1 - alpha)),
                "bootstrap_two_sided_p": float(p_value),
                "ci_excludes_zero": bool(
                    np.quantile(array, alpha) > 0 or np.quantile(array, 1 - alpha) < 0
                ),
                "requested_resamples": resamples,
                "valid_resamples": len(array),
            }
        )
    return pd.DataFrame(rows)


def _applicability_rows(
    frame: pd.DataFrame,
    *,
    cohort: str,
    endpoint: str,
    thresholds: dict[str, float],
    operating_points: dict[str, str],
    similarity_threshold: float,
) -> pd.DataFrame:
    rows = []
    similarity = frame.maximum_tanimoto_to_fitted_train
    for stratum, mask in (
        ("outside_max_tanimoto_lt_0p40", similarity < similarity_threshold),
        ("inside_max_tanimoto_ge_0p40", similarity >= similarity_threshold),
    ):
        subset = frame.loc[mask]
        if subset.empty:
            continue
        for model in MODELS:
            rows.append(
                {
                    "cohort": cohort,
                    "endpoint": endpoint,
                    "similarity_stratum": stratum,
                    "model": model,
                    "operating_point": operating_points[model],
                    "similarity_minimum": float(
                        subset.maximum_tanimoto_to_fitted_train.min()
                    ),
                    "similarity_median": float(
                        subset.maximum_tanimoto_to_fitted_train.median()
                    ),
                    "similarity_maximum": float(
                        subset.maximum_tanimoto_to_fitted_train.max()
                    ),
                    **_safe_metrics(
                        subset.label.to_numpy(dtype=int),
                        subset[PROBABILITY_COLUMNS[model]].to_numpy(dtype=float),
                        thresholds[model],
                    ),
                }
            )
    return pd.DataFrame(rows)


def _summary(
    thresholds: pd.DataFrame,
    d1_metrics: pd.DataFrame,
    external_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    applicability: pd.DataFrame,
) -> str:
    paper_refit = bool(
        (thresholds.threshold_source == "published_svc_predict_decision_function_zero").any()
    )
    lines = [
        "# Locked blend component ablation and applicability analysis",
        "",
        "All component probabilities come from the same component state fitted on the 324",
        "D1 paper-training rows.",
        (
            "The published paper SVM was independently refitted on those same 324 rows and "
            "proved probability-identical to the sealed component."
            if paper_refit
            else "No component was refit for this v1 sensitivity analysis."
        ),
        "No external label selected a model, threshold, weight, or applicability boundary.",
        "",
        "## D1-train operating points",
        "",
        "| Model | Probability threshold | Source | Train-OOF MCC at rule |",
        "|---|---:|---|---:|",
    ]
    for row in thresholds.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.threshold:.12f} | {row.threshold_source} | "
            f"{row.oof_mcc_at_selection:.4f} |"
        )

    def metric_section(title: str, frame: pd.DataFrame) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Model | n | AP | AUROC | Brier | MCC | Macro F1 | Recall | Specificity |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        selected = frame[frame.is_primary_operating_point]
        for row in selected.itertuples(index=False):
            lines.append(
                f"| {row.model} | {row.n_test} | {row.auprc_average_precision_positive:.4f} | "
                f"{row.auroc:.4f} | {row.brier:.4f} | {row.mcc:.4f} | "
                f"{row.macro_f1:.4f} | {row.recall_sensitivity:.4f} | "
                f"{row.specificity:.4f} |"
            )

    metric_section("D1 held-out paper test", d1_metrics)
    retrieval = external_metrics[
        external_metrics.endpoint
        == "significant_positive_retrieval_background_not_certified_negative"
    ]
    metric_section("DrugAge positive-retrieval endpoint", retrieval)
    agextend = external_metrics[
        external_metrics.endpoint == "published_independent_table6_binary"
    ]
    metric_section("AgeXtend Table 6 endpoint", agextend)

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "DrugAge background compounds are not certified experimental negatives. AgeXtend",
            "has only four negatives after exclusions. Component comparisons are post-lock",
            "retrospective analyses and cannot replace the locked blend or select a new model.",
            "The 0.40 Tanimoto boundary is a label-free structural-similarity proxy, not a",
            "calibrated guarantee that a prediction is valid.",
            "",
            (
                f"Paired-bootstrap rows: {len(bootstrap)}. "
                f"Applicability rows: {len(applicability)}."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def run(root: Path, config_path: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ScreeningBlendAblationError("Unsafe run ID")
    protocol, protocol_sha256 = load_protocol(root, config_path)
    inputs = _verify_inputs(root, protocol)
    destination = root / "outputs" / run_id
    completed_path = destination / "COMPLETED.json"
    if completed_path.is_file() and not completed_path.is_symlink():
        completed = _read_json(completed_path, "ablation completion")
        if completed.get("protocol_sha256") != protocol_sha256:
            raise ScreeningBlendAblationError("Completed ablation protocol differs")
        _verify_artifact_map(destination, completed["artifact_hashes"])
        return destination
    if destination.exists() or destination.is_symlink():
        raise ScreeningBlendAblationError(f"Unsealed output directory exists: {destination}")

    external_protocol_path = root / protocol["sealed_inputs"]["external_protocol"]["path"]
    external_protocol, _external_protocol_sha256 = load_external_protocol(
        root, external_protocol_path
    )
    d1 = _d1_frame(external_protocol)
    paper_svm_model: SVC | None = None
    paper_svm_boundary: float | None = None
    paper_svm_audit: dict[str, Any] | None = None
    if protocol["schema_version"].endswith(".v2"):
        paper_svm_model, paper_svm_boundary, paper_svm_audit = (
            _refit_and_validate_published_svm(d1, inputs["bundle"], protocol)
        )
    weighted = inputs["weighted_directory"]
    oof = pd.read_csv(weighted / "full_train_oof_component_predictions.csv")
    test = _add_blend(pd.read_csv(weighted / "test_component_predictions.csv"))
    thresholds, operating_points, threshold_frame = _thresholds(
        d1, oof, inputs["bundle"], paper_svm_boundary
    )
    similarity_threshold = float(protocol["applicability"]["threshold"])
    test["maximum_tanimoto_to_fitted_train"] = _d1_test_similarity(
        d1, test, inputs["bundle"]
    )
    d1_metrics = _metric_table(
        test,
        cohort="d1_paper_test",
        endpoint="paper_binary",
        thresholds=thresholds,
        operating_points=operating_points,
    )

    external_frames: dict[tuple[str, str], pd.DataFrame] = {}
    external_directory = inputs["external_directory"]
    drugage = _add_blend(
        pd.read_csv(external_directory / "drugage" / "scored" / "scored_predictions.csv")
    )
    drugage_retrieval = drugage.assign(label=drugage.has_significant_positive.astype(int))
    external_frames[
        ("drugage", "significant_positive_retrieval_background_not_certified_negative")
    ] = drugage_retrieval
    strict = drugage[
        drugage.strict_status.isin(["positive_clean", "negative_clean"])
    ].copy()
    strict["label"] = (strict.strict_status == "positive_clean").astype(int)
    external_frames[("drugage", "strict_binary_sensitivity")] = strict
    agextend = _add_blend(
        pd.read_csv(external_directory / "agextend" / "scored" / "scored_predictions.csv")
    )
    external_frames[("agextend", "published_independent_table6_binary")] = agextend

    external_metrics = pd.concat(
        [
            _metric_table(
                frame,
                cohort=cohort,
                endpoint=endpoint,
                thresholds=thresholds,
                operating_points=operating_points,
            )
            for (cohort, endpoint), frame in external_frames.items()
        ],
        ignore_index=True,
    )

    bootstrap_config = protocol["paired_bootstrap"]
    resamples = int(bootstrap_config["resamples"])
    seed = int(bootstrap_config["seed"])
    confidence = float(bootstrap_config["confidence_level"])
    bootstrap_rows = []
    bootstrap_jobs = [
        (
            "d1_paper_test",
            "paper_binary",
            test,
            None,
            "conditional_paired_stratified_raw_paper_test_row",
        ),
        (
            "drugage",
            "significant_positive_retrieval_background_not_certified_negative",
            drugage_retrieval,
            None,
            "paired_stratified_parent_connectivity_compound",
        ),
        (
            "drugage",
            "significant_positive_retrieval_background_not_certified_negative",
            drugage_retrieval,
            _union_cluster_ids(drugage_retrieval),
            "paired_publication_compound_union_cluster_sensitivity",
        ),
        (
            "agextend",
            "published_independent_table6_binary",
            agextend,
            None,
            "unstable_descriptive_stratified_compound_only",
        ),
    ]
    for job_index, (cohort, endpoint, frame, cluster_ids, scheme) in enumerate(
        bootstrap_jobs
    ):
        for model_index, component in enumerate(COMPONENTS):
            result = paired_bootstrap(
                labels=frame.label.to_numpy(dtype=int),
                blend_probability=frame.blend_probability.to_numpy(dtype=float),
                component_probability=frame[PROBABILITY_COLUMNS[component]].to_numpy(
                    dtype=float
                ),
                blend_threshold=thresholds["blend_010_060_030"],
                component_threshold=thresholds[component],
                resamples=resamples,
                seed=seed + job_index * 1009 + model_index * 101,
                confidence_level=confidence,
                cluster_ids=cluster_ids,
            )
            result.insert(0, "sampling_scheme", scheme)
            result.insert(0, "comparison", f"blend_010_060_030_minus_{component}")
            result.insert(0, "endpoint", endpoint)
            result.insert(0, "cohort", cohort)
            result["inference_role"] = (
                "unstable_descriptive_only"
                if cohort == "agextend"
                else "retrospective_post_lock_sensitivity"
            )
            bootstrap_rows.append(result)
    bootstrap_frame = pd.concat(bootstrap_rows, ignore_index=True)

    applicability_frames = [
        _applicability_rows(
            test,
            cohort="d1_paper_test",
            endpoint="paper_binary",
            thresholds=thresholds,
            operating_points=operating_points,
            similarity_threshold=similarity_threshold,
        ),
        _applicability_rows(
            drugage_retrieval,
            cohort="drugage",
            endpoint="significant_positive_retrieval_background_not_certified_negative",
            thresholds=thresholds,
            operating_points=operating_points,
            similarity_threshold=similarity_threshold,
        ),
        _applicability_rows(
            agextend,
            cohort="agextend",
            endpoint="published_independent_table6_binary",
            thresholds=thresholds,
            operating_points=operating_points,
            similarity_threshold=similarity_threshold,
        ),
    ]
    applicability_frame = pd.concat(applicability_frames, ignore_index=True)

    summary = _summary(
        threshold_frame, d1_metrics, external_metrics, bootstrap_frame, applicability_frame
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.work-", dir=destination.parent))
    try:
        conditional_artifacts: list[str] = []
        if paper_svm_model is not None and paper_svm_audit is not None:
            model_directory = temporary / "models"
            model_directory.mkdir()
            model_path = model_directory / "paper_svm_exact.joblib"
            joblib.dump(paper_svm_model, model_path, compress=3)
            model_sha256 = sha256_file(model_path)
            atomic_write_json(temporary / "paper_svm_source_resolution.json", paper_svm_audit)
            atomic_write_json(
                model_directory / "PAPER_SVM_LOCK.json",
                {
                    "schema_version": "geroprotector.paper_svm_exact.lock.v1",
                    "model_id": "paper_svm_exact_executable_publication",
                    "model_artifact": "paper_svm_exact.joblib",
                    "model_artifact_sha256": model_sha256,
                    "fit_scope": "exact_324_d1_paper_train_rows",
                    "fit_paper_indices_sha256": inputs["bundle"][
                        "fit_paper_indices_sha256"
                    ],
                    "fit_labels_sha256": inputs["bundle"]["fit_labels_sha256"],
                    "features": protocol["paper_svm_contract"]["features"],
                    "preprocessing": protocol["paper_svm_contract"]["preprocessing"],
                    "parameters": paper_svm_audit["parameters"],
                    "native_probability_boundary": paper_svm_boundary,
                    "native_binary_rule": paper_svm_audit["native_binary_rule"],
                    "upstream_component_state_sha256": protocol["locked_model"][
                        "component_state_sha256"
                    ],
                    "refit_equals_upstream_sealed_component": True,
                    "external_outcomes_used_for_fit_or_lock": False,
                },
            )
            conditional_artifacts.extend(
                [
                    "models/paper_svm_exact.joblib",
                    "models/PAPER_SVM_LOCK.json",
                    "paper_svm_source_resolution.json",
                ]
            )
        _write_csv(temporary / "component_thresholds.csv", threshold_frame)
        _write_csv(temporary / "d1_test_predictions_with_applicability.csv", test)
        _write_csv(temporary / "d1_component_metrics.csv", d1_metrics)
        _write_csv(temporary / "external_component_metrics.csv", external_metrics)
        _write_csv(temporary / "paired_bootstrap_blend_vs_components.csv", bootstrap_frame)
        _write_csv(temporary / "applicability_stratified_metrics.csv", applicability_frame)
        (temporary / "summary.md").write_text(summary, encoding="utf-8")
        atomic_write_json(
            temporary / "analysis_audit.json",
            {
                "schema_version": "geroprotector.screening_blend_ablation.audit.v1",
                "protocol_sha256": protocol_sha256,
                "model_lock_sha256": inputs["model_lock_sha256"],
                "component_state_sha256": protocol["locked_model"][
                    "component_state_sha256"
                ],
                "sealed_input_completed_sha256": inputs["input_completed_sha256"],
                "paper_svm_refit_on_exact_d1_train": paper_svm_model is not None,
                "external_refit_or_recalibration": False,
                "external_labels_used_for_model_weight_threshold_or_ad_selection": False,
                "component_threshold_timing": (
                    "deterministically_reconstructed_post_hoc_from_preexisting_sealed_"
                    "d1_train_oof_predictions"
                ),
                "primary_component_comparison_metrics_are_threshold_free": True,
                "applicability_threshold": similarity_threshold,
                "applicability_is_a_similarity_proxy_not_a_validity_guarantee": True,
                "d1_test_exact_similarity_one_count": int(
                    np.count_nonzero(
                        np.isclose(test.maximum_tanimoto_to_fitted_train, 1.0)
                    )
                ),
                "age_extend_formal_inference_allowed": False,
                "drugage_background_is_certified_negative": False,
            },
        )
        artifact_names = (
            "component_thresholds.csv",
            "d1_test_predictions_with_applicability.csv",
            "d1_component_metrics.csv",
            "external_component_metrics.csv",
            "paired_bootstrap_blend_vs_components.csv",
            "applicability_stratified_metrics.csv",
            "summary.md",
            "analysis_audit.json",
            *conditional_artifacts,
        )
        artifact_hashes = {
            name: sha256_file(temporary / name) for name in artifact_names
        }
        atomic_write_json(
            temporary / "COMPLETED.json",
            {
                "schema_version": "geroprotector.screening_blend_ablation.completed.v1",
                "status": "COMPLETE",
                "run_id": run_id,
                "protocol_sha256": protocol_sha256,
                "artifact_hashes": artifact_hashes,
                "external_results_used_to_change_locked_model": False,
            },
        )
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    destination = run(args.root.resolve(), args.config.resolve(), args.run_id)
    print(f"Complete: {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
