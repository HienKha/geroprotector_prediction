"""Lock and analyse the train-selected paper-405 screening blend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from rdkit import __version__ as rdkit_version
from rdkit.Chem import Descriptors
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)

from geroprotector.audit import runtime_environment, validate_core_runtime
from geroprotector.fixed_blend_paper405 import (
    _features,
    _regular_file,
    _tanimoto,
)
from geroprotector.hashing import (
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.traditional_paper405 import (
    _read_sources,
    evaluation_metrics,
    paper_split_indices,
)
from geroprotector.weighted_blend_paper405 import (
    COMPONENTS,
    select_threshold,
)
from geroprotector.weighted_blend_paper405 import (
    load_protocol as load_weighted_protocol,
)

matplotlib.use("Agg")


class ScreeningBlendPaper405Error(RuntimeError):
    """Raised when a locked screening-blend contract is violated."""


LOCKED_WEIGHTS = np.asarray([0.10, 0.60, 0.30], dtype=np.float64)
LOCKED_CANDIDATE = "svm_0.10_tanimoto_0.60_tabpfn_0.30"
LOCKED_OOF_THRESHOLD = 0.5299579802368826


def _regular_directory(path: Path, role: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ScreeningBlendPaper405Error(
            f"{role} must be a regular non-symlink directory: {path}"
        )
    return path.resolve()


def _safe_file(directory: Path, relative: str, role: str) -> Path:
    path = directory / relative
    if ".." in Path(relative).parts or path.is_symlink() or not path.is_file():
        raise ScreeningBlendPaper405Error(
            f"{role} must be a regular file inside the sealed run: {path}"
        )
    resolved = path.resolve()
    if directory != resolved.parent and directory not in resolved.parents:
        raise ScreeningBlendPaper405Error(f"{role} escapes its sealed run")
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ScreeningBlendPaper405Error(f"Missing regular JSON artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScreeningBlendPaper405Error(f"JSON artifact is not an object: {path}")
    return payload


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = yaml.safe_load(
        _regular_file(path, "screening-blend protocol").read_text(encoding="utf-8")
    )
    if not isinstance(protocol, dict) or protocol.get("schema_version") != (
        "geroprotector.screening_blend_paper405.protocol.v1"
    ):
        raise ScreeningBlendPaper405Error("Unknown screening-blend protocol schema")
    locked = protocol.get("locked_model", {})
    if (
        locked.get("candidate_id") != LOCKED_CANDIDATE
        or locked.get("components") != list(COMPONENTS)
        or not np.allclose(
            np.asarray(locked.get("weights"), dtype=float),
            LOCKED_WEIGHTS,
            rtol=0.0,
            atol=1e-15,
        )
        or locked.get("selection_source") != "full_324_train_cross_fitted_oof_only"
        or locked.get("outer_test_used_for_weight_selection") is not False
        or locked.get("intended_use")
        != "compound_ranking_and_screening_prioritization"
    ):
        raise ScreeningBlendPaper405Error("Locked screening model differs from 0.10/0.60/0.30")
    bundles = protocol.get("decision_bundles", {})
    if tuple(bundles) != ("oof_mcc", "fixed_0p5"):
        raise ScreeningBlendPaper405Error("Decision-bundle set/order differs")
    if not np.isclose(
        float(bundles["oof_mcc"].get("decision_threshold", np.nan)),
        LOCKED_OOF_THRESHOLD,
        rtol=0.0,
        atol=1e-15,
    ) or float(bundles["fixed_0p5"].get("decision_threshold", np.nan)) != 0.5:
        raise ScreeningBlendPaper405Error("Decision thresholds differ from the lock")
    analysis = protocol.get("post_lock_analysis", {})
    if analysis.get("test_labels_may_change_model_or_threshold") is not False:
        raise ScreeningBlendPaper405Error("Test labels may not change the locked model")
    firewall = protocol.get("external_firewall", {})
    if firewall != {
        "development_outcomes_loaded": False,
        "hagr_or_drugage_used_for_model_selection": False,
        "agextend_used_for_model_selection": False,
        "prediction_lock_required_before_outcome_access": True,
        "thresholds_may_be_reoptimized_externally": False,
    }:
        raise ScreeningBlendPaper405Error("External firewall differs from the lock")
    return protocol, canonical_sha256(protocol)


def _verify_weighted_run(
    directory: Path, protocol: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    directory = _regular_directory(directory, "weighted run")
    expected = protocol["source_runs"]
    completed_path = _safe_file(directory, "COMPLETED.json", "weighted completion")
    if sha256_file(completed_path) != expected["weighted_completed_sha256"]:
        raise ScreeningBlendPaper405Error("Weighted COMPLETED hash differs from the lock")
    completed = _load_json(completed_path)
    if completed.get("status") != "COMPLETE" or completed.get("run_id") != expected[
        "weighted_run_id"
    ]:
        raise ScreeningBlendPaper405Error("Weighted run identity/status differs")
    manifest_path = _safe_file(directory, "run_manifest.json", "weighted manifest")
    if sha256_file(manifest_path) != completed.get("run_manifest_sha256"):
        raise ScreeningBlendPaper405Error("Weighted run manifest hash differs")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("selected_primary_candidate") != LOCKED_CANDIDATE
        or manifest.get("outer_test_used_for_weight_or_threshold_selection") is not False
        or manifest.get("all_171_test_metrics_are_exploratory") is not True
    ):
        raise ScreeningBlendPaper405Error("Weighted run selection/firewall differs")
    for relative, expected_sha256 in manifest.get("artifact_hashes", {}).items():
        if sha256_file(_safe_file(directory, relative, relative)) != expected_sha256:
            raise ScreeningBlendPaper405Error(f"Weighted artifact hash differs: {relative}")
    for relative, expected_sha256 in manifest.get("model_hashes", {}).items():
        if sha256_file(_safe_file(directory, relative, relative)) != expected_sha256:
            raise ScreeningBlendPaper405Error(f"Weighted model hash differs: {relative}")
    selection_path = _safe_file(directory, "SELECTION_LOCK.json", "selection lock")
    if sha256_file(selection_path) != completed.get("selection_lock_sha256"):
        raise ScreeningBlendPaper405Error("Selection-lock byte hash differs")
    selection = _load_json(selection_path)
    winner = selection.get("winner", {})
    if (
        selection.get("outer_test_rows_or_labels_used") is not False
        or selection.get("immutable_before_outer_test_evaluation") is not True
        or winner.get("candidate_id") != LOCKED_CANDIDATE
        or winner.get("selected_using_test_labels") is not False
        or not np.allclose(
            [
                winner.get("weight_svm"),
                winner.get("weight_tanimoto"),
                winner.get("weight_tabpfn"),
            ],
            LOCKED_WEIGHTS,
            rtol=0.0,
            atol=1e-15,
        )
        or not np.isclose(
            float(winner.get("primary_threshold", np.nan)),
            LOCKED_OOF_THRESHOLD,
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise ScreeningBlendPaper405Error("Train-only selection lock differs")
    return completed, manifest, selection


def _verify_traditional_run(
    directory: Path, protocol: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = _regular_directory(directory, "traditional run")
    expected = protocol["source_runs"]
    completed_path = _safe_file(directory, "COMPLETED.json", "traditional completion")
    if sha256_file(completed_path) != expected["traditional_completed_sha256"]:
        raise ScreeningBlendPaper405Error("Traditional COMPLETED hash differs from the lock")
    completed = _load_json(completed_path)
    if completed.get("status") != "COMPLETE" or completed.get("run_id") != expected[
        "traditional_run_id"
    ]:
        raise ScreeningBlendPaper405Error("Traditional run identity/status differs")
    manifest_path = _safe_file(directory, "run_manifest.json", "traditional manifest")
    if sha256_file(manifest_path) != completed.get("run_manifest_sha256"):
        raise ScreeningBlendPaper405Error("Traditional manifest hash differs")
    manifest = _load_json(manifest_path)
    if manifest.get("outer_test_used_for_selection") is not False:
        raise ScreeningBlendPaper405Error("Traditional test-selection firewall differs")
    for record in manifest.get("artifact_inventory", []):
        path = _safe_file(directory, str(record["path"]), str(record["path"]))
        if path.stat().st_size != int(record["size_bytes"]) or sha256_file(path) != record[
            "sha256"
        ]:
            raise ScreeningBlendPaper405Error(
                f"Traditional artifact integrity differs: {record['path']}"
            )
    return completed, manifest


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(json.dumps(values.shape).encode())
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _descriptor_context(raw_train: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(raw_train)
    keep = finite.any(axis=0)
    retained = raw_train[:, keep]
    medians = np.nanmedian(np.where(np.isfinite(retained), retained, np.nan), axis=0)
    filled = np.where(np.isfinite(retained), retained, medians)
    varying = np.ptp(filled, axis=0) > 0
    context = filled[:, varying].astype(np.float32)
    if not np.isfinite(context).all():
        raise ScreeningBlendPaper405Error("TabPFN context preprocessing is non-finite")
    return {
        "finite_any_mask": keep,
        "medians_after_finite_any": medians.astype(np.float64),
        "varying_after_imputation_mask": varying,
        "context_features": context,
    }


def _bundle_sidecar_payload(bundle_path: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "geroprotector.screening_blend_paper405.bundle_sidecar.v1",
        "artifact": bundle_path.name,
        "artifact_sha256": sha256_file(bundle_path),
        "bundle_id": bundle["bundle_id"],
        "component_state_sha256": bundle["component_state_sha256"],
        "decision_threshold": bundle["decision_threshold"],
        "weights": bundle["weights"],
        "source_binding": bundle["source_binding"],
        "fit_paper_indices_sha256": bundle["fit_paper_indices_sha256"],
        "fit_labels_sha256": bundle["fit_labels_sha256"],
        "portable_contract": bundle["portable_contract"],
    }


def load_locked_bundle(
    bundle_path: Path, *, expected_artifact_sha256: str | None = None
) -> dict[str, Any]:
    bundle_path = _regular_file(bundle_path, "locked blend bundle")
    sidecar_path = bundle_path.with_suffix(".manifest.json")
    sidecar = _load_json(sidecar_path)
    observed_sha256 = sha256_file(bundle_path)
    if sidecar.get("artifact_sha256") != observed_sha256:
        raise ScreeningBlendPaper405Error("Bundle bytes differ from sidecar")
    if expected_artifact_sha256 is not None and observed_sha256 != expected_artifact_sha256:
        raise ScreeningBlendPaper405Error("Bundle bytes differ from expected sealed hash")
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != (
        "geroprotector.screening_blend_paper405.portable_bundle.v1"
    ):
        raise ScreeningBlendPaper405Error("Unknown locked bundle schema")
    if sidecar != _bundle_sidecar_payload(bundle_path, bundle):
        raise ScreeningBlendPaper405Error("Bundle semantic fields differ from sidecar")
    if not np.allclose(bundle.get("weights"), LOCKED_WEIGHTS, rtol=0.0, atol=1e-15):
        raise ScreeningBlendPaper405Error("Bundle weights differ from the lock")
    context = bundle["tabpfn_context"]
    checks = {
        "fit_paper_indices_sha256": _array_sha256(bundle["fit_paper_indices"]),
        "fit_labels_sha256": _array_sha256(bundle["fit_labels"]),
        "tanimoto_train_bits_sha256": _array_sha256(bundle["tanimoto_train_bits"]),
        "tabpfn_context_sha256": _array_sha256(context["context_features"]),
    }
    for key, observed in checks.items():
        if observed != bundle.get(key):
            raise ScreeningBlendPaper405Error(f"Bundle array integrity differs: {key}")
    return bundle


def _write_bundle(
    *,
    directory: Path,
    bundle_id: str,
    threshold: float,
    threshold_source: str,
    shared: dict[str, Any],
) -> dict[str, Any]:
    bundle = {
        **shared,
        "schema_version": "geroprotector.screening_blend_paper405.portable_bundle.v1",
        "bundle_id": bundle_id,
        "decision_threshold": float(threshold),
        "threshold_source": threshold_source,
    }
    path = directory / f"{bundle_id}.joblib"
    joblib.dump(bundle, path, compress=3)
    sidecar = _bundle_sidecar_payload(path, bundle)
    atomic_write_json(path.with_suffix(".manifest.json"), sidecar)
    loaded = load_locked_bundle(path, expected_artifact_sha256=sidecar["artifact_sha256"])
    if loaded["decision_threshold"] != threshold:
        raise ScreeningBlendPaper405Error("Bundle reload threshold parity failed")
    return {
        "bundle_id": bundle_id,
        "path": path.name,
        "sha256": sidecar["artifact_sha256"],
        "manifest": path.with_suffix(".manifest.json").name,
        "manifest_sha256": sha256_file(path.with_suffix(".manifest.json")),
        "decision_threshold": float(threshold),
        "threshold_source": threshold_source,
    }


def _metric_values(
    labels: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    decisions = probability >= threshold
    return {
        "auprc_average_precision_positive": float(
            average_precision_score(labels, probability)
        ),
        "auroc": float(roc_auc_score(labels, probability)),
        "brier": float(np.mean((probability - labels) ** 2)),
        "mcc": float(matthews_corrcoef(labels, decisions)),
        "macro_f1": float(f1_score(labels, decisions, average="macro", zero_division=0)),
    }


def paired_stratified_bootstrap(
    *,
    labels: np.ndarray,
    first_probability: np.ndarray,
    second_probability: np.ndarray,
    first_threshold: float,
    second_threshold: float,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> pd.DataFrame:
    y = np.asarray(labels, dtype=int)
    first = np.asarray(first_probability, dtype=float)
    second = np.asarray(second_probability, dtype=float)
    if (
        len(y) != len(first)
        or len(y) != len(second)
        or set(y) != {0, 1}
        or resamples < 100
        or not (0 < confidence_level < 1)
    ):
        raise ScreeningBlendPaper405Error("Paired-bootstrap inputs are invalid")
    by_class = [np.flatnonzero(y == value) for value in (0, 1)]
    rng = np.random.default_rng(seed)
    metrics = tuple(_metric_values(y, first, first_threshold))
    draws = {name: np.empty(resamples, dtype=float) for name in metrics}
    for iteration in range(resamples):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in by_class]
        )
        first_metrics = _metric_values(y[indices], first[indices], first_threshold)
        second_metrics = _metric_values(y[indices], second[indices], second_threshold)
        for name in metrics:
            draws[name][iteration] = first_metrics[name] - second_metrics[name]
    alpha = (1.0 - confidence_level) / 2.0
    first_point = _metric_values(y, first, first_threshold)
    second_point = _metric_values(y, second, second_threshold)
    rows = []
    for name in metrics:
        values = draws[name]
        p_value = min(
            1.0,
            2.0
            * min(
                (float(np.count_nonzero(values <= 0)) + 1.0) / (resamples + 1.0),
                (float(np.count_nonzero(values >= 0)) + 1.0) / (resamples + 1.0),
            ),
        )
        rows.append(
            {
                "metric": name,
                "first": first_point[name],
                "second": second_point[name],
                "delta_first_minus_second": first_point[name] - second_point[name],
                "ci_lower": float(np.quantile(values, alpha)),
                "ci_upper": float(np.quantile(values, 1.0 - alpha)),
                "paired_bootstrap_two_sided_p": p_value,
                "resamples": resamples,
                "sampling_unit": "stratified_raw_paper_test_row",
                "test_used_for_model_or_threshold_selection": False,
            }
        )
    return pd.DataFrame(rows)


def _threshold_curve(
    labels: np.ndarray, probability: np.ndarray, thresholds: np.ndarray, model_id: str
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        decision = probability >= threshold
        tn, fp, fn, tp = confusion_matrix(labels, decision, labels=[0, 1]).ravel()
        rows.append(
            {
                "model_id": model_id,
                "threshold": float(threshold),
                "mcc": float(matthews_corrcoef(labels, decision)),
                "macro_f1": float(
                    f1_score(labels, decision, average="macro", zero_division=0)
                ),
                "recall_sensitivity": float(tp / (tp + fn)),
                "specificity": float(tn / (tn + fp)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows)


def _decision_curve(
    labels: np.ndarray, probability: np.ndarray, thresholds: np.ndarray, model_id: str
) -> pd.DataFrame:
    n = len(labels)
    rows = []
    for threshold in thresholds:
        decision = probability >= threshold
        _tn, fp, _fn, tp = confusion_matrix(labels, decision, labels=[0, 1]).ravel()
        odds = threshold / (1.0 - threshold)
        rows.append(
            {
                "model_id": model_id,
                "threshold_probability": float(threshold),
                "net_benefit": float(tp / n - fp / n * odds),
                "treat_all_net_benefit": float(labels.mean() - (1 - labels.mean()) * odds),
                "treat_none_net_benefit": 0.0,
                "role": "descriptive_test_decision_curve_not_selection",
            }
        )
    return pd.DataFrame(rows)


def _top_k(
    labels: np.ndarray, probability: np.ndarray, model_id: str, values: list[int]
) -> pd.DataFrame:
    order = np.argsort(-probability, kind="mergesort")
    prevalence = float(labels.mean())
    rows = []
    for requested in values:
        k = min(int(requested), len(labels))
        selected = labels[order[:k]]
        hits = int(selected.sum())
        precision = hits / k
        rows.append(
            {
                "model_id": model_id,
                "k": k,
                "test_fraction": k / len(labels),
                "positive_hits": hits,
                "precision_at_k": precision,
                "recall_at_k": hits / int(labels.sum()),
                "enrichment_over_test_prevalence": precision / prevalence,
                "role": "descriptive_candidate_prioritization_not_selection",
            }
        )
    return pd.DataFrame(rows)


def _plot_pr(path: Path, labels: np.ndarray, probabilities: dict[str, np.ndarray]) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 4.8))
    for name, values in probabilities.items():
        precision, recall, _ = precision_recall_curve(labels, values)
        ap = average_precision_score(labels, values)
        axis.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    axis.axhline(labels.mean(), color="grey", linestyle="--", label="prevalence")
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    axis.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_threshold(path: Path, frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True)
    for model_id, group in frame.groupby("model_id", sort=False):
        for axis, metric, label in zip(
            axes,
            ("mcc", "recall_sensitivity", "specificity"),
            ("MCC", "Recall", "Specificity"),
            strict=True,
        ):
            axis.plot(group.threshold, group[metric], label=model_id)
            axis.set(xlabel="Threshold", ylabel=label, xlim=(0, 1))
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_decision(path: Path, frame: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 4.8))
    for model_id, group in frame.groupby("model_id", sort=False):
        axis.plot(group.threshold_probability, group.net_benefit, label=model_id)
    anchor = frame[frame.model_id == frame.model_id.iloc[0]]
    axis.plot(
        anchor.threshold_probability,
        anchor.treat_all_net_benefit,
        "--",
        label="treat all",
    )
    axis.axhline(0, color="black", linewidth=0.8, label="treat none")
    axis.set(xlabel="Threshold probability", ylabel="Net benefit", xlim=(0.01, 0.99))
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _summary_markdown(
    metrics: pd.DataFrame,
    intervals: pd.DataFrame,
    svm_oof_threshold: float,
) -> str:
    selected = metrics.set_index("operating_point")
    blend = selected.loc["blend_oof_mcc"]
    svm = selected.loc["svm_original_fixed_0p5"]
    interval = intervals[intervals.comparison == "locked_blend_vs_original_svm"]
    lines = [
        "# Locked 0.10 / 0.60 / 0.30 screening blend",
        "",
        "The model was selected exclusively from the 324 paper-training rows. Test labels",
        "were opened only after the weight and OOF-MCC threshold had been sealed. This",
        "analysis is descriptive and cannot be used to change the locked model.",
        "",
        "## Screening interpretation",
        "",
        f"- Blend test AP: `{blend.auprc_average_precision_positive:.6f}` versus original "
        f"SVM `{svm.auprc_average_precision_positive:.6f}`.",
        f"- Blend test AUROC: `{blend.auroc:.6f}` versus original SVM `{svm.auroc:.6f}`.",
        f"- Blend Brier: `{blend.brier:.6f}` versus original SVM "
        f"`{svm.brier:.6f}` (lower is better).",
        f"- The original SVM train-OOF MCC threshold is `{svm_oof_threshold:.16g}`; it is",
        "  reported only as a fairness sensitivity and does not replace the paper's "
        "0.5 cutoff.",
        "- Accuracy and Macro F1 remain operating-point results. The blend is positioned as",
        "  a ranking/prioritization model, not as universally superior binary classification.",
        "",
        "## Paired 95% bootstrap intervals: locked blend minus original SVM",
        "",
        "| Metric | Delta | 95% CI |",
        "|---|---:|---:|",
    ]
    for row in interval.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.delta_first_minus_second:.6f} | "
            f"[{row.ci_lower:.6f}, {row.ci_upper:.6f}] |"
        )
    lines.extend(
        [
            "",
            "Intervals are conditional paired stratified bootstrap intervals over the 81 raw",
            "paper-test rows. The original split may contain related chemistry, the "
            "endpoint is",
            "source-confounded, and the sample is small. If an interval includes zero, wording",
            "must be 'numerically higher/lower', not 'significantly better'.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(
    *,
    root: Path,
    config_path: Path,
    weighted_run: Path,
    traditional_run: Path,
    positive_path: Path,
    negative_path: Path,
    run_id: str,
) -> Path:
    if not re.fullmatch(r"screeningblend405_[a-z0-9_.-]+", run_id):
        raise ScreeningBlendPaper405Error("RUN_ID must start with screeningblend405_")
    root = _regular_directory(root, "project root")
    validate_core_runtime(root / "requirements-lock.txt")
    protocol, protocol_sha256 = load_protocol(config_path)
    weighted_run = _regular_directory(weighted_run, "weighted run")
    traditional_run = _regular_directory(traditional_run, "traditional run")
    weighted_completed, weighted_manifest, _selection = _verify_weighted_run(
        weighted_run, protocol
    )
    traditional_completed, _traditional_manifest = _verify_traditional_run(
        traditional_run, protocol
    )

    weighted_protocol, weighted_protocol_sha256 = load_weighted_protocol(
        root / "configs" / "weighted_blend_paper405.yaml"
    )
    if weighted_protocol_sha256 != weighted_manifest.get("protocol_sha256"):
        raise ScreeningBlendPaper405Error("Current weighted protocol differs from sealed run")
    paper = _paper_contract(root, weighted_protocol)
    frame, source_audit = _read_sources(
        _regular_file(positive_path, "positive source"),
        _regular_file(negative_path, "negative source"),
        paper,
    )
    train_indices, test_indices, split_sha256 = paper_split_indices(paper)
    if split_sha256 != weighted_manifest.get("paper_split_sha256"):
        raise ScreeningBlendPaper405Error("Reconstructed paper split differs from sealed run")
    smiles, smiles_audit = _validated_raw_smiles(frame)
    features = _features(frame, smiles, weighted_protocol)
    labels = frame.label.to_numpy(dtype=int)

    test_components = pd.read_csv(
        _safe_file(weighted_run, "test_component_predictions.csv", "test components")
    ).sort_values("paper_row_index", kind="mergesort")
    if test_components.paper_row_index.tolist() != sorted(map(int, test_indices)):
        raise ScreeningBlendPaper405Error("Weighted test identities differ from paper split")
    test_labels = labels[test_components.paper_row_index.to_numpy(dtype=int)]
    if not np.array_equal(test_labels, test_components.label.to_numpy(dtype=int)):
        raise ScreeningBlendPaper405Error(
            "Weighted test labels differ from source reconstruction"
        )
    matrix = test_components[
        ["probability_paper_svm", "probability_tanimoto_svc", "probability_tabpfn_v2"]
    ].to_numpy(dtype=float)
    blend_probability = matrix @ LOCKED_WEIGHTS
    svm_probability = matrix[:, 0]

    traditional_predictions = pd.read_csv(
        _safe_file(traditional_run, "predictions.csv", "traditional predictions")
    )
    svm_test = traditional_predictions[
        traditional_predictions.model_id == "svm_original_paper"
    ].sort_values("paper_row_index", kind="mergesort")
    if (
        svm_test.paper_row_index.tolist() != test_components.paper_row_index.tolist()
        or not np.array_equal(svm_test.label.to_numpy(dtype=int), test_labels)
        or not np.allclose(
            svm_test.ranking_score.to_numpy(dtype=float),
            svm_probability,
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise ScreeningBlendPaper405Error("Original SVM and blend component are not identical")

    oof = pd.read_csv(
        _safe_file(
            weighted_run,
            "full_train_oof_component_predictions.csv",
            "full-train OOF components",
        )
    ).sort_values("paper_row_index", kind="mergesort")
    oof_indices = oof.paper_row_index.to_numpy(dtype=int)
    if set(oof_indices) != set(map(int, train_indices)):
        raise ScreeningBlendPaper405Error("OOF identities do not cover exact paper train")
    oof_labels = labels[oof_indices]
    oof_svm_probability = oof.probability_paper_svm.to_numpy(dtype=float)
    svm_threshold_result = select_threshold(oof_labels, oof_svm_probability)
    if svm_threshold_result is None:
        raise ScreeningBlendPaper405Error("Could not choose train-only SVM threshold")
    svm_oof_threshold = float(svm_threshold_result[0])

    output_root = root / "outputs"
    output_root.mkdir(exist_ok=True)
    destination = output_root / run_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite screening-blend run: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.work-", dir=output_root))
    try:
        model_directory = temporary / "models"
        model_directory.mkdir()
        paper_svm_path = _safe_file(
            weighted_run, "models/components/paper_svm.joblib", "paper SVM model"
        )
        tanimoto_path = _safe_file(
            weighted_run, "models/components/tanimoto_svc.joblib", "Tanimoto SVC model"
        )
        paper_svm = joblib.load(paper_svm_path)
        tanimoto_svc = joblib.load(tanimoto_path)
        train_bits = features["morgan"][train_indices]
        context = _descriptor_context(features["rdkit2d"][train_indices])

        paper_parity = np.asarray(
            paper_svm.predict_proba(features["paper"][test_indices])[:, 1], dtype=float
        )
        tanimoto_parity = expit(
            tanimoto_svc.decision_function(
                _tanimoto(features["morgan"][test_indices], train_bits)
            )
        )
        indexed_test = test_components.set_index("paper_row_index").loc[test_indices]
        if not np.allclose(
            paper_parity,
            indexed_test.probability_paper_svm.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-15,
        ) or not np.allclose(
            tanimoto_parity,
            indexed_test.probability_tanimoto_svc.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-7,
        ):
            raise ScreeningBlendPaper405Error("Reloaded component prediction parity failed")

        component_semantics = {
            "paper_svm_model_sha256": sha256_file(paper_svm_path),
            "tanimoto_svc_model_sha256": sha256_file(tanimoto_path),
            "tanimoto_train_bits_sha256": _array_sha256(train_bits),
            "tabpfn_context_sha256": _array_sha256(context["context_features"]),
            "tabpfn_checkpoint_sha256": weighted_protocol["components"]["tabpfn_v2"][
                "checkpoint_sha256"
            ],
            "fit_paper_indices_sha256": _array_sha256(train_indices.astype(np.int64)),
            "fit_labels_sha256": _array_sha256(labels[train_indices].astype(np.int64)),
        }
        component_state_sha256 = canonical_sha256(component_semantics)
        shared = {
            "weights": LOCKED_WEIGHTS.tolist(),
            "components": list(COMPONENTS),
            "component_state_sha256": component_state_sha256,
            "source_binding": {
                "weighted_run_id": weighted_completed["run_id"],
                "weighted_run_manifest_sha256": weighted_completed["run_manifest_sha256"],
                "selection_lock_sha256": weighted_completed["selection_lock_sha256"],
                "traditional_run_id": traditional_completed["run_id"],
                "traditional_run_manifest_sha256": traditional_completed[
                    "run_manifest_sha256"
                ],
                "paper_split_sha256": split_sha256,
                "screening_protocol_sha256": protocol_sha256,
            },
            "fit_paper_indices": train_indices.astype(np.int64),
            "fit_labels": labels[train_indices].astype(np.int64),
            "fit_paper_indices_sha256": component_semantics["fit_paper_indices_sha256"],
            "fit_labels_sha256": component_semantics["fit_labels_sha256"],
            "paper_svm": paper_svm,
            "paper_svm_feature_names": list(weighted_protocol["features"]["paper_descriptors"]),
            "tanimoto_svc": tanimoto_svc,
            "tanimoto_train_bits": train_bits,
            "tanimoto_train_bits_sha256": component_semantics[
                "tanimoto_train_bits_sha256"
            ],
            "tabpfn_context": context,
            "tabpfn_context_sha256": component_semantics["tabpfn_context_sha256"],
            "tabpfn_checkpoint": weighted_protocol["components"]["tabpfn_v2"],
            "descriptor_names": [name for name, _function in Descriptors._descList],
            "portable_contract": {
                "rdkit_version": rdkit_version,
                "paper_descriptor_generator": "DataWarrior_5.5.0_exact_values_required",
                "paper_descriptor_substitution_allowed": False,
                "morgan_radius": int(weighted_protocol["features"]["morgan_radius"]),
                "morgan_bits": int(weighted_protocol["features"]["morgan_bits"]),
                "morgan_include_chirality": False,
                "tabpfn_reconstruction_required": True,
                "tabpfn_checkpoint_embedded": False,
                "external_overlap_with_any_paper405_identity_allowed": False,
            },
        }
        bundle_records = []
        for bundle_id, specification in protocol["decision_bundles"].items():
            bundle_records.append(
                _write_bundle(
                    directory=model_directory,
                    bundle_id=f"blend_010_060_030_{bundle_id}",
                    threshold=float(specification["decision_threshold"]),
                    threshold_source=str(specification["threshold_source"]),
                    shared=shared,
                )
            )
        atomic_write_json(
            model_directory / "MODEL_LOCK.json",
            {
                "schema_version": "geroprotector.screening_blend_paper405.model_lock.v1",
                "candidate_id": LOCKED_CANDIDATE,
                "weights": LOCKED_WEIGHTS.tolist(),
                "component_state_sha256": component_state_sha256,
                "bundles": bundle_records,
                "selection_used_outer_test": False,
                "two_files_represent_one_fitted_ranking_model_with_two_decision_rules": True,
            },
        )

        operating_points = {
            "blend_oof_mcc": (blend_probability, LOCKED_OOF_THRESHOLD),
            "blend_fixed_0p5": (blend_probability, 0.5),
            "svm_original_fixed_0p5": (svm_probability, 0.5),
            "svm_train_oof_mcc": (svm_probability, svm_oof_threshold),
        }
        metric_rows = []
        prediction_rows = []
        for name, (probability, threshold) in operating_points.items():
            metric_rows.append(
                {
                    "operating_point": name,
                    **evaluation_metrics(
                        test_labels, probability, probability, threshold=threshold
                    ),
                    "test_used_for_model_or_threshold_selection": False,
                    "role": "post_lock_descriptive_test_evaluation",
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "operating_point": name,
                        "paper_row_index": test_components.paper_row_index,
                        "label": test_labels,
                        "probability": probability,
                        "threshold": threshold,
                        "decision": (probability >= threshold).astype(int),
                    }
                )
            )
        metrics = pd.DataFrame(metric_rows)
        predictions = pd.concat(prediction_rows, ignore_index=True)
        metrics.to_csv(temporary / "test_operating_point_metrics.csv", index=False)
        predictions.to_csv(temporary / "test_operating_point_predictions.csv", index=False)

        bootstrap = protocol["post_lock_analysis"]["bootstrap"]
        comparison_specs = {
            "locked_blend_vs_original_svm": (LOCKED_OOF_THRESHOLD, 0.5),
            "common_threshold_0p5": (0.5, 0.5),
            "separate_train_oof_mcc_thresholds": (
                LOCKED_OOF_THRESHOLD,
                svm_oof_threshold,
            ),
        }
        interval_frames = []
        for offset, (name, thresholds) in enumerate(comparison_specs.items()):
            frame_interval = paired_stratified_bootstrap(
                labels=test_labels,
                first_probability=blend_probability,
                second_probability=svm_probability,
                first_threshold=thresholds[0],
                second_threshold=thresholds[1],
                resamples=int(bootstrap["resamples"]),
                seed=int(bootstrap["seed"]) + offset,
                confidence_level=float(bootstrap["confidence_level"]),
            )
            frame_interval.insert(0, "comparison", name)
            frame_interval.insert(1, "first_model", "blend_010_060_030")
            frame_interval.insert(2, "second_model", "svm_original_paper")
            frame_interval.insert(3, "first_threshold", thresholds[0])
            frame_interval.insert(4, "second_threshold", thresholds[1])
            interval_frames.append(frame_interval)
        intervals = pd.concat(interval_frames, ignore_index=True)
        intervals.to_csv(temporary / "paired_bootstrap_intervals.csv", index=False)

        pr_rows = []
        probabilities = {
            "blend_010_060_030": blend_probability,
            "svm_original_paper": svm_probability,
        }
        for name, probability in probabilities.items():
            precision, recall, thresholds = precision_recall_curve(test_labels, probability)
            for index in range(len(precision)):
                pr_rows.append(
                    {
                        "model_id": name,
                        "precision": precision[index],
                        "recall": recall[index],
                        "threshold": None if index == len(thresholds) else thresholds[index],
                        "role": "post_lock_descriptive_curve",
                    }
                )
        pd.DataFrame(pr_rows).to_csv(temporary / "precision_recall_curve.csv", index=False)
        _plot_pr(temporary / "precision_recall_curve.svg", test_labels, probabilities)

        grid = protocol["post_lock_analysis"]["threshold_grid"]
        thresholds = np.arange(
            float(grid["minimum"]),
            float(grid["maximum"]) + float(grid["step"]) / 2,
            float(grid["step"]),
        )
        threshold_frame = pd.concat(
            [
                _threshold_curve(test_labels, probability, thresholds, name)
                for name, probability in probabilities.items()
            ],
            ignore_index=True,
        )
        threshold_frame["role"] = "post_lock_descriptive_curve_not_threshold_selection"
        threshold_frame.to_csv(temporary / "threshold_operating_curves.csv", index=False)
        _plot_threshold(temporary / "threshold_operating_curves.svg", threshold_frame)

        decision_spec = protocol["post_lock_analysis"]["decision_curve"]
        decision_thresholds = np.arange(
            float(decision_spec["minimum"]),
            float(decision_spec["maximum"]) + float(decision_spec["step"]) / 2,
            float(decision_spec["step"]),
        )
        decision_frame = pd.concat(
            [
                _decision_curve(test_labels, probability, decision_thresholds, name)
                for name, probability in probabilities.items()
            ],
            ignore_index=True,
        )
        decision_frame.to_csv(temporary / "decision_curve.csv", index=False)
        _plot_decision(temporary / "decision_curve.svg", decision_frame)

        top_k = pd.concat(
            [
                _top_k(
                    test_labels,
                    probability,
                    name,
                    list(protocol["post_lock_analysis"]["top_k"]),
                )
                for name, probability in probabilities.items()
            ],
            ignore_index=True,
        )
        top_k.to_csv(temporary / "top_k_enrichment.csv", index=False)
        atomic_write_json(
            temporary / "leakage_and_logic_audit.json",
            {
                "schema_version": "geroprotector.screening_blend_paper405.audit.v1",
                "weights_selected_from_train_oof_only": True,
                "threshold_oof_mcc_selected_from_train_oof_only": True,
                "fixed_0p5_threshold_prespecified": True,
                "selection_lock_precedes_test_evaluation": True,
                "test_curves_and_bootstrap_are_post_lock_descriptive": True,
                "test_curves_or_intervals_used_to_change_model": False,
                "external_outcomes_loaded": False,
                "hagr_loaded": False,
                "agextend_loaded": False,
                "paper_svm_equals_original_svm_predictions": True,
                "component_model_reload_parity": True,
                "source_audit": source_audit,
                "smiles_audit": smiles_audit,
                "train_rows": len(train_indices),
                "test_rows": len(test_indices),
                "svm_oof_threshold": svm_oof_threshold,
                "bootstrap_sampling_caveat": (
                    "Conditional on the raw 81-row paper test; related chemistry and "
                    "source-label confounding remain."
                ),
            },
        )
        (temporary / "summary.md").write_text(
            _summary_markdown(metrics, intervals, svm_oof_threshold), encoding="utf-8"
        )
        artifact_records = []
        for path in sorted(temporary.rglob("*")):
            if path.is_symlink():
                raise ScreeningBlendPaper405Error("Output bundle contains a symlink")
            if path.is_file():
                artifact_records.append(
                    {
                        "path": path.relative_to(temporary).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        manifest = {
            "schema_version": "geroprotector.screening_blend_paper405.run.v1",
            "run_id": run_id,
            "protocol_sha256": protocol_sha256,
            "runtime_environment": runtime_environment(),
            "source_runs": {
                "weighted": weighted_completed["run_manifest_sha256"],
                "traditional": traditional_completed["run_manifest_sha256"],
            },
            "candidate_id": LOCKED_CANDIDATE,
            "weights": LOCKED_WEIGHTS.tolist(),
            "component_state_sha256": component_state_sha256,
            "test_used_for_model_or_threshold_selection": False,
            "post_lock_test_analysis_role": "descriptive_not_selection",
            "external_outcomes_loaded": False,
            "artifacts": artifact_records,
        }
        atomic_write_json(temporary / "run_manifest.json", manifest)
        atomic_write_json(
            temporary / "COMPLETED.json",
            {
                "status": "COMPLETE",
                "run_id": run_id,
                "run_manifest_sha256": sha256_file(temporary / "run_manifest.json"),
                "model_lock_sha256": sha256_file(model_directory / "MODEL_LOCK.json"),
                "weights": LOCKED_WEIGHTS.tolist(),
                "thresholds": [LOCKED_OOF_THRESHOLD, 0.5],
                "external_outcomes_loaded": False,
            },
        )
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"Complete: {destination}", flush=True)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weighted-run", required=True)
    parser.add_argument("--traditional-run", required=True)
    parser.add_argument("--positive", required=True)
    parser.add_argument("--negative", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    run(
        root=Path(args.root),
        config_path=Path(args.config),
        weighted_run=Path(args.weighted_run),
        traditional_run=Path(args.traditional_run),
        positive_path=Path(args.positive),
        negative_path=Path(args.negative),
        run_id=args.run_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
