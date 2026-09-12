"""Nested train-only weight selection for the paper-405 three-component blend."""

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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from geroprotector.audit import runtime_environment, source_tree_sha256, validate_core_runtime
from geroprotector.fixed_blend_paper405 import (
    _features,
    _regular_file,
    _selected_component_predictions,
)
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_blend_protocol
from geroprotector.hashing import (
    atomic_write_json,
    canonical_sha256,
    sha256_bytes,
    sha256_file,
)
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.traditional_paper405 import (
    _cross_split_audit,
    _read_sources,
    evaluation_metrics,
    paper_split_indices,
)


class WeightedBlendPaper405Error(RuntimeError):
    pass


COMPONENTS = ("paper_svm", "tanimoto_svc", "tabpfn_v2")
WEIGHT_COLUMNS = ("weight_svm", "weight_tanimoto", "weight_tabpfn")


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = yaml.safe_load(_regular_file(path, "weighted-blend protocol").read_text())
    if protocol.get("schema_version") != "geroprotector.weighted_blend_paper405.protocol.v1":
        raise WeightedBlendPaper405Error("Unknown weighted-blend protocol schema")
    fixed, _ = load_fixed_blend_protocol(path.parent / "fixed_blend_paper405.yaml")
    for key in ("sources", "paper_split", "features", "components"):
        if protocol.get(key) != fixed.get(key):
            raise WeightedBlendPaper405Error(
                f"Weighted blend changes the sealed component/data contract: {key}"
            )
    search = protocol.get("weight_search", {})
    expected_search = {
        "components": list(COMPONENTS),
        "minimum_weight": 0.05,
        "step": 0.05,
        "sum": 1.0,
        "expected_grid_candidates": 171,
        "primary_metric": "auprc_average_precision_positive",
        "auprc_practical_tie": 0.005,
        "tie_break": ["mcc", "macro_f1", "brier", "distance_from_equal", "candidate_id"],
    }
    if search != expected_search:
        raise WeightedBlendPaper405Error("Weight-search contract differs from the lock")
    nested = protocol.get("nested_meta_cv", {})
    if nested != {
        "outer_folds": 5,
        "outer_seed": 314159,
        "inner_folds": 4,
        "inner_seed_base": 271828,
        "full_train_oof_folds": 5,
        "full_train_oof_seed": 42,
        "outer_inner_component_seed_base": 10_000,
        "outer_refit_component_seed_base": 20_000,
        "full_train_oof_component_seed_base": 42,
        "final_full_train_component_seed": 42,
    }:
        raise WeightedBlendPaper405Error("Nested meta-CV contract differs from the lock")
    threshold = protocol.get("thresholds", {})
    if threshold != {
        "primary": "maximize_mcc_on_active_train_oof",
        "screening": "maximize_mcc_subject_to_recall_and_specificity_floors",
        "screening_recall_floor": 0.50,
        "screening_specificity_floor": 0.75,
    }:
        raise WeightedBlendPaper405Error("Threshold contract differs from the lock")
    evaluation = protocol.get("evaluation", {})
    if evaluation != {
        "fit_components_once_on_full_324_train": True,
        "evaluate_every_grid_weight_on_81_test": True,
        "per_weight_retraining_required": False,
        "per_weight_predictions_are_deterministic_recombinations": True,
        "primary_weight_selected_before_test_evaluation": True,
        "outer_test_used_for_weight_or_threshold_selection": False,
        "test_sweep_role": "exploratory_multiple_comparison_table_not_model_selection",
        "hagr_loaded": False,
    }:
        raise WeightedBlendPaper405Error("Evaluation/firewall contract differs from the lock")
    return protocol, canonical_sha256(protocol)


def weight_grid(protocol: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    search = protocol["weight_search"]
    step = float(search["step"])
    minimum = float(search["minimum_weight"])
    units = round(1.0 / step)
    minimum_units = round(minimum / step)
    if not np.isclose(units * step, 1.0) or not np.isclose(minimum_units * step, minimum):
        raise WeightedBlendPaper405Error("Weight grid is not exactly representable in units")
    rows: list[dict[str, Any]] = []
    for svm_units in range(minimum_units, units - 2 * minimum_units + 1):
        for tanimoto_units in range(
            minimum_units, units - svm_units - minimum_units + 1
        ):
            tabpfn_units = units - svm_units - tanimoto_units
            if tabpfn_units < minimum_units:
                continue
            weights = np.asarray(
                [svm_units, tanimoto_units, tabpfn_units], dtype=np.float64
            ) / units
            candidate_id = (
                f"svm_{weights[0]:.2f}_tanimoto_{weights[1]:.2f}_tabpfn_{weights[2]:.2f}"
            )
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "weight_svm": float(weights[0]),
                    "weight_tanimoto": float(weights[1]),
                    "weight_tabpfn": float(weights[2]),
                }
            )
    if len(rows) != int(search["expected_grid_candidates"]):
        raise WeightedBlendPaper405Error(
            f"Weight grid has {len(rows)} candidates, expected 171"
        )
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise WeightedBlendPaper405Error("Weight candidate IDs are not unique")
    return tuple(rows)


def _validate_binary(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=np.float64)
    if (
        y.ndim != 1
        or p.ndim != 1
        or len(y) != len(p)
        or set(y.tolist()) != {0, 1}
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise WeightedBlendPaper405Error("Threshold input is not finite aligned binary data")
    return y.astype(int), p


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    recall_floor: float | None = None,
    specificity_floor: float | None = None,
) -> tuple[float, float] | None:
    """Exact deterministic MCC threshold, optionally under operational constraints."""
    y, p = _validate_binary(labels, probabilities)
    candidates = np.unique(np.r_[0.0, 0.5, 1.0, p])
    predictions = p[None, :] >= candidates[:, None]
    positive = y[None, :] == 1
    negative = ~positive
    tp = (predictions & positive).sum(axis=1)
    tn = ((~predictions) & negative).sum(axis=1)
    fp = (predictions & negative).sum(axis=1)
    fn = ((~predictions) & positive).sum(axis=1)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    feasible = np.ones(len(candidates), dtype=bool)
    if recall_floor is not None:
        feasible &= recall + 1e-15 >= float(recall_floor)
    if specificity_floor is not None:
        feasible &= specificity + 1e-15 >= float(specificity_floor)
    if not feasible.any():
        return None
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(
        tp * tn - fp * fn,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 0,
    )
    eligible = np.flatnonzero(feasible)
    # Mirrors the existing threshold contract: MCC, nearest 0.5, then lower threshold.
    winner = max(
        eligible,
        key=lambda index: (
            float(mcc[index]),
            -abs(float(candidates[index]) - 0.5),
            -float(candidates[index]),
        ),
    )
    return float(candidates[winner]), float(mcc[winner])


def _component_matrix(probabilities: dict[str, np.ndarray]) -> np.ndarray:
    if tuple(probabilities) != COMPONENTS:
        probabilities = {name: probabilities[name] for name in COMPONENTS}
    matrix = np.column_stack(
        [np.asarray(probabilities[name], dtype=float) for name in COMPONENTS]
    )
    if matrix.ndim != 2 or matrix.shape[1] != 3 or not np.isfinite(matrix).all():
        raise WeightedBlendPaper405Error("Three-component probability matrix is invalid")
    if ((matrix < 0) | (matrix > 1)).any():
        raise WeightedBlendPaper405Error("Component probabilities are outside [0,1]")
    return matrix


def select_weight_candidate(
    labels: np.ndarray,
    component_probabilities: dict[str, np.ndarray],
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    y = np.asarray(labels, dtype=int)
    matrix = _component_matrix(component_probabilities)
    if len(y) != len(matrix):
        raise WeightedBlendPaper405Error("Weight-selection inputs are misaligned")
    rows: list[dict[str, Any]] = []
    threshold_contract = protocol["thresholds"]
    for candidate in weight_grid(protocol):
        weights = np.asarray([candidate[column] for column in WEIGHT_COLUMNS], dtype=float)
        probability = matrix @ weights
        primary = select_threshold(y, probability)
        assert primary is not None
        primary_threshold, _ = primary
        metrics = evaluation_metrics(y, probability, probability, threshold=primary_threshold)
        screening = select_threshold(
            y,
            probability,
            recall_floor=float(threshold_contract["screening_recall_floor"]),
            specificity_floor=float(threshold_contract["screening_specificity_floor"]),
        )
        rows.append(
            {
                **candidate,
                **metrics,
                "distance_from_equal": float(np.sum((weights - 1.0 / 3.0) ** 2)),
                "screening_feasible": screening is not None,
                "screening_threshold": None if screening is None else screening[0],
            }
        )
    table = pd.DataFrame(rows)
    primary = str(protocol["weight_search"]["primary_metric"])
    best_primary = float(table[primary].max())
    practical_tie = float(protocol["weight_search"]["auprc_practical_tie"])
    eligible = table[table[primary] + 1e-15 >= best_primary - practical_tie].copy()
    winner = eligible.sort_values(
        ["mcc", "macro_f1", "brier", "distance_from_equal", "candidate_id"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    ).iloc[0]
    record = {
        "candidate_id": str(winner["candidate_id"]),
        "weight_svm": float(winner["weight_svm"]),
        "weight_tanimoto": float(winner["weight_tanimoto"]),
        "weight_tabpfn": float(winner["weight_tabpfn"]),
        "primary_threshold": float(winner["threshold"]),
        "screening_feasible": bool(winner["screening_feasible"]),
        "screening_threshold": (
            None
            if pd.isna(winner["screening_threshold"])
            else float(winner["screening_threshold"])
        ),
        "selection_auprc": float(winner[primary]),
        "selection_auroc": float(winner["auroc"]),
        "selection_mcc": float(winner["mcc"]),
        "selection_macro_f1": float(winner["macro_f1"]),
        "selection_brier": float(winner["brier"]),
        "best_grid_auprc": best_primary,
        "auprc_practical_tie": practical_tie,
        "eligible_candidates_in_tie_band": len(eligible),
        "selected_using_test_labels": False,
    }
    return table, record


def _prediction_binding(
    *,
    role: str,
    fit: np.ndarray,
    target: np.ndarray,
    seed: int,
    protocol_sha256: str,
) -> dict[str, Any]:
    return {
        "role": role,
        "fit_paper_indices": sorted(map(int, fit)),
        "target_paper_indices": sorted(map(int, target)),
        "seed": int(seed),
        "protocol_sha256": protocol_sha256,
        "components": list(COMPONENTS),
    }


def _load_sealed_component_prediction(
    directory: Path, expected_binding: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    marker_path = directory / "COMPLETED.json"
    if directory.is_symlink() or not marker_path.is_file() or marker_path.is_symlink():
        raise WeightedBlendPaper405Error(f"Component cache is not sealed: {directory}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("binding_sha256") != canonical_sha256(expected_binding):
        raise WeightedBlendPaper405Error(f"Component cache binding differs: {directory}")
    predictions_path = directory / "predictions.csv"
    audit_path = directory / "audit.json"
    for path, key in (
        (predictions_path, "predictions_sha256"),
        (audit_path, "audit_sha256"),
    ):
        if path.is_symlink() or not path.is_file() or sha256_file(path) != marker.get(key):
            raise WeightedBlendPaper405Error(f"Component cache failed integrity: {path}")
    for relative, expected_sha256 in marker.get("model_hashes", {}).items():
        model_path = directory / "models" / f"{relative}.joblib"
        if (
            model_path.is_symlink()
            or not model_path.is_file()
            or sha256_file(model_path) != expected_sha256
        ):
            raise WeightedBlendPaper405Error(
                f"Component model cache failed integrity: {model_path}"
            )
    predictions = pd.read_csv(predictions_path)
    expected_columns = ["paper_row_index", *[f"probability_{name}" for name in COMPONENTS]]
    if list(predictions) != expected_columns:
        raise WeightedBlendPaper405Error("Component cache columns differ from the lock")
    expected_target = list(map(int, expected_binding["target_paper_indices"]))
    if predictions.paper_row_index.tolist() != expected_target:
        raise WeightedBlendPaper405Error("Component cache target order differs")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    return predictions, audit, marker


def _sealed_component_prediction(
    *,
    directory: Path,
    role: str,
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    fit: np.ndarray,
    target: np.ndarray,
    settings: dict[str, Any],
    seed: int,
    protocol_sha256: str,
    save_models: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    binding = _prediction_binding(
        role=role,
        fit=fit,
        target=target,
        seed=seed,
        protocol_sha256=protocol_sha256,
    )
    if directory.exists():
        print(f"Resume verified component cache: {role}", flush=True)
        return _load_sealed_component_prediction(directory, binding)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.work-", dir=directory.parent))
    try:
        probabilities, models, audit = _selected_component_predictions(
            features,
            labels,
            fit,
            target,
            settings,
            seed,
            requested=COMPONENTS,
        )
        frame = pd.DataFrame(
            {
                "paper_row_index": list(map(int, target)),
                **{
                    f"probability_{name}": np.asarray(probabilities[name], dtype=float)
                    for name in COMPONENTS
                },
            }
        ).sort_values("paper_row_index", kind="mergesort")
        frame.to_csv(temporary / "predictions.csv", index=False, lineterminator="\n")
        atomic_write_json(temporary / "audit.json", {"binding": binding, "fit_audit": audit})
        model_hashes: dict[str, str] = {}
        if save_models:
            model_directory = temporary / "models"
            model_directory.mkdir()
            for name, model in models.items():
                path = model_directory / f"{name}.joblib"
                joblib.dump(model, path, compress=3)
                model_hashes[name] = sha256_file(path)
        marker = {
            "status": "COMPLETE",
            "binding_sha256": canonical_sha256(binding),
            "predictions_sha256": sha256_file(temporary / "predictions.csv"),
            "audit_sha256": sha256_file(temporary / "audit.json"),
            "model_hashes": model_hashes,
        }
        atomic_write_json(temporary / "COMPLETED.json", marker)
        os.rename(temporary, directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return _load_sealed_component_prediction(directory, binding)


def _wide_probabilities(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        name: frame[f"probability_{name}"].to_numpy(dtype=float) for name in COMPONENTS
    }


def _metrics_with_decisions(
    labels: np.ndarray, probability: np.ndarray, decision: np.ndarray
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probability, dtype=float)
    d = np.asarray(decision, dtype=int)
    if len(y) != len(p) or len(y) != len(d) or set(y) != {0, 1} or not set(d) <= {0, 1}:
        raise WeightedBlendPaper405Error("Nested meta metrics are not aligned binary data")
    tn, fp, fn, tp = confusion_matrix(y, d, labels=[0, 1]).ravel()
    return {
        "n": len(y),
        "auprc_average_precision_positive": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, d)),
        "balanced_accuracy": float(balanced_accuracy_score(y, d)),
        "mcc": float(matthews_corrcoef(y, d)),
        "macro_f1": float(f1_score(y, d, average="macro", zero_division=0)),
        "precision_positive": float(precision_score(y, d, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, d, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "cohen_kappa": float(cohen_kappa_score(y, d)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _run_nested_meta_cv(
    *,
    work: Path,
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    train_indices: np.ndarray,
    protocol: dict[str, Any],
    protocol_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    nested = protocol["nested_meta_cv"]
    outer = StratifiedKFold(
        n_splits=int(nested["outer_folds"]),
        shuffle=True,
        random_state=int(nested["outer_seed"]),
    )
    prediction_rows: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    search_rows: list[pd.DataFrame] = []
    lineage: list[dict[str, Any]] = []
    for outer_fold, (relative_fit, relative_validation) in enumerate(
        outer.split(train_indices, labels[train_indices])
    ):
        meta_train = train_indices[relative_fit]
        meta_validation = train_indices[relative_validation]
        inner = StratifiedKFold(
            n_splits=int(nested["inner_folds"]),
            shuffle=True,
            random_state=int(nested["inner_seed_base"]) + outer_fold,
        )
        inner_frames: list[pd.DataFrame] = []
        inner_lineage: list[dict[str, Any]] = []
        print(f"Nested weight selection outer fold {outer_fold + 1}/5", flush=True)
        for inner_fold, (inner_relative_fit, inner_relative_validation) in enumerate(
            inner.split(meta_train, labels[meta_train])
        ):
            print(
                f"  component inner fold {inner_fold + 1}/{nested['inner_folds']}",
                flush=True,
            )
            fit = meta_train[inner_relative_fit]
            validation = meta_train[inner_relative_validation]
            cache = work / "cache" / f"outer_{outer_fold}" / f"inner_{inner_fold}"
            frame, _audit, marker = _sealed_component_prediction(
                directory=cache,
                role=f"nested_outer_{outer_fold}_inner_{inner_fold}",
                features=features,
                labels=labels,
                fit=fit,
                target=validation,
                settings=protocol["components"],
                seed=(
                    int(nested["outer_inner_component_seed_base"])
                    + outer_fold * 100
                    + inner_fold
                ),
                protocol_sha256=protocol_sha256,
            )
            inner_frames.append(frame)
            inner_lineage.append(
                {
                    "inner_fold": inner_fold,
                    "fit_paper_indices": sorted(map(int, fit)),
                    "validation_paper_indices": sorted(map(int, validation)),
                    "cache_marker_sha256": canonical_sha256(marker),
                }
            )
        inner_oof = pd.concat(inner_frames, ignore_index=True).sort_values(
            "paper_row_index", kind="mergesort"
        )
        if set(inner_oof.paper_row_index) != set(map(int, meta_train)) or inner_oof[
            "paper_row_index"
        ].duplicated().any():
            raise WeightedBlendPaper405Error("Nested inner OOF coverage is incomplete")
        inner_y = labels[inner_oof.paper_row_index.to_numpy(dtype=int)]
        search, winner = select_weight_candidate(
            inner_y, _wide_probabilities(inner_oof), protocol
        )
        search.insert(0, "outer_fold", outer_fold)
        search_rows.append(search)
        print("  refit outer-meta train and predict held fold", flush=True)
        outer_cache = work / "cache" / f"outer_{outer_fold}" / "outer_validation"
        outer_frame, _outer_audit, outer_marker = _sealed_component_prediction(
            directory=outer_cache,
            role=f"nested_outer_{outer_fold}_validation",
            features=features,
            labels=labels,
            fit=meta_train,
            target=meta_validation,
            settings=protocol["components"],
            seed=int(nested["outer_refit_component_seed_base"]) + outer_fold,
            protocol_sha256=protocol_sha256,
        )
        weights = np.asarray([winner[column] for column in WEIGHT_COLUMNS], dtype=float)
        probability = _component_matrix(_wide_probabilities(outer_frame)) @ weights
        primary_decision = probability >= float(winner["primary_threshold"])
        screening_threshold = winner["screening_threshold"]
        screening_decision = (
            np.zeros(len(probability), dtype=bool)
            if screening_threshold is None
            else probability >= float(screening_threshold)
        )
        rows = outer_frame.copy()
        rows["label"] = labels[rows.paper_row_index.to_numpy(dtype=int)]
        rows["outer_fold"] = outer_fold
        rows["candidate_id"] = winner["candidate_id"]
        for column in WEIGHT_COLUMNS:
            rows[column] = winner[column]
        rows["probability"] = probability
        rows["primary_threshold"] = winner["primary_threshold"]
        rows["primary_decision"] = primary_decision.astype(int)
        rows["screening_feasible"] = winner["screening_feasible"]
        rows["screening_threshold"] = screening_threshold
        rows["screening_decision"] = screening_decision.astype(int)
        prediction_rows.append(rows)
        selection_rows.append({"outer_fold": outer_fold, **winner})
        lineage.append(
            {
                "outer_fold": outer_fold,
                "meta_train_paper_indices": sorted(map(int, meta_train)),
                "meta_validation_paper_indices": sorted(map(int, meta_validation)),
                "inner_folds": inner_lineage,
                "outer_prediction_cache_marker_sha256": canonical_sha256(outer_marker),
            }
        )
    predictions = pd.concat(prediction_rows, ignore_index=True).sort_values(
        "paper_row_index", kind="mergesort"
    )
    if (
        set(predictions.paper_row_index) != set(map(int, train_indices))
        or predictions.paper_row_index.duplicated().any()
    ):
        raise WeightedBlendPaper405Error("Nested meta validation coverage is incomplete")
    primary_metrics = _metrics_with_decisions(
        predictions.label.to_numpy(dtype=int),
        predictions.probability.to_numpy(dtype=float),
        predictions.primary_decision.to_numpy(dtype=int),
    )
    screening = predictions[predictions.screening_feasible]
    screening_metrics = (
        None
        if len(screening) != len(predictions)
        else _metrics_with_decisions(
            screening.label.to_numpy(dtype=int),
            screening.probability.to_numpy(dtype=float),
            screening.screening_decision.to_numpy(dtype=int),
        )
    )
    metrics = {
        "role": "honest_nested_train_only_weight_selection_estimate",
        "primary": primary_metrics,
        "screening": screening_metrics,
        "outer_test_rows_or_labels_used": False,
    }
    return (
        predictions,
        pd.DataFrame(selection_rows),
        pd.concat(search_rows, ignore_index=True),
        metrics,
        lineage,
    )


def _run_full_train_selection(
    *,
    work: Path,
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    train_indices: np.ndarray,
    protocol: dict[str, Any],
    protocol_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    nested = protocol["nested_meta_cv"]
    folds = StratifiedKFold(
        n_splits=int(nested["full_train_oof_folds"]),
        shuffle=True,
        random_state=int(nested["full_train_oof_seed"]),
    )
    frames: list[pd.DataFrame] = []
    lineage: list[dict[str, Any]] = []
    for fold, (relative_fit, relative_validation) in enumerate(
        folds.split(train_indices, labels[train_indices])
    ):
        fit = train_indices[relative_fit]
        validation = train_indices[relative_validation]
        print(f"Full-train component OOF fold {fold + 1}/5", flush=True)
        frame, _audit, marker = _sealed_component_prediction(
            directory=work / "cache" / "full_train_oof" / f"fold_{fold}",
            role=f"full_train_oof_{fold}",
            features=features,
            labels=labels,
            fit=fit,
            target=validation,
            settings=protocol["components"],
            seed=int(nested["full_train_oof_component_seed_base"]) + fold,
            protocol_sha256=protocol_sha256,
        )
        frames.append(frame)
        lineage.append(
            {
                "fold": fold,
                "fit_paper_indices": sorted(map(int, fit)),
                "validation_paper_indices": sorted(map(int, validation)),
                "cache_marker_sha256": canonical_sha256(marker),
            }
        )
    oof = pd.concat(frames, ignore_index=True).sort_values(
        "paper_row_index", kind="mergesort"
    )
    if set(oof.paper_row_index) != set(map(int, train_indices)) or oof[
        "paper_row_index"
    ].duplicated().any():
        raise WeightedBlendPaper405Error("Full-train component OOF coverage is incomplete")
    train_y = labels[oof.paper_row_index.to_numpy(dtype=int)]
    search, winner = select_weight_candidate(train_y, _wide_probabilities(oof), protocol)
    return oof, search, winner, lineage


def _write_selection_bundle(
    *,
    directory: Path,
    oof: pd.DataFrame,
    search: pd.DataFrame,
    winner: dict[str, Any],
    protocol_sha256: str,
    nested_lineage_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if directory.exists():
        if directory.is_symlink():
            raise WeightedBlendPaper405Error("Existing selection bundle is a symlink")
        marker = json.loads((directory / "SELECTION_LOCK.json").read_text(encoding="utf-8"))
        for name, key in (
            ("full_train_oof_component_predictions.csv", "oof_sha256"),
            ("full_train_weight_search.csv", "search_sha256"),
        ):
            path = directory / name
            if path.is_symlink() or not path.is_file() or sha256_file(path) != marker.get(key):
                raise WeightedBlendPaper405Error("Existing selection bundle failed integrity")
        if marker.get("protocol_sha256") != protocol_sha256:
            raise WeightedBlendPaper405Error("Existing selection bundle protocol differs")
        if marker.get("winner") != winner:
            raise WeightedBlendPaper405Error("Existing selection winner differs on resume")
        if marker.get("nested_lineage_sha256") != nested_lineage_sha256:
            raise WeightedBlendPaper405Error("Existing selection lineage differs on resume")
        expected_oof_sha256 = sha256_bytes(
            oof.to_csv(index=False, lineterminator="\n").encode("utf-8")
        )
        expected_search_sha256 = sha256_bytes(
            search.to_csv(index=False, lineterminator="\n").encode("utf-8")
        )
        if (
            marker.get("oof_sha256") != expected_oof_sha256
            or marker.get("search_sha256") != expected_search_sha256
        ):
            raise WeightedBlendPaper405Error(
                "Existing selection bytes differ from deterministic recomputation"
            )
        return (
            pd.read_csv(directory / "full_train_oof_component_predictions.csv"),
            pd.read_csv(directory / "full_train_weight_search.csv"),
            marker,
        )
    temporary = Path(tempfile.mkdtemp(prefix=".selection.work-", dir=directory.parent))
    try:
        oof.to_csv(
            temporary / "full_train_oof_component_predictions.csv",
            index=False,
            lineterminator="\n",
        )
        search.to_csv(
            temporary / "full_train_weight_search.csv", index=False, lineterminator="\n"
        )
        lock = {
            "schema_version": "geroprotector.weighted_blend_paper405.selection.v1",
            "protocol_sha256": protocol_sha256,
            "winner": winner,
            "oof_sha256": sha256_file(
                temporary / "full_train_oof_component_predictions.csv"
            ),
            "search_sha256": sha256_file(temporary / "full_train_weight_search.csv"),
            "nested_lineage_sha256": nested_lineage_sha256,
            "outer_test_rows_or_labels_used": False,
            "immutable_before_outer_test_evaluation": True,
        }
        atomic_write_json(temporary / "SELECTION_LOCK.json", lock)
        os.rename(temporary, directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return oof, search, lock


def _test_sweep(
    *,
    labels: np.ndarray,
    test_frame: pd.DataFrame,
    search: pd.DataFrame,
    winner: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    matrix = _component_matrix(_wide_probabilities(test_frame))
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for row in search.itertuples(index=False):
        weights = np.asarray(
            [row.weight_svm, row.weight_tanimoto, row.weight_tabpfn], dtype=float
        )
        probability = matrix @ weights
        metrics = evaluation_metrics(
            labels, probability, probability, threshold=float(row.threshold)
        )
        screening_threshold = (
            None if pd.isna(row.screening_threshold) else float(row.screening_threshold)
        )
        screening_metrics: dict[str, Any] = {}
        if screening_threshold is not None:
            screening_metrics = {
                f"screening_{key}": value
                for key, value in evaluation_metrics(
                    labels,
                    probability,
                    probability,
                    threshold=screening_threshold,
                ).items()
            }
        metric_rows.append(
            {
                "candidate_id": row.candidate_id,
                "weight_svm": row.weight_svm,
                "weight_tanimoto": row.weight_tanimoto,
                "weight_tabpfn": row.weight_tabpfn,
                "selected_primary_candidate": row.candidate_id == winner["candidate_id"],
                "test_used_for_selection": False,
                **metrics,
                "screening_feasible": screening_threshold is not None,
                "screening_threshold": screening_threshold,
                **screening_metrics,
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "candidate_id": row.candidate_id,
                    "paper_row_index": test_frame.paper_row_index.to_numpy(dtype=int),
                    "label": labels,
                    "weight_svm": row.weight_svm,
                    "weight_tanimoto": row.weight_tanimoto,
                    "weight_tabpfn": row.weight_tabpfn,
                    "probability": probability,
                    "primary_threshold": float(row.threshold),
                    "primary_decision": (probability >= float(row.threshold)).astype(int),
                    "screening_threshold": screening_threshold,
                    "screening_decision": (
                        np.nan
                        if screening_threshold is None
                        else (probability >= screening_threshold).astype(int)
                    ),
                }
            )
        )
    metrics_table = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    selected = metrics_table[metrics_table.selected_primary_candidate]
    if len(selected) != 1:
        raise WeightedBlendPaper405Error("Exactly one train-selected primary must be present")
    primary_metrics = {
        key: None if pd.isna(value) else value
        for key, value in selected.iloc[0].to_dict().items()
    }
    equal_probability = matrix @ np.full(3, 1.0 / 3.0)
    # Equal gets its own train-only threshold from exact equal OOF outside this helper.
    anchor = pd.DataFrame(
        {
            "paper_row_index": test_frame.paper_row_index.to_numpy(dtype=int),
            "label": labels,
            "probability": equal_probability,
        }
    )
    return metrics_table, predictions, primary_metrics, anchor, {
        "evaluated_candidates": len(metrics_table),
        "test_sweep_used_to_change_primary": False,
        "test_sweep_role": "exploratory_multiple_comparison_table_not_model_selection",
    }


def _summary(
    run_id: str,
    primary: dict[str, Any],
    nested_metrics: dict[str, Any],
    equal_metrics: dict[str, Any],
    conservativeness: dict[str, Any],
) -> str:
    delta = conservativeness["primary_minus_equal"]
    return "\n".join(
        [
            f"# Nested weighted blend paper-405 — {run_id}",
            "",
            "Primary weights and thresholds were locked from nested/full-train OOF before "
            "outer-test evaluation. The 171-row test sweep is exploratory and must not be used "
            "to replace the train-selected primary.",
            "",
            "## Train-selected primary on the 81-row paper test",
            "",
            "| SVM | Tanimoto | TabPFN | AUPRC+ | AUROC | MCC | Macro F1 | "
            "Recall | Specificity | TN/FP/FN/TP |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| {primary['weight_svm']:.2f} | {primary['weight_tanimoto']:.2f} | "
            f"{primary['weight_tabpfn']:.2f} | "
            f"{primary['auprc_average_precision_positive']:.4f} | "
            f"{primary['auroc']:.4f} | {primary['mcc']:.4f} | "
            f"{primary['macro_f1']:.4f} | {primary['recall_sensitivity']:.4f} | "
            f"{primary['specificity']:.4f} | "
            f"{primary['tn']}/{primary['fp']}/{primary['fn']}/{primary['tp']} |",
            "",
            "## Exact equal-weight anchor on the same test",
            "",
            f"AUPRC={equal_metrics['auprc_average_precision_positive']:.4f}; "
            f"AUROC={equal_metrics['auroc']:.4f}; MCC={equal_metrics['mcc']:.4f}; "
            f"Macro F1={equal_metrics['macro_f1']:.4f}; "
            f"recall={equal_metrics['recall_sensitivity']:.4f}; "
            f"specificity={equal_metrics['specificity']:.4f}; "
            f"TN/FP/FN/TP={equal_metrics['tn']}/{equal_metrics['fp']}/"
            f"{equal_metrics['fn']}/{equal_metrics['tp']}.",
            "",
            "## Conservativeness check versus exact equal weights",
            "",
            f"Recall delta={delta['recall_sensitivity']:+.4f}; "
            f"specificity delta={delta['specificity']:+.4f}; "
            f"TP delta={delta['tp']:+d}; FP delta={delta['fp']:+d}; "
            f"FN delta={delta['fn']:+d}; TN delta={delta['tn']:+d}. "
            f"Prespecified descriptive guard passed="
            f"{conservativeness['descriptive_guard']['passed_all']}.",
            "",
            "## Nested train-only selection estimate",
            "",
            f"AUPRC={nested_metrics['primary']['auprc_average_precision_positive']:.4f}; "
            f"AUROC={nested_metrics['primary']['auroc']:.4f}; "
            f"MCC={nested_metrics['primary']['mcc']:.4f}; "
            f"Macro F1={nested_metrics['primary']['macro_f1']:.4f}; "
            f"recall={nested_metrics['primary']['recall_sensitivity']:.4f}; "
            f"specificity={nested_metrics['primary']['specificity']:.4f}.",
            "",
            "## Claim boundary",
            "",
            "This remains a contextual result on the already-observed raw-paper random split. "
            "Source and label remain perfectly confounded, chemical overlap is retained, "
            "and no "
            "row of the 171-candidate test table may be promoted post hoc as a new winner.",
            "",
        ]
    )


def _conservativeness_comparison(
    primary: dict[str, Any], equal: dict[str, Any]
) -> dict[str, Any]:
    metric_names = (
        "auprc_average_precision_positive",
        "auroc",
        "mcc",
        "macro_f1",
        "balanced_accuracy",
        "recall_sensitivity",
        "specificity",
        "precision_positive",
        "npv",
        "tn",
        "fp",
        "fn",
        "tp",
    )
    delta = {name: primary[name] - equal[name] for name in metric_names}
    primary_rates = {
        "false_negative_rate": 1.0 - float(primary["recall_sensitivity"]),
        "false_positive_rate": 1.0 - float(primary["specificity"]),
        "predicted_positive_fraction": float(
            (primary["tp"] + primary["fp"]) / primary["n_test"]
        ),
    }
    equal_rates = {
        "false_negative_rate": 1.0 - float(equal["recall_sensitivity"]),
        "false_positive_rate": 1.0 - float(equal["specificity"]),
        "predicted_positive_fraction": float(
            (equal["tp"] + equal["fp"]) / equal["n_test"]
        ),
    }
    checks = {
        "recall_improves_by_at_least_0p05": delta["recall_sensitivity"] >= 0.05,
        "false_negatives_decrease": delta["fn"] < 0,
        "specificity_at_least_0p80": primary["specificity"] >= 0.80,
        "mcc_not_lower": delta["mcc"] >= 0.0,
        "macro_f1_not_lower": delta["macro_f1"] >= 0.0,
        "auprc_not_lower_by_more_than_0p005": (
            delta["auprc_average_precision_positive"] >= -0.005
        ),
    }
    return {
        "role": "descriptive_outer_test_monitor_not_model_selection",
        "primary": {"metrics": primary, **primary_rates},
        "equal_weight_anchor": {"metrics": equal, **equal_rates},
        "primary_minus_equal": delta,
        "descriptive_guard": {**checks, "passed_all": all(checks.values())},
        "outer_test_used_to_change_weights_or_thresholds": False,
    }


def run(
    *,
    root: Path,
    config_path: Path,
    positive_path: Path,
    negative_path: Path,
    run_id: str,
) -> Path:
    if not re.fullmatch(r"weightedblend405_[a-z0-9_.-]+", run_id):
        raise WeightedBlendPaper405Error("RUN_ID must start with weightedblend405_")
    root = root.resolve()
    validate_core_runtime(root / "requirements-lock.txt")
    positive_path = _regular_file(positive_path, "positive source")
    negative_path = _regular_file(negative_path, "negative source")
    protocol, protocol_sha256 = load_protocol(config_path)
    source_hash = source_tree_sha256(root)
    traditional = _paper_contract(root, protocol)
    frame, source_audit = _read_sources(positive_path, negative_path, traditional)
    train_indices, test_indices, split_hash = paper_split_indices(traditional)
    smiles, smiles_audit = _validated_raw_smiles(frame)
    features = _features(frame, smiles, protocol)
    labels = frame.label.to_numpy(dtype=int)
    output_root = root / "outputs"
    if output_root.is_symlink():
        raise WeightedBlendPaper405Error("Output root must not be a symlink")
    output_root.mkdir(exist_ok=True)
    destination = output_root / run_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite weighted-blend run: {destination}")
    work = output_root / f".{run_id}.work"
    running = {
        "schema_version": "geroprotector.weighted_blend_paper405.running.v1",
        "run_id": run_id,
        "protocol_sha256": protocol_sha256,
        "source_tree_sha256": source_hash,
        "paper_split_sha256": split_hash,
        "positive_source_sha256": sha256_file(positive_path),
        "negative_source_sha256": sha256_file(negative_path),
    }
    if work.exists():
        running_path = work / "RUNNING.json"
        if (
            work.is_symlink()
            or not running_path.is_file()
            or json.loads(running_path.read_text(encoding="utf-8")) != running
        ):
            raise WeightedBlendPaper405Error("Existing resume workspace binding differs")
        print(f"Resuming verified weighted-blend workspace: {work}", flush=True)
    else:
        work.mkdir()
        atomic_write_json(work / "RUNNING.json", running)
    nested_predictions, nested_selections, nested_search, nested_metrics, nested_lineage = (
        _run_nested_meta_cv(
            work=work,
            features=features,
            labels=labels,
            train_indices=train_indices,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
        )
    )
    oof, search, winner, full_oof_lineage = _run_full_train_selection(
        work=work,
        features=features,
        labels=labels,
        train_indices=train_indices,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    selection_directory = work / "selection"
    oof, search, selection_lock = _write_selection_bundle(
        directory=selection_directory,
        oof=oof,
        search=search,
        winner=winner,
        protocol_sha256=protocol_sha256,
        nested_lineage_sha256=canonical_sha256(nested_lineage),
    )
    winner = selection_lock["winner"]
    print(
        "Train-only primary locked: "
        f"SVM={winner['weight_svm']:.2f}, "
        f"Tanimoto={winner['weight_tanimoto']:.2f}, "
        f"TabPFN={winner['weight_tabpfn']:.2f}, "
        f"threshold={winner['primary_threshold']:.6f}",
        flush=True,
    )
    # From this line onward test identities may be predicted and test labels evaluated.
    print("Selection lock sealed; final component fit on all 324 paper-train rows", flush=True)
    test_components, final_audit, final_marker = _sealed_component_prediction(
        directory=work / "cache" / "final_full_train_to_test",
        role="final_full_324_train_to_81_test",
        features=features,
        labels=labels,
        fit=train_indices,
        target=test_indices,
        settings=protocol["components"],
        seed=int(protocol["nested_meta_cv"]["final_full_train_component_seed"]),
        protocol_sha256=protocol_sha256,
        save_models=True,
    )
    test_labels = labels[test_components.paper_row_index.to_numpy(dtype=int)]
    sweep_metrics, sweep_predictions, primary_metrics, equal_predictions, sweep_audit = (
        _test_sweep(
            labels=test_labels,
            test_frame=test_components,
            search=search,
            winner=winner,
        )
    )
    equal_oof_probability = _component_matrix(_wide_probabilities(oof)) @ np.full(3, 1 / 3)
    equal_threshold = select_threshold(
        labels[oof.paper_row_index.to_numpy(dtype=int)], equal_oof_probability
    )
    assert equal_threshold is not None
    equal_probability = equal_predictions.probability.to_numpy(dtype=float)
    equal_metrics = evaluation_metrics(
        test_labels,
        equal_probability,
        equal_probability,
        threshold=equal_threshold[0],
    )
    equal_predictions["threshold"] = equal_threshold[0]
    equal_predictions["decision"] = (equal_probability >= equal_threshold[0]).astype(int)
    conservativeness = _conservativeness_comparison(primary_metrics, equal_metrics)
    publish = Path(tempfile.mkdtemp(prefix=f".{run_id}.publish-", dir=output_root))
    try:
        (publish / "models").mkdir()
        shutil.copytree(
            work / "cache" / "final_full_train_to_test" / "models",
            publish / "models" / "components",
        )
        shutil.copy2(work / "RUNNING.json", publish / "RUNNING.json")
        shutil.copy2(
            selection_directory / "SELECTION_LOCK.json", publish / "SELECTION_LOCK.json"
        )
        oof.to_csv(
            publish / "full_train_oof_component_predictions.csv",
            index=False,
            lineterminator="\n",
        )
        search.to_csv(
            publish / "full_train_weight_search.csv", index=False, lineterminator="\n"
        )
        nested_predictions.to_csv(
            publish / "nested_meta_predictions.csv", index=False, lineterminator="\n"
        )
        nested_selections.to_csv(
            publish / "nested_meta_fold_selections.csv", index=False, lineterminator="\n"
        )
        nested_search.to_csv(
            publish / "nested_meta_weight_search.csv", index=False, lineterminator="\n"
        )
        atomic_write_json(publish / "nested_meta_metrics.json", nested_metrics)
        test_components.assign(label=test_labels).to_csv(
            publish / "test_component_predictions.csv", index=False, lineterminator="\n"
        )
        sweep_metrics.to_csv(
            publish / "test_weight_sweep_metrics.csv", index=False, lineterminator="\n"
        )
        sweep_predictions.to_csv(
            publish / "test_weight_sweep_predictions.csv", index=False, lineterminator="\n"
        )
        equal_predictions.to_csv(
            publish / "equal_weight_anchor_predictions.csv", index=False, lineterminator="\n"
        )
        atomic_write_json(publish / "primary_metrics.json", primary_metrics)
        atomic_write_json(publish / "equal_weight_anchor_metrics.json", equal_metrics)
        atomic_write_json(
            publish / "conservativeness_comparison.json", conservativeness
        )
        split_frame = frame[
            ["paper_row_index", "compound_name", "smiles", "label", "source_role"]
        ].copy()
        split_frame["role"] = "paper_train"
        split_frame.loc[split_frame.paper_row_index.isin(test_indices), "role"] = "paper_test"
        split_frame.to_csv(publish / "split_registry.csv", index=False, lineterminator="\n")
        audit = {
            **source_audit,
            **smiles_audit,
            "paper_split_sha256": split_hash,
            "paper_train_rows": len(train_indices),
            "paper_test_rows": len(test_indices),
            "cross_split_overlap_audit": _cross_split_audit(
                frame, train_indices, test_indices
            ),
            "nested_meta_lineage": nested_lineage,
            "nested_meta_lineage_sha256": canonical_sha256(nested_lineage),
            "full_train_oof_lineage": full_oof_lineage,
            "full_train_oof_lineage_sha256": canonical_sha256(full_oof_lineage),
            "selection_lock_sha256": sha256_file(publish / "SELECTION_LOCK.json"),
            "final_component_fit_audit": final_audit,
            "final_component_cache_marker_sha256": canonical_sha256(final_marker),
            "full_train_component_fit_count": 1,
            "per_weight_component_retraining": False,
            "per_weight_predictions_are_mathematically_exact_recombinations": True,
            "weight_grid_candidates": len(search),
            "test_sweep_audit": sweep_audit,
            "primary_weight_or_threshold_selected_from_test": False,
            "test_sweep_is_exploratory_only": True,
            "hagr_loaded": False,
        }
        atomic_write_json(publish / "audit.json", audit)
        (publish / "summary.md").write_text(
            _summary(
                run_id,
                primary_metrics,
                nested_metrics,
                equal_metrics,
                conservativeness,
            ),
            encoding="utf-8",
        )
        if source_tree_sha256(root) != source_hash:
            raise WeightedBlendPaper405Error("Source tree changed during weighted-blend run")
        artifact_names = (
            "SELECTION_LOCK.json",
            "full_train_oof_component_predictions.csv",
            "full_train_weight_search.csv",
            "nested_meta_predictions.csv",
            "nested_meta_fold_selections.csv",
            "nested_meta_weight_search.csv",
            "nested_meta_metrics.json",
            "test_component_predictions.csv",
            "test_weight_sweep_metrics.csv",
            "test_weight_sweep_predictions.csv",
            "equal_weight_anchor_predictions.csv",
            "primary_metrics.json",
            "equal_weight_anchor_metrics.json",
            "conservativeness_comparison.json",
            "split_registry.csv",
            "audit.json",
            "summary.md",
        )
        artifact_hashes = {name: sha256_file(publish / name) for name in artifact_names}
        model_hashes = {
            path.relative_to(publish).as_posix(): sha256_file(path)
            for path in sorted((publish / "models").rglob("*"))
            if path.is_file() and not path.is_symlink()
        }
        manifest = {
            "schema_version": "geroprotector.weighted_blend_paper405.run.v1",
            "run_id": run_id,
            "protocol_sha256": protocol_sha256,
            "source_tree_sha256": source_hash,
            "runtime_environment": runtime_environment(),
            "paper_split_sha256": split_hash,
            "artifact_hashes": artifact_hashes,
            "model_hashes": model_hashes,
            "selected_primary_candidate": winner["candidate_id"],
            "outer_test_used_for_weight_or_threshold_selection": False,
            "all_171_test_metrics_are_exploratory": True,
        }
        atomic_write_json(publish / "run_manifest.json", manifest)
        atomic_write_json(
            publish / "COMPLETED.json",
            {
                "status": "COMPLETE",
                "run_id": run_id,
                "run_manifest_sha256": sha256_file(publish / "run_manifest.json"),
                "selection_lock_sha256": artifact_hashes["SELECTION_LOCK.json"],
                "primary_metrics_sha256": artifact_hashes["primary_metrics.json"],
                "test_weight_sweep_metrics_sha256": artifact_hashes[
                    "test_weight_sweep_metrics.csv"
                ],
            },
        )
        os.rename(publish, destination)
    finally:
        if publish.exists():
            shutil.rmtree(publish)
    shutil.rmtree(work)
    print(f"Complete: {destination}", flush=True)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--positive", required=True)
    parser.add_argument("--negative", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    run(
        root=Path(args.root),
        config_path=_regular_file(Path(args.config), "protocol"),
        positive_path=_regular_file(Path(args.positive), "positive source"),
        negative_path=_regular_file(Path(args.negative), "negative source"),
        run_id=args.run_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
