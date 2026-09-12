"""Fold-safe TabPFN-3 selection and evaluation under the locked paper-405 protocol.

The sealed 2026-08-17/18 runs used `tabpfn` ModelVersion.**V2** with
`tabpfn-v2-classifier.ckpt`.  V2_5, V2_6 and V3 were declared in `configs/v6.yaml`
but never executed: the V6BIS run aborted at plan time on unresolved checkpoint
placeholders.  This module is the first TabPFN-3 evaluation.

Fold safety.  Candidate selection is *nested*: inside every outer OOF fold the
candidate grid is scored by an inner StratifiedKFold restricted to that fold's fit
rows, the winner is refitted on those fit rows, and only then are the held-out outer
rows predicted.  The outer validation rows therefore never influence which candidate
produced their own OOF probability.  A separate final selection over all 324 training
rows produces the single configuration that scores D1 test and both external cohorts.

Nothing in this module writes to, reads through, or re-derives an existing run.  Every
sealed input is hash-verified and opened read-only, and the run refuses to start if its
own output directory already exists.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from geroprotector.fixed_blend_paper405 import (
    _features,
    _selected_component_predictions,
    _tanimoto,
)
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.screening_blend_ablation import (
    _safe_metrics,
    _svc_probability_at_native_boundary,
)
from geroprotector.screening_blend_altmodels import (
    _apply_context,
    _imputer_context,
    _morgan_from_smiles,
    _rdkit2d_from_smiles,
)
from geroprotector.screening_blend_paper405 import load_locked_bundle
from geroprotector.traditional_paper405 import _read_sources, paper_split_indices
from geroprotector.weighted_blend_paper405 import select_threshold


class TabPFN3Paper405Error(RuntimeError):
    """Raised when a sealed input, a parity proof or a leakage contract fails."""


SCHEMA = "geroprotector.tabpfn3_paper405"
CHEMISTRY_COMPONENTS = ("paper_svm", "tanimoto_svc")
D1_ENDPOINT = "paper_binary"
DRUGAGE_ENDPOINT = "significant_positive_retrieval_background_not_certified_negative"
AGEXTEND_ENDPOINT = "published_independent_table6_binary"
OOF_MCC_SOURCE = "full_324_d1_train_cross_fitted_oof_mcc"
PLACEHOLDER = "REQUIRED_AT_RUNTIME"


# --------------------------------------------------------------------------- io


def _regular_file(path: Path, role: str, expected_sha256: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise TabPFN3Paper405Error(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected_sha256 is not None and sha256_file(resolved) != expected_sha256:
        raise TabPFN3Paper405Error(f"{role} SHA256 differs from the protocol lock")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def _json_safe(value: Any) -> Any:
    """YAML turns bare ISO dates into date objects; the protocol hasher needs JSON."""

    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json_safe(
        yaml.safe_load(_regular_file(path, "TabPFN-3 protocol").read_text(encoding="utf-8"))
    )
    expected_schema = f"{SCHEMA}.protocol.v1"
    if not isinstance(protocol, dict) or protocol.get("schema_version") != expected_schema:
        raise TabPFN3Paper405Error("Unknown TabPFN-3 protocol schema")

    model = protocol.get("model", {})
    if model.get("package") != "tabpfn":
        raise TabPFN3Paper405Error("Protocol does not declare the tabpfn package")
    if str(model.get("explicit_model_version")) not in {"V2", "V2_5", "V2_6", "V3"}:
        raise TabPFN3Paper405Error(
            "explicit_model_version must be one of V2, V2_5, V2_6, V3"
        )
    unresolved = [
        key
        for key in (
            "checkpoint_path",
            "checkpoint_sha256",
            "checkpoint_source",
            "access_date_utc",
        )
        if str(model.get(key)) == PLACEHOLDER
    ]
    if unresolved:
        raise TabPFN3Paper405Error(
            "Resolve the checkpoint placeholders first by running "
            "scripts/stage_tabpfn3_checkpoint.sh, then paste its output into "
            f"the protocol: {', '.join(unresolved)}"
        )

    cross = protocol.get("cross_fitting", {})
    if (
        cross.get("outer_validation_rows_used_for_selection") is not False
        or cross.get("test_or_external_labels_used_for_selection") is not False
        or cross.get("selection_scope") != "outer_fit_rows_only"
    ):
        raise TabPFN3Paper405Error("Fold-safety contract differs")
    if protocol.get("blends", {}).get("weights_selected_from_data_in_this_run") is not False:
        raise TabPFN3Paper405Error("Blend weights must stay prespecified")
    if (
        protocol.get("thresholds", {}).get("external_labels_may_select_or_change_threshold")
        is not False
    ):
        raise TabPFN3Paper405Error("Threshold leakage contract differs")
    parity = protocol.get("parity_contract", {})
    if (
        parity.get("rebuilt_paper_svm_and_tanimoto_streams_must_match_sealed_run") is not True
        or parity.get("rebuilt_rdkit2d_context_must_equal_sealed_tabpfn_context") is not True
        or parity.get("rebuilt_morgan_bits_must_equal_sealed_tanimoto_train_bits") is not True
    ):
        raise TabPFN3Paper405Error("Parity contract differs")
    immutability = protocol.get("immutability", {})
    if immutability.get("existing_run_directories_are_read_only") is not True:
        raise TabPFN3Paper405Error("Immutability contract differs")

    for record in protocol.get("sealed_inputs", {}).values():
        _regular_file(root / record["path"], f"sealed input {record['path']}", record["sha256"])
    return protocol, canonical_sha256(protocol)


# ------------------------------------------------------------------ TabPFN-3 core


def _tabpfn3_estimator(model: dict[str, Any], n_estimators: int, seed: int):
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "true")
    if os.environ.get("TABPFN_DISABLE_TELEMETRY") not in {"true", "1"}:
        raise TabPFN3Paper405Error("TabPFN telemetry is not disabled")
    observed = importlib.metadata.version("tabpfn")
    if observed != str(model["required_package_version"]):
        raise TabPFN3Paper405Error(
            f"Installed tabpfn {observed} differs from the protocol lock "
            f"{model['required_package_version']}"
        )
    checkpoint = _regular_file(
        Path(model["checkpoint_path"]), "TabPFN-3 checkpoint", model["checkpoint_sha256"]
    )
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    requested = str(model["explicit_model_version"])
    if requested not in {"V2", "V2_5", "V2_6", "V3"}:
        raise TabPFN3Paper405Error(f"Unsupported TabPFN model version: {requested}")
    return TabPFNClassifier.create_default_for_version(
        getattr(ModelVersion, requested),
        model_path=str(checkpoint),
        device=str(model["device"]),
        n_estimators=int(n_estimators),
        softmax_temperature=float(model["softmax_temperature"]),
        random_state=int(seed),
    )


def _positive_probability(
    estimator, X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, *, singleton: bool
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the in-context model and score the query rows.

    ``singleton`` reproduces the sealed run's canonical one-query-per-call mode, which
    removes any dependence of a prediction on the composition or order of its batch.
    """

    estimator.fit(X_train, y_train)
    classes = np.asarray(estimator.classes_)
    if classes.shape != (2,) or set(map(int, classes)) != {0, 1}:
        raise TabPFN3Paper405Error("TabPFN-3 returned unexpected classes")
    positive = int(np.flatnonzero(classes == 1)[0])

    if not singleton:
        raw = np.asarray(estimator.predict_proba(X_query), dtype=float)
        if raw.shape != (len(X_query), 2) or not np.isfinite(raw).all():
            raise TabPFN3Paper405Error("TabPFN-3 batch prediction is invalid")
        return np.clip(raw[:, positive], 1e-7, 1 - 1e-7), {"mode": "batched"}

    order = sorted(range(len(X_query)), key=lambda i: X_query[i].astype("<f8").tobytes())
    probability = np.empty(len(X_query), dtype=np.float64)
    for index in order:
        raw = np.asarray(estimator.predict_proba(X_query[index : index + 1]), dtype=float)
        if raw.shape != (1, 2) or not np.isfinite(raw).all():
            raise TabPFN3Paper405Error("TabPFN-3 singleton prediction is invalid")
        probability[index] = raw[0, positive]
    sentinels = tuple(
        dict.fromkeys([order[0], order[len(order) // 2], order[-1]])
    )
    repeated = np.asarray(
        [
            estimator.predict_proba(X_query[i : i + 1])[0, positive]
            for i in sentinels
        ],
        dtype=float,
    )
    difference = float(np.max(np.abs(repeated - probability[list(sentinels)])))
    return np.clip(probability, 1e-7, 1 - 1e-7), {
        "mode": "canonical_isolated_singleton",
        "repeat_sentinels": len(sentinels),
        "repeat_max_abs_difference": difference,
    }


# ---------------------------------------------------------------------- panels


def _build_panel(
    panel: str,
    settings: dict[str, Any],
    fit_raw: dict[str, np.ndarray],
    target_raw: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit every transform on the fit rows only and apply it to each target block."""

    context = _imputer_context(fit_raw["rdkit2d"])
    fit_matrix = _apply_context(context, fit_raw["rdkit2d"])
    targets = {
        name: _apply_context(context, block["rdkit2d"]) for name, block in target_raw.items()
    }
    if panel == "rdkit2d":
        return fit_matrix, targets
    if panel != "rdkit2d_morgan_svd64":
        raise TabPFN3Paper405Error(f"Unknown panel: {panel}")

    specification = settings[panel]
    svd = TruncatedSVD(
        n_components=int(specification["svd_components"]),
        random_state=int(specification["svd_random_state"]),
    ).fit(fit_raw["morgan"].astype(np.float32))
    fit_matrix = np.column_stack(
        [fit_matrix, svd.transform(fit_raw["morgan"].astype(np.float32))]
    ).astype(np.float32)
    targets = {
        name: np.column_stack(
            [targets[name], svd.transform(target_raw[name]["morgan"].astype(np.float32))]
        ).astype(np.float32)
        for name in targets
    }
    return fit_matrix, targets


def _candidate_grid(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    grid = []
    for panel in protocol["candidates"]["panels"]:
        for n_estimators in protocol["candidates"]["n_estimators"]:
            grid.append(
                {
                    "candidate_id": f"{panel}__n{int(n_estimators)}",
                    "panel": panel,
                    "n_estimators": int(n_estimators),
                }
            )
    return grid


def _inner_selection(
    protocol: dict[str, Any],
    raw: dict[str, np.ndarray],
    fit_rows: np.ndarray,
    labels: np.ndarray,
    *,
    label: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Score every candidate by inner CV restricted to ``fit_rows`` and pick a winner."""

    cross = protocol["cross_fitting"]
    inner = StratifiedKFold(
        n_splits=int(cross["inner_folds"]),
        shuffle=True,
        random_state=int(cross["inner_seed"]),
    )
    splits = list(inner.split(fit_rows, labels[fit_rows]))
    rows = []
    for candidate in _candidate_grid(protocol):
        fold_ap, fold_auroc = [], []
        for inner_fold, (relative_fit, relative_validation) in enumerate(splits):
            sub_fit = fit_rows[relative_fit]
            sub_validation = fit_rows[relative_validation]
            fit_matrix, targets = _build_panel(
                candidate["panel"],
                protocol["panels"],
                {key: raw[key][sub_fit] for key in ("rdkit2d", "morgan")},
                {
                    "validation": {
                        key: raw[key][sub_validation] for key in ("rdkit2d", "morgan")
                    }
                },
            )
            estimator = _tabpfn3_estimator(
                protocol["model"], candidate["n_estimators"], seed=42 + inner_fold
            )
            probability, _audit = _positive_probability(
                estimator,
                fit_matrix,
                labels[sub_fit],
                targets["validation"],
                singleton=False,
            )
            y = labels[sub_validation]
            fold_ap.append(float(average_precision_score(y, probability)))
            fold_auroc.append(float(roc_auc_score(y, probability)))
            del estimator
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "panel": candidate["panel"],
                "n_estimators": candidate["n_estimators"],
                "inner_ap_mean": float(np.mean(fold_ap)),
                "inner_ap_std": float(np.std(fold_ap, ddof=0)),
                "inner_auroc_mean": float(np.mean(fold_auroc)),
                "n_inner_fit_rows": len(fit_rows),
            }
        )
        print(
            f"    [{label}] {candidate['candidate_id']}: "
            f"inner AP={rows[-1]['inner_ap_mean']:.4f}",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    declaration = {c["candidate_id"]: i for i, c in enumerate(_candidate_grid(protocol))}
    winner = max(
        frame.itertuples(index=False),
        key=lambda r: (
            r.inner_ap_mean,
            -int(r.n_estimators),
            -declaration[r.candidate_id],
        ),
    )
    frame["selected"] = frame.candidate_id == winner.candidate_id
    return {
        "candidate_id": winner.candidate_id,
        "panel": winner.panel,
        "n_estimators": int(winner.n_estimators),
    }, frame


# ------------------------------------------------------------------------ report


def _metric_row(
    cohort: str,
    endpoint: str,
    model: str,
    operating_point: str,
    labels: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    source: str,
) -> dict[str, Any]:
    return {
        "cohort": cohort,
        "endpoint": endpoint,
        "model": model,
        "operating_point": operating_point,
        "result_source": source,
        **_safe_metrics(labels, probability, threshold),
    }


def _markdown(frame: pd.DataFrame, columns: list[tuple[str, str]], title: str) -> str:
    header = "| " + " | ".join(label for _key, label in columns) + " |"
    lines = [f"## {title}", "", header, "|" + "---|" * len(columns)]
    for row in frame.itertuples(index=False):
        cells = []
        for key, _label in columns:
            value = getattr(row, key)
            if isinstance(value, float):
                cells.append("nan" if not np.isfinite(value) else f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------- run


def run(
    *,
    root: Path,
    config_path: Path,
    positive_path: Path,
    negative_path: Path,
    run_id: str,
) -> Path:
    if not re.fullmatch(r"tabpfn3_paper405_[a-z0-9_.-]+", run_id):
        raise TabPFN3Paper405Error("RUN_ID must start with tabpfn3_paper405_")
    root = root.resolve()
    protocol, protocol_sha256 = load_protocol(root, config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise TabPFN3Paper405Error(f"Run directory already exists: {destination}")

    sealed = protocol["sealed_inputs"]
    bundle = load_locked_bundle(
        root / sealed["screening_bundle"]["path"],
        expected_artifact_sha256=sealed["screening_bundle"]["sha256"],
    )

    # -- D1 frame, locked split, feature panels ---------------------------------
    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    frame, _audit = _read_sources(
        _regular_file(positive_path, "positive source", protocol["sources"]["positive_sha256"]),
        _regular_file(negative_path, "negative source", protocol["sources"]["negative_sha256"]),
        traditional,
    )
    if len(frame) != int(protocol["sources"]["expected_rows"]):
        raise TabPFN3Paper405Error("D1 row count differs from 405")
    train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    if split_sha256 != protocol["paper_split"]["expected_assignment_sha256"]:
        raise TabPFN3Paper405Error("Paper split differs from the sealed assignment")
    labels = frame["label"].to_numpy(dtype=int)
    smiles, _ = _validated_raw_smiles(frame)
    features = _features(frame, smiles, fixed_protocol)
    raw = {"rdkit2d": features["rdkit2d"], "morgan": features["morgan"]}

    fit_indices = np.asarray(bundle["fit_paper_indices"], dtype=int)
    if not np.array_equal(np.sort(fit_indices), np.sort(train_indices)):
        raise TabPFN3Paper405Error("Sealed fit indices differ from the paper train split")

    # -- parity proofs against the sealed locked component state -----------------
    if not np.array_equal(
        features["morgan"][fit_indices],
        np.asarray(bundle["tanimoto_train_bits"], dtype=np.uint8),
    ):
        raise TabPFN3Paper405Error("Rebuilt Morgan bits differ from the sealed bundle")
    train_context = _imputer_context(features["rdkit2d"][fit_indices])
    if not np.array_equal(
        _apply_context(train_context, features["rdkit2d"][fit_indices]),
        np.asarray(bundle["tabpfn_context"]["context_features"], dtype=np.float32),
    ):
        raise TabPFN3Paper405Error("Rebuilt descriptor panel differs from the sealed context")

    atol = float(protocol["parity_contract"]["parity_atol"])
    sealed_oof = pd.read_csv(root / sealed["weighted_train_oof"]["path"])
    sealed_test = pd.read_csv(root / sealed["weighted_test_components"]["path"])

    # -- chemistry components: rebuild and prove parity --------------------------
    settings = fixed_protocol["components"]
    cross = protocol["cross_fitting"]
    outer = StratifiedKFold(
        n_splits=int(cross["outer_folds"]), shuffle=True, random_state=int(cross["outer_seed"])
    )
    fold_split = list(outer.split(train_indices, labels[train_indices]))
    oof = {name: np.full(len(train_indices), np.nan) for name in CHEMISTRY_COMPONENTS}
    for fold, (relative_fit, relative_validation) in enumerate(fold_split):
        chemistry, _models, _a = _selected_component_predictions(
            features,
            labels,
            train_indices[relative_fit],
            train_indices[relative_validation],
            settings,
            seed=42 + fold,
            requested=CHEMISTRY_COMPONENTS,
        )
        for name in CHEMISTRY_COMPONENTS:
            oof[name][relative_validation] = chemistry[name]
        print(f"chemistry OOF fold {fold + 1}/{len(fold_split)}", flush=True)

    rebuilt = pd.DataFrame(
        {
            "paper_row_index": train_indices,
            "probability_paper_svm": oof["paper_svm"],
            "probability_tanimoto_svc": oof["tanimoto_svc"],
        }
    )
    merged = sealed_oof.merge(rebuilt, on="paper_row_index", suffixes=("_sealed", "_rebuilt"))
    if len(merged) != len(train_indices):
        raise TabPFN3Paper405Error("Sealed OOF join is incomplete")
    oof_parity = {}
    for column in ("probability_paper_svm", "probability_tanimoto_svc"):
        difference = float(
            np.max(np.abs(merged[f"{column}_sealed"] - merged[f"{column}_rebuilt"]))
        )
        if difference > atol:
            raise TabPFN3Paper405Error(f"Rebuilt OOF {column} differs from the sealed run")
        oof_parity[column] = difference

    full_chemistry, full_models, _a = _selected_component_predictions(
        features, labels, train_indices, test_indices, settings, seed=42,
        requested=CHEMISTRY_COMPONENTS,
    )
    test_frame = pd.DataFrame(
        {
            "paper_row_index": test_indices,
            "label": labels[test_indices],
            "probability_paper_svm": full_chemistry["paper_svm"],
            "probability_tanimoto_svc": full_chemistry["tanimoto_svc"],
        }
    )
    merged_test = sealed_test.merge(
        test_frame.drop(columns=["label"]),
        on="paper_row_index",
        suffixes=("_sealed", "_rebuilt"),
    )
    test_parity = {}
    for column in ("probability_paper_svm", "probability_tanimoto_svc"):
        difference = float(
            np.max(np.abs(merged_test[f"{column}_sealed"] - merged_test[f"{column}_rebuilt"]))
        )
        if difference > atol:
            raise TabPFN3Paper405Error(f"Rebuilt test {column} differs from the sealed run")
        test_parity[column] = difference
    paper_svm_boundary = _svc_probability_at_native_boundary(full_models["paper_svm"])

    # -- external cohorts: chemistry streams read from the sealed run -------------
    cohorts: dict[str, dict[str, Any]] = {}
    for cohort, key, endpoint in (
        ("drugage", "drugage_scored", DRUGAGE_ENDPOINT),
        ("agextend", "agextend_scored", AGEXTEND_ENDPOINT),
    ):
        scored = pd.read_csv(root / sealed[key]["path"])
        if cohort == "drugage":
            scored = scored.assign(label=scored.has_significant_positive.astype(int))
        scored = scored.reset_index(drop=True)
        external_smiles = scored.source_smiles.astype(str).tolist()
        payload = {
            "endpoint": endpoint,
            "frame": scored,
            "rdkit2d": _rdkit2d_from_smiles(external_smiles, bundle["descriptor_names"]),
            "morgan": _morgan_from_smiles(external_smiles, bundle["portable_contract"]),
        }
        similarity = _tanimoto(
            payload["morgan"], np.asarray(bundle["tanimoto_train_bits"], dtype=np.uint8)
        ).max(axis=1)
        stored = scored.maximum_tanimoto_to_fitted_train.to_numpy(dtype=float)
        if float(np.max(np.abs(similarity - stored))) > 1e-6:
            raise TabPFN3Paper405Error(
                f"Rebuilt {cohort} similarity differs from the sealed file"
            )
        cohorts[cohort] = payload
        print(f"external features rebuilt for {cohort}: {len(scored)} rows", flush=True)

    # -- nested, fold-safe TabPFN-3 OOF ------------------------------------------
    tabpfn3_oof = np.full(len(train_indices), np.nan)
    outer_fold_id = np.full(len(train_indices), -1, dtype=int)
    selected_per_row = np.empty(len(train_indices), dtype=object)
    selection_tables = []
    for fold, (relative_fit, relative_validation) in enumerate(fold_split):
        sub_fit = train_indices[relative_fit]
        sub_validation = train_indices[relative_validation]
        print(f"[tabpfn3] outer fold {fold + 1}/{len(fold_split)} inner selection", flush=True)
        winner, table = _inner_selection(
            protocol, raw, sub_fit, labels, label=f"outer {fold + 1}"
        )
        table.insert(0, "outer_fold", fold)
        selection_tables.append(table)
        fit_matrix, targets = _build_panel(
            winner["panel"],
            protocol["panels"],
            {key: raw[key][sub_fit] for key in ("rdkit2d", "morgan")},
            {"validation": {key: raw[key][sub_validation] for key in ("rdkit2d", "morgan")}},
        )
        estimator = _tabpfn3_estimator(protocol["model"], winner["n_estimators"], seed=42)
        probability, _audit = _positive_probability(
            estimator, fit_matrix, labels[sub_fit], targets["validation"], singleton=True
        )
        tabpfn3_oof[relative_validation] = probability
        outer_fold_id[relative_validation] = fold
        selected_per_row[relative_validation] = winner["candidate_id"]
        del estimator
        print(f"[tabpfn3] outer fold {fold + 1} winner {winner['candidate_id']}", flush=True)
    if not np.isfinite(tabpfn3_oof).all():
        raise TabPFN3Paper405Error("TabPFN-3 OOF is incomplete")

    # -- final selection on all 324 training rows, then score once ---------------
    print("[tabpfn3] final selection on all 324 training rows", flush=True)
    final_winner, final_table = _inner_selection(
        protocol, raw, train_indices, labels, label="final"
    )
    target_raw = {
        "d1_test": {key: raw[key][test_indices] for key in ("rdkit2d", "morgan")},
        "drugage": {
            "rdkit2d": cohorts["drugage"]["rdkit2d"],
            "morgan": cohorts["drugage"]["morgan"],
        },
        "agextend": {
            "rdkit2d": cohorts["agextend"]["rdkit2d"],
            "morgan": cohorts["agextend"]["morgan"],
        },
    }
    fit_matrix, targets = _build_panel(
        final_winner["panel"],
        protocol["panels"],
        {key: raw[key][train_indices] for key in ("rdkit2d", "morgan")},
        target_raw,
    )
    estimator = _tabpfn3_estimator(protocol["model"], final_winner["n_estimators"], seed=42)
    estimator.fit(fit_matrix, labels[train_indices])
    scored_probability, inference_audits = {}, {}
    for name, matrix in targets.items():
        classes = np.asarray(estimator.classes_)
        positive = int(np.flatnonzero(classes == 1)[0])
        order = sorted(range(len(matrix)), key=lambda i: matrix[i].astype("<f8").tobytes())
        probability = np.empty(len(matrix), dtype=np.float64)
        for index in order:
            probability[index] = float(
                estimator.predict_proba(matrix[index : index + 1])[0, positive]
            )
        sentinels = tuple(dict.fromkeys([order[0], order[len(order) // 2], order[-1]]))
        repeated = np.asarray(
            [estimator.predict_proba(matrix[i : i + 1])[0, positive] for i in sentinels],
            dtype=float,
        )
        difference = float(np.max(np.abs(repeated - probability[list(sentinels)])))
        if difference > float(protocol["inference"]["repeat_atol"]):
            raise TabPFN3Paper405Error(f"TabPFN-3 singleton repeatability failed on {name}")
        scored_probability[name] = np.clip(probability, 1e-7, 1 - 1e-7)
        inference_audits[name] = {
            "mode": "canonical_isolated_singleton",
            "repeat_sentinels": len(sentinels),
            "repeat_max_abs_difference": difference,
            "n_rows": len(matrix),
        }
        print(f"[tabpfn3] scored {name}: {len(matrix)} rows", flush=True)
    del estimator

    # -- blends, thresholds ------------------------------------------------------
    locked_weights = np.asarray(
        protocol["blends"]["locked_weights_010_060_030"]["weights"], float
    )
    equal_weights = np.asarray(protocol["blends"]["equal_thirds"]["weights"], float)

    def _blend(svm, tanimoto, third, weights):
        return weights[0] * svm + weights[1] * tanimoto + weights[2] * third

    y_train = labels[train_indices]
    blend_locked_oof = _blend(
        oof["paper_svm"], oof["tanimoto_svc"], tabpfn3_oof, locked_weights
    )
    blend_equal_oof = _blend(oof["paper_svm"], oof["tanimoto_svc"], tabpfn3_oof, equal_weights)

    thresholds, threshold_rows = {}, []
    for name, stream, source in (
        ("tabpfn3", tabpfn3_oof, OOF_MCC_SOURCE),
        ("tanimoto_svc", oof["tanimoto_svc"], OOF_MCC_SOURCE),
        ("blend_010_060_030_tabpfn3", blend_locked_oof, OOF_MCC_SOURCE),
        ("blend_equal_thirds_tabpfn3", blend_equal_oof, OOF_MCC_SOURCE),
    ):
        threshold, mcc = select_threshold(y_train, stream)
        thresholds[name] = float(threshold)
        threshold_rows.append(
            {
                "model": name,
                "threshold": float(threshold),
                "threshold_source": source,
                "oof_mcc_at_selection": float(mcc),
                "external_labels_used": False,
            }
        )
    thresholds["paper_svm"] = float(paper_svm_boundary)
    threshold_rows.insert(
        0,
        {
            "model": "paper_svm",
            "threshold": float(paper_svm_boundary),
            "threshold_source": "published_svc_predict_decision_function_zero",
            "oof_mcc_at_selection": float(select_threshold(y_train, oof["paper_svm"])[1]),
            "external_labels_used": False,
        },
    )

    # -- scored frames and metrics -----------------------------------------------
    test_frame["probability_tabpfn3"] = scored_probability["d1_test"]
    test_frame["blend_010_060_030_tabpfn3"] = _blend(
        test_frame.probability_paper_svm,
        test_frame.probability_tanimoto_svc,
        test_frame.probability_tabpfn3,
        locked_weights,
    )
    test_frame["blend_equal_thirds_tabpfn3"] = _blend(
        test_frame.probability_paper_svm,
        test_frame.probability_tanimoto_svc,
        test_frame.probability_tabpfn3,
        equal_weights,
    )
    test_frame["maximum_tanimoto_to_fitted_train"] = _tanimoto(
        features["morgan"][test_indices],
        np.asarray(bundle["tanimoto_train_bits"], dtype=np.uint8),
    ).max(axis=1)

    for cohort, payload in cohorts.items():
        scored = payload["frame"]
        scored["probability_tabpfn3"] = scored_probability[cohort]
        scored["blend_010_060_030_tabpfn3"] = _blend(
            scored.probability_paper_svm,
            scored.probability_tanimoto_svc,
            scored.probability_tabpfn3,
            locked_weights,
        )
        scored["blend_equal_thirds_tabpfn3"] = _blend(
            scored.probability_paper_svm,
            scored.probability_tanimoto_svc,
            scored.probability_tabpfn3,
            equal_weights,
        )

    def _streams(scored: pd.DataFrame):
        return [
            (
                "paper_svm",
                scored.probability_paper_svm.to_numpy(float),
                thresholds["paper_svm"],
            ),
            (
                "tanimoto_svc",
                scored.probability_tanimoto_svc.to_numpy(float),
                thresholds["tanimoto_svc"],
            ),
            ("tabpfn3", scored.probability_tabpfn3.to_numpy(float), thresholds["tabpfn3"]),
            (
                "blend_010_060_030_tabpfn3",
                scored.blend_010_060_030_tabpfn3.to_numpy(float),
                thresholds["blend_010_060_030_tabpfn3"],
            ),
            (
                "blend_equal_thirds_tabpfn3",
                scored.blend_equal_thirds_tabpfn3.to_numpy(float),
                thresholds["blend_equal_thirds_tabpfn3"],
            ),
        ]

    metric_rows = []
    for cohort, endpoint, scored in (
        ("d1_paper_test", D1_ENDPOINT, test_frame),
        ("drugage", DRUGAGE_ENDPOINT, cohorts["drugage"]["frame"]),
        ("agextend", AGEXTEND_ENDPOINT, cohorts["agextend"]["frame"]),
    ):
        y = scored.label.to_numpy(dtype=int)
        for model, probability, threshold in _streams(scored):
            point = "published_svc_native_predict" if model == "paper_svm" else "oof_mcc"
            metric_rows.append(
                _metric_row(
                    cohort, endpoint, model, point, y, probability, threshold, "this_run"
                )
            )
            metric_rows.append(
                _metric_row(
                    cohort, endpoint, model, "fixed_0p5", y, probability, 0.5, "this_run"
                )
            )
    metrics = pd.DataFrame(metric_rows)

    sealed_reference = pd.concat(
        [
            pd.read_csv(root / sealed["ablation_d1_metrics"]["path"]),
            pd.read_csv(root / sealed["ablation_external_metrics"]["path"]),
        ],
        ignore_index=True,
    )
    sealed_reference = sealed_reference[
        sealed_reference.is_primary_operating_point.astype(bool)
    ]
    sealed_reference = sealed_reference[
        sealed_reference.endpoint.isin([D1_ENDPOINT, DRUGAGE_ENDPOINT, AGEXTEND_ENDPOINT])
    ].copy()
    sealed_reference["result_source"] = "sealed_locked_run_20260818"
    comparison_columns = [
        "cohort", "endpoint", "model", "result_source", "n_test",
        "auprc_average_precision_positive", "auroc", "brier", "mcc", "macro_f1",
        "recall_sensitivity", "specificity", "threshold",
    ]
    primary = metrics[metrics.operating_point != "fixed_0p5"]
    comparison = pd.concat(
        [sealed_reference[comparison_columns], primary[comparison_columns]], ignore_index=True
    )
    order = {"d1_paper_test": 0, "drugage": 1, "agextend": 2}
    comparison["_o"] = comparison.cohort.map(order)
    comparison = comparison.sort_values(["_o", "model"], kind="stable").drop(columns=["_o"])

    # -- write --------------------------------------------------------------------
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".tabpfn3.work-", dir=destination.parent))
    try:
        _write_csv(temporary / "selection_outer_folds.csv", pd.concat(selection_tables))
        _write_csv(temporary / "selection_final.csv", final_table)
        _write_csv(
            temporary / "d1_train_oof_predictions.csv",
            pd.DataFrame(
                {
                    "paper_row_index": train_indices,
                    "label": y_train,
                    "outer_fold": outer_fold_id,
                    "selected_candidate_id": selected_per_row,
                    "probability_paper_svm": oof["paper_svm"],
                    "probability_tanimoto_svc": oof["tanimoto_svc"],
                    "probability_tabpfn3": tabpfn3_oof,
                    "blend_010_060_030_tabpfn3": blend_locked_oof,
                    "blend_equal_thirds_tabpfn3": blend_equal_oof,
                }
            ),
        )
        _write_csv(temporary / "component_thresholds.csv", pd.DataFrame(threshold_rows))
        _write_csv(temporary / "d1_test_predictions.csv", test_frame)
        for cohort, payload in cohorts.items():
            _write_csv(temporary / f"external_predictions_{cohort}.csv", payload["frame"])
        _write_csv(temporary / "metrics_all.csv", metrics)
        _write_csv(temporary / "comparison_vs_locked.csv", comparison)

        atomic_write_json(
            temporary / "inference_audit.json",
            {
                "schema_version": f"{SCHEMA}.inference_audit.v1",
                "per_target": inference_audits,
                "checkpoint_sha256": protocol["model"]["checkpoint_sha256"],
                "package_version": importlib.metadata.version("tabpfn"),
                "model_version": str(protocol["model"]["explicit_model_version"]),
                "telemetry_disabled": True,
            },
        )

        columns = [
            ("model", "Model"), ("n_test", "n"),
            ("auprc_average_precision_positive", "AP"), ("auroc", "AUROC"),
            ("brier", "Brier"), ("mcc", "MCC"), ("macro_f1", "Macro F1"),
            ("recall_sensitivity", "Recall"), ("specificity", "Specificity"),
        ]
        parts = [
            "# TabPFN-3 under the locked paper-405 protocol",
            "",
            f"Final selected configuration: `{final_winner['candidate_id']}`.",
            "Outer-fold winners: "
            + ", ".join(
                f"fold {int(t.outer_fold.iloc[0])}={t.loc[t.selected, 'candidate_id'].iloc[0]}"
                for t in selection_tables
            )
            + ".",
            "",
            "Selection used inner cross-validation restricted to each fold's fit rows.",
            "Rows labelled `sealed_locked_run_20260818` are copied unchanged from the",
            "sealed run.",
            "",
        ]
        for cohort, title in (
            ("d1_paper_test", "D1 held-out paper test (n=81)"),
            ("drugage", "DrugAge positive-retrieval endpoint (n=446)"),
            ("agextend", "AgeXtend Table 6 endpoint (n=69)"),
        ):
            parts.append(_markdown(comparison[comparison.cohort == cohort], columns, title))
        (temporary / "summary.md").write_text("\n".join(parts), encoding="utf-8")

        manifest = {
            "schema_version": f"{SCHEMA}.run_manifest.v1",
            "run_id": run_id,
            "created_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": protocol_sha256,
            "paper_split_sha256": split_sha256,
            "locked_component_state_sha256": bundle["component_state_sha256"],
            "model": {
                "package_version": importlib.metadata.version("tabpfn"),
                "model_version": str(protocol["model"]["explicit_model_version"]),
                "checkpoint_path": protocol["model"]["checkpoint_path"],
                "checkpoint_sha256": protocol["model"]["checkpoint_sha256"],
                "checkpoint_source": protocol["model"]["checkpoint_source"],
                "access_date_utc": protocol["model"]["access_date_utc"],
            },
            "final_selected_candidate": final_winner,
            "outer_fold_selected_candidates": [
                {
                    "outer_fold": int(t.outer_fold.iloc[0]),
                    "candidate_id": t.loc[t.selected, "candidate_id"].iloc[0],
                }
                for t in selection_tables
            ],
            "thresholds": thresholds,
            "paper_svm_native_probability_boundary": float(paper_svm_boundary),
            "parity": {
                "morgan_bits_equal_sealed_bundle": True,
                "descriptor_panel_equal_sealed_context": True,
                "oof_max_abs_difference": oof_parity,
                "d1_test_max_abs_difference": test_parity,
            },
            "output_schema": protocol["output_schema"],
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "sealed_inputs_sha256": {
                name: sha256_file(root / record["path"]) for name, record in sealed.items()
            },
            "outer_validation_rows_used_for_selection": False,
            "test_or_external_labels_used_for_selection_or_threshold": False,
            "existing_runs_modified": False,
        }
        atomic_write_json(temporary / "RUN_MANIFEST.json", manifest)
        atomic_write_json(
            temporary / "COMPLETED.json",
            {
                "schema_version": f"{SCHEMA}.completed.v1",
                "status": "COMPLETE",
                "run_id": run_id,
                "run_manifest_sha256": sha256_file(temporary / "RUN_MANIFEST.json"),
                "artifact_hashes": {
                    str(path.relative_to(temporary)): sha256_file(path)
                    for path in sorted(temporary.rglob("*"))
                    if path.is_file() and path.name != "COMPLETED.json"
                },
            },
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps({"run": str(destination), "status": "COMPLETE"}, indent=2))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args(argv)
    run(
        root=arguments.root,
        config_path=arguments.config,
        positive_path=arguments.positive,
        negative_path=arguments.negative,
        run_id=arguments.run_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
