"""T-DDI-style feature selection for the DL components, then an equal-thirds blend.

FEATURE-SELECTION PIPELINE -- transcribed from the T-DDI paper
--------------------------------------------------------------
Source: "Robust Prediction of Drug Interactions using Chemical Descriptors"
(Fulltext.pdf, Methods p.17; sweep table in Supplementary_document.pdf,
Supplementary Table 3, p.14).  The paper's own words:

  "we ran a separate feature-selection workflow using only the training and
   validation partitions; the held-out test set was excluded from all selection
   and threshold-tuning steps. Starting from all 3,780 descriptors, we first
   performed a descriptor audit ... removing descriptors with all missing
   values, high missingness (missing ratio > 0.995), or near-zero variance
   (variance threshold 1e-10). This reduced the auditable candidate pool from
   3,780 to 2,858 descriptors; missing values in retained descriptors were
   imputed using training-sample medians before ranking. The 2,858 retained
   descriptors were ranked using ANOVA F-statistics and then passed through a
   Pearson-correlation-aware priority step using an absolute correlation
   threshold of 0.995. Recursive feature elimination (RFE) was then applied to
   the ranked feature pool to generate candidate subsets at predefined counts
   K in {2858, 2700, 2400, 2100, 1800, 1500}. To determine the final operating
   dimensionality without bias, the T-DDI architecture was trained and evaluated
   under the same three-fold cross-validation protocol for the full
   3,780-descriptor representation and each reduced feature-count variant."

So the pipeline reproduced here is, in order:

  1. descriptor audit   : drop all-missing / missingness > 0.995 / variance < 1e-10
  2. median imputation  : training-row medians, applied before ranking
  3. ANOVA F ranking    : sklearn f_classif
  4. correlation prune  : Pearson-correlation-aware priority, |r| > 0.995
                          (of a correlated pair, the lower-F member is dropped)
  5. RFE at predefined K counts, applied to the ranked pool
  6. K chosen by re-training from scratch under the SAME k-fold CV protocol,
     fold splits and seed schedule, and comparing the sweep

Adapted only in scale: our candidate pool is the 217 RDKit2D descriptors rather
than 3,780 PyBioMed ones, so the K ladder is rescaled.  Its MINIMUM is 7 -- the
descriptor count of the Geroprotectors paper -- per the user's instruction.

WHAT IS AND IS NOT SUBJECT TO FEATURE SELECTION
------------------------------------------------
Feature selection targets ONLY the deep/foundation components, which is the
stated purpose ("the FS process is to research the DL component"):

  * svm_paper_setting : NEVER refitted and NEVER feature-selected.  Its
    probability streams are read verbatim from the sealed runs, so it is exactly
    the published SVC(kernel='linear', C=1, gamma=1, probability=True,
    random_state=42) on exactly the paper's seven DataWarrior descriptors, on
    every cohort including both external ones.
  * tabpfn_v2, tabfm  : fitted on the FS-selected RDKit2D descriptors.
  * Tanimoto          : untouched -- it is not a component of this blend and no
    Tanimoto artifact is read or modified.

Blend = 1/3 svm_paper_setting + 1/3 tabpfn_v2 + 1/3 tabfm.

EVALUATIONS
-----------
  1. K sweep by 5-fold CV on the 324 training rows (T-DDI's selection step);
  2. an additional NESTED-FS cross-validation at the chosen K, where the whole
     FS pipeline is refitted inside each fold -- reported because T-DDI's own
     protocol fits FS once on development data, which makes the sweep CV mildly
     selection-optimistic;
  3. lock on all 324 rows -> score the 81 held-out rows once;
  4. external validation on DrugAge (n=446) and AgeXtend (n=69), scored once.

No test or external label is used for feature selection, model fitting, weight
choice or threshold choice anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import RFE, f_classif
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from geroprotector.fixed_blend_paper405 import _features, _tanimoto
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.screening_blend_altmodels import (
    _morgan_from_smiles,
    _rdkit2d_from_smiles,
)
from geroprotector.screening_blend_paper405 import load_locked_bundle
from geroprotector.screening_blend_tabfm import _load_tabfm, _tabfm_probability
from geroprotector.tabpfn3_paper405 import _positive_probability, _tabpfn3_estimator
from geroprotector.traditional_paper405 import _read_sources, paper_split_indices
from geroprotector.weighted_blend_paper405 import select_threshold


class TDDIFSBlendError(RuntimeError):
    """Raised when a sealed input, a contract or an invariant fails."""


SCHEMA = "geroprotector.tddi_fs_blend_paper405"
DL_COMPONENTS = ("tabpfn_v2", "tabfm")
COMPONENTS = ("svm_paper_setting", "tabpfn_v2", "tabfm")
EQUAL_THIRDS = np.full(3, 1.0 / 3.0)
D1_ENDPOINT = "paper_binary"
DRUGAGE_ENDPOINT = "significant_positive_retrieval_background_not_certified_negative"
AGEXTEND_ENDPOINT = "published_independent_table6_binary"


def _regular_file(path: Path, role: str, expected: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise TDDIFSBlendError(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected is not None and sha256_file(resolved) != expected:
        raise TDDIFSBlendError(f"{role} SHA256 differs from the protocol lock")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    text = _regular_file(path, "T-DDI FS protocol").read_text(encoding="utf-8")
    protocol = yaml.safe_load(text)
    expected_schema = f"{SCHEMA}.protocol.v1"
    if not isinstance(protocol, dict) or protocol.get("schema_version") != expected_schema:
        raise TDDIFSBlendError("Unknown T-DDI FS protocol schema")
    fs = protocol.get("feature_selection", {})
    for key, expected in (
        ("audit_missing_ratio_threshold", 0.995),
        ("audit_variance_threshold", 1e-10),
        ("correlation_abs_threshold", 0.995),
    ):
        if not np.isclose(float(fs.get(key, np.nan)), expected, rtol=0.0, atol=1e-15):
            raise TDDIFSBlendError(f"T-DDI FS contract differs at {key}")
    if fs.get("ranking") != "anova_f_classif" or fs.get("reduction") != "RFE":
        raise TDDIFSBlendError("T-DDI FS ranking/reduction contract differs")
    if fs.get("applies_to") != "dl_components_only":
        raise TDDIFSBlendError("FS scope contract differs")
    if int(min(fs["k_ladder"])) != 7:
        raise TDDIFSBlendError("Minimum K must be 7 (the Geroprotectors descriptor count)")
    svm = protocol.get("svm_component", {})
    if (
        svm.get("source") != "sealed_streams_never_refitted"
        or svm.get("feature_set") != "paper_7_datawarrior_descriptors"
        or svm.get("feature_selected") is not False
    ):
        raise TDDIFSBlendError("SVM component contract differs")
    blend = protocol.get("blend", {})
    if (
        blend.get("components") != list(COMPONENTS)
        or not np.allclose(blend.get("weights"), EQUAL_THIRDS, rtol=0.0, atol=1e-15)
        or blend.get("weights_selected_from_data") is not False
    ):
        raise TDDIFSBlendError("Blend contract differs")
    if protocol.get("contract", {}).get(
        "test_or_external_labels_used_for_selection_or_threshold"
    ) is not False:
        raise TDDIFSBlendError("Leakage contract differs")
    immutability = protocol.get("immutability", {})
    if immutability.get("existing_run_directories_are_read_only") is not True:
        raise TDDIFSBlendError("Immutability contract differs")
    for record in protocol.get("sealed_inputs", {}).values():
        _regular_file(root / record["path"], f"sealed input {record['path']}", record["sha256"])
    return protocol, canonical_sha256(protocol)


# ------------------------------------------------- T-DDI feature selection


def tddi_feature_selection(
    x_raw_fit: np.ndarray, y_fit: np.ndarray, names: list[str], settings: dict[str, Any]
) -> dict[str, Any]:
    """The T-DDI pipeline, fitted on FIT ROWS ONLY.

    Steps, in the paper's order: audit -> median impute -> ANOVA F rank ->
    Pearson-correlation-aware priority prune -> RFE at predefined K counts.
    Returns the fitted state plus the per-K index lists.
    """

    missing_threshold = float(settings["audit_missing_ratio_threshold"])
    variance_threshold = float(settings["audit_variance_threshold"])
    correlation_threshold = float(settings["correlation_abs_threshold"])

    # ---- 1. descriptor audit ------------------------------------------------
    finite = np.isfinite(x_raw_fit)
    missing_ratio = 1.0 - finite.mean(axis=0)
    keep = (missing_ratio <= missing_threshold) & finite.any(axis=0)
    audited = np.flatnonzero(keep)
    if audited.size == 0:
        raise TDDIFSBlendError("Descriptor audit removed every candidate")

    # ---- 2. median imputation on training-row medians ------------------------
    subset = x_raw_fit[:, audited]
    medians = np.nanmedian(np.where(np.isfinite(subset), subset, np.nan), axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    imputed = np.where(np.isfinite(subset), subset, medians)

    # near-zero-variance drop is part of the audit but must be measured AFTER
    # imputation, otherwise a mostly-missing column looks spuriously variable
    variances = imputed.var(axis=0)
    variable = variances > variance_threshold
    audited = audited[variable]
    medians = medians[variable]
    imputed = imputed[:, variable]
    if audited.size == 0:
        raise TDDIFSBlendError("Variance audit removed every candidate")

    # ---- 3. ANOVA F ranking --------------------------------------------------
    f_values, _p = f_classif(imputed, y_fit)
    f_values = np.where(np.isfinite(f_values), f_values, -np.inf)
    order = np.argsort(-f_values)  # descending F

    # ---- 4. Pearson-correlation-aware priority prune -------------------------
    # Walk the ANOVA-ranked list; a descriptor is dropped if it correlates
    # |r| > threshold with an already-accepted (higher-F) descriptor.
    correlation = np.corrcoef(imputed, rowvar=False)
    correlation = np.where(np.isfinite(correlation), correlation, 0.0)
    accepted: list[int] = []
    for candidate in order:
        if all(abs(correlation[candidate, kept]) <= correlation_threshold for kept in accepted):
            accepted.append(int(candidate))
    ranked = np.asarray(accepted, dtype=int)  # local indices into `imputed`

    # ---- 5. RFE at predefined K counts ---------------------------------------
    scaler = StandardScaler().fit(imputed[:, ranked])
    scaled = scaler.transform(imputed[:, ranked])
    per_k: dict[int, np.ndarray] = {}
    for k in sorted({int(v) for v in settings["k_ladder"]}, reverse=True):
        if k >= ranked.size:
            per_k[int(ranked.size)] = ranked.copy()
            continue
        rfe = RFE(
            estimator=SVC(kernel="linear", C=float(settings["rfe_estimator_C"])),
            n_features_to_select=int(k),
            step=int(settings["rfe_step"]),
        ).fit(scaled, y_fit)
        per_k[int(k)] = ranked[rfe.support_]

    return {
        "audited_local_to_global": audited,   # global column index per audited column
        "medians": medians,
        "ranked_local": ranked,
        "per_k_local": per_k,                 # K -> local indices into `imputed`
        "n_input": int(x_raw_fit.shape[1]),
        "n_after_audit": int(audited.size),
        "n_after_correlation_prune": int(ranked.size),
        "names": names,
    }


def apply_selection(state: dict[str, Any], x_raw: np.ndarray, k: int) -> np.ndarray:
    """Apply a fitted selection state to any cohort (audit -> impute -> select K)."""

    subset = x_raw[:, state["audited_local_to_global"]]
    imputed = np.where(np.isfinite(subset), subset, state["medians"])
    local = state["per_k_local"][k]
    out = imputed[:, local].astype(np.float32)
    if not np.isfinite(out).all():
        raise TDDIFSBlendError("Selected feature matrix contains non-finite values")
    return out


def selected_names(state: dict[str, Any], k: int) -> list[str]:
    globals_ = state["audited_local_to_global"][state["per_k_local"][k]]
    return [state["names"][i] for i in globals_]


# --------------------------------------------------------------- metrics


def _all_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    d = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, d, labels=[0, 1]).ravel()
    both = len(set(y)) == 2
    return {
        "n": len(y),
        "accuracy": float(accuracy_score(y, d)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "cohen_kappa": float(cohen_kappa_score(y, d)),
        "f1_positive": float(f1_score(y, d, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y, d, average="macro", zero_division=0)),
        "auprc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
        "auroc": float(roc_auc_score(y, p)) if both else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
    }


def _fit_dl(
    protocol: dict[str, Any], x_fit: np.ndarray, y_fit: np.ndarray,
    targets: dict[str, np.ndarray], tabfm_model, seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Fit TabPFN-v2 and TabFM from scratch on the selected features."""

    output: dict[str, dict[str, np.ndarray]] = {}
    settings = protocol["tabpfn_v2"]
    estimator = _tabpfn3_estimator(
        settings, int(settings["hyperparameters"]["n_estimators"]), seed=seed
    )
    output["tabpfn_v2"] = {}
    for key, matrix in targets.items():
        probability, _audit = _positive_probability(
            estimator, x_fit, y_fit, matrix, singleton=False
        )
        output["tabpfn_v2"][key] = probability
    del estimator

    probabilities, _audit = _tabfm_probability(
        tabfm_model, protocol["tabfm"], x_fit, y_fit, targets, seed=seed
    )
    output["tabfm"] = probabilities
    return output


# ------------------------------------------------------------------- run


def run(
    *, root: Path, config_path: Path, positive_path: Path, negative_path: Path,
    run_id: str, quick: bool = False,
) -> Path:
    if not re.fullmatch(r"tddi_fs_blend_[a-z0-9_.-]+", run_id):
        raise TDDIFSBlendError("RUN_ID must start with tddi_fs_blend_")
    root = root.resolve()
    protocol, protocol_sha256 = load_protocol(root, config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise TDDIFSBlendError(f"Run directory already exists: {destination}")
    sealed = protocol["sealed_inputs"]
    fs_settings = dict(protocol["feature_selection"])
    if quick:
        fs_settings["k_ladder"] = sorted({7, 30, int(max(fs_settings["k_ladder"]))})
        print(f"QUICK MODE: K ladder reduced to {fs_settings['k_ladder']}", flush=True)

    bundle = load_locked_bundle(
        root / sealed["screening_bundle"]["path"],
        expected_artifact_sha256=sealed["screening_bundle"]["sha256"],
    )
    descriptor_names = list(bundle["descriptor_names"])

    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    frame, _audit = _read_sources(
        _regular_file(positive_path, "positive source", protocol["sources"]["positive_sha256"]),
        _regular_file(negative_path, "negative source", protocol["sources"]["negative_sha256"]),
        traditional,
    )
    train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    if split_sha256 != protocol["paper_split"]["expected_assignment_sha256"]:
        raise TDDIFSBlendError("Paper split differs from the sealed assignment")
    labels = frame["label"].to_numpy(dtype=int)
    smiles, _ = _validated_raw_smiles(frame)
    features = _features(frame, smiles, fixed_protocol)
    y_train, y_test = labels[train_indices], labels[test_indices]

    # ---- SVM component: sealed streams, never refitted, always the paper's 7 --
    sealed_oof = pd.read_csv(root / sealed["weighted_train_oof"]["path"])
    sealed_oof = sealed_oof.set_index("paper_row_index").loc[train_indices].reset_index()
    sealed_test = pd.read_csv(root / sealed["weighted_test_components"]["path"])
    sealed_test = sealed_test.set_index("paper_row_index").loc[test_indices].reset_index()
    svm_stream = {
        "cv5": sealed_oof.probability_paper_svm.to_numpy(float),
        "d1_test": sealed_test.probability_paper_svm.to_numpy(float),
    }

    cohorts: dict[str, dict[str, Any]] = {}
    for cohort, key, endpoint in (
        ("drugage", "drugage_scored", DRUGAGE_ENDPOINT),
        ("agextend", "agextend_scored", AGEXTEND_ENDPOINT),
    ):
        scored = pd.read_csv(root / sealed[key]["path"])
        if cohort == "drugage":
            scored = scored.assign(label=scored.has_significant_positive.astype(int))
        scored = scored.reset_index(drop=True)
        smi = scored.source_smiles.astype(str).tolist()
        payload = {
            "endpoint": endpoint, "frame": scored,
            "rdkit2d": _rdkit2d_from_smiles(smi, descriptor_names),
            "morgan": _morgan_from_smiles(smi, bundle["portable_contract"]),
        }
        rebuilt = _tanimoto(
            payload["morgan"], np.asarray(bundle["tanimoto_train_bits"], dtype=np.uint8)
        ).max(axis=1)
        stored = scored.maximum_tanimoto_to_fitted_train.to_numpy(dtype=float)
        if float(np.max(np.abs(rebuilt - stored))) > 1e-6:
            raise TDDIFSBlendError(f"Rebuilt {cohort} similarity differs from the sealed file")
        payload["svm"] = scored.probability_paper_svm.to_numpy(float)
        cohorts[cohort] = payload
        print(f"external features rebuilt for {cohort}: {len(scored)} rows", flush=True)

    tabfm_model = _load_tabfm(protocol["tabfm"])
    print("TabFM weights loaded", flush=True)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_split = list(folds.split(train_indices, y_train))
    fold_id = np.full(len(train_indices), -1, dtype=int)
    for fold, (_relative_fit, relative_validation) in enumerate(fold_split):
        fold_id[relative_validation] = fold

    # ---- STEP 1: T-DDI FS on the training rows, then the K sweep by 5-fold CV --
    print("fitting the T-DDI FS pipeline on the 324 training rows", flush=True)
    state = tddi_feature_selection(
        features["rdkit2d"][train_indices], y_train, descriptor_names, fs_settings
    )
    print(
        f"  audit: {state['n_input']} -> {state['n_after_audit']}; "
        f"correlation prune -> {state['n_after_correlation_prune']}",
        flush=True,
    )
    k_values = sorted(state["per_k_local"].keys(), reverse=True)

    sweep_rows, sweep_fold_rows = [], []
    sweep_oof: dict[int, dict[str, np.ndarray]] = {}
    for k in k_values:
        oof = {name: np.full(len(train_indices), np.nan) for name in DL_COMPONENTS}
        for fold, (relative_fit, relative_validation) in enumerate(fold_split):
            sub_fit = train_indices[relative_fit]
            sub_validation = train_indices[relative_validation]
            x_fit = apply_selection(state, features["rdkit2d"][sub_fit], k)
            x_val = apply_selection(state, features["rdkit2d"][sub_validation], k)
            streams = _fit_dl(
                protocol, x_fit, labels[sub_fit], {"validation": x_val},
                tabfm_model, seed=42 + fold,
            )
            for name in DL_COMPONENTS:
                oof[name][relative_validation] = streams[name]["validation"]
            print(f"  [K={k}] fold {fold + 1}/5", flush=True)
        blend = np.column_stack(
            [svm_stream["cv5"], oof["tabpfn_v2"], oof["tabfm"]]
        ) @ EQUAL_THIRDS
        sweep_oof[k] = {**oof, "blend_eq_thirds": blend}
        threshold, _mcc = select_threshold(y_train, blend)
        pooled = _all_metrics(y_train, blend, float(threshold))
        sweep_rows.append({"k": k, "aggregation": "pooled_oof", **pooled})
        for fold in range(5):
            mask = fold_id == fold
            sweep_fold_rows.append({
                "k": k, "fold": fold,
                **_all_metrics(y_train[mask], blend[mask], float(threshold)),
            })
        print(f"[K={k}] pooled OOF AP+={pooled['auprc']:.4f} acc={pooled['accuracy']:.4f}",
              flush=True)

    sweep = pd.DataFrame(sweep_rows)
    sweep_folds = pd.DataFrame(sweep_fold_rows)
    # T-DDI reports mean +/- SD across folds; choose K on the primary metric.
    primary = str(protocol["contract"]["k_selection_metric"])
    fold_summary = sweep_folds.groupby("k").agg(
        **{f"{primary}_mean": (primary, "mean"), f"{primary}_sd": (primary, "std"),
           "accuracy_mean": ("accuracy", "mean"), "accuracy_sd": ("accuracy", "std"),
           "f1_macro_mean": ("f1_macro", "mean"), "f1_macro_sd": ("f1_macro", "std")}
    ).reset_index()
    chosen_k = int(fold_summary.loc[fold_summary[f"{primary}_mean"].idxmax(), "k"])
    print(f"CHOSEN K = {chosen_k} (by mean fold {primary})", flush=True)

    # ---- STEP 2: nested-FS CV at the chosen K (leakage sensitivity) -----------
    nested_oof = {name: np.full(len(train_indices), np.nan) for name in DL_COMPONENTS}
    nested_selection = []
    for fold, (relative_fit, relative_validation) in enumerate(fold_split):
        sub_fit = train_indices[relative_fit]
        sub_validation = train_indices[relative_validation]
        fold_state = tddi_feature_selection(
            features["rdkit2d"][sub_fit], labels[sub_fit], descriptor_names, fs_settings
        )
        available = sorted(fold_state["per_k_local"].keys())
        k_fold = min(available, key=lambda v: abs(v - chosen_k))
        x_fit = apply_selection(fold_state, features["rdkit2d"][sub_fit], k_fold)
        x_val = apply_selection(fold_state, features["rdkit2d"][sub_validation], k_fold)
        streams = _fit_dl(
            protocol, x_fit, labels[sub_fit], {"validation": x_val},
            tabfm_model, seed=42 + fold,
        )
        for name in DL_COMPONENTS:
            nested_oof[name][relative_validation] = streams[name]["validation"]
        nested_selection.append({
            "fold": fold, "k_used": k_fold,
            "n_after_audit": fold_state["n_after_audit"],
            "n_after_correlation_prune": fold_state["n_after_correlation_prune"],
            "selected_features": json.dumps(selected_names(fold_state, k_fold)),
        })
        print(f"[nested] fold {fold + 1}/5 at K={k_fold}", flush=True)
    nested_blend = np.column_stack(
        [svm_stream["cv5"], nested_oof["tabpfn_v2"], nested_oof["tabfm"]]
    ) @ EQUAL_THIRDS

    # ---- STEP 3: LOCK on all 324 at the chosen K, then score once -------------
    locked_features = selected_names(state, chosen_k)
    print(f"LOCKED {chosen_k} features: {locked_features}", flush=True)
    x_train = apply_selection(state, features["rdkit2d"][train_indices], chosen_k)
    targets = {
        "d1_test": apply_selection(state, features["rdkit2d"][test_indices], chosen_k),
        "drugage": apply_selection(state, cohorts["drugage"]["rdkit2d"], chosen_k),
        "agextend": apply_selection(state, cohorts["agextend"]["rdkit2d"], chosen_k),
    }
    scored = _fit_dl(protocol, x_train, y_train, targets, tabfm_model, seed=42)
    for key in targets:
        print(f"scored {key}: {len(targets[key])} rows", flush=True)

    svm_by_cohort = {
        "d1_test": svm_stream["d1_test"],
        "drugage": cohorts["drugage"]["svm"],
        "agextend": cohorts["agextend"]["svm"],
    }
    blend_scored = {
        key: np.column_stack(
            [svm_by_cohort[key], scored["tabpfn_v2"][key], scored["tabfm"][key]]
        ) @ EQUAL_THIRDS
        for key in targets
    }

    # ---- thresholds: train OOF only ------------------------------------------
    chosen_oof = sweep_oof[chosen_k]
    thresholds, threshold_rows = {}, []
    stream_map = {
        "svm_paper_setting": svm_stream["cv5"],
        "tabpfn_v2": chosen_oof["tabpfn_v2"],
        "tabfm": chosen_oof["tabfm"],
        "blend_eq_thirds": chosen_oof["blend_eq_thirds"],
    }
    for name, stream in stream_map.items():
        value, mcc = select_threshold(y_train, stream)
        thresholds[name] = float(value)
        threshold_rows.append({
            "model": name, "threshold": float(value),
            "threshold_source": f"324_train_oof_mcc_at_K{chosen_k}",
            "oof_mcc_at_selection": float(mcc), "external_labels_used": False,
        })

    # ---- metrics across every evaluation --------------------------------------
    metric_rows = []
    evaluations = [
        ("cv5_train_oof", y_train, {
            "svm_paper_setting": svm_stream["cv5"],
            "tabpfn_v2": chosen_oof["tabpfn_v2"], "tabfm": chosen_oof["tabfm"],
            "blend_eq_thirds": chosen_oof["blend_eq_thirds"]}),
        ("cv5_train_oof_nestedFS", y_train, {
            "svm_paper_setting": svm_stream["cv5"],
            "tabpfn_v2": nested_oof["tabpfn_v2"], "tabfm": nested_oof["tabfm"],
            "blend_eq_thirds": nested_blend}),
        ("d1_test_80_20", y_test, {
            "svm_paper_setting": svm_by_cohort["d1_test"],
            "tabpfn_v2": scored["tabpfn_v2"]["d1_test"], "tabfm": scored["tabfm"]["d1_test"],
            "blend_eq_thirds": blend_scored["d1_test"]}),
        ("drugage", cohorts["drugage"]["frame"].label.to_numpy(int), {
            "svm_paper_setting": svm_by_cohort["drugage"],
            "tabpfn_v2": scored["tabpfn_v2"]["drugage"], "tabfm": scored["tabfm"]["drugage"],
            "blend_eq_thirds": blend_scored["drugage"]}),
        ("agextend", cohorts["agextend"]["frame"].label.to_numpy(int), {
            "svm_paper_setting": svm_by_cohort["agextend"],
            "tabpfn_v2": scored["tabpfn_v2"]["agextend"], "tabfm": scored["tabfm"]["agextend"],
            "blend_eq_thirds": blend_scored["agextend"]}),
    ]
    for cohort, y, streams in evaluations:
        for model_name, probability in streams.items():
            for point, value in (("oof_mcc", thresholds[model_name]), ("fixed_0p5", 0.5)):
                metric_rows.append({
                    "cohort": cohort, "model": model_name, "operating_point": point,
                    "k_features": chosen_k, **_all_metrics(y, probability, value),
                })
    metrics = pd.DataFrame(metric_rows)

    per_fold = pd.DataFrame([
        {"model": "blend_eq_thirds", "cohort": "cv5_train_oof", "fold": f,
         **_all_metrics(y_train[fold_id == f], chosen_oof["blend_eq_thirds"][fold_id == f],
                        thresholds["blend_eq_thirds"])}
        for f in range(5)
    ] + [
        {"model": "blend_eq_thirds", "cohort": "cv5_train_oof_nestedFS", "fold": f,
         **_all_metrics(y_train[fold_id == f], nested_blend[fold_id == f],
                        thresholds["blend_eq_thirds"])}
        for f in range(5)
    ])

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".tddifs.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "fs_k_sweep_pooled.csv", sweep)
        _write_csv(tmp / "fs_k_sweep_per_fold.csv", sweep_folds)
        _write_csv(tmp / "fs_k_sweep_mean_sd.csv", fold_summary)
        _write_csv(tmp / "fs_nested_fold_selection.csv", pd.DataFrame(nested_selection))
        _write_csv(tmp / "fs_locked_features.csv", pd.DataFrame({
            "rank": range(1, len(locked_features) + 1), "feature": locked_features}))
        _write_csv(tmp / "component_thresholds.csv", pd.DataFrame(threshold_rows))
        _write_csv(tmp / "metrics_all.csv", metrics)
        _write_csv(tmp / "cv5_per_fold_metrics.csv", per_fold)
        _write_csv(tmp / "cv5_train_oof_predictions.csv", pd.DataFrame({
            "paper_row_index": train_indices, "fold": fold_id, "label": y_train,
            "probability_svm_paper_setting": svm_stream["cv5"],
            "probability_tabpfn_v2": chosen_oof["tabpfn_v2"],
            "probability_tabfm": chosen_oof["tabfm"],
            "blend_eq_thirds": chosen_oof["blend_eq_thirds"],
            "nested_probability_tabpfn_v2": nested_oof["tabpfn_v2"],
            "nested_probability_tabfm": nested_oof["tabfm"],
            "nested_blend_eq_thirds": nested_blend}))
        _write_csv(tmp / "d1_test_predictions.csv", pd.DataFrame({
            "paper_row_index": test_indices, "label": y_test,
            "probability_svm_paper_setting": svm_by_cohort["d1_test"],
            "probability_tabpfn_v2": scored["tabpfn_v2"]["d1_test"],
            "probability_tabfm": scored["tabfm"]["d1_test"],
            "blend_eq_thirds": blend_scored["d1_test"]}))
        for cohort in ("drugage", "agextend"):
            out = cohorts[cohort]["frame"].copy()
            out["probability_svm_paper_setting"] = svm_by_cohort[cohort]
            for name in DL_COMPONENTS:
                out[f"probability_{name}"] = scored[name][cohort]
            out["blend_eq_thirds"] = blend_scored[cohort]
            _write_csv(tmp / f"external_predictions_{cohort}.csv", out)

        cols = [("model", "Model"), ("n", "n"), ("auprc", "AP+"), ("auroc", "AUROC"),
                ("brier", "Brier"), ("accuracy", "Acc"), ("cohen_kappa", "Kappa"),
                ("f1_macro", "F1mac"), ("sensitivity", "Sens"), ("specificity", "Spec")]
        lines = [
            "# T-DDI-style feature selection + equal-thirds SVM / TabPFN-v2 / TabFM blend",
            "",
            "FS pipeline transcribed from T-DDI Methods (p.17): descriptor audit "
            "(missingness > 0.995, variance < 1e-10) -> training-median imputation -> "
            "ANOVA F ranking -> Pearson-correlation-aware prune (|r| > 0.995) -> RFE at "
            "predefined K counts -> K chosen by re-training under the same 5-fold CV.",
            "",
            f"Pool {state['n_input']} -> audit {state['n_after_audit']} -> "
            f"correlation prune {state['n_after_correlation_prune']}. "
            f"**Chosen K = {chosen_k}.**",
            "",
            "```", ", ".join(locked_features), "```",
            "",
            "The SVM component is NEVER feature-selected and NEVER refitted: it is the "
            "published SVC on the paper's seven DataWarrior descriptors, read from the "
            "sealed runs. Tanimoto is untouched and not part of this blend.",
            "",
            "## Feature-count sweep (5-fold CV, mean +/- SD, T-DDI Supplementary Table 3 style)",
            "",
            f"| K | {primary} | Accuracy | Macro F1 |", "|---|---|---|---|",
        ]
        for row in fold_summary.sort_values("k", ascending=False).itertuples(index=False):
            lines.append(
                f"| {int(row.k)} | {getattr(row, primary + '_mean'):.4f} ± "
                f"{getattr(row, primary + '_sd'):.4f} | "
                f"{row.accuracy_mean:.4f} ± {row.accuracy_sd:.4f} | "
                f"{row.f1_macro_mean:.4f} ± {row.f1_macro_sd:.4f} |"
            )
        lines.append("")
        for cohort, title in (
            ("cv5_train_oof", f"5-fold CV, D1 train (n=324), FS on train, K={chosen_k}"),
            ("cv5_train_oof_nestedFS", "5-fold CV with NESTED FS (leakage sensitivity)"),
            ("d1_test_80_20", "D1 held-out test, 80/20 split (n=81)"),
            ("drugage", "DrugAge positive-retrieval (n=446)"),
            ("agextend", "AgeXtend Table 6 (n=69)"),
        ):
            sub = metrics[(metrics.cohort == cohort) & (metrics.operating_point == "oof_mcc")]
            sub = sub.sort_values("auprc", ascending=False)
            lines += [f"## {title}", "",
                      "| " + " | ".join(c[1] for c in cols) + " |",
                      "|" + "---|" * len(cols)]
            for row in sub.itertuples(index=False):
                cells = []
                for key, _label in cols:
                    v = getattr(row, key)
                    cells.append(v if isinstance(v, str) else
                                 (str(int(v)) if key == "n" else
                                  ("nan" if not np.isfinite(v) else f"{v:.4f}")))
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        (tmp / "summary.md").write_text("\n".join(lines), encoding="utf-8")

        manifest = {
            "schema_version": f"{SCHEMA}.run_manifest.v1",
            "run_id": run_id, "protocol_sha256": protocol_sha256,
            "paper_split_sha256": split_sha256,
            "feature_selection": {
                "source": ("T-DDI Fulltext.pdf Methods p.17; sweep table "
                           "Supplementary_document.pdf Supplementary Table 3 p.14"),
                "pipeline": ["descriptor_audit", "training_median_imputation",
                             "anova_f_ranking", "pearson_correlation_prune_0.995",
                             "rfe_at_predefined_k", "k_chosen_by_kfold_cv_retraining"],
                **{k: fs_settings[k] for k in (
                    "audit_missing_ratio_threshold", "audit_variance_threshold",
                    "correlation_abs_threshold", "ranking", "reduction",
                    "rfe_step", "rfe_estimator_C", "k_ladder", "applies_to")},
                "n_input": state["n_input"],
                "n_after_audit": state["n_after_audit"],
                "n_after_correlation_prune": state["n_after_correlation_prune"],
                "k_evaluated": k_values,
                "k_chosen": chosen_k,
                "k_selection_metric": primary,
                "locked_features": locked_features,
                "nested_fold_k_used": [r["k_used"] for r in nested_selection],
            },
            "svm_component": {
                "source": "sealed streams, never refitted",
                "feature_set": "paper 7 DataWarrior descriptors",
                "feature_selected": False,
                "hyperparameters": "SVC(kernel=linear, C=1, gamma=1, probability=True, "
                                   "random_state=42) as published",
            },
            "tanimoto": "untouched; not a component of this blend",
            "blend": {"components": list(COMPONENTS), "weights": EQUAL_THIRDS.tolist()},
            "thresholds": thresholds,
            "selection_optimism_note": (
                "The K sweep selects K using the same 5-fold CV it reports, which mirrors "
                "T-DDI's own protocol but makes the cv5_train_oof row mildly optimistic. "
                "cv5_train_oof_nestedFS refits the whole FS pipeline inside each fold and "
                "is the leakage-clean cross-validation estimate. The held-out test and both "
                "external cohorts are unaffected either way."
            ),
            "test_or_external_labels_used_for_selection_or_threshold": False,
            "runtime": {"python": platform.python_version(), "platform": platform.platform()},
            "sealed_inputs_sha256": {
                k: sha256_file(root / v["path"]) for k, v in sealed.items()},
            "existing_runs_modified": False,
        }
        atomic_write_json(tmp / "RUN_MANIFEST.json", manifest)
        atomic_write_json(tmp / "COMPLETED.json", {
            "schema_version": f"{SCHEMA}.completed.v1", "status": "COMPLETE", "run_id": run_id,
            "run_manifest_sha256": sha256_file(tmp / "RUN_MANIFEST.json"),
            "artifact_hashes": {
                str(p.relative_to(tmp)): sha256_file(p)
                for p in sorted(tmp.rglob("*")) if p.is_file() and p.name != "COMPLETED.json"},
        })
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    print(json.dumps({"run": str(destination), "status": "COMPLETE"}, indent=2))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--quick", action="store_true",
                        help="Shrink the K ladder to {7, 30, full} for a fast first pass.")
    a = parser.parse_args(argv)
    run(root=a.root, config_path=a.config, positive_path=a.positive,
        negative_path=a.negative, run_id=a.run_id, quick=a.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
