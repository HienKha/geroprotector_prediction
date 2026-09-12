"""Fold-safe Optuna search over TabPFNv2 hyperparameters inside the equal-thirds
SVM/Tanimoto/TabPFNv2 blend.

Two objectives, chosen with --objective:

  standalone  Search {tabpfn n_estimators, softmax_temperature, panel} against
              standalone TabPFNv2 OOF AP+.  Tanimoto stays fixed at its current
              default C=1.0.  The winning TabPFN config is reused, unmodified,
              both as the reported "TabPFNv2 alone (tuned)" row and inside a
              blend built with paper_svm and the fixed-C Tanimoto.  This is the
              cheapest, most defensible way to answer "should the blend reuse
              the same TabPFNv2 params" with "yes" -- one search.

  blend       Search {tabpfn n_estimators, softmax_temperature, panel,
              tanimoto_C} jointly against the equal-thirds blend's own OOF AP+.
              Also reports what that blend-optimized TabPFN looks like alone,
              labelled honestly as "blend-optimized", since the blend optimum
              and the standalone optimum need not coincide.

paper_svm is never in either search space: it is contractually the published
paper's exact linear SVC (C=1, gamma=1) and its probabilities for D1 test,
DrugAge and AgeXtend are read verbatim from already-sealed prediction files
rather than refit.

Optuna maximizes AP+ (a threshold-free ranking metric, matching this whole
project's `primary_metric: ap_positive` convention).  Threshold selection
(fixed 0.5 and this experiment's own OOF-MCC) happens strictly after a
configuration is locked; it is never inside the Optuna objective.

Fold safety mirrors tabpfn3_paper405.py: 5 outer StratifiedKFold folds (seed 42,
the same folds every sealed OOF stream in this repository uses), each with its
own Optuna study restricted to that fold's fit rows (scored by a further 5-fold
inner CV); a separate final Optuna study over all 324 training rows produces the
one configuration that scores D1 test, DrugAge and AgeXtend once.  No outer-fold
validation row and no test/external label ever appears inside a trial's
objective.  Every sealed input is opened read-only and hash-verified; the run
refuses to start if its own output directory already exists.
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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from geroprotector.fixed_blend_paper405 import _features as _fixed_features
from geroprotector.fixed_blend_paper405 import (
    _fit_tanimoto,
    _selected_component_predictions,
    _tanimoto,
)
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.screening_blend_ablation import _safe_metrics
from geroprotector.screening_blend_altmodels import _morgan_from_smiles, _rdkit2d_from_smiles
from geroprotector.screening_blend_paper405 import load_locked_bundle
from geroprotector.tabpfn3_paper405 import (
    _build_panel,
    _positive_probability,
    _tabpfn3_estimator,
)
from geroprotector.traditional_paper405 import _read_sources, paper_split_indices
from geroprotector.weighted_blend_paper405 import select_threshold


class BlendOptunaError(RuntimeError):
    """Raised when a sealed input, a search-space, or a leakage contract fails."""


SCHEMA = "geroprotector.blend_svm_tani_tabpfnv2_optuna"
D1_ENDPOINT = "paper_binary"
DRUGAGE_ENDPOINT = "significant_positive_retrieval_background_not_certified_negative"
AGEXTEND_ENDPOINT = "published_independent_table6_binary"
PLACEHOLDER = "REQUIRED_AT_RUNTIME"


# --------------------------------------------------------------------------- io


def _regular_file(path: Path, role: str, expected_sha256: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise BlendOptunaError(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected_sha256 is not None and sha256_file(resolved) != expected_sha256:
        raise BlendOptunaError(f"{role} SHA256 differs from the protocol lock")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = yaml.safe_load(
        _regular_file(path, "Optuna protocol").read_text(encoding="utf-8")
    )
    expected_schema = f"{SCHEMA}.protocol.v1"
    if not isinstance(protocol, dict) or protocol.get("schema_version") != expected_schema:
        raise BlendOptunaError("Unknown Optuna protocol schema")
    if str(protocol.get("optuna", {}).get("required_optuna_version")) == PLACEHOLDER:
        raise BlendOptunaError(
            "Resolve required_optuna_version first: run scripts/check_optuna_env.sh "
            "and paste the printed version into the protocol under `optuna:`"
        )
    contract = protocol.get("paper_svm_contract", {})
    frozen_ok = contract.get("frozen") is True
    unsearched_ok = contract.get("never_included_in_search_space") is True
    if not (frozen_ok and unsearched_ok):
        raise BlendOptunaError("paper_svm freeze contract differs")
    cross = protocol.get("cross_fitting", {})
    if (
        cross.get("outer_validation_rows_used_for_trial_objective") is not False
        or cross.get("test_or_external_labels_used_for_search_or_threshold") is not False
    ):
        raise BlendOptunaError("Fold-safety contract differs")
    thresholds_block = protocol.get("thresholds", {})
    if thresholds_block.get("external_labels_may_select_or_change_threshold") is not False:
        raise BlendOptunaError("Threshold leakage contract differs")
    immutability = protocol.get("immutability", {})
    if immutability.get("existing_run_directories_are_read_only") is not True:
        raise BlendOptunaError("Immutability contract differs")
    for record in protocol.get("sealed_inputs", {}).values():
        if str(record.get("sha256")) == PLACEHOLDER:
            raise BlendOptunaError(f"Sealed input hash unresolved: {record['path']}")
        _regular_file(root / record["path"], f"sealed input {record['path']}", record["sha256"])
    return protocol, canonical_sha256(protocol)


def _require_optuna(expected_version: str):
    import optuna

    observed = importlib.metadata.version("optuna")
    if observed != str(expected_version):
        raise BlendOptunaError(
            f"Installed optuna {observed} differs from the protocol lock {expected_version}"
        )
    return optuna


# ------------------------------------------------------------------------ search


def _inner_split(
    fit_rows: np.ndarray, labels: np.ndarray, folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(fit_rows, labels[fit_rows]))


def _paper_svm_oof(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    fit_rows: np.ndarray,
    inner_split: list[tuple[np.ndarray, np.ndarray]],
    settings: dict[str, Any],
) -> np.ndarray:
    """paper_svm never varies across trials: compute its inner-CV OOF once per scope."""

    oof = np.full(len(fit_rows), np.nan)
    for relative_fit, relative_validation in inner_split:
        sub_fit = fit_rows[relative_fit]
        sub_validation = fit_rows[relative_validation]
        chemistry, _models, _audit = _selected_component_predictions(
            features,
            labels,
            sub_fit,
            sub_validation,
            settings,
            seed=42,
            requested=("paper_svm",),
        )
        oof[relative_validation] = chemistry["paper_svm"]
    if not np.isfinite(oof).all():
        raise BlendOptunaError("paper_svm inner-CV OOF is incomplete")
    return oof


def _objective(
    trial,
    *,
    objective_mode: str,
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    fit_rows: np.ndarray,
    inner_split: list[tuple[np.ndarray, np.ndarray]],
    paper_svm_oof: np.ndarray,
    panels_settings: dict[str, Any],
    tabpfn_model: dict[str, Any],
    search_space: dict[str, Any],
    trial_seed_base: int,
) -> float:
    import torch

    panel = trial.suggest_categorical("panel", search_space["tabpfn_panel"])
    n_estimators = trial.suggest_categorical(
        "n_estimators", search_space["tabpfn_n_estimators"]
    )
    low, high = search_space["tabpfn_softmax_temperature_range"]
    softmax_temperature = trial.suggest_float("softmax_temperature", low, high)
    tanimoto_c = None
    if objective_mode == "blend":
        low_c, high_c = search_space["tanimoto_C_range"]
        tanimoto_c = trial.suggest_float("tanimoto_C", low_c, high_c, log=True)

    model_settings = {**tabpfn_model, "softmax_temperature": softmax_temperature}
    oof = np.full(len(fit_rows), np.nan)
    for fold_index, (relative_fit, relative_validation) in enumerate(inner_split):
        sub_fit = fit_rows[relative_fit]
        sub_validation = fit_rows[relative_validation]
        fit_matrix, targets = _build_panel(
            panel,
            panels_settings,
            {"rdkit2d": features["rdkit2d"][sub_fit], "morgan": features["morgan"][sub_fit]},
            {
                "validation": {
                    "rdkit2d": features["rdkit2d"][sub_validation],
                    "morgan": features["morgan"][sub_validation],
                }
            },
        )
        estimator = _tabpfn3_estimator(
            model_settings, n_estimators, seed=trial_seed_base + fold_index
        )
        tabpfn_probability, _audit = _positive_probability(
            estimator, fit_matrix, labels[sub_fit], targets["validation"], singleton=False
        )
        del estimator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if objective_mode == "blend":
            tanimoto_probability, _model = _fit_tanimoto(
                features["morgan"][sub_fit],
                labels[sub_fit],
                features["morgan"][sub_validation],
                {"C": tanimoto_c},
            )
            oof[relative_validation] = (
                paper_svm_oof[relative_validation] + tanimoto_probability + tabpfn_probability
            ) / 3.0
        else:
            oof[relative_validation] = tabpfn_probability
    if not np.isfinite(oof).all():
        raise BlendOptunaError("Trial OOF is incomplete")
    return float(average_precision_score(labels[fit_rows], oof))


def _run_scope_search(
    optuna,
    *,
    label: str,
    objective_mode: str,
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    fit_rows: np.ndarray,
    panels_settings: dict[str, Any],
    tabpfn_model: dict[str, Any],
    search_space: dict[str, Any],
    cross_fitting: dict[str, Any],
    n_trials: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    inner_split = _inner_split(
        fit_rows, labels, int(cross_fitting["inner_folds"]), int(cross_fitting["inner_seed"])
    )
    paper_svm_settings = {
        "paper_svm": {
            "kernel": "linear",
            "C": 1.0,
            "gamma": 1.0,
            "probability": True,
            "random_state": 42,
            "preprocessing": "none_matching_public_notebook",
        }
    }
    paper_svm_oof = _paper_svm_oof(features, labels, fit_rows, inner_split, paper_svm_settings)

    def objective(trial):
        return _objective(
            trial,
            objective_mode=objective_mode,
            features=features,
            labels=labels,
            fit_rows=fit_rows,
            inner_split=inner_split,
            paper_svm_oof=paper_svm_oof,
            panels_settings=panels_settings,
            tabpfn_model=tabpfn_model,
            search_space=search_space,
            trial_seed_base=seed,
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.NopPruner(),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(
        f"[{label}] best AP={study.best_trial.value:.4f} params={study.best_trial.params}",
        flush=True,
    )
    return study.best_trial.params, study.trials_dataframe()


def _refit_and_score(
    params: dict[str, Any],
    objective_mode: str,
    *,
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    fit_rows: np.ndarray,
    panels_settings: dict[str, Any],
    tabpfn_model: dict[str, Any],
    default_tanimoto_c: float,
    target_raw: dict[str, dict[str, np.ndarray]],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Refit the winning trial on all of ``fit_rows`` and score every target block."""

    fit_matrix, targets = _build_panel(
        params["panel"],
        panels_settings,
        {"rdkit2d": features["rdkit2d"][fit_rows], "morgan": features["morgan"][fit_rows]},
        target_raw,
    )
    model_settings = {**tabpfn_model, "softmax_temperature": params["softmax_temperature"]}
    estimator = _tabpfn3_estimator(model_settings, params["n_estimators"], seed=seed)
    tabpfn_scored, audits = {}, {}
    for name, matrix in targets.items():
        probability, audit = _positive_probability(
            estimator, fit_matrix, labels[fit_rows], matrix, singleton=True
        )
        tabpfn_scored[name] = probability
        audits[name] = audit
    del estimator

    tanimoto_c = params.get("tanimoto_C", default_tanimoto_c)
    train_bits = features["morgan"][fit_rows]
    tanimoto_scored = {}
    for name in target_raw:
        probability, _model = _fit_tanimoto(
            train_bits, labels[fit_rows], target_raw[name]["morgan"], {"C": tanimoto_c}
        )
        tanimoto_scored[name] = probability
    return tabpfn_scored, tanimoto_scored, {"tanimoto_C": tanimoto_c, "tabpfn_audit": audits}


# ---------------------------------------------------------------------------- run


def run(
    *,
    root: Path,
    config_path: Path,
    positive_path: Path,
    negative_path: Path,
    objective_mode: str,
    run_id: str,
) -> Path:
    if objective_mode not in {"standalone", "blend"}:
        raise BlendOptunaError("--objective must be 'standalone' or 'blend'")
    if not re.fullmatch(r"blend_svm_tani_tabpfnv2_optuna_[a-z0-9_.-]+", run_id):
        raise BlendOptunaError("RUN_ID must start with blend_svm_tani_tabpfnv2_optuna_")
    root = root.resolve()
    protocol, protocol_sha256 = load_protocol(root, config_path)
    optuna = _require_optuna(protocol["optuna"]["required_optuna_version"])
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise BlendOptunaError(f"Run directory already exists: {destination}")

    sealed = protocol["sealed_inputs"]
    bundle = load_locked_bundle(
        root / sealed["screening_bundle"]["path"],
        expected_artifact_sha256=sealed["screening_bundle"]["sha256"],
    )

    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    frame, _audit = _read_sources(
        _regular_file(positive_path, "positive source", protocol["sources"]["positive_sha256"]),
        _regular_file(negative_path, "negative source", protocol["sources"]["negative_sha256"]),
        traditional,
    )
    if len(frame) != int(protocol["sources"]["expected_rows"]):
        raise BlendOptunaError("D1 row count differs from 405")
    train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    if split_sha256 != protocol["paper_split"]["expected_assignment_sha256"]:
        raise BlendOptunaError("Paper split differs from the sealed assignment")
    labels = frame["label"].to_numpy(dtype=int)
    smiles, _ = _validated_raw_smiles(frame)
    features = _fixed_features(frame, smiles, fixed_protocol)

    sealed_train_bits = np.asarray(bundle["tanimoto_train_bits"], dtype=np.uint8)
    if not np.array_equal(features["morgan"][train_indices], sealed_train_bits):
        raise BlendOptunaError("Rebuilt Morgan bits differ from the sealed bundle")

    test_components = pd.read_csv(root / sealed["weighted_test_components"]["path"])
    drugage = pd.read_csv(root / sealed["drugage_scored"]["path"])
    agextend = pd.read_csv(root / sealed["agextend_scored"]["path"])
    drugage = drugage.assign(label=drugage.has_significant_positive.astype(int))

    drugage_smiles = drugage.source_smiles.astype(str).tolist()
    agextend_smiles = agextend.source_smiles.astype(str).tolist()
    drugage_raw = {
        "rdkit2d": _rdkit2d_from_smiles(drugage_smiles, bundle["descriptor_names"]),
        "morgan": _morgan_from_smiles(drugage_smiles, bundle["portable_contract"]),
    }
    agextend_raw = {
        "rdkit2d": _rdkit2d_from_smiles(agextend_smiles, bundle["descriptor_names"]),
        "morgan": _morgan_from_smiles(agextend_smiles, bundle["portable_contract"]),
    }
    stored_drugage_similarity = drugage.maximum_tanimoto_to_fitted_train.to_numpy(dtype=float)
    rebuilt_drugage_similarity = _tanimoto(
        drugage_raw["morgan"], np.asarray(bundle["tanimoto_train_bits"], dtype=np.uint8)
    ).max(axis=1)
    if float(np.max(np.abs(rebuilt_drugage_similarity - stored_drugage_similarity))) > 1e-6:
        raise BlendOptunaError("Rebuilt DrugAge similarity differs from the sealed file")

    search_space = protocol["search_space"]
    panels_settings = protocol["panels"]
    tabpfn_model = protocol["model"]
    cross_fitting = protocol["cross_fitting"]
    budget = protocol["search_budget"]

    # -- 1. nested outer-fold search: builds this experiment's own OOF stream -----
    folds = StratifiedKFold(
        n_splits=int(cross_fitting["outer_folds"]),
        shuffle=True,
        random_state=int(cross_fitting["outer_seed"]),
    )
    fold_split = list(folds.split(train_indices, labels[train_indices]))
    tabpfn_oof = np.full(len(train_indices), np.nan)
    tanimoto_oof = np.full(len(train_indices), np.nan)
    outer_fold_id = np.full(len(train_indices), -1, dtype=int)
    outer_winners = []
    outer_trial_frames = []
    for fold, (relative_fit, relative_validation) in enumerate(fold_split):
        sub_fit = train_indices[relative_fit]
        sub_validation = train_indices[relative_validation]
        print(
            f"outer fold {fold + 1}/{len(fold_split)} Optuna search ({objective_mode})",
            flush=True,
        )
        params, trials = _run_scope_search(
            optuna,
            label=f"outer {fold + 1}",
            objective_mode=objective_mode,
            features=features,
            labels=labels,
            fit_rows=sub_fit,
            panels_settings=panels_settings,
            tabpfn_model=tabpfn_model,
            search_space=search_space,
            cross_fitting=cross_fitting,
            n_trials=int(budget["n_trials_outer_fold"]),
            seed=1000 + fold,
        )
        trials.insert(0, "outer_fold", fold)
        outer_trial_frames.append(trials)
        outer_winners.append({"outer_fold": fold, **params})

        tabpfn_scored, tanimoto_scored, _audit = _refit_and_score(
            params,
            objective_mode,
            features=features,
            labels=labels,
            fit_rows=sub_fit,
            panels_settings=panels_settings,
            tabpfn_model=tabpfn_model,
            default_tanimoto_c=float(search_space["tanimoto_C_default_for_standalone_mode"]),
            target_raw={
                "validation": {
                    "rdkit2d": features["rdkit2d"][sub_validation],
                    "morgan": features["morgan"][sub_validation],
                }
            },
            seed=42,
        )
        tabpfn_oof[relative_validation] = tabpfn_scored["validation"]
        tanimoto_oof[relative_validation] = tanimoto_scored["validation"]
        outer_fold_id[relative_validation] = fold

    if not np.isfinite(tabpfn_oof).all() or not np.isfinite(tanimoto_oof).all():
        raise BlendOptunaError("Outer OOF is incomplete")

    paper_svm_full_oof = pd.read_csv(root / sealed["weighted_train_oof"]["path"])
    paper_svm_by_row = paper_svm_full_oof.set_index("paper_row_index").probability_paper_svm
    paper_svm_oof_324 = paper_svm_by_row.loc[train_indices].to_numpy(dtype=float)

    blend_oof = (paper_svm_oof_324 + tanimoto_oof + tabpfn_oof) / 3.0

    # -- 2. final Optuna study over all 324 training rows --------------------------
    print(f"final Optuna search on all 324 rows ({objective_mode})", flush=True)
    final_params, final_trials = _run_scope_search(
        optuna,
        label="final",
        objective_mode=objective_mode,
        features=features,
        labels=labels,
        fit_rows=train_indices,
        panels_settings=panels_settings,
        tabpfn_model=tabpfn_model,
        search_space=search_space,
        cross_fitting=cross_fitting,
        n_trials=int(budget["n_trials_final"]),
        seed=2000,
    )

    target_raw = {
        "d1_test": {
            "rdkit2d": features["rdkit2d"][test_indices],
            "morgan": features["morgan"][test_indices],
        },
        "drugage": drugage_raw,
        "agextend": agextend_raw,
    }
    tabpfn_scored, tanimoto_scored, score_audit = _refit_and_score(
        final_params,
        objective_mode,
        features=features,
        labels=labels,
        fit_rows=train_indices,
        panels_settings=panels_settings,
        tabpfn_model=tabpfn_model,
        default_tanimoto_c=float(search_space["tanimoto_C_default_for_standalone_mode"]),
        target_raw=target_raw,
        seed=42,
    )

    paper_svm_scored = {
        "d1_test": test_components.set_index("paper_row_index").loc[
            test_indices, "probability_paper_svm"
        ].to_numpy(dtype=float),
        "drugage": drugage.probability_paper_svm.to_numpy(dtype=float),
        "agextend": agextend.probability_paper_svm.to_numpy(dtype=float),
    }

    blend_scored = {
        name: (paper_svm_scored[name] + tanimoto_scored[name] + tabpfn_scored[name]) / 3.0
        for name in target_raw
    }

    # -- 3. thresholds from this experiment's own OOF (never test/external) -------
    y_train = labels[train_indices]
    tabpfn_threshold, tabpfn_oof_mcc = select_threshold(y_train, tabpfn_oof)
    blend_threshold, blend_oof_mcc = select_threshold(y_train, blend_oof)

    tag = f"{objective_mode}_tuned"
    threshold_rows = [
        {
            "model": f"tabpfn_v2_{tag}",
            "threshold": float(tabpfn_threshold),
            "threshold_source": "this_experiments_own_5fold_outer_oof",
            "oof_mcc_at_selection": float(tabpfn_oof_mcc),
        },
        {
            "model": f"blend_equal_thirds_{tag}",
            "threshold": float(blend_threshold),
            "threshold_source": "this_experiments_own_5fold_outer_oof",
            "oof_mcc_at_selection": float(blend_oof_mcc),
        },
    ]

    # -- 4. metrics -----------------------------------------------------------------
    d1_test_labels = (
        test_components.set_index("paper_row_index")
        .loc[test_indices, "label"]
        .to_numpy(dtype=int)
    )
    cohort_labels = {
        "d1_test": ("d1_paper_test", D1_ENDPOINT, d1_test_labels),
        "drugage": ("drugage", DRUGAGE_ENDPOINT, drugage.label.to_numpy(dtype=int)),
        "agextend": ("agextend", AGEXTEND_ENDPOINT, agextend.label.to_numpy(dtype=int)),
    }
    metric_rows = []
    for name, (cohort, endpoint, y) in cohort_labels.items():
        for model, probability, threshold in (
            (f"tabpfn_v2_{tag}", tabpfn_scored[name], tabpfn_threshold),
            (f"blend_equal_thirds_{tag}", blend_scored[name], blend_threshold),
        ):
            for operating_point, value in (("fixed_0p5", 0.5), ("oof_mcc", threshold)):
                metrics = _safe_metrics(y, probability, value)
                metric_rows.append(
                    {
                        "cohort": cohort,
                        "endpoint": endpoint,
                        "model": model,
                        "operating_point": operating_point,
                        **metrics,
                    }
                )
    metrics_frame = pd.DataFrame(metric_rows)

    untuned = pd.read_csv(root / sealed["untuned_comparison"]["path"])
    untuned_reference = untuned[
        untuned.model.isin(["blend_equal_thirds_svm_tani_tabpfnv2", "tabpfn_v2"])
    ].copy()
    untuned_reference["result_source"] = "sealed_untuned_reference"
    comparison_columns = [
        "cohort", "endpoint", "model", "threshold_rule", "threshold", "result_source",
        "auprc_average_precision_positive", "auroc", "brier", "mcc", "macro_f1",
        "recall_sensitivity", "specificity",
    ]
    tuned_reference = metrics_frame.rename(columns={"operating_point": "threshold_rule"}).copy()
    tuned_reference["result_source"] = "computed_this_run"
    comparison = pd.concat(
        [untuned_reference[comparison_columns], tuned_reference[comparison_columns]],
        ignore_index=True,
    )

    # -- 5. write -------------------------------------------------------------------
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".blendoptuna.work-", dir=destination.parent))
    try:
        for fold, trials in enumerate(outer_trial_frames):
            _write_csv(temporary / f"optuna_trials_outer_fold_{fold}.csv", trials)
        _write_csv(temporary / "optuna_trials_final.csv", final_trials)
        atomic_write_json(
            temporary / "selected_config.json",
            {
                "objective_mode": objective_mode,
                "outer_fold_winners": outer_winners,
                "final_winner": final_params,
            },
        )
        _write_csv(
            temporary / "d1_train_oof_predictions.csv",
            pd.DataFrame(
                {
                    "paper_row_index": train_indices,
                    "label": y_train,
                    "outer_fold": outer_fold_id,
                    "probability_paper_svm": paper_svm_oof_324,
                    "probability_tanimoto_svc_tuned": tanimoto_oof,
                    "probability_tabpfn_v2_tuned": tabpfn_oof,
                    "blend_equal_thirds_tuned": blend_oof,
                }
            ),
        )
        _write_csv(temporary / "component_thresholds.csv", pd.DataFrame(threshold_rows))
        _write_csv(
            temporary / "d1_test_predictions.csv",
            pd.DataFrame(
                {
                    "paper_row_index": test_indices,
                    "label": cohort_labels["d1_test"][2],
                    "probability_tabpfn_v2_tuned": tabpfn_scored["d1_test"],
                    "blend_equal_thirds_tuned": blend_scored["d1_test"],
                }
            ),
        )
        for name in ("drugage", "agextend"):
            source = drugage if name == "drugage" else agextend
            _write_csv(
                temporary / f"external_predictions_{name}.csv",
                source.assign(
                    probability_tabpfn_v2_tuned=tabpfn_scored[name],
                    blend_equal_thirds_tuned=blend_scored[name],
                ),
            )
        _write_csv(temporary / "metrics_all.csv", metrics_frame)
        _write_csv(temporary / "comparison_metrics_tuned_vs_untuned.csv", comparison)

        manifest = {
            "schema_version": f"{SCHEMA}.run_manifest.v1",
            "run_id": run_id,
            "objective_mode": objective_mode,
            "protocol_sha256": protocol_sha256,
            "paper_split_sha256": split_sha256,
            "final_winner": final_params,
            "outer_fold_winners": outer_winners,
            "thresholds": {row["model"]: row["threshold"] for row in threshold_rows},
            "tanimoto_C_used_final": float(score_audit["tanimoto_C"]),
            "optuna_version": importlib.metadata.version("optuna"),
            "runtime": {"python": platform.python_version(), "platform": platform.platform()},
            "sealed_inputs_sha256": {
                name: sha256_file(root / record["path"]) for name, record in sealed.items()
            },
            "outer_validation_rows_used_for_trial_objective": False,
            "test_or_external_labels_used_for_search_or_threshold": False,
            "existing_runs_modified": False,
        }
        atomic_write_json(temporary / "RUN_MANIFEST.json", manifest)

        lines = [
            f"# Optuna-tuned SVM/Tanimoto/TabPFNv2 -- objective={objective_mode}",
            "",
            f"Final winning configuration: `{final_params}`.",
            f"Tanimoto C used for final scoring: {score_audit['tanimoto_C']:.6f}.",
            "",
            comparison.to_string(index=False),
            "",
        ]
        (temporary / "summary.md").write_text("\n".join(lines), encoding="utf-8")

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
    parser.add_argument("--objective", required=True, choices=["standalone", "blend"])
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args(argv)
    run(
        root=arguments.root,
        config_path=arguments.config,
        positive_path=arguments.positive,
        negative_path=arguments.negative,
        objective_mode=arguments.objective,
        run_id=arguments.run_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
