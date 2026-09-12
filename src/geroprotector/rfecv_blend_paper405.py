"""RFECV feature selection, then an equal-thirds SVM / TabPFN-v2 / TabFM blend.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
The user asked for feature selection "likewise" to an npj Digital Medicine
paper.  That article is paywalled and could not be read, and the accompanying
released code (github.com/HienKha/tddi v1.0.0) performs NO recursive feature
elimination: its `--num_features_to_drop 6` simply removes six identity columns
(drugid/drugname/drugsmiles for each drug), and
`list_of_all_features_ascending_order.txt` is a plain column list, not an
importance ranking.  So this module does NOT claim to replicate that paper's
methodology.  It implements the specification the user gave directly:

    RFECV, minimum 5 features, with no feature forcibly retained
    (in particular the paper's seven DataWarrior descriptors are NOT protected).

FEATURE POOL -- and why the 7 DataWarrior descriptors are absent
----------------------------------------------------------------
The pool is the 205-descriptor RDKit2D panel (217 raw, fold-locally
median-imputed and variance-filtered).  The seven DataWarrior descriptors could
NOT be added: they are not stored for the DrugAge/AgeXtend cohorts, and the
DataWarrior CLI needs a Java runtime that is not installed here, so they cannot
be recomputed for the external validation.  Including them would have made the
external step impossible.  This is a hard environmental constraint, not a
modelling choice.

LEAKAGE: FEATURE SELECTION IS NESTED FOR THE CROSS-VALIDATION
-------------------------------------------------------------
Running RFECV once on all 324 rows and then cross-validating on those same rows
would be optimistically biased -- the selector would have seen every validation
fold's labels.  So:

  * 5-fold CV      : RFECV is refitted INSIDE each outer training fold, and the
                     held-out fold is scored with only that fold's features.
                     This is the honest cross-validation estimate.
  * lock -> test   : RFECV is fitted ONCE on all 324 training rows, the feature
                     set is frozen, models are fitted on it, and the 81 held-out
                     rows plus both external cohorts are scored once.

BLEND
-----
Equal thirds of three components, all on the selected features:
  * svm_paper_setting : SVC(kernel='linear', C=1, gamma=1, probability=True,
                        random_state=42) -- the published paper's hyperparameters
  * tabpfn_v2
  * tabfm
Tanimoto-SVC is NOT part of this blend: it consumes binary Morgan fingerprints,
which are outside a continuous-descriptor feature-selection experiment.

DOCUMENTED DEVIATION: the published SVM is used WITHOUT scaling on its seven
DataWarrior descriptors.  Here every component sits on RDKit2D descriptors whose
scales differ by orders of magnitude, where libsvm converges pathologically
slowly unscaled.  A StandardScaler fitted on the fit rows only is therefore
applied for the SVM and inside RFECV.  This matches the source manuscript's own
wording ("scaled data"), though not its notebook.
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
from sklearn.feature_selection import RFECV
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
    _apply_context,
    _imputer_context,
    _morgan_from_smiles,
    _rdkit2d_from_smiles,
)
from geroprotector.screening_blend_paper405 import load_locked_bundle
from geroprotector.screening_blend_tabfm import _load_tabfm, _tabfm_probability
from geroprotector.tabpfn3_paper405 import _positive_probability, _tabpfn3_estimator
from geroprotector.traditional_paper405 import _read_sources, paper_split_indices
from geroprotector.weighted_blend_paper405 import select_threshold


class RFECVBlendError(RuntimeError):
    """Raised when a sealed input, a contract or an invariant fails."""


SCHEMA = "geroprotector.rfecv_blend_paper405"
COMPONENTS = ("svm_paper_setting", "tabpfn_v2", "tabfm")
EQUAL_THIRDS = np.full(3, 1.0 / 3.0)
D1_ENDPOINT = "paper_binary"
DRUGAGE_ENDPOINT = "significant_positive_retrieval_background_not_certified_negative"
AGEXTEND_ENDPOINT = "published_independent_table6_binary"


def _regular_file(path: Path, role: str, expected: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RFECVBlendError(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected is not None and sha256_file(resolved) != expected:
        raise RFECVBlendError(f"{role} SHA256 differs from the protocol lock")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    text = _regular_file(path, "RFECV blend protocol").read_text(encoding="utf-8")
    protocol = yaml.safe_load(text)
    expected_schema = f"{SCHEMA}.protocol.v1"
    if not isinstance(protocol, dict) or protocol.get("schema_version") != expected_schema:
        raise RFECVBlendError("Unknown RFECV blend protocol schema")
    fs = protocol.get("feature_selection", {})
    if (
        fs.get("method") != "RFECV"
        or int(fs.get("min_features_to_select", 0)) < 1
        or fs.get("forced_retained_features") not in (None, [], "none")
        or fs.get("nested_inside_cross_validation") is not True
        or fs.get("fitted_on_training_rows_only") is not True
    ):
        raise RFECVBlendError("Feature-selection contract differs")
    blend = protocol.get("blend", {})
    if (
        blend.get("components") != list(COMPONENTS)
        or not np.allclose(blend.get("weights"), EQUAL_THIRDS, rtol=0.0, atol=1e-15)
        or blend.get("weights_selected_from_data") is not False
    ):
        raise RFECVBlendError("Blend contract differs")
    if protocol.get("contract", {}).get(
        "test_or_external_labels_used_for_selection_or_threshold"
    ) is not False:
        raise RFECVBlendError("Leakage contract differs")
    immutability = protocol.get("immutability", {})
    if immutability.get("existing_run_directories_are_read_only") is not True:
        raise RFECVBlendError("Immutability contract differs")
    for record in protocol.get("sealed_inputs", {}).values():
        _regular_file(root / record["path"], f"sealed input {record['path']}", record["sha256"])
    return protocol, canonical_sha256(protocol)


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


def _run_rfecv(x_fit: np.ndarray, y_fit: np.ndarray, settings: dict[str, Any]):
    """Scale on fit rows, then RFECV.  Returns (support mask, scaler, n_selected)."""

    scaler = StandardScaler().fit(x_fit)
    selector = RFECV(
        estimator=SVC(
            kernel="linear",
            C=float(settings["estimator_C"]),
            random_state=int(settings["estimator_random_state"]),
        ),
        step=int(settings["step"]),
        min_features_to_select=int(settings["min_features_to_select"]),
        cv=StratifiedKFold(
            n_splits=int(settings["cv_folds"]), shuffle=True,
            random_state=int(settings["cv_seed"]),
        ),
        scoring=str(settings["scoring"]),
        n_jobs=1,
    )
    selector.fit(scaler.transform(x_fit), y_fit)
    return selector.support_.copy(), scaler, int(selector.n_features_)


def _fit_components(
    protocol: dict[str, Any], x_fit: np.ndarray, y_fit: np.ndarray,
    targets: dict[str, np.ndarray], tabfm_model, seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Fit all three components on the already-selected features."""

    output: dict[str, dict[str, np.ndarray]] = {}

    # SVM: paper hyperparameters, standardised (see module docstring).
    svm_scaler = StandardScaler().fit(x_fit)
    svm = SVC(kernel="linear", C=1.0, gamma=1.0, probability=True, random_state=42)
    svm.fit(svm_scaler.transform(x_fit), y_fit)
    output["svm_paper_setting"] = {
        k: np.asarray(svm.predict_proba(svm_scaler.transform(v))[:, 1], dtype=float)
        for k, v in targets.items()
    }

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


def run(
    *, root: Path, config_path: Path, positive_path: Path, negative_path: Path, run_id: str
) -> Path:
    if not re.fullmatch(r"rfecv_blend_[a-z0-9_.-]+", run_id):
        raise RFECVBlendError("RUN_ID must start with rfecv_blend_")
    root = root.resolve()
    protocol, protocol_sha256 = load_protocol(root, config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise RFECVBlendError(f"Run directory already exists: {destination}")
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
    train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    if split_sha256 != protocol["paper_split"]["expected_assignment_sha256"]:
        raise RFECVBlendError("Paper split differs from the sealed assignment")
    labels = frame["label"].to_numpy(dtype=int)
    smiles, _ = _validated_raw_smiles(frame)
    features = _features(frame, smiles, fixed_protocol)
    descriptor_names = list(bundle["descriptor_names"])
    y_train, y_test = labels[train_indices], labels[test_indices]

    # -- external cohorts: raw RDKit2D (computable from SMILES for every cohort) --
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
            raise RFECVBlendError(f"Rebuilt {cohort} similarity differs from the sealed file")
        cohorts[cohort] = payload
        print(f"external features rebuilt for {cohort}: {len(scored)} rows", flush=True)

    fs_settings = protocol["feature_selection"]
    tabfm_model = _load_tabfm(protocol["tabfm"])
    print("TabFM weights loaded", flush=True)

    # ---- 1. NESTED 5-fold CV: RFECV refitted inside every outer fold -----------
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_split = list(folds.split(train_indices, y_train))
    fold_id = np.full(len(train_indices), -1, dtype=int)
    oof = {name: np.full(len(train_indices), np.nan) for name in COMPONENTS}
    fold_selection = []
    for fold, (relative_fit, relative_validation) in enumerate(fold_split):
        sub_fit = train_indices[relative_fit]
        sub_validation = train_indices[relative_validation]
        fold_id[relative_validation] = fold
        context = _imputer_context(features["rdkit2d"][sub_fit])
        x_fit_all = _apply_context(context, features["rdkit2d"][sub_fit])
        x_val_all = _apply_context(context, features["rdkit2d"][sub_validation])
        support, _scaler, n_selected = _run_rfecv(x_fit_all, labels[sub_fit], fs_settings)
        chosen = [descriptor_names[i] for i, keep in enumerate(support) if keep]
        fold_selection.append({
            "fold": fold, "n_selected": n_selected,
            "n_candidates": int(x_fit_all.shape[1]),
            "selected_features": json.dumps(chosen),
        })
        print(f"[fold {fold}] RFECV selected {n_selected}/{x_fit_all.shape[1]}", flush=True)
        streams = _fit_components(
            protocol, x_fit_all[:, support], labels[sub_fit],
            {"validation": x_val_all[:, support]}, tabfm_model, seed=42 + fold,
        )
        for name in COMPONENTS:
            oof[name][relative_validation] = streams[name]["validation"]
        print(f"[fold {fold}] components fitted", flush=True)
    for name in COMPONENTS:
        if not np.isfinite(oof[name]).all():
            raise RFECVBlendError(f"{name} OOF is incomplete")
    blend_oof = np.column_stack([oof[n] for n in COMPONENTS]) @ EQUAL_THIRDS

    # ---- 2. LOCK: RFECV once on all 324, freeze, then score once --------------
    print("final RFECV on all 324 training rows", flush=True)
    context = _imputer_context(features["rdkit2d"][train_indices])
    x_train_all = _apply_context(context, features["rdkit2d"][train_indices])
    support, _scaler, n_selected = _run_rfecv(x_train_all, y_train, fs_settings)
    selected_names = [descriptor_names[i] for i, keep in enumerate(support) if keep]
    print(f"LOCKED feature set: {n_selected}/{x_train_all.shape[1]} -> {selected_names}",
          flush=True)

    targets_all = {
        "d1_test": _apply_context(context, features["rdkit2d"][test_indices])[:, support],
        "drugage": _apply_context(context, cohorts["drugage"]["rdkit2d"])[:, support],
        "agextend": _apply_context(context, cohorts["agextend"]["rdkit2d"])[:, support],
    }
    scored = _fit_components(
        protocol, x_train_all[:, support], y_train, targets_all, tabfm_model, seed=42
    )
    for key in targets_all:
        print(f"scored {key}: {len(targets_all[key])} rows", flush=True)
    blend_scored = {
        key: np.column_stack([scored[n][key] for n in COMPONENTS]) @ EQUAL_THIRDS
        for key in targets_all
    }

    # ---- 3. thresholds from the train OOF only --------------------------------
    threshold_rows, thresholds = [], {}
    for name, stream in [*[(n, oof[n]) for n in COMPONENTS], ("blend_eq_thirds", blend_oof)]:
        value, mcc = select_threshold(y_train, stream)
        thresholds[name] = float(value)
        threshold_rows.append({
            "model": name, "threshold": float(value),
            "threshold_source": "nested_rfecv_324_train_oof_mcc",
            "oof_mcc_at_selection": float(mcc), "external_labels_used": False,
        })

    # ---- 4. metrics ------------------------------------------------------------
    metric_rows = []
    evaluations = [
        ("cv5_train_oof", y_train, {**{n: oof[n] for n in COMPONENTS},
                                    "blend_eq_thirds": blend_oof}),
        ("d1_test_80_20", y_test, {**{n: scored[n]["d1_test"] for n in COMPONENTS},
                                   "blend_eq_thirds": blend_scored["d1_test"]}),
        ("drugage", cohorts["drugage"]["frame"].label.to_numpy(int),
         {**{n: scored[n]["drugage"] for n in COMPONENTS},
          "blend_eq_thirds": blend_scored["drugage"]}),
        ("agextend", cohorts["agextend"]["frame"].label.to_numpy(int),
         {**{n: scored[n]["agextend"] for n in COMPONENTS},
          "blend_eq_thirds": blend_scored["agextend"]}),
    ]
    for cohort, y, streams in evaluations:
        for model_name, probability in streams.items():
            for point, value in (("oof_mcc", thresholds[model_name]), ("fixed_0p5", 0.5)):
                metric_rows.append({
                    "cohort": cohort, "model": model_name, "operating_point": point,
                    "feature_set": "rfecv_selected", "n_features": n_selected,
                    **_all_metrics(y, probability, value),
                })
    metrics = pd.DataFrame(metric_rows)

    per_fold = pd.DataFrame([
        {"model": "blend_eq_thirds", "fold": f,
         **_all_metrics(y_train[fold_id == f], blend_oof[fold_id == f],
                        thresholds["blend_eq_thirds"])}
        for f in range(5)
    ])

    # ---- 5. comparison against the sealed full-panel equal-thirds blends -------
    comparison_columns = ["cohort", "model", "threshold_rule", "feature_set",
                          "auprc", "auroc", "brier", "accuracy", "cohen_kappa",
                          "f1_macro", "sensitivity", "specificity"]
    mine = metrics[metrics.operating_point == "oof_mcc"].copy()
    mine["threshold_rule"] = "oof_mcc"
    mine["model"] = mine["model"] + "_rfecv"
    reference = pd.read_csv(root / sealed["eq_thirds_comparison"]["path"])
    reference = reference[
        (reference.threshold_rule == "oof_mcc")
        & reference.model.isin([
            "blend_equal_thirds_svm_tani_tabpfnv2", "tabpfn_v2", "paper_svm",
        ])
    ].copy()
    reference = reference.rename(columns={
        "auprc_average_precision_positive": "auprc",
        "recall_sensitivity": "sensitivity", "macro_f1": "f1_macro",
    })
    reference["feature_set"] = "full_rdkit2d_panel_plus_tanimoto"
    reference["cohort"] = reference["cohort"].replace({"d1_paper_test": "d1_test_80_20"})
    for column in comparison_columns:
        if column not in reference:
            reference[column] = np.nan
    comparison = pd.concat(
        [mine[comparison_columns], reference[comparison_columns]], ignore_index=True
    )

    predictions_oof = pd.DataFrame({
        "paper_row_index": train_indices, "fold": fold_id, "label": y_train,
        **{f"probability_{n}": oof[n] for n in COMPONENTS},
        "blend_eq_thirds": blend_oof,
    })
    predictions_test = pd.DataFrame({
        "paper_row_index": test_indices, "label": y_test,
        **{f"probability_{n}": scored[n]["d1_test"] for n in COMPONENTS},
        "blend_eq_thirds": blend_scored["d1_test"],
    })

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".rfecv.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "rfecv_fold_selection.csv", pd.DataFrame(fold_selection))
        _write_csv(tmp / "rfecv_locked_features.csv", pd.DataFrame({
            "rank": range(1, n_selected + 1), "feature": selected_names,
        }))
        _write_csv(tmp / "component_thresholds.csv", pd.DataFrame(threshold_rows))
        _write_csv(tmp / "metrics_all.csv", metrics)
        _write_csv(tmp / "cv5_per_fold_metrics.csv", per_fold)
        _write_csv(tmp / "comparison_vs_full_panel.csv", comparison)
        _write_csv(tmp / "cv5_train_oof_predictions.csv", predictions_oof)
        _write_csv(tmp / "d1_test_predictions.csv", predictions_test)
        for cohort in ("drugage", "agextend"):
            out = cohorts[cohort]["frame"].copy()
            for name in COMPONENTS:
                out[f"probability_{name}"] = scored[name][cohort]
            out["blend_eq_thirds"] = blend_scored[cohort]
            _write_csv(tmp / f"external_predictions_{cohort}.csv", out)

        cols = [("model", "Model"), ("n", "n"), ("auprc", "AP+"), ("auroc", "AUROC"),
                ("brier", "Brier"), ("accuracy", "Acc"), ("cohen_kappa", "Kappa"),
                ("f1_macro", "F1mac"), ("sensitivity", "Sens"), ("specificity", "Spec")]
        lines = [
            "# RFECV feature selection + equal-thirds SVM / TabPFN-v2 / TabFM blend",
            "",
            f"**Locked feature set: {n_selected} of "
            f"{x_train_all.shape[1]} RDKit2D descriptors.**",
            "",
            "```",
            ", ".join(selected_names),
            "```",
            "",
            "RFECV(SVC linear, step=1, min_features_to_select="
            f"{fs_settings['min_features_to_select']}, scoring={fs_settings['scoring']}), "
            "no feature forcibly retained.",
            "Feature selection is NESTED inside the 5-fold CV and fitted once on all 324",
            "rows for the locked model. Thresholds come from the train OOF only.",
            "",
        ]
        for cohort, title in (
            ("cv5_train_oof", "5-fold CV, D1 train only (n=324, nested FS)"),
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
            "provenance_note": (
                "The npj Digital Medicine article is paywalled and could not be read; the "
                "accompanying tddi v1.0.0 code performs no RFECV (its num_features_to_drop=6 "
                "removes six identity columns). This run implements the user's explicit "
                "specification and does NOT claim to replicate that paper's methodology."
            ),
            "feature_selection": {
                **fs_settings,
                "candidate_pool": (
                    "205 RDKit2D descriptors (fold-local imputed/variance-filtered)"
                ),
                "datawarrior_7_excluded_reason": (
                    "not stored for DrugAge/AgeXtend and the DataWarrior CLI needs a Java "
                    "runtime that is not installed, so they cannot be recomputed for the "
                    "external validation"
                ),
                "nested_inside_cross_validation": True,
                "locked_n_features": n_selected,
                "locked_features": selected_names,
                "per_fold_n_selected": [r["n_selected"] for r in fold_selection],
            },
            "blend": {"components": list(COMPONENTS), "weights": EQUAL_THIRDS.tolist(),
                      "tanimoto_excluded_reason":
                          "consumes binary fingerprints, outside a descriptor FS experiment"},
            "svm_scaling_deviation": (
                "StandardScaler fitted on fit rows only; the published notebook does not "
                "scale its seven DataWarrior descriptors, but RDKit2D scales differ by "
                "orders of magnitude and libsvm converges pathologically slowly unscaled"
            ),
            "thresholds": thresholds,
            "test_or_external_labels_used_for_selection_or_threshold": False,
            "runtime": {"python": platform.python_version(), "platform": platform.platform()},
            "sealed_inputs_sha256": {
                k: sha256_file(root / v["path"]) for k, v in sealed.items()
            },
            "existing_runs_modified": False,
        }
        atomic_write_json(tmp / "RUN_MANIFEST.json", manifest)
        atomic_write_json(tmp / "COMPLETED.json", {
            "schema_version": f"{SCHEMA}.completed.v1", "status": "COMPLETE", "run_id": run_id,
            "run_manifest_sha256": sha256_file(tmp / "RUN_MANIFEST.json"),
            "artifact_hashes": {
                str(p.relative_to(tmp)): sha256_file(p)
                for p in sorted(tmp.rglob("*")) if p.is_file() and p.name != "COMPLETED.json"
            },
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
    a = parser.parse_args(argv)
    run(root=a.root, config_path=a.config, positive_path=a.positive,
        negative_path=a.negative, run_id=a.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
