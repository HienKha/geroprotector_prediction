"""Equal-thirds SVM / Tanimoto / TabFM blend under the locked paper-405 protocol.

TabFM (google-research/tabfm, v1.0.1) is Google Research's tabular foundation
model -- a direct peer of TabPFN.  It had never been evaluated in this project;
this module is its first run here.

Design mirrors `screening_blend_altmodels.py` exactly, so the resulting numbers
are directly comparable to the BiSHop / TabM / TabNet equal-thirds blends and to
the sealed TabPFN-v2 equal-thirds anchor:

  * the two chemistry components (paper_svm, tanimoto_svc) are the SAME fitted
    state as the locked blend.  They are rebuilt from raw sources and PROVEN
    identical to the sealed streams before anything else runs;
  * TabFM occupies the third slot and sees the exact same fold-local RDKit2D
    panel the TabPFN slot saw (217 raw -> 205 after median imputation and
    variance filtering), so the comparison isolates the model, not the features;
  * weights are a prespecified 1/3 each -- never fitted;
  * the decision threshold is selected by MCC on the 324-row cross-fitted OOF,
    the same deterministic selector every other run in this repository uses,
    plus a fixed 0.5 second operating point;
  * D1 test / DrugAge / AgeXtend are scored once, afterwards.

External chemistry-component probabilities are READ from the sealed external
prediction files, so those streams stay byte-identical to the published ones.

LICENSE NOTE.  The TabFM source is Apache-2.0, but the pretrained weights ship
under `tabfm-non-commercial-v1.0`, restricted to non-commercial, non-production
use.  Academic evaluation and publication fall inside those terms; deploying
these weights in a commercial or production screening service does not.  This is
recorded in the run manifest so it travels with the results.
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


class ScreeningBlendTabFMError(RuntimeError):
    """Raised when a sealed input, a parity proof or a leakage contract fails."""


SCHEMA = "geroprotector.screening_blend_tabfm"
CHEMISTRY_COMPONENTS = ("paper_svm", "tanimoto_svc")
EQUAL_WEIGHT = 1.0 / 3.0
D1_ENDPOINT = "paper_binary"
DRUGAGE_ENDPOINT = "significant_positive_retrieval_background_not_certified_negative"
AGEXTEND_ENDPOINT = "published_independent_table6_binary"


def _regular_file(path: Path, role: str, expected: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ScreeningBlendTabFMError(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected is not None and sha256_file(resolved) != expected:
        raise ScreeningBlendTabFMError(f"{role} SHA256 differs from the protocol lock")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    text = _regular_file(path, "TabFM protocol").read_text(encoding="utf-8")
    protocol = yaml.safe_load(text)
    expected_schema = f"{SCHEMA}.protocol.v1"
    if not isinstance(protocol, dict) or protocol.get("schema_version") != expected_schema:
        raise ScreeningBlendTabFMError("Unknown TabFM protocol schema")
    blend = protocol.get("blend", {})
    if (
        blend.get("components") != ["paper_svm", "tanimoto_svc", "tabfm"]
        or not np.allclose(blend.get("weights"), [EQUAL_WEIGHT] * 3, rtol=0.0, atol=1e-15)
        or blend.get("weights_selected_from_data") is not False
        or blend.get("test_or_external_labels_used_for_fit_weight_or_threshold") is not False
    ):
        raise ScreeningBlendTabFMError("Equal-weight blend contract differs")
    parity = protocol.get("parity_contract", {})
    for key in (
        "rebuilt_paper_svm_and_tanimoto_streams_must_match_sealed_run",
        "rebuilt_rdkit2d_context_must_equal_sealed_tabpfn_context",
        "rebuilt_morgan_bits_must_equal_sealed_tanimoto_train_bits",
    ):
        if parity.get(key) is not True:
            raise ScreeningBlendTabFMError(f"Parity contract differs at {key}")
    immutability = protocol.get("immutability", {})
    if immutability.get("existing_run_directories_are_read_only") is not True:
        raise ScreeningBlendTabFMError("Immutability contract differs")
    for record in protocol.get("sealed_inputs", {}).values():
        _regular_file(root / record["path"], f"sealed input {record['path']}", record["sha256"])
    return protocol, canonical_sha256(protocol)


def _load_tabfm(settings: dict[str, Any]):
    """Load the pinned TabFM checkpoint on the requested device."""

    observed = importlib.metadata.version("tabfm")
    if observed != str(settings["required_package_version"]):
        raise ScreeningBlendTabFMError(
            f"Installed tabfm {observed} differs from the protocol lock "
            f"{settings['required_package_version']}"
        )
    if settings["backend"] != "pytorch":
        raise ScreeningBlendTabFMError("Only the pinned pytorch backend is supported here")
    from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

    return tabfm_v1_0_0.load(model_type="classification", device=str(settings["device"]))


def _tabfm_probability(
    model,
    settings: dict[str, Any],
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    targets: dict[str, np.ndarray],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit TabFM in-context on the fit rows and score every target block."""

    from tabfm import TabFMClassifier

    hyper = settings["hyperparameters"]
    estimator = TabFMClassifier(
        model=model,
        n_estimators=int(hyper["n_estimators"]),
        softmax_temperature=float(hyper["softmax_temperature"]),
        max_num_features=int(hyper["max_num_features"]),
        max_num_rows=hyper["max_num_rows"],  # None => use every context row
        random_state=int(seed),
    )
    estimator.fit(x_fit, y_fit)
    classes = np.asarray(estimator.classes_)
    if classes.shape != (2,) or set(map(int, classes)) != {0, 1}:
        raise ScreeningBlendTabFMError("TabFM returned unexpected classes")
    positive = int(np.flatnonzero(classes == 1)[0])

    output = {}
    for name, matrix in targets.items():
        raw = np.asarray(estimator.predict_proba(matrix), dtype=float)
        if raw.shape != (len(matrix), 2) or not np.isfinite(raw).all():
            raise ScreeningBlendTabFMError(f"TabFM prediction invalid for {name}")
        output[name] = np.clip(raw[:, positive], 1e-7, 1 - 1e-7)
    audit = {
        "n_estimators": int(hyper["n_estimators"]),
        "softmax_temperature": float(hyper["softmax_temperature"]),
        "max_num_rows": hyper["max_num_rows"],
        "max_num_features": int(hyper["max_num_features"]),
        "random_state": int(seed),
        "context_rows": len(x_fit),
        "context_features": int(x_fit.shape[1]),
        "row_subsampling_active": bool(
            hyper["max_num_rows"] is not None and int(hyper["max_num_rows"]) < len(x_fit)
        ),
        "feature_subsampling_active": bool(int(hyper["max_num_features"]) < x_fit.shape[1]),
    }
    return output, audit


def _panel(fit_raw: np.ndarray, targets_raw: dict[str, np.ndarray]):
    """Fold-local median imputation + variance filter, fitted on fit rows only."""

    context = _imputer_context(fit_raw)
    return (
        _apply_context(context, fit_raw),
        {k: _apply_context(context, v) for k, v in targets_raw.items()},
    )


def run(
    *, root: Path, config_path: Path, positive_path: Path, negative_path: Path, run_id: str
) -> Path:
    if not re.fullmatch(r"screeningblend_tabfm_[a-z0-9_.-]+", run_id):
        raise ScreeningBlendTabFMError("RUN_ID must start with screeningblend_tabfm_")
    root = root.resolve()
    protocol, protocol_sha256 = load_protocol(root, config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise ScreeningBlendTabFMError(f"Run directory already exists: {destination}")

    sealed = protocol["sealed_inputs"]
    bundle = load_locked_bundle(
        root / sealed["screening_bundle"]["path"],
        expected_artifact_sha256=sealed["screening_bundle"]["sha256"],
    )

    # -- D1 frame, locked split, features ---------------------------------------
    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    frame, _audit = _read_sources(
        _regular_file(positive_path, "positive source", protocol["sources"]["positive_sha256"]),
        _regular_file(negative_path, "negative source", protocol["sources"]["negative_sha256"]),
        traditional,
    )
    if len(frame) != int(protocol["sources"]["expected_rows"]):
        raise ScreeningBlendTabFMError("D1 row count differs from 405")
    train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    if split_sha256 != protocol["paper_split"]["expected_assignment_sha256"]:
        raise ScreeningBlendTabFMError("Paper split differs from the sealed assignment")
    labels = frame["label"].to_numpy(dtype=int)
    smiles, _ = _validated_raw_smiles(frame)
    features = _features(frame, smiles, fixed_protocol)
    fit_indices = np.asarray(bundle["fit_paper_indices"], dtype=int)
    sealed_bits = np.asarray(bundle["tanimoto_train_bits"], dtype=np.uint8)

    # -- parity proofs -----------------------------------------------------------
    if not np.array_equal(features["morgan"][fit_indices], sealed_bits):
        raise ScreeningBlendTabFMError("Rebuilt Morgan bits differ from the sealed bundle")
    train_context = _imputer_context(features["rdkit2d"][fit_indices])
    if not np.array_equal(
        _apply_context(train_context, features["rdkit2d"][fit_indices]),
        np.asarray(bundle["tabpfn_context"]["context_features"], dtype=np.float32),
    ):
        raise ScreeningBlendTabFMError(
            "Rebuilt descriptor panel differs from the sealed foundation-model context"
        )

    atol = float(protocol["parity_contract"]["parity_atol"])
    sealed_oof = pd.read_csv(root / sealed["weighted_train_oof"]["path"])
    sealed_test = pd.read_csv(root / sealed["weighted_test_components"]["path"])

    # -- chemistry OOF + full fit, proven identical to the sealed run -------------
    settings = fixed_protocol["components"]
    folds = StratifiedKFold(
        n_splits=int(settings["cross_fitted_oof_folds"]),
        shuffle=True,
        random_state=int(settings["cross_fitted_oof_seed"]),
    )
    fold_split = list(folds.split(train_indices, labels[train_indices]))
    oof = {name: np.full(len(train_indices), np.nan) for name in CHEMISTRY_COMPONENTS}
    for fold, (relative_fit, relative_validation) in enumerate(fold_split):
        chemistry, _models, _a = _selected_component_predictions(
            features, labels,
            train_indices[relative_fit], train_indices[relative_validation],
            settings, seed=42 + fold, requested=CHEMISTRY_COMPONENTS,
        )
        for name in CHEMISTRY_COMPONENTS:
            oof[name][relative_validation] = chemistry[name]
        print(f"chemistry OOF fold {fold + 1}/{len(fold_split)}", flush=True)

    rebuilt = pd.DataFrame({
        "paper_row_index": train_indices,
        "probability_paper_svm": oof["paper_svm"],
        "probability_tanimoto_svc": oof["tanimoto_svc"],
    })
    merged = sealed_oof.merge(rebuilt, on="paper_row_index", suffixes=("_sealed", "_rebuilt"))
    if len(merged) != len(train_indices):
        raise ScreeningBlendTabFMError("Sealed OOF join is incomplete")
    oof_parity = {}
    for column in ("probability_paper_svm", "probability_tanimoto_svc"):
        diff = float(np.max(np.abs(merged[f"{column}_sealed"] - merged[f"{column}_rebuilt"])))
        if diff > atol:
            raise ScreeningBlendTabFMError(f"Rebuilt OOF {column} differs from the sealed run")
        oof_parity[column] = diff

    full_chemistry, full_models, _a = _selected_component_predictions(
        features, labels, train_indices, test_indices, settings, seed=42,
        requested=CHEMISTRY_COMPONENTS,
    )
    test_frame = pd.DataFrame({
        "paper_row_index": test_indices,
        "label": labels[test_indices],
        "probability_paper_svm": full_chemistry["paper_svm"],
        "probability_tanimoto_svc": full_chemistry["tanimoto_svc"],
    })
    merged_test = sealed_test.merge(
        test_frame.drop(columns=["label"]), on="paper_row_index",
        suffixes=("_sealed", "_rebuilt"),
    )
    test_parity = {}
    for column in ("probability_paper_svm", "probability_tanimoto_svc"):
        diff = float(
            np.max(np.abs(merged_test[f"{column}_sealed"] - merged_test[f"{column}_rebuilt"]))
        )
        if diff > atol:
            raise ScreeningBlendTabFMError(f"Rebuilt test {column} differs from the sealed run")
        test_parity[column] = diff
    paper_svm_boundary = _svc_probability_at_native_boundary(full_models["paper_svm"])

    # -- external cohorts --------------------------------------------------------
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
            "endpoint": endpoint,
            "frame": scored,
            "rdkit2d": _rdkit2d_from_smiles(smi, bundle["descriptor_names"]),
            "morgan": _morgan_from_smiles(smi, bundle["portable_contract"]),
        }
        similarity = _tanimoto(payload["morgan"], sealed_bits).max(axis=1)
        stored = scored.maximum_tanimoto_to_fitted_train.to_numpy(dtype=float)
        if float(np.max(np.abs(similarity - stored))) > 1e-6:
            raise ScreeningBlendTabFMError(
                f"Rebuilt {cohort} similarity differs from sealed file"
            )
        cohorts[cohort] = payload
        print(f"external features rebuilt for {cohort}: {len(scored)} rows", flush=True)

    # -- TabFM: cross-fitted OOF, then one full fit scoring all three cohorts -----
    tabfm_settings = protocol["tabfm"]
    model = _load_tabfm(tabfm_settings)
    print("TabFM weights loaded", flush=True)

    tabfm_oof = np.full(len(train_indices), np.nan)
    fold_audits = []
    for fold, (relative_fit, relative_validation) in enumerate(fold_split):
        sub_fit = train_indices[relative_fit]
        sub_validation = train_indices[relative_validation]
        x_fit, targets = _panel(
            features["rdkit2d"][sub_fit],
            {"validation": features["rdkit2d"][sub_validation]},
        )
        probabilities, audit = _tabfm_probability(
            model, tabfm_settings, x_fit, labels[sub_fit], targets, seed=42 + fold
        )
        tabfm_oof[relative_validation] = probabilities["validation"]
        fold_audits.append({"fold": fold, **audit})
        print(f"[tabfm] OOF fold {fold + 1}/{len(fold_split)}", flush=True)
    if not np.isfinite(tabfm_oof).all():
        raise ScreeningBlendTabFMError("TabFM OOF is incomplete")

    print("[tabfm] full 324-row fit, scoring all three cohorts", flush=True)
    x_fit, targets = _panel(
        features["rdkit2d"][fit_indices],
        {
            "d1_test": features["rdkit2d"][test_indices],
            "drugage": cohorts["drugage"]["rdkit2d"],
            "agextend": cohorts["agextend"]["rdkit2d"],
        },
    )
    tabfm_scored, full_audit = _tabfm_probability(
        model, tabfm_settings, x_fit, labels[fit_indices], targets, seed=42
    )
    for name, values in tabfm_scored.items():
        print(f"[tabfm] scored {name}: {len(values)} rows", flush=True)

    # -- blend + thresholds (OOF only) -------------------------------------------
    y_train = labels[train_indices]
    blend_oof = (oof["paper_svm"] + oof["tanimoto_svc"] + tabfm_oof) / 3.0
    tabfm_threshold, tabfm_mcc = select_threshold(y_train, tabfm_oof)
    blend_threshold, blend_mcc = select_threshold(y_train, blend_oof)
    tanimoto_threshold, tanimoto_mcc = select_threshold(y_train, oof["tanimoto_svc"])
    thresholds = {
        "tabfm": float(tabfm_threshold),
        "blend_equal_thirds_tabfm": float(blend_threshold),
        "tanimoto_svc": float(tanimoto_threshold),
        "paper_svm": float(paper_svm_boundary),
    }
    threshold_rows = [
        {"model": "paper_svm", "threshold": float(paper_svm_boundary),
         "threshold_source": "published_svc_predict_decision_function_zero",
         "oof_mcc_at_selection": float(select_threshold(y_train, oof["paper_svm"])[1]),
         "external_labels_used": False},
        {"model": "tanimoto_svc", "threshold": float(tanimoto_threshold),
         "threshold_source": "full_324_d1_train_cross_fitted_oof_mcc",
         "oof_mcc_at_selection": float(tanimoto_mcc), "external_labels_used": False},
        {"model": "tabfm", "threshold": float(tabfm_threshold),
         "threshold_source": "full_324_d1_train_cross_fitted_oof_mcc",
         "oof_mcc_at_selection": float(tabfm_mcc), "external_labels_used": False},
        {"model": "blend_equal_thirds_tabfm", "threshold": float(blend_threshold),
         "threshold_source": "full_324_d1_train_cross_fitted_oof_mcc",
         "oof_mcc_at_selection": float(blend_mcc), "external_labels_used": False},
    ]

    # -- scored frames ------------------------------------------------------------
    test_frame["probability_tabfm"] = tabfm_scored["d1_test"]
    test_frame["blend_equal_thirds_tabfm"] = (
        test_frame.probability_paper_svm + test_frame.probability_tanimoto_svc
        + test_frame.probability_tabfm
    ) / 3.0
    for cohort in ("drugage", "agextend"):
        scored = cohorts[cohort]["frame"]
        scored["probability_tabfm"] = tabfm_scored[cohort]
        scored["blend_equal_thirds_tabfm"] = (
            scored.probability_paper_svm + scored.probability_tanimoto_svc
            + scored.probability_tabfm
        ) / 3.0

    # -- metrics -------------------------------------------------------------------
    metric_rows = []
    for cohort, endpoint, scored in (
        ("d1_paper_test", D1_ENDPOINT, test_frame),
        ("drugage", DRUGAGE_ENDPOINT, cohorts["drugage"]["frame"]),
        ("agextend", AGEXTEND_ENDPOINT, cohorts["agextend"]["frame"]),
    ):
        y = scored.label.to_numpy(dtype=int)
        streams = [
            ("paper_svm", scored.probability_paper_svm.to_numpy(float),
             thresholds["paper_svm"]),
            ("tanimoto_svc", scored.probability_tanimoto_svc.to_numpy(float),
             thresholds["tanimoto_svc"]),
            ("tabfm", scored.probability_tabfm.to_numpy(float), thresholds["tabfm"]),
            ("blend_equal_thirds_tabfm", scored.blend_equal_thirds_tabfm.to_numpy(float),
             thresholds["blend_equal_thirds_tabfm"]),
        ]
        for model_name, probability, threshold in streams:
            point = "published_svc_native_predict" if model_name == "paper_svm" else "oof_mcc"
            metric_rows.append({
                "cohort": cohort, "endpoint": endpoint, "model": model_name,
                "operating_point": point, "result_source": "this_run",
                **_safe_metrics(y, probability, threshold),
            })
            metric_rows.append({
                "cohort": cohort, "endpoint": endpoint, "model": model_name,
                "operating_point": "fixed_0p5", "result_source": "this_run",
                **_safe_metrics(y, probability, 0.5),
            })
    metrics = pd.DataFrame(metric_rows)

    # -- comparison against the other equal-thirds blends --------------------------
    comparison_columns = [
        "cohort", "endpoint", "model", "threshold_rule", "threshold", "result_source",
        "auprc_average_precision_positive", "auroc", "brier", "mcc", "macro_f1",
        "recall_sensitivity", "specificity",
    ]
    reference = pd.read_csv(root / sealed["eq_thirds_comparison"]["path"])
    reference = reference[
        reference.model.isin([
            "blend_equal_thirds_svm_tani_tabpfnv2",
            "blend_equal_thirds_svm_tani_bishop",
            "tabpfn_v2", "bishop",
        ])
    ].copy()
    reference["result_source"] = "sealed_" + reference["result_source"].astype(str)

    altmodels = pd.read_csv(root / sealed["altmodels_metrics"]["path"])
    altmodels = altmodels[
        (altmodels.result_source == "this_run")
        & (altmodels.model.isin([
            "blend_equal_thirds_tabm", "blend_equal_thirds_tabnet", "tabm", "tabnet",
        ]))
    ].copy()
    altmodels = altmodels.rename(columns={"operating_point": "threshold_rule"})
    altmodels["result_source"] = "sealed_screeningblend_altmodels_20260819"

    mine = metrics.rename(columns={"operating_point": "threshold_rule"}).copy()
    mine["result_source"] = "computed_this_run"
    native = mine.threshold_rule == "published_svc_native_predict"
    mine.loc[native, "threshold_rule"] = "oof_mcc"

    comparison = pd.concat(
        [
            mine[comparison_columns],
            reference[comparison_columns],
            altmodels[comparison_columns],
        ],
        ignore_index=True,
    )
    order = {"d1_paper_test": 0, "drugage": 1, "agextend": 2}
    comparison["_c"] = comparison.cohort.map(order)
    comparison = comparison.sort_values(
        ["_c", "threshold_rule", "model"], kind="stable"
    ).drop(columns=["_c"])

    # -- write ----------------------------------------------------------------------
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".tabfm.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "d1_train_oof_predictions.csv", pd.DataFrame({
            "paper_row_index": train_indices, "label": y_train,
            "probability_paper_svm": oof["paper_svm"],
            "probability_tanimoto_svc": oof["tanimoto_svc"],
            "probability_tabfm": tabfm_oof,
            "blend_equal_thirds_tabfm": blend_oof,
        }))
        _write_csv(tmp / "component_thresholds.csv", pd.DataFrame(threshold_rows))
        _write_csv(tmp / "d1_test_predictions.csv", test_frame)
        for cohort in ("drugage", "agextend"):
            _write_csv(tmp / f"external_predictions_{cohort}.csv", cohorts[cohort]["frame"])
        _write_csv(tmp / "metrics_all.csv", metrics)
        _write_csv(tmp / "comparison_equal_thirds.csv", comparison)

        columns = [
            ("model", "Model"), ("n_test", "n"),
            ("auprc_average_precision_positive", "AP"), ("auroc", "AUROC"),
            ("brier", "Brier"), ("mcc", "MCC"), ("macro_f1", "Macro F1"),
            ("recall_sensitivity", "Recall"), ("specificity", "Specificity"),
        ]
        lines = [
            "# Equal-thirds SVM / Tanimoto / TabFM blend",
            "",
            "TabFM (google-research/tabfm v1.0.1, PyTorch backend) in the third slot.",
            "The paper-SVM and Tanimoto components are the same fitted state as the locked",
            "blend (parity proven). Weights are a prespecified 1/3 each; thresholds come from",
            "the 324-row D1-train cross-fitted OOF only.",
            "",
            "Pretrained TabFM weights are under `tabfm-non-commercial-v1.0` "
            "(non-commercial, non-production use only).",
            "",
        ]
        primary = metrics[metrics.operating_point != "fixed_0p5"]
        for cohort, title in (
            ("d1_paper_test", "D1 held-out paper test (n=81)"),
            ("drugage", "DrugAge positive-retrieval endpoint (n=446)"),
            ("agextend", "AgeXtend Table 6 endpoint (n=69)"),
        ):
            sub = primary[primary.cohort == cohort]
            lines += [f"## {title}", "",
                      "| " + " | ".join(lbl for _k, lbl in columns) + " |",
                      "|" + "---|" * len(columns)]
            for row in sub.itertuples(index=False):
                cells = []
                for key, _lbl in columns:
                    v = getattr(row, key)
                    cells.append(str(int(v)) if key == "n_test"
                                 else (str(v) if isinstance(v, str)
                                       else ("nan" if not np.isfinite(v) else f"{v:.4f}")))
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        (tmp / "summary.md").write_text("\n".join(lines), encoding="utf-8")

        manifest = {
            "schema_version": f"{SCHEMA}.run_manifest.v1",
            "run_id": run_id, "protocol_sha256": protocol_sha256,
            "paper_split_sha256": split_sha256,
            "locked_component_state_sha256": bundle["component_state_sha256"],
            "blend_weights": [EQUAL_WEIGHT] * 3,
            "thresholds": thresholds,
            "paper_svm_native_probability_boundary": float(paper_svm_boundary),
            "tabfm": {
                "package_version": importlib.metadata.version("tabfm"),
                "backend": tabfm_settings["backend"],
                "repository": tabfm_settings["repository"],
                "repository_commit": tabfm_settings["repository_commit"],
                "huggingface_repo": tabfm_settings["huggingface_repo"],
                "device": tabfm_settings["device"],
                "hyperparameters": tabfm_settings["hyperparameters"],
                "weights_license": tabfm_settings["weights_license"],
                "weights_license_permits_commercial_or_production_use": False,
                "fold_audits": fold_audits,
                "full_fit_audit": full_audit,
            },
            "parity": {
                "morgan_bits_equal_sealed_bundle": True,
                "descriptor_panel_equal_sealed_context": True,
                "oof_max_abs_difference": oof_parity,
                "d1_test_max_abs_difference": test_parity,
            },
            "runtime": {"python": platform.python_version(), "platform": platform.platform()},
            "sealed_inputs_sha256": {
                k: sha256_file(root / v["path"]) for k, v in sealed.items()
            },
            "test_or_external_labels_used_for_fit_weight_or_threshold": False,
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
