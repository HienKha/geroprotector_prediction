"""5-fold cross-validation on D1 TRAIN ONLY, reported in the original paper's metrics.

Scope, deliberately narrow:

  * data      : the 324 D1 paper-training rows only.  The 81-row held-out test
                set and both external cohorts are NOT touched here.
  * folds     : StratifiedKFold(n_splits=5, shuffle=True, random_state=42) --
                the exact fold structure every cross-fitted OOF stream in this
                repository already uses, so the per-fold predictions below are
                the ones that were already computed, not a new resampling.
  * models    : paper SVM, blend 1/3 (SVM+Tanimoto+TabPFNv2),
                blend 1/3 (SVM+Tanimoto+TabFM).
  * metrics   : ONLY those the source paper reports.  Its official notebook
                (BioAgeLab/Geroprotectors-Project-INGER @ c8f4589,
                2.SVM_model/SVM_model_CODE.ipynb) imports
                `accuracy_score, confusion_matrix, ConfusionMatrixDisplay`
                and additionally computes specificity = tn/(tn+fp) and Cohen's
                kappa by hand.  It computes no ROC/AUC, no precision/recall/F1,
                and no classification_report.  So this module reports exactly:
                accuracy, specificity, Cohen's kappa, and the confusion matrix
                (tn/fp/fn/tp).  Sensitivity is printed alongside only because it
                is a direct read of the confusion matrix the paper displays.

NOTE ON THE PAPER ITSELF: the source paper performs NO cross-validation -- it
uses a single 80/20 split.  This 5-fold CV is therefore an addition to the
paper's protocol, not a reproduction of it.

THRESHOLDS.  Threshold-dependent metrics need a decision rule, and the choice
matters:

  * `fixed_0p5` is the PRIMARY comparison: a common, fixed operating point for
    all three models, selected without reference to these 324 outcomes.  It is
    the only rule here that is free of circularity.
  * `paper_svm_native` additionally scores the paper SVM with its own
    `SVC.predict()` rule (decision_function >= 0), refitted per fold -- i.e.
    literally what the paper's notebook does -- so the SVM is not
    disadvantaged by being forced onto a foreign threshold.
  * The project's OOF-MCC thresholds are deliberately NOT used: they were
    selected by maximising MCC on these same 324 OOF rows, so scoring the same
    rows at them would be optimistically biased.  This is stated in the manifest.

Nothing is refitted except the paper SVM, which is refitted per fold solely to
obtain its native `.predict()` decision; its refitted probabilities are proven
identical to the sealed OOF stream before use.
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
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from geroprotector.fixed_blend_paper405 import (
    _features,
    _selected_component_predictions,
)
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.traditional_paper405 import _read_sources, paper_split_indices


class CV5PaperMetricsError(RuntimeError):
    """Raised when a sealed input or a protocol invariant fails."""


SCHEMA = "geroprotector.cv5_paper_metrics"
EQUAL_THIRDS = np.full(3, 1.0 / 3.0)
MODELS = ("paper_svm", "blend_eq_thirds_tabpfnv2", "blend_eq_thirds_tabfm")


def _regular_file(path: Path, role: str, expected: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise CV5PaperMetricsError(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected is not None and sha256_file(resolved) != expected:
        raise CV5PaperMetricsError(f"{role} SHA256 differs from the protocol lock")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    text = _regular_file(path, "CV5 protocol").read_text(encoding="utf-8")
    protocol = yaml.safe_load(text)
    expected_schema = f"{SCHEMA}.protocol.v1"
    if not isinstance(protocol, dict) or protocol.get("schema_version") != expected_schema:
        raise CV5PaperMetricsError("Unknown CV5 protocol schema")
    contract = protocol.get("contract", {})
    for key in ("train_rows_only", "oof_selected_thresholds_are_not_used"):
        if contract.get(key) is not True:
            raise CV5PaperMetricsError(f"Contract differs at {key}")
    for key in ("test_rows_used", "external_rows_used"):
        if contract.get(key) is not False:
            raise CV5PaperMetricsError(f"Contract differs at {key}")
    immutability = protocol.get("immutability", {})
    if immutability.get("existing_run_directories_are_read_only") is not True:
        raise CV5PaperMetricsError("Immutability contract differs")
    for record in protocol.get("sealed_inputs", {}).values():
        _regular_file(root / record["path"], f"sealed input {record['path']}", record["sha256"])
    return protocol, canonical_sha256(protocol)


def _paper_metrics(y: np.ndarray, decision: np.ndarray) -> dict[str, Any]:
    """Exactly the metric set the source paper's notebook computes."""

    y = np.asarray(y, dtype=int)
    decision = np.asarray(decision, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, decision, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, decision)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "cohen_kappa": float(cohen_kappa_score(y, decision)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        # derived read of the confusion matrix the paper displays, not an added metric
        "sensitivity_derived_from_confusion_matrix": (
            float(tp / (tp + fn)) if (tp + fn) else float("nan")
        ),
        "n": len(y),
    }


def run(
    *, root: Path, config_path: Path, positive_path: Path, negative_path: Path, run_id: str
) -> Path:
    if not re.fullmatch(r"cv5_paper_metrics_[a-z0-9_.-]+", run_id):
        raise CV5PaperMetricsError("RUN_ID must start with cv5_paper_metrics_")
    root = root.resolve()
    protocol, protocol_sha256 = load_protocol(root, config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise CV5PaperMetricsError(f"Run directory already exists: {destination}")
    sealed = protocol["sealed_inputs"]

    # -- D1, the locked split, and the exact 5 folds -----------------------------
    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    frame, _audit = _read_sources(
        _regular_file(positive_path, "positive source", protocol["sources"]["positive_sha256"]),
        _regular_file(negative_path, "negative source", protocol["sources"]["negative_sha256"]),
        traditional,
    )
    train_indices, _test_indices, split_sha256 = paper_split_indices(traditional)
    if split_sha256 != protocol["paper_split"]["expected_assignment_sha256"]:
        raise CV5PaperMetricsError("Paper split differs from the sealed assignment")
    if len(train_indices) != 324:
        raise CV5PaperMetricsError("Train split is not the 324 paper-training rows")
    labels = frame["label"].to_numpy(dtype=int)
    smiles, _ = _validated_raw_smiles(frame)
    features = _features(frame, smiles, fixed_protocol)

    cv = protocol["cross_validation"]
    folds = StratifiedKFold(
        n_splits=int(cv["n_splits"]), shuffle=bool(cv["shuffle"]),
        random_state=int(cv["random_state"]),
    )
    fold_split = list(folds.split(train_indices, labels[train_indices]))

    # -- component OOF streams (already cross-fitted on these exact folds) -------
    tabpfn_oof = pd.read_csv(root / sealed["weighted_train_oof"]["path"])
    tabfm_oof = pd.read_csv(root / sealed["tabfm_train_oof"]["path"])
    tabpfn_oof = tabpfn_oof.set_index("paper_row_index").loc[train_indices].reset_index()
    tabfm_oof = tabfm_oof.set_index("paper_row_index").loc[train_indices].reset_index()
    y_train = labels[train_indices]
    if not np.array_equal(tabfm_oof.label.to_numpy(dtype=int), y_train):
        raise CV5PaperMetricsError("TabFM OOF labels differ from D1")

    # The two chemistry streams must be the same in both files.
    for column in ("probability_paper_svm", "probability_tanimoto_svc"):
        drift = float(np.max(np.abs(
            tabpfn_oof[column].to_numpy(float) - tabfm_oof[column].to_numpy(float)
        )))
        if drift > float(protocol["contract"]["parity_atol"]):
            raise CV5PaperMetricsError(f"{column} differs between the two sealed OOF files")

    svm_oof = tabpfn_oof.probability_paper_svm.to_numpy(float)
    tani_oof = tabpfn_oof.probability_tanimoto_svc.to_numpy(float)
    blend_tabpfn_oof = np.column_stack(
        [svm_oof, tani_oof, tabpfn_oof.probability_tabpfn_v2.to_numpy(float)]
    ) @ EQUAL_THIRDS
    blend_tabfm_oof = np.column_stack(
        [svm_oof, tani_oof, tabfm_oof.probability_tabfm.to_numpy(float)]
    ) @ EQUAL_THIRDS

    # -- paper SVM native .predict() per fold (what the paper's notebook does) ----
    settings = fixed_protocol["components"]
    svm_native = np.full(len(train_indices), -1, dtype=int)
    fold_id = np.full(len(train_indices), -1, dtype=int)
    svm_parity = 0.0
    for fold, (relative_fit, relative_validation) in enumerate(fold_split):
        fit_rows = train_indices[relative_fit]
        validation_rows = train_indices[relative_validation]
        fold_id[relative_validation] = fold
        chemistry, models, _a = _selected_component_predictions(
            features, labels, fit_rows, validation_rows, settings,
            seed=42 + fold, requested=("paper_svm",),
        )
        # parity: the refit must reproduce the sealed OOF probability stream
        svm_parity = max(
            svm_parity,
            float(np.max(np.abs(chemistry["paper_svm"] - svm_oof[relative_validation]))),
        )
        svm_native[relative_validation] = models["paper_svm"].predict(
            features["paper"][validation_rows]
        ).astype(int)
        print(f"paper-SVM native fold {fold + 1}/{len(fold_split)}", flush=True)
    if svm_parity > float(protocol["contract"]["parity_atol"]):
        raise CV5PaperMetricsError("Refit paper-SVM differs from the sealed OOF stream")
    if (svm_native < 0).any() or (fold_id < 0).any():
        raise CV5PaperMetricsError("Fold coverage is incomplete")

    # -- assemble per-row predictions --------------------------------------------
    predictions = pd.DataFrame({
        "paper_row_index": train_indices,
        "fold": fold_id,
        "label": y_train,
        "probability_paper_svm": svm_oof,
        "probability_tanimoto_svc": tani_oof,
        "probability_tabpfn_v2": tabpfn_oof.probability_tabpfn_v2.to_numpy(float),
        "probability_tabfm": tabfm_oof.probability_tabfm.to_numpy(float),
        "blend_eq_thirds_tabpfnv2": blend_tabpfn_oof,
        "blend_eq_thirds_tabfm": blend_tabfm_oof,
        "paper_svm_native_decision": svm_native,
    })

    # -- metrics: per fold, mean +/- SD across folds, and pooled OOF -------------
    rules: list[tuple[str, str, Any]] = [
        ("paper_svm", "fixed_0p5", (svm_oof >= 0.5).astype(int)),
        ("paper_svm", "paper_svm_native", svm_native),
        ("blend_eq_thirds_tabpfnv2", "fixed_0p5", (blend_tabpfn_oof >= 0.5).astype(int)),
        ("blend_eq_thirds_tabfm", "fixed_0p5", (blend_tabfm_oof >= 0.5).astype(int)),
    ]

    per_fold_rows, pooled_rows = [], []
    for model_name, rule, decision in rules:
        for fold in range(len(fold_split)):
            mask = fold_id == fold
            per_fold_rows.append({
                "model": model_name, "threshold_rule": rule, "fold": fold,
                **_paper_metrics(y_train[mask], decision[mask]),
            })
        pooled_rows.append({
            "model": model_name, "threshold_rule": rule, "aggregation": "pooled_oof_324_rows",
            **_paper_metrics(y_train, decision),
        })
    per_fold = pd.DataFrame(per_fold_rows)
    pooled = pd.DataFrame(pooled_rows)

    metric_names = ["accuracy", "specificity", "cohen_kappa",
                    "sensitivity_derived_from_confusion_matrix"]
    summary_rows = []
    for (model_name, rule), group in per_fold.groupby(["model", "threshold_rule"], sort=False):
        row = {"model": model_name, "threshold_rule": rule,
               "aggregation": "mean_sd_across_5_folds"}
        for metric in metric_names:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    # -- write ---------------------------------------------------------------------
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".cv5.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "cv5_per_fold_metrics.csv", per_fold)
        _write_csv(tmp / "cv5_mean_sd_across_folds.csv", summary)
        _write_csv(tmp / "cv5_pooled_oof_metrics.csv", pooled)
        _write_csv(tmp / "cv5_per_row_predictions.csv", predictions)

        lines = [
            "# 5-fold cross-validation on D1 TRAIN only (n=324)",
            "",
            "Metric set restricted to what the source paper reports: **accuracy, specificity,",
            "Cohen's kappa, confusion matrix**. Its official notebook computes no ROC/AUC and",
            "no precision/recall/F1. Sensitivity is shown only as a direct read of the",
            "confusion matrix the paper displays.",
            "",
            "The source paper performs **no cross-validation** (single 80/20 split), so this",
            "5-fold CV is an addition to its protocol, not a reproduction of it.",
            "",
            "Folds: StratifiedKFold(5, shuffle=True, random_state=42) -- the same folds every",
            "cross-fitted OOF stream in this project uses. The 81-row held-out test set and",
            "both external cohorts are untouched here.",
            "",
            "Primary rule is a common fixed 0.5; the project's OOF-MCC thresholds are NOT used",
            "because they were tuned on these same 324 rows. `paper_svm_native` additionally",
            "scores the SVM with its own `SVC.predict()` rule, refitted per fold.",
            "",
            "## Mean +/- SD across the 5 folds",
            "",
            "| Model | Rule | Accuracy | Specificity | Cohen's kappa | Sensitivity |",
            "|---|---|---|---|---|---|",
        ]
        for row in summary.itertuples(index=False):
            lines.append(
                f"| {row.model} | {row.threshold_rule} | "
                f"{row.accuracy_mean:.4f} ± {row.accuracy_sd:.4f} | "
                f"{row.specificity_mean:.4f} ± {row.specificity_sd:.4f} | "
                f"{row.cohen_kappa_mean:.4f} ± {row.cohen_kappa_sd:.4f} | "
                f"{row.sensitivity_derived_from_confusion_matrix_mean:.4f} ± "
                f"{row.sensitivity_derived_from_confusion_matrix_sd:.4f} |"
            )
        lines += ["", "## Pooled over all 324 OOF rows", "",
                  "| Model | Rule | Accuracy | Specificity | Cohen's kappa | Sensitivity | "
                  "TN | FP | FN | TP |", "|---|---|---|---|---|---|---|---|---|---|"]
        for row in pooled.itertuples(index=False):
            lines.append(
                f"| {row.model} | {row.threshold_rule} | {row.accuracy:.4f} | "
                f"{row.specificity:.4f} | {row.cohen_kappa:.4f} | "
                f"{row.sensitivity_derived_from_confusion_matrix:.4f} | "
                f"{row.tn} | {row.fp} | {row.fn} | {row.tp} |"
            )
        lines.append("")
        (tmp / "summary.md").write_text("\n".join(lines), encoding="utf-8")

        manifest = {
            "schema_version": f"{SCHEMA}.run_manifest.v1",
            "run_id": run_id, "protocol_sha256": protocol_sha256,
            "paper_split_sha256": split_sha256,
            "n_train_rows": len(train_indices),
            "folds": {"n_splits": int(cv["n_splits"]), "shuffle": bool(cv["shuffle"]),
                      "random_state": int(cv["random_state"]),
                      "fold_sizes": [
                          int((fold_id == f).sum()) for f in range(len(fold_split))
                      ]},
            "paper_metric_set": {
                "source": ("BioAgeLab/Geroprotectors-Project-INGER @ c8f4589, "
                           "2.SVM_model/SVM_model_CODE.ipynb"),
                "reported": ["accuracy", "confusion_matrix", "specificity", "cohen_kappa"],
                "not_reported_by_the_paper": [
                    "roc_auc", "auprc", "precision", "recall_as_a_named_metric",
                    "f1", "macro_f1", "brier",
                ],
                "sensitivity_note": (
                    "shown only as a direct read of the displayed confusion matrix"
                ),
                "paper_performs_cross_validation": False,
                "this_cv_is_an_addition_not_a_reproduction": True,
            },
            "threshold_rules": {
                "primary": "fixed_0p5",
                "secondary": "paper_svm_native (SVC.predict, refitted per fold)",
                "oof_mcc_thresholds_deliberately_excluded_reason": (
                    "they were selected by maximising MCC on these same 324 OOF rows, so "
                    "scoring the same rows at them would be optimistically biased"
                ),
            },
            "paper_svm_refit_vs_sealed_oof_max_abs_difference": float(svm_parity),
            "train_rows_only": True,
            "test_rows_used": False,
            "external_rows_used": False,
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
