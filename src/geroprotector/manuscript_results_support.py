"""Post-lock support analyses for the manuscript Results section.

This module does not fit or select a model. It verifies and reads sealed prediction
streams, then produces three descriptive artifacts:

1. a complete full-RDKit2D five-fold OOF table across conventional, deep, and
   foundation-model families;
2. repeated- and chemistry-grouped positive-only QED associations; and
3. risk-coverage summaries for the two prespecified equal-weight ensembles.

The held-out labels are used only for post-lock evaluation. No output from this
module is allowed to change a model, feature set, threshold, or ensemble weight.
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

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from geroprotector.hashing import atomic_write_json, sha256_file
from geroprotector.nine_ml_featuresets import _metrics


SCHEMA = "geroprotector.manuscript_results_support"
MODELS = {
    "source_svm": "p_paper_svm",
    "tanimoto_svc": "p_tanimoto_svc",
    "tabpfn_v2": "p_tabpfn_v2",
    "tabfm": "p_tabfm",
    "catboost": "p_catboost_full",
    "tabm": "p_tabm_full",
    "SVM_PFN_FM": "p_blend3_equal",
    "SVM_PFN_FM_Tani": "p_gb4_equal",
}


class SupportError(RuntimeError):
    pass


def _require_sealed(run: Path, required: list[str]) -> dict[str, str]:
    completed_path = run / "COMPLETED.json"
    if completed_path.is_symlink() or not completed_path.is_file():
        raise SupportError(f"Missing completion record: {completed_path}")
    completed = json.loads(completed_path.read_text())
    if completed.get("status") != "COMPLETE":
        raise SupportError(f"Run is not COMPLETE: {run}")
    declared = completed.get("artifact_hashes", {})
    verified = {"COMPLETED.json": sha256_file(completed_path)}
    for relative in required:
        path = run / relative
        if relative not in declared:
            raise SupportError(f"Required artifact is not declared: {path}")
        if path.is_symlink() or not path.is_file():
            raise SupportError(f"Required artifact is not a regular file: {path}")
        actual = sha256_file(path)
        if actual != declared[relative]:
            raise SupportError(f"Hash mismatch: {path}")
        verified[relative] = actual
    return verified


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def _qed_registry(modelwide: pd.DataFrame) -> pd.DataFrame:
    registry = modelwide.loc[
        modelwide["cohort"].eq("d1_train_oof"),
        ["paper_row_index", "label", "qed"],
    ].copy()
    if len(registry) != 324 or registry["paper_row_index"].duplicated().any():
        raise SupportError("D1 QED registry must contain 324 unique development rows")
    if registry["qed"].isna().any():
        raise SupportError("D1 QED registry contains missing values")
    return registry


def _rho(frame: pd.DataFrame, probability: str) -> tuple[float, float]:
    positive = frame.loc[frame["label"].eq(1), ["qed", probability]].dropna()
    if len(positive) != 160:
        raise SupportError(f"Expected 160 positive compounds, found {len(positive)}")
    result = spearmanr(positive["qed"], positive[probability])
    return float(result.statistic), float(result.pvalue)


def _bootstrap_rho(frame: pd.DataFrame, probability: str, *, seed: int,
                   resamples: int) -> tuple[float, float, int]:
    positive = frame.loc[frame["label"].eq(1), ["qed", probability]].dropna()
    values = positive.to_numpy(float)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sample = values[rng.integers(0, len(values), size=len(values))]
        statistic = spearmanr(sample[:, 0], sample[:, 1]).statistic
        if np.isfinite(statistic):
            estimates.append(float(statistic))
    if len(estimates) < int(0.99 * resamples):
        raise SupportError("Too many undefined bootstrap Spearman estimates")
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high), len(estimates)


def _full_oof_table(root: Path) -> pd.DataFrame:
    conventional = pd.read_csv(
        root / "outputs/nineml_cv5_full_all_20260905/cv5_pooled_oof_metrics.csv"
    )
    if len(conventional) != 9 or conventional["model_id"].nunique() != 9:
        raise SupportError("Complete conventional full-feature OOF table is malformed")
    conventional = conventional.copy()
    conventional["source_stream"] = "nineml_cv5_full_all_20260905"

    quad = pd.read_csv(root / "outputs/quad_blend_20260822/cv5_train_oof_predictions.csv")
    alt = pd.read_csv(
        root / "outputs/screeningblend_altmodels_20260819/d1_train_oof_predictions.csv"
    )
    if len(quad) != 324 or len(alt) != 324:
        raise SupportError("Deep-model OOF streams must each contain 324 rows")
    aligned = quad[["paper_row_index", "label"]].merge(
        alt, on=["paper_row_index", "label"], how="inner", validate="one_to_one"
    )
    if len(aligned) != 324:
        raise SupportError("Deep-model OOF streams are not row-aligned")

    extra = []
    for model_id, column, source in [
        ("tabpfn_v2", "probability_tabpfn_v2", "quad_blend_20260822"),
        ("tabfm", "probability_tabfm", "quad_blend_20260822"),
        ("bishop", "probability_bishop", "screeningblend_altmodels_20260819"),
        ("tabm", "probability_tabm", "screeningblend_altmodels_20260819"),
        ("tabnet", "probability_tabnet", "screeningblend_altmodels_20260819"),
    ]:
        source_frame = quad if source == "quad_blend_20260822" else aligned
        y = source_frame["label"].to_numpy(int)
        p = source_frame[column].to_numpy(float)
        extra.append({
            "model_id": model_id,
            "aggregation": "pooled_oof",
            "feature_set": "full_rdkit2d",
            **_metrics(y, p, p),
            "source_stream": source,
        })
    return pd.concat([conventional, pd.DataFrame(extra)], ignore_index=True, sort=False)


def _qed_robustness(root: Path, *, bootstrap_resamples: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    modelwide = pd.read_csv(
        root / "outputs/modelwide_druglikeness_bias_20260826/model_cohort_predictions.csv"
    )
    registry = _qed_registry(modelwide)
    repeated = pd.read_csv(
        root / "outputs/repeated_rank_stability_20260826/per_row_repeated_oof_predictions.csv"
    ).merge(registry, on=["paper_row_index", "label"], how="left", validate="many_to_one")
    if len(repeated) != 3240 or repeated["qed"].isna().any():
        raise SupportError("Repeated OOF stream must contain 10 complete 324-row registries")

    per_repeat = []
    for seed, block in repeated.groupby("seed", sort=True):
        if len(block) != 324:
            raise SupportError(f"Repeat {seed} does not contain 324 rows")
        for model_id, column in MODELS.items():
            rho, p_value = _rho(block, column)
            per_repeat.append({"seed": int(seed), "model_id": model_id,
                               "n_positive": 160, "spearman_rho": rho,
                               "p_value_unadjusted": p_value})
    per_repeat_frame = pd.DataFrame(per_repeat)

    summaries = []
    mean_scores = repeated.groupby(["paper_row_index", "label"], as_index=False)[
        list(MODELS.values())
    ].mean().merge(registry, on=["paper_row_index", "label"], validate="one_to_one")
    for position, (model_id, column) in enumerate(MODELS.items()):
        values = per_repeat_frame.loc[
            per_repeat_frame["model_id"].eq(model_id), "spearman_rho"
        ].to_numpy(float)
        mean_rho, mean_p = _rho(mean_scores, column)
        low, high, valid = _bootstrap_rho(
            mean_scores, column, seed=20260905 + position,
            resamples=bootstrap_resamples,
        )
        summaries.append({
            "model_id": model_id,
            "n_repeats": len(values),
            "repeat_rho_median": float(np.median(values)),
            "repeat_rho_q1": float(np.quantile(values, 0.25)),
            "repeat_rho_q3": float(np.quantile(values, 0.75)),
            "repeat_rho_min": float(values.min()),
            "repeat_rho_max": float(values.max()),
            "repeats_negative": int((values < 0).sum()),
            "compound_mean_rho": mean_rho,
            "compound_mean_p_unadjusted": mean_p,
            "compound_bootstrap_ci_low": low,
            "compound_bootstrap_ci_high": high,
            "bootstrap_valid": valid,
        })

    grouped = pd.read_csv(
        root / "outputs/chemical_space_cv_20260826/per_row_grouped_oof_predictions.csv"
    )
    grouped = grouped.loc[grouped["registry"].isin(
        ["C1_scaffold", "C2_similarity_component"]
    )].merge(registry, on=["paper_row_index", "label"], how="left", validate="many_to_one")
    if len(grouped) != 648 or grouped["qed"].isna().any():
        raise SupportError("Grouped OOF stream must contain two complete 324-row registries")
    grouped_rows = []
    for registry_id, block in grouped.groupby("registry", sort=True):
        if len(block) != 324:
            raise SupportError(f"Grouped registry {registry_id} does not contain 324 rows")
        for position, (model_id, column) in enumerate(MODELS.items()):
            rho, p_value = _rho(block, column)
            low, high, valid = _bootstrap_rho(
                block, column,
                seed=20261005 + position + 100 * len(grouped_rows),
                resamples=bootstrap_resamples,
            )
            grouped_rows.append({
                "registry": registry_id, "model_id": model_id,
                "n_positive": 160, "spearman_rho": rho,
                "p_value_unadjusted": p_value,
                "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                "bootstrap_valid": valid,
            })
    return per_repeat_frame, pd.DataFrame(summaries), pd.DataFrame(grouped_rows)


def _risk_coverage(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "outputs/quad_blend_20260822/predictions_d1_test.csv")
    if len(frame) != 81 or frame["paper_row_index"].duplicated().any():
        raise SupportError("D1 held-out prediction stream must contain 81 unique rows")
    y = frame["label"].to_numpy(int)
    probabilities = {
        "SVM_PFN_FM": frame[["probability_paper_svm", "probability_tabpfn_v2",
                              "probability_tabfm"]].mean(axis=1).to_numpy(float),
        "SVM_PFN_FM_Tani": frame["blend_eq_quarters"].to_numpy(float),
    }
    expected_full = (
        frame[["probability_paper_svm", "probability_tanimoto_svc",
               "probability_tabpfn_v2", "probability_tabfm"]].mean(axis=1).to_numpy(float)
    )
    if not np.allclose(probabilities["SVM_PFN_FM_Tani"], expected_full, atol=1e-15, rtol=0):
        raise SupportError("Four-component equal-weight stream fails parity")

    rows = []
    for model_id, probability in probabilities.items():
        confidence = np.abs(probability - 0.5)
        order = np.lexsort((frame["paper_row_index"].to_numpy(int), -confidence))
        for target_coverage in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
            retained = max(1, int(np.ceil(target_coverage * len(frame))))
            chosen = order[:retained]
            y_keep = y[chosen]
            p_keep = probability[chosen]
            decision = (p_keep >= 0.5).astype(int)
            rows.append({
                "model_id": model_id,
                "confidence_proxy": "absolute_distance_from_0.5",
                "target_coverage": target_coverage,
                "retained_n": retained,
                "realized_coverage": retained / len(frame),
                "accuracy": float(accuracy_score(y_keep, decision)),
                "error_rate": float(1 - accuracy_score(y_keep, decision)),
                "mcc": float(matthews_corrcoef(y_keep, decision)),
                "macro_f1": float(f1_score(y_keep, decision, average="macro", zero_division=0)),
                "average_precision": float(average_precision_score(y_keep, p_keep)),
                "auroc": float(roc_auc_score(y_keep, p_keep)) if len(set(y_keep)) == 2 else np.nan,
                "selection_uses_labels": False,
                "fixed_threshold": 0.5,
            })
    return pd.DataFrame(rows)


def run(root: Path, run_id: str, bootstrap_resamples: int) -> Path:
    if not re.fullmatch(r"manuscript_results_support_[a-z0-9_.-]+", run_id):
        raise SupportError("run-id must start with manuscript_results_support_")
    root = root.resolve()
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise SupportError(f"Run directory already exists: {destination}")

    required = {
        "nineml_cv5_full_all_20260905": ["RUN_MANIFEST.json", "cv5_pooled_oof_metrics.csv"],
        "quad_blend_20260822": ["RUN_MANIFEST.json", "cv5_train_oof_predictions.csv",
                                  "predictions_d1_test.csv"],
        "screeningblend_altmodels_20260819": ["RUN_MANIFEST.json",
                                                "d1_train_oof_predictions.csv"],
        "repeated_rank_stability_20260826": ["RUN_MANIFEST.json",
                                               "per_row_repeated_oof_predictions.csv"],
        "chemical_space_cv_20260826": ["RUN_MANIFEST.json",
                                         "per_row_grouped_oof_predictions.csv"],
        "modelwide_druglikeness_bias_20260826": ["RUN_MANIFEST.json",
                                                   "model_cohort_predictions.csv"],
    }
    provenance = {
        run_id_: _require_sealed(root / "outputs" / run_id_, files)
        for run_id_, files in required.items()
    }
    full_oof = _full_oof_table(root)
    per_repeat, repeat_summary, grouped_summary = _qed_robustness(
        root, bootstrap_resamples=bootstrap_resamples
    )
    risk_coverage = _risk_coverage(root)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".manuscript-support.work-", dir=destination.parent))
    try:
        _write_csv(temporary / "full_rdkit2d_cv5_pooled_oof_metrics.csv", full_oof)
        _write_csv(temporary / "qed_repeated_per_repeat.csv", per_repeat)
        _write_csv(temporary / "qed_repeated_summary.csv", repeat_summary)
        _write_csv(temporary / "qed_grouped_summary.csv", grouped_summary)
        _write_csv(temporary / "d1_heldout_risk_coverage.csv", risk_coverage)
        atomic_write_json(temporary / "RUN_MANIFEST.json", {
            "schema_version": f"{SCHEMA}.run_manifest.v1",
            "run_id": run_id,
            "analysis_type": "post-lock descriptive support; no model fitting or selection",
            "upstream_verified_hashes": provenance,
            "producer_sha256": sha256_file(Path(__file__)),
            "statistical_contract": {
                "qed_endpoint": "positive-only Spearman correlation between QED and score",
                "repeated_registry": "10 prespecified five-fold OOF repeats",
                "grouped_registries": ["C1_scaffold", "C2_similarity_component"],
                "bootstrap": "compound-level percentile, 95% CI",
                "bootstrap_resamples": bootstrap_resamples,
                "bootstrap_seed_base": 20260905,
                "risk_coverage_order": "descending absolute distance from 0.5; row index tie break",
                "risk_coverage_labels_used_for_selection": False,
                "decision_threshold": 0.5,
            },
            "leakage_guards": {
                "models_refitted": False,
                "features_selected": False,
                "thresholds_tuned": False,
                "ensemble_weights_tuned": False,
                "heldout_labels_used_only_for_post_lock_evaluation": True,
            },
            "runtime": {"python": platform.python_version(), "platform": platform.platform()},
            "existing_runs_modified": False,
        })
        atomic_write_json(temporary / "COMPLETED.json", {
            "schema_version": f"{SCHEMA}.completed.v1",
            "status": "COMPLETE",
            "run_id": run_id,
            "run_manifest_sha256": sha256_file(temporary / "RUN_MANIFEST.json"),
            "artifact_hashes": {
                str(path.relative_to(temporary)): sha256_file(path)
                for path in sorted(temporary.rglob("*"))
                if path.is_file() and path.name != "COMPLETED.json"
            },
        })
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps({"run": str(destination), "status": "COMPLETE"}, indent=2))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args(argv)
    run(args.root, args.run_id, args.bootstrap_resamples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
