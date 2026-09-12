"""Balanced-operating-point variants of the equal-quarters four-component blend.

GOAL.  Find a variant of `quad_eq_quarters` with better sensitivity/specificity
balance at the FIXED 0.5 threshold, admitted only if it is no worse than equal
quarters on accuracy, Cohen's kappa, AUPRC, Brier, MCC and macro-F1.

The threshold stays pinned at 0.5, so balance has to come from somewhere else.
There are exactly two levers, and both are searched here on the 324-row D1 train
cross-fitted OOF and nowhere else:

  ROUTE 1 -- REWEIGHT.  Search the 0.01-step simplex for weight vectors whose
    train-OOF |sensitivity - specificity| at 0.5 beats equal quarters, subject to
    all six admission metrics being >= equal quarters (<= for Brier) on train
    OOF.  Among feasible vectors the most balanced is taken.

  ROUTE 2 -- RECALIBRATE.  Fit a Platt (logistic-on-logit) recalibration of the
    equal-quarters blend on the train OOF and apply it everywhere.  The map is
    monotone, so AUPRC and AUROC are mathematically UNCHANGED; Brier and the
    0.5-decision move.  Thresholding the recalibrated score at 0.5 is exactly
    equivalent to thresholding the raw score at the point the train OOF implies,
    which is why this satisfies "threshold stays 0.5" honestly rather than
    by wordplay.  This is stated plainly rather than presented as free balance.

Both routes select an operating point from training data.  That is legitimate --
no test or external label is touched -- but it is a selection, so the reweight
route is additionally reported NESTED (search repeated inside each CV fold) and
the admission constraints are re-checked out of sample rather than assumed to
hold.

No model is refitted and no GPU is used: this re-weights the four component
streams sealed by quad_blend_20260822.
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
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.quad_blend_paper405 import (
    COMPONENTS,
    FIXED_THRESHOLD,
    PAPER_SVM_NATIVE_THRESHOLD,
    _ap_vec,
    _counts,
    _macro_f1_vec,
    _mcc_vec,
    _metrics,
    _plain_markdown,
    _simplex_grid,
)


class QuadBalancedError(RuntimeError):
    """Raised when a sealed input or an admission contract fails."""


SCHEMA = "geroprotector.quad_blend_balanced"
EQUAL_QUARTERS = np.full(4, 0.25)
ADMISSION_HIGHER = ("accuracy", "cohen_kappa", "auprc", "mcc", "macro_f1")
ADMISSION_LOWER = ("brier",)
COHORTS = ("d1_test", "drugage", "agextend")


def _regular_file(path: Path, role: str, expected: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise QuadBalancedError(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected is not None and sha256_file(resolved) != expected:
        raise QuadBalancedError(f"{role} SHA256 differs from the protocol lock")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = yaml.safe_load(_regular_file(path, "balanced protocol").read_text("utf-8"))
    if not isinstance(protocol, dict) or protocol.get("schema_version") != f"{SCHEMA}.protocol.v1":
        raise QuadBalancedError("Unknown balanced-blend protocol schema")
    contract = protocol["contract"]
    if float(contract["fixed_threshold"]) != FIXED_THRESHOLD:
        raise QuadBalancedError("Threshold contract differs: this module pins 0.5")
    for key in ("any_model_is_refitted",
                "test_or_external_labels_used_for_weights_threshold_or_calibration"):
        if contract.get(key) is not False:
            raise QuadBalancedError(f"Contract differs at {key}")
    if tuple(protocol["admission"]["not_worse_than_eq_quarters_on"]) != (
        ADMISSION_HIGHER + ADMISSION_LOWER
    ):
        raise QuadBalancedError("Admission metric set differs from the contract")
    if protocol.get("immutability", {}).get("existing_run_directories_are_read_only") is not True:
        raise QuadBalancedError("Immutability contract differs")
    for record in protocol["sealed_inputs"].values():
        _regular_file(root / record["path"], f"sealed input {record['path']}", record["sha256"])
    return protocol, canonical_sha256(protocol)


# --------------------------------------------------------- vectorised screening


def _kappa_vec(y: np.ndarray, decisions: np.ndarray) -> np.ndarray:
    tp, fp, fn, tn = _counts(y, decisions)
    n = float(len(y))
    observed = (tp + tn) / n
    expected = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n)
    return np.divide(observed - expected, 1.0 - expected,
                     out=np.zeros_like(observed), where=(1.0 - expected) > 0)


def screen_weights(
    y: np.ndarray, panel: np.ndarray, grid: np.ndarray, floors: dict[str, float],
    chunk: int = 4096,
) -> pd.DataFrame:
    """Every grid weight vector, with its train-OOF metrics and admission verdict."""

    y = np.asarray(y, int)
    records = []
    for start in range(0, len(grid), chunk):
        block = grid[start:start + chunk]
        scores = panel @ block.T
        decisions = (scores >= FIXED_THRESHOLD).astype(float)
        tp, fp, fn, tn = _counts(y, decisions)
        n = float(len(y))
        sensitivity = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=tp + fn > 0)
        specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=tn + fp > 0)
        values = {
            "accuracy": (tp + tn) / n,
            "cohen_kappa": _kappa_vec(y, decisions),
            "auprc": _ap_vec(y, scores),
            "brier": ((scores - y[:, None]) ** 2).mean(axis=0),
            "mcc": _mcc_vec(y, decisions),
            "macro_f1": _macro_f1_vec(y, decisions),
        }
        feasible = np.ones(len(block), dtype=bool)
        for metric in ADMISSION_HIGHER:
            feasible &= values[metric] >= floors[metric] - 1e-12
        for metric in ADMISSION_LOWER:
            feasible &= values[metric] <= floors[metric] + 1e-12
        frame = pd.DataFrame({
            **{f"w_{c}": block[:, i] for i, c in enumerate(COMPONENTS)},
            **values,
            "sensitivity": sensitivity, "specificity": specificity,
            "imbalance_abs_sens_minus_spec": np.abs(sensitivity - specificity),
            "admissible": feasible,
        })
        records.append(frame[frame.admissible | (frame.index < 0)])
        del scores, decisions
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


# ------------------------------------------------------------------------- run


def run(*, root: Path, config_path: Path, run_id: str) -> Path:
    if not re.fullmatch(r"quad_balanced_[a-z0-9_.-]+", run_id):
        raise QuadBalancedError("RUN_ID must start with quad_balanced_")
    root = root.resolve()
    protocol, protocol_sha256 = load_protocol(root, config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise QuadBalancedError(f"Run directory already exists: {destination}")
    sealed = protocol["sealed_inputs"]
    columns = [f"probability_{c}" for c in COMPONENTS]

    oof_frame = pd.read_csv(root / sealed["quad_train_oof"]["path"])
    y_train = oof_frame.label.to_numpy(int)
    oof_panel = oof_frame[columns].to_numpy(float)
    fold_id = oof_frame.fold.to_numpy(int)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_split = list(folds.split(np.arange(len(y_train)), y_train))
    rebuilt = np.full(len(y_train), -1)
    for fold, (_fit, validation) in enumerate(fold_split):
        rebuilt[validation] = fold
    if not np.array_equal(rebuilt, fold_id):
        raise QuadBalancedError("Fold identity differs from the sealed quad_blend run")

    panels, labels, frames = {}, {}, {}
    for cohort in COHORTS:
        frame = pd.read_csv(root / sealed[f"quad_{cohort}"]["path"])
        panels[cohort] = frame[columns].to_numpy(float)
        labels[cohort] = frame.label.to_numpy(int)
        frames[cohort] = frame
    print(f"panels loaded: train {oof_panel.shape}, "
          + ", ".join(f"{c} {panels[c].shape}" for c in COHORTS), flush=True)

    # ---- baseline floors from equal quarters, on TRAIN OOF only --------------
    eq_oof = oof_panel @ EQUAL_QUARTERS
    eq_metrics = _metrics(y_train, eq_oof, FIXED_THRESHOLD)
    floors = {m: eq_metrics[m] for m in ADMISSION_HIGHER + ADMISSION_LOWER}
    eq_imbalance = abs(eq_metrics["sensitivity"] - eq_metrics["specificity"])
    print(f"eq_quarters train OOF: sens {eq_metrics['sensitivity']:.4f} "
          f"spec {eq_metrics['specificity']:.4f} imbalance {eq_imbalance:.4f}", flush=True)
    print("admission floors: " + ", ".join(f"{k}={v:.4f}" for k, v in floors.items()), flush=True)

    # ---- ROUTE 1: reweight -----------------------------------------------------
    step = float(protocol["weight_search"]["grid_step"])
    grid = _simplex_grid(step)
    print(f"screening {len(grid)} weight vectors...", flush=True)
    admissible = screen_weights(y_train, oof_panel, grid, floors)
    print(f"admissible weight vectors: {len(admissible)}", flush=True)

    weight_sets: dict[str, np.ndarray] = {"quad_eq_quarters": EQUAL_QUARTERS.copy()}
    route1_note = ""
    if len(admissible):
        better = admissible[admissible.imbalance_abs_sens_minus_spec < eq_imbalance - 1e-12]
        pool = better if len(better) else admissible
        best = pool.sort_values(
            ["imbalance_abs_sens_minus_spec", "mcc"], ascending=[True, False]).iloc[0]
        weight_sets["quad_balanced_reweight"] = np.array(
            [best[f"w_{c}"] for c in COMPONENTS], dtype=float)
        route1_note = (
            f"{len(admissible)} of {len(grid)} weight vectors satisfy all six admission "
            f"constraints on train OOF; {len(better)} of those also improve balance."
        )
    else:
        route1_note = (
            f"NO weight vector among {len(grid)} satisfies all six admission constraints "
            "on train OOF, so no reweighted balanced variant is admitted."
        )
    print(route1_note, flush=True)

    # ---- ROUTE 2: recalibrate (monotone; AUPRC/AUROC provably unchanged) -------
    clip = lambda p: np.clip(p, 1e-6, 1 - 1e-6)  # noqa: E731
    platt = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(
        logit(clip(eq_oof)).reshape(-1, 1), y_train)
    recalibrate = lambda p: platt.predict_proba(logit(clip(p)).reshape(-1, 1))[:, 1]  # noqa: E731
    slope = float(platt.coef_[0][0])
    if slope <= 0:
        raise QuadBalancedError("Platt slope is non-positive; the map would not be monotone")
    # the equivalent raw-probability cut that recalibrated-at-0.5 corresponds to
    equivalent_cut = float(1.0 / (1.0 + np.exp(-(-float(platt.intercept_[0])) / slope)))
    print(f"Platt slope {slope:.4f}, intercept {float(platt.intercept_[0]):.4f}; "
          f"recalibrated@0.5 == raw@{equivalent_cut:.6f}", flush=True)

    # ---- nested reweight search (honest CV estimate) ---------------------------
    nested_stream = np.full(len(y_train), np.nan)
    nested_rows = []
    if "quad_balanced_reweight" in weight_sets:
        for fold, (fit, validation) in enumerate(fold_split):
            inner_eq = _metrics(y_train[fit], oof_panel[fit] @ EQUAL_QUARTERS, FIXED_THRESHOLD)
            inner_floors = {m: inner_eq[m] for m in ADMISSION_HIGHER + ADMISSION_LOWER}
            inner = screen_weights(y_train[fit], oof_panel[fit], grid, inner_floors)
            inner_imbalance = abs(inner_eq["sensitivity"] - inner_eq["specificity"])
            if len(inner):
                better = inner[inner.imbalance_abs_sens_minus_spec < inner_imbalance - 1e-12]
                pool = better if len(better) else inner
                pick = pool.sort_values(
                    ["imbalance_abs_sens_minus_spec", "mcc"], ascending=[True, False]).iloc[0]
                w = np.array([pick[f"w_{c}"] for c in COMPONENTS], dtype=float)
            else:
                w = EQUAL_QUARTERS.copy()
            nested_stream[validation] = oof_panel[validation] @ w
            nested_rows.append({"fold": fold, "n_admissible": int(len(inner)),
                                **{f"w_{c}": float(v) for c, v in zip(COMPONENTS, w, strict=True)}})
            print(f"  nested fold {fold}: {len(inner)} admissible, w={np.round(w, 2)}", flush=True)

    # ---- evaluate --------------------------------------------------------------
    rows = []

    def add(cohort, model, y, p, threshold=FIXED_THRESHOLD, note=""):
        rows.append({"cohort": cohort, "model": model, "note": note,
                     **_metrics(y, p, threshold)})

    reference = {
        "paper_svm": lambda panel: panel[:, 0],
        "paper_svm_published_point": lambda panel: panel[:, 0],
        "eq_thirds_tabfm": lambda panel: panel[:, [0, 1, 3]] @ np.full(3, 1 / 3),
        "eq_thirds_tabpfn_v2": lambda panel: panel[:, [0, 1, 2]] @ np.full(3, 1 / 3),
    }
    for cohort, panel, y in (
        ("cv5_train_oof", oof_panel, y_train),
        *[(c, panels[c], labels[c]) for c in COHORTS],
    ):
        for name, fn in reference.items():
            add(cohort, name, y, fn(panel),
                PAPER_SVM_NATIVE_THRESHOLD if name.endswith("published_point")
                else FIXED_THRESHOLD,
                note="decision_function>=0" if name.endswith("published_point") else "fixed_0.5")
        for name, w in weight_sets.items():
            add(cohort, name, y, panel @ w,
                note="prespecified" if name == "quad_eq_quarters"
                     else "weights screened on train OOF for balance")
        add(cohort, "quad_eq_quarters_recal", y, recalibrate(panel @ EQUAL_QUARTERS),
            note="Platt fitted on train OOF; monotone, AUPRC/AUROC unchanged")
    if "quad_balanced_reweight" in weight_sets:
        add("cv5_train_oof", "quad_balanced_reweight_NESTED", y_train, nested_stream,
            note="search repeated inside each fold; honest CV estimate")

    metrics = pd.DataFrame(rows)
    metrics["imbalance_abs_sens_minus_spec"] = (
        metrics.sensitivity - metrics.specificity).abs()

    # ---- did the admission constraints survive out of sample? ------------------
    survival = []
    for cohort in ("cv5_train_oof", *COHORTS):
        base = metrics[(metrics.cohort == cohort)
                       & (metrics.model == "quad_eq_quarters")].iloc[0]
        for model in [m for m in metrics[metrics.cohort == cohort].model.unique()
                      if m.startswith("quad_") and m != "quad_eq_quarters"]:
            row = metrics[(metrics.cohort == cohort) & (metrics.model == model)].iloc[0]
            failures = [m for m in ADMISSION_HIGHER if row[m] < base[m] - 1e-12]
            failures += [m for m in ADMISSION_LOWER if row[m] > base[m] + 1e-12]
            survival.append({
                "cohort": cohort, "model": model,
                "admission_holds": not failures, "failed_on": ", ".join(failures),
                "imbalance": row.imbalance_abs_sens_minus_spec,
                "imbalance_eq_quarters": base.imbalance_abs_sens_minus_spec,
                "balance_improved": bool(
                    row.imbalance_abs_sens_minus_spec
                    < base.imbalance_abs_sens_minus_spec - 1e-12),
                **{f"delta_{m}": float(row[m] - base[m])
                   for m in ADMISSION_HIGHER + ADMISSION_LOWER},
            })

    # ---- external rankings, one table per metric per cohort --------------------
    ranking = []
    rank_metrics = ("auprc", "auroc", "accuracy", "cohen_kappa", "mcc", "macro_f1",
                    "brier", "sensitivity", "specificity", "balanced_accuracy")
    for cohort in COHORTS:
        sub = metrics[metrics.cohort == cohort]
        for metric in rank_metrics:
            ascending = metric == "brier"
            ordered = sub.sort_values(metric, ascending=ascending).reset_index(drop=True)
            for position, row in ordered.iterrows():
                ranking.append({"cohort": cohort, "metric": metric,
                                "rank": position + 1, "model": row.model,
                                "value": float(row[metric])})
    ranking_frame = pd.DataFrame(ranking)
    mean_rank = (ranking_frame[ranking_frame.metric.isin(
        ("auprc", "auroc", "accuracy", "cohen_kappa", "mcc", "macro_f1", "brier"))]
        .groupby(["cohort", "model"])["rank"].mean().reset_index()
        .rename(columns={"rank": "mean_rank_7_metrics"}))

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".quadbal.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "metrics_all.csv", metrics)
        _write_csv(tmp / "admission_survival.csv", pd.DataFrame(survival))
        _write_csv(tmp / "external_rankings_by_metric.csv", ranking_frame)
        _write_csv(tmp / "external_mean_rank.csv", mean_rank)
        _write_csv(tmp / "admissible_weight_vectors.csv",
                   admissible.sort_values("imbalance_abs_sens_minus_spec").head(500)
                   if len(admissible) else pd.DataFrame())
        if nested_rows:
            _write_csv(tmp / "nested_fold_weights.csv", pd.DataFrame(nested_rows))
        _write_csv(tmp / "weight_sets.csv", pd.DataFrame([
            {"weight_set": k, **{f"w_{c}": float(v) for c, v in zip(COMPONENTS, w, strict=True)}}
            for k, w in weight_sets.items()]))
        atomic_write_json(tmp / "recalibration.json", {
            "method": "Platt (logistic on logit) fitted on the 324 train-OOF rows only",
            "slope": slope, "intercept": float(platt.intercept_[0]),
            "monotone": True,
            "auprc_and_auroc_unchanged_by_construction": True,
            "recalibrated_at_0p5_equals_raw_at": equivalent_cut,
            "honest_reading": (
                "thresholding the recalibrated score at 0.5 is exactly thresholding the raw "
                "equal-quarters score at "
                f"{equivalent_cut:.6f}; the operating point was chosen from training data, "
                "not from test or external labels"
            ),
        })

        show = ["model", "n", "accuracy", "sensitivity", "specificity",
                "imbalance_abs_sens_minus_spec", "cohen_kappa", "auprc", "brier",
                "mcc", "macro_f1"]
        lines = ["# Balanced-operating-point variants of quad_eq_quarters", "",
                 f"Admission rule: a variant is admitted only if, on the 324-row train OOF, it "
                 f"is no worse than equal quarters on **{', '.join(ADMISSION_HIGHER)}** and no "
                 f"worse on **brier**. Threshold pinned at 0.5 throughout.", "",
                 f"- Route 1 (reweight): {route1_note}",
                 f"- Route 2 (recalibrate): Platt on train OOF; monotone, so AUPRC and AUROC "
                 f"are unchanged by construction. Recalibrated@0.5 == raw@{equivalent_cut:.4f}.",
                 ""]
        lines += ["## Weight sets", ""] + _plain_markdown(pd.DataFrame([
            {"weight_set": k, **{f"w_{c}": float(v) for c, v in zip(COMPONENTS, w, strict=True)}}
            for k, w in weight_sets.items()]))
        for cohort, title in (("cv5_train_oof", "5-fold CV, D1 train (n=324)"),
                              ("d1_test", "D1 held-out test (n=81)"),
                              ("drugage", "DrugAge external (n=446)"),
                              ("agextend", "AgeXtend external (n=69)")):
            sub = metrics[metrics.cohort == cohort].sort_values("mcc", ascending=False)
            lines += ["", f"## {title} -- threshold 0.5", ""] + _plain_markdown(sub[show])
        lines += ["", "## Did the admission constraints survive out of sample?", ""]
        lines += _plain_markdown(pd.DataFrame(survival)[
            ["cohort", "model", "admission_holds", "failed_on", "imbalance",
             "imbalance_eq_quarters", "balance_improved"]])
        for cohort in ("drugage", "agextend"):
            lines += ["", f"## {cohort}: mean rank across 7 headline metrics", ""]
            lines += _plain_markdown(
                mean_rank[mean_rank.cohort == cohort]
                .sort_values("mean_rank_7_metrics")[["model", "mean_rank_7_metrics"]])
        (tmp / "summary.md").write_text("\n".join(lines), encoding="utf-8")

        manifest = {
            "schema_version": f"{SCHEMA}.run_manifest.v1",
            "run_id": run_id, "protocol_sha256": protocol_sha256,
            "components": list(COMPONENTS), "fixed_threshold": FIXED_THRESHOLD,
            "any_model_is_refitted": False, "gpu_used": False,
            "admission_metrics": list(ADMISSION_HIGHER + ADMISSION_LOWER),
            "admission_floors_from": "quad_eq_quarters on the 324 train OOF",
            "admission_floors": floors,
            "route_1_reweight": {
                "grid_step": step, "n_weight_vectors": int(len(grid)),
                "n_admissible": int(len(admissible)), "note": route1_note,
                "selected": (weight_sets["quad_balanced_reweight"].tolist()
                             if "quad_balanced_reweight" in weight_sets else None),
                "nested_variant_reported": bool(nested_rows),
            },
            "route_2_recalibrate": {
                "method": "Platt on train OOF", "slope": slope,
                "intercept": float(platt.intercept_[0]),
                "monotone_so_auprc_auroc_unchanged": True,
                "recalibrated_at_0p5_equals_raw_at": equivalent_cut,
            },
            "test_or_external_labels_used_for_weights_threshold_or_calibration": False,
            "runtime": {"python": platform.python_version(), "platform": platform.platform()},
            "sealed_inputs_sha256": {k: sha256_file(root / v["path"])
                                     for k, v in sealed.items()},
            "existing_runs_modified": False,
        }
        atomic_write_json(tmp / "RUN_MANIFEST.json", manifest)
        atomic_write_json(tmp / "COMPLETED.json", {
            "schema_version": f"{SCHEMA}.completed.v1", "status": "COMPLETE", "run_id": run_id,
            "run_manifest_sha256": sha256_file(tmp / "RUN_MANIFEST.json"),
            "artifact_hashes": {str(p.relative_to(tmp)): sha256_file(p)
                                for p in sorted(tmp.rglob("*"))
                                if p.is_file() and p.name != "COMPLETED.json"},
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
    parser.add_argument("--run-id", required=True)
    a = parser.parse_args(argv)
    run(root=a.root, config_path=a.config, run_id=a.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
