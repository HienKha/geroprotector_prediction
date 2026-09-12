"""Five-component ablation for any candidate fifth component, on D1 test and 5-fold CV.

The base blend is always the four components already in the manuscript:

    paper_svm, tanimoto_svc, tabpfn_v2, tabfm

A single candidate is added as a fifth, and all 31 non-empty subsets of the five are scored
as equal-weight probability averages at a threshold fixed at 0.5. Supported candidates:

    catboost, lightgbm, xgboost, bishop, tabm, tabnet      (all on the full RDKit2D panel)

NOTHING IS REFITTED.  Every component was fitted once on the 324 D1 training compounds and
scored once, so every subset is itself a locked model with prespecified weights. This module
only reads those streams, re-weights them, and computes statistics. Streams are taken from
whichever sealed run already contains them:

    d1_test        catboost/lightgbm/xgboost : nineml_full_scaled_20260823
                   bishop/tabm/tabnet        : screeningblend_altmodels_20260819
                   base four                 : quad_blend_20260822
    cv5_train_oof  catboost/lightgbm/xgboost : nineml_cv5_full_*      (must be run first)
                   bishop/tabm/tabnet        : screeningblend_altmodels_20260819
                   base four                 : quad_blend_20260822

Every source is row-aligned against the base run and the two shared chemistry streams
(paper_svm, tanimoto_svc) are proven identical across runs before anything is computed; a
mismatch aborts rather than being silently tolerated.

Statistics: percentile bootstrap 95% confidence intervals from 10,000 compound-level
resamples, and a paired bootstrap of each subset against the full five-component blend with
Holm-Bonferroni correction across the thirty reduced subsets within each metric. A paired
analysis of the candidate's marginal effect (subset S versus S plus the candidate, over the
fifteen candidate-free subsets) is reported alongside.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from geroprotector.hashing import atomic_write_json, sha256_file


class AblationCandidateError(RuntimeError):
    """Raised when a stream is missing, misaligned, or fails a parity proof."""


SCHEMA = "geroprotector.ablation_candidate"
BASE = ["paper_svm", "tanimoto_svc", "tabpfn_v2", "tabfm"]
SHORT = {"paper_svm": "SVM", "tanimoto_svc": "Tan", "tabpfn_v2": "PFN", "tabfm": "FM"}
CANDIDATES = ("catboost", "lightgbm", "xgboost", "bishop", "tabm", "tabnet")
CANDIDATE_SHORT = {"catboost": "CatB", "lightgbm": "LGBM", "xgboost": "XGB",
                   "bishop": "BiS", "tabm": "TabM", "tabnet": "Net"}
ALTMODELS = ("bishop", "tabm", "tabnet")
GBM = ("catboost", "lightgbm", "xgboost")
METRICS = ["Accuracy", "Sensitivity", "Specificity", "Kappa", "AUROC", "AUPRC",
           "Brier", "MCC", "MacroF1"]
LOWER_BETTER = {"Brier"}
THRESHOLD, BOOTSTRAP, SEED, ATOL = 0.5, 10000, 20260823, 1e-12


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def _need(path: Path, role: str) -> Path:
    if not path.is_file():
        raise AblationCandidateError(
            f"{role} not found: {path}\n"
            "    Run nine_ml_cv5_fullfeat first if this is a gradient-boosting CV stream.")
    return path


def load_streams(outputs: Path, candidate: str, gbm_cv_run: str):
    """Return {cohort: (y, panel, parity)} with the five components column-aligned."""
    quad = outputs / "quad_blend_20260822"
    alt = outputs / "screeningblend_altmodels_20260819"
    full = outputs / "nineml_full_scaled_20260823"
    data, parity = {}, {}

    # ---------------- D1 held-out test -------------------------------------
    base = pd.read_csv(_need(quad / "predictions_d1_test.csv", "base D1 test stream"))
    base = base.sort_values("paper_row_index").reset_index(drop=True)
    idx = base.paper_row_index.to_numpy()
    y = base.label.to_numpy(int)
    a = pd.read_csv(_need(alt / "d1_test_predictions.csv", "alt-model D1 test stream"))
    a = a.set_index("paper_row_index").loc[idx]
    for column in ("probability_paper_svm", "probability_tanimoto_svc"):
        parity[f"d1_test:{column}"] = float(np.max(np.abs(
            a[column].to_numpy(float) - base[column].to_numpy(float))))
    if candidate in ALTMODELS:
        extra = a[f"probability_{candidate}"].to_numpy(float)
    else:
        f = pd.read_csv(_need(full / "predictions.csv", "full-feature D1 test stream"))
        f = f[f.model_id == candidate].set_index("paper_row_index").loc[idx]
        if not np.array_equal(f.label.to_numpy(int), y):
            raise AblationCandidateError("d1_test: candidate labels differ from the base run")
        extra = f.probability.to_numpy(float)
    data["d1_test"] = (y, np.column_stack(
        [base[f"probability_{c}"].to_numpy(float) for c in BASE] + [extra]))

    # ---------------- 5-fold CV on D1 train --------------------------------
    base = pd.read_csv(_need(quad / "cv5_train_oof_predictions.csv", "base CV stream"))
    idx = base.paper_row_index.to_numpy()
    y = base.label.to_numpy(int)
    a = pd.read_csv(_need(alt / "d1_train_oof_predictions.csv", "alt-model CV stream"))
    a = a.set_index("paper_row_index").loc[idx]
    for column in ("probability_paper_svm", "probability_tanimoto_svc"):
        parity[f"cv5:{column}"] = float(np.max(np.abs(
            a[column].to_numpy(float) - base[column].to_numpy(float))))
    if candidate in ALTMODELS:
        extra = a[f"probability_{candidate}"].to_numpy(float)
    else:
        g = outputs / gbm_cv_run
        f = pd.read_csv(_need(g / "cv5_train_oof_predictions.csv",
                              f"gradient-boosting CV stream ({gbm_cv_run})"))
        f = f.set_index("paper_row_index").loc[idx]
        if not np.array_equal(f.label.to_numpy(int), y):
            raise AblationCandidateError("cv5: candidate labels differ from the base run")
        column = f"probability_{candidate}"
        if column not in f:
            raise AblationCandidateError(
                f"{column} missing from {gbm_cv_run}; rerun nine_ml_cv5_fullfeat "
                "with a model set that includes it")
        extra = f[column].to_numpy(float)
    data["cv5_train_oof"] = (y, np.column_stack(
        [base[f"probability_{c}"].to_numpy(float) for c in BASE] + [extra]))

    worst = max(parity.values())
    if worst > ATOL:
        raise AblationCandidateError(
            f"shared component streams differ across runs by {worst!r} (> {ATOL}); "
            "the runs are not aligned and must not be pooled")
    return data, parity


# ------------------------------------------------------- vectorised statistics
def _auroc_w(y, p, W):
    o = np.argsort(p, kind="mergesort")
    ys, ps, Ws = y[o].astype(float), p[o], W[:, o]
    e = np.flatnonzero(np.r_[np.diff(ps) != 0, True])
    cp, cn = np.cumsum(Ws * ys, 1)[:, e], np.cumsum(Ws * (1 - ys), 1)[:, e]
    gp, gn = np.diff(cp, axis=1, prepend=0.0), np.diff(cn, axis=1, prepend=0.0)
    den = cp[:, -1] * cn[:, -1]
    return np.divide((gp * ((cn - gn) + 0.5 * gn)).sum(1), den,
                     out=np.full(len(W), np.nan), where=den > 0)


def _ap_w(y, p, W):
    o = np.argsort(-p, kind="mergesort")
    ys, ps, Ws = y[o].astype(float), p[o], W[:, o]
    tp, tot = np.cumsum(Ws * ys, 1), np.cumsum(Ws, 1)
    e = np.flatnonzero(np.r_[np.diff(ps) != 0, True])
    tpe, tote = tp[:, e], tot[:, e]
    pre = np.divide(tpe, tote, out=np.zeros_like(tpe), where=tote > 0)
    npos = tp[:, -1]
    return np.divide((np.diff(tpe, axis=1, prepend=0.0) * pre).sum(1), npos,
                     out=np.full(len(W), np.nan), where=npos > 0)


def bundle(y, p, W):
    d = (p >= THRESHOLD).astype(int)
    tp = W @ ((y == 1) & (d == 1)).astype(float)
    fp = W @ ((y == 0) & (d == 1)).astype(float)
    fn = W @ ((y == 1) & (d == 0)).astype(float)
    tn = W @ ((y == 0) & (d == 0)).astype(float)
    n = W.sum(1)
    safe = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)  # noqa: E731
    acc = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n)
    f1p, f1n = safe(2 * tp, 2 * tp + fp + fn), safe(2 * tn, 2 * tn + fn + fp)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"Accuracy": acc, "Sensitivity": safe(tp, tp + fn),
            "Specificity": safe(tn, tn + fp), "Kappa": safe(acc - pe, 1 - pe),
            "AUROC": _auroc_w(y, p, W), "AUPRC": _ap_w(y, p, W),
            "Brier": (W @ ((p - y) ** 2)) / n, "MCC": safe(tp * tn - fp * fn, den),
            "MacroF1": (f1p + f1n) / 2}


def point(y, p):
    d = (p >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, d, labels=[0, 1]).ravel()
    return {"Accuracy": accuracy_score(y, d), "Sensitivity": tp / (tp + fn),
            "Specificity": tn / (tn + fp), "Kappa": cohen_kappa_score(y, d),
            "AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p), "MCC": matthews_corrcoef(y, d),
            "MacroF1": f1_score(y, d, average="macro", zero_division=0)}


def holm(values):
    v = np.asarray(values, float)
    order = np.argsort(v)
    adjusted = np.empty_like(v)
    running = 0.0
    for i, index in enumerate(order):
        running = max(running, (len(v) - i) * v[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def run(*, root: Path, candidate: str, run_id: str, gbm_cv_run: str) -> Path:
    if candidate not in CANDIDATES:
        raise AblationCandidateError(f"candidate must be one of {CANDIDATES}")
    if not re.fullmatch(r"ablation_[a-z0-9_.-]+", run_id):
        raise AblationCandidateError("RUN_ID must start with ablation_")
    root = root.resolve()
    outputs = root / "outputs"
    destination = outputs / run_id
    if destination.exists() or destination.is_symlink():
        raise AblationCandidateError(f"Run directory already exists: {destination}")

    data, parity = load_streams(outputs, candidate, gbm_cv_run)
    print(f"candidate={candidate}; shared-stream parity max {max(parity.values()):.3e}",
          flush=True)
    names = [SHORT[c] for c in BASE] + [CANDIDATE_SHORT[candidate]]
    label = lambda s: "+".join(names[i] for i in s)  # noqa: E731
    subs = [tuple(c) for k in range(1, 6) for c in itertools.combinations(range(5), k)]
    full = (0, 1, 2, 3, 4)
    rng = np.random.default_rng(SEED)
    grid_rows, pair_rows = [], []

    for cohort, (y, panel) in data.items():
        n = len(y)
        W = rng.multinomial(n, np.full(n, 1 / n), size=BOOTSTRAP).astype(float)
        W = W[(W @ (y == 1).astype(float) > 0) & (W @ (y == 0).astype(float) > 0)]
        score = {s: panel[:, list(s)] @ np.full(len(s), 1 / len(s)) for s in subs}
        boot = {s: bundle(y, score[s], W) for s in subs}
        pt = {s: point(y, score[s]) for s in subs}
        one = np.ones((1, n))
        for s in subs:                       # validate the fast estimators
            check = bundle(y, score[s], one)
            for m in METRICS:
                if abs(check[m][0] - pt[s][m]) > 1e-9:
                    raise AblationCandidateError(f"metric mismatch at {cohort} {s} {m}")
        for metric in METRICS:
            raw = {}
            for s in subs:
                v = boot[s][metric]
                lo, hi = np.percentile(v, [2.5, 97.5])
                d = boot[full][metric] - v
                if metric in LOWER_BETTER:
                    d = -d
                raw[s] = min(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0)
                grid_rows.append({"cohort": cohort, "candidate": candidate, "n": n,
                                  "metric": metric, "subset": label(s), "k": len(s),
                                  "has_candidate": int(4 in s),
                                  **{f"has_{BASE[i]}": int(i in s) for i in range(4)},
                                  "value": pt[s][metric], "ci_low": lo, "ci_high": hi,
                                  "p_vs_full": raw[s]})
            others = [s for s in subs if s != full]
            for s, q in zip(others, holm([raw[s] for s in others])):
                for r in grid_rows:
                    if (r["cohort"] == cohort and r["metric"] == metric
                            and r["subset"] == label(s)):
                        r["q_vs_full"] = q
        # marginal effect of the candidate
        base_subs = [s for s in subs if 4 not in s]
        for metric in METRICS:
            raw = {}
            for s in base_subs:
                s2 = tuple(sorted(s + (4,)))
                d = boot[s2][metric] - boot[s][metric]
                if metric in LOWER_BETTER:
                    d = -d
                lo, hi = np.percentile(d, [2.5, 97.5])
                raw[s] = min(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0)
                delta = pt[s2][metric] - pt[s][metric]
                pair_rows.append({
                    "cohort": cohort, "candidate": candidate, "metric": metric,
                    "base_subset": label(s), "with_candidate": label(s2),
                    "value_base": pt[s][metric], "value_with": pt[s2][metric],
                    "delta_favouring_candidate": -delta if metric in LOWER_BETTER else delta,
                    "ci_low": lo, "ci_high": hi, "p": raw[s]})
            for s, q in zip(base_subs, holm([raw[s] for s in base_subs])):
                for r in pair_rows:
                    if (r["cohort"] == cohort and r["metric"] == metric
                            and r["base_subset"] == label(s)):
                        r["q"] = q
        print(f"  {cohort}: 31 subsets, 15 paired comparisons, {len(W)} resamples",
              flush=True)

    grid = pd.DataFrame(grid_rows)
    pair = pd.DataFrame(pair_rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".ablcand.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "ablation_grid.csv", grid)
        _write_csv(tmp / "candidate_paired_effect.csv", pair)
        _write_csv(tmp / "parity_checks.csv", pd.DataFrame(
            [{"check": k, "max_abs_difference": v, "passed": v <= ATOL}
             for k, v in parity.items()]))
        atomic_write_json(tmp / "RUN_MANIFEST.json", {
            "schema_version": f"{SCHEMA}.run_manifest.v1", "run_id": run_id,
            "base_components": BASE, "candidate_fifth_component": candidate,
            "n_subsets": len(subs), "cohorts": list(data),
            "fixed_threshold": THRESHOLD, "threshold_is_tuned": False,
            "any_model_is_refitted": False,
            "reference_for_significance": "full five-component blend",
            "multiplicity": "Holm-Bonferroni across the 30 reduced subsets per metric",
            "bootstrap_resamples": BOOTSTRAP, "seed": SEED,
            "shared_stream_parity": parity, "parity_atol": ATOL,
            "gbm_cv_source": gbm_cv_run if candidate in GBM else None,
            "runtime": {"python": platform.python_version(),
                        "platform": platform.platform()},
            "existing_runs_modified": False})
        atomic_write_json(tmp / "COMPLETED.json", {
            "schema_version": f"{SCHEMA}.completed.v1", "status": "COMPLETE",
            "run_id": run_id,
            "run_manifest_sha256": sha256_file(tmp / "RUN_MANIFEST.json"),
            "artifact_hashes": {str(p.relative_to(tmp)): sha256_file(p)
                                for p in sorted(tmp.rglob("*"))
                                if p.is_file() and p.name != "COMPLETED.json"}})
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)

    for cohort in data:
        s = grid[(grid.cohort == cohort) & (grid.metric == "MCC")].nlargest(5, "value")
        print(f"\n  top 5 by MCC, {cohort}:")
        for i, r in enumerate(s.itertuples(), 1):
            mark = "  <- contains candidate" if r.has_candidate else ""
            print(f"    #{i} {r.subset:26s} {r.value:+.3f}{mark}")
    print("\n" + json.dumps({"run": str(destination), "status": "COMPLETE"}, indent=2))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gbm-cv-run", default="nineml_cv5_full_20260823",
                        help="run id holding full-panel CV OOF for the GBMs")
    a = parser.parse_args(argv)
    run(root=a.root, candidate=a.candidate, run_id=a.run_id, gbm_cv_run=a.gbm_cv_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
