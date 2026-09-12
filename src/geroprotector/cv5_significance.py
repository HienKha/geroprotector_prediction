"""Significance testing for the 5-fold D1-train CV: equal-thirds TabFM vs the others.

Reads the sealed per-row CV predictions from `cv5_paper_metrics_20260821` and
refits nothing.  Same 324 D1-training rows, same folds, same decisions.

THREE TESTS, because no single one is sufficient here:

1. PAIRED BOOTSTRAP over compounds (primary for effect size).
   Resample the 324 compounds with replacement, recompute the metric for both
   models on the same resample, take the difference.  Gives a 95% percentile CI
   for the difference and a two-sided bootstrap p-value.  Works for every
   metric, including ones with no closed-form test (kappa, F1, specificity).

2. McNEMAR'S EXACT TEST (primary for accuracy / decisions).
   The textbook paired test for two classifiers on the same items.  It
   conditions on the discordant pairs only -- cases where exactly one model is
   correct -- and asks whether their split departs from 50/50.  Implemented as
   an exact two-sided binomial test, which is valid at any discordant count
   (no normal approximation, no continuity-correction argument).

3. PER-FOLD PAIRED t-TEST AND WILCOXON (supplementary ONLY).
   Reported for completeness because CV papers often show it, but flagged as
   the weakest evidence here: n=5, and cross-validation folds share training
   data, so the independence assumption underlying the t-test is violated.
   Nadeau & Bengio (2003) and Dietterich (1998) show this makes the naive
   CV t-test anti-conservative -- it finds "significance" too easily.  It is
   NOT used to support any claim in this module.

MULTIPLICITY.  Six metrics are tested per comparison, so a Holm-Bonferroni
adjusted p-value is reported alongside the raw one, per comparison family.
Read the adjusted column when asking "is anything significant here".

KNOWN LIMITATION, stated rather than hidden: the 324 rows are cross-fitted OOF
predictions, so rows in different folds come from models trained on overlapping
data.  Bootstrapping over compounds treats them as exchangeable, which is the
standard practical choice but is mildly optimistic.  Neither the CIs nor the
p-values here account for that residual dependence.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest, ttest_rel, wilcoxon
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file


class CV5SignificanceError(RuntimeError):
    """Raised when a sealed input or a protocol invariant fails."""


SCHEMA = "geroprotector.cv5_significance"


def _specificity(y: np.ndarray, d: np.ndarray) -> float:
    tn, fp, _fn, _tp = confusion_matrix(y, d, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else float("nan")


def _sensitivity(y: np.ndarray, d: np.ndarray) -> float:
    _tn, _fp, fn, tp = confusion_matrix(y, d, labels=[0, 1]).ravel()
    return float(tp / (tp + fn)) if (tp + fn) else float("nan")


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": lambda y, d: float(accuracy_score(y, d)),
    "specificity": _specificity,
    "sensitivity": _sensitivity,
    "cohen_kappa": lambda y, d: float(cohen_kappa_score(y, d)),
    "f1_positive": lambda y, d: float(f1_score(y, d, pos_label=1, zero_division=0)),
    "f1_macro": lambda y, d: float(f1_score(y, d, average="macro", zero_division=0)),
}


def _regular_file(path: Path, role: str, expected: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise CV5SignificanceError(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected is not None and sha256_file(resolved) != expected:
        raise CV5SignificanceError(f"{role} SHA256 differs from the protocol lock")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    text = _regular_file(path, "CV5 significance protocol").read_text(encoding="utf-8")
    protocol = yaml.safe_load(text)
    expected_schema = f"{SCHEMA}.protocol.v1"
    if not isinstance(protocol, dict) or protocol.get("schema_version") != expected_schema:
        raise CV5SignificanceError("Unknown CV5 significance protocol schema")
    contract = protocol.get("contract", {})
    for key in ("nothing_is_refitted", "train_rows_only"):
        if contract.get(key) is not True:
            raise CV5SignificanceError(f"Contract differs at {key}")
    for key in ("test_rows_used", "external_rows_used"):
        if contract.get(key) is not False:
            raise CV5SignificanceError(f"Contract differs at {key}")
    immutability = protocol.get("immutability", {})
    if immutability.get("existing_run_directories_are_read_only") is not True:
        raise CV5SignificanceError("Immutability contract differs")
    for record in protocol.get("sealed_inputs", {}).values():
        _regular_file(root / record["path"], f"sealed input {record['path']}", record["sha256"])
    return protocol, canonical_sha256(protocol)


def _holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""

    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        value = (m - rank) * pvalues[index]
        running = max(running, value)
        adjusted[index] = min(1.0, running)
    return adjusted


def run(*, root: Path, config_path: Path, run_id: str) -> Path:
    if not re.fullmatch(r"cv5_significance_[a-z0-9_.-]+", run_id):
        raise CV5SignificanceError("RUN_ID must start with cv5_significance_")
    root = root.resolve()
    protocol, protocol_sha256 = load_protocol(root, config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise CV5SignificanceError(f"Run directory already exists: {destination}")
    sealed = protocol["sealed_inputs"]

    predictions = pd.read_csv(root / sealed["cv5_per_row_predictions"]["path"])
    if len(predictions) != int(protocol["expected"]["n_rows"]):
        raise CV5SignificanceError("Unexpected CV row count")
    y = predictions.label.to_numpy(dtype=int)
    fold = predictions.fold.to_numpy(dtype=int)
    n_folds = int(protocol["expected"]["n_folds"])

    decisions = {
        "blend_eq_thirds_tabfm":
            (predictions.blend_eq_thirds_tabfm.to_numpy(float) >= 0.5).astype(int),
        "blend_eq_thirds_tabpfnv2":
            (predictions.blend_eq_thirds_tabpfnv2.to_numpy(float) >= 0.5).astype(int),
        "paper_svm_fixed_0p5":
            (predictions.probability_paper_svm.to_numpy(float) >= 0.5).astype(int),
        "paper_svm_native":
            predictions.paper_svm_native_decision.to_numpy(int),
    }
    reference = protocol["comparison"]["reference_model"]
    others = list(protocol["comparison"]["compared_against"])
    if reference not in decisions or any(o not in decisions for o in others):
        raise CV5SignificanceError("Comparison models are not all available")

    resamples = int(protocol["bootstrap"]["resamples"])
    seed = int(protocol["bootstrap"]["seed"])

    rows, mcnemar_rows, fold_rows = [], [], []
    for other in others:
        a = decisions[reference]
        b = decisions[other]

        # ---- 1. paired bootstrap over compounds --------------------------------
        rng = np.random.default_rng(seed)
        draws: dict[str, list[float]] = {name: [] for name in METRICS}
        n = len(y)
        for _ in range(resamples):
            idx = rng.integers(0, n, n)
            yb = y[idx]
            if len(set(yb)) < 2:
                continue
            for name, fn in METRICS.items():
                draws[name].append(fn(yb, a[idx]) - fn(yb, b[idx]))

        raw_p = []
        staged = []
        for name, fn in METRICS.items():
            observed = fn(y, a) - fn(y, b)
            sample = np.asarray(draws[name], dtype=float)
            low, high = np.percentile(sample, [2.5, 97.5])
            # two-sided bootstrap p: how often the resampled difference lands on
            # the opposite side of zero from the observed effect
            share = float(np.mean(sample <= 0.0)) if observed > 0 else float(
                np.mean(sample >= 0.0)
            )
            p_value = min(1.0, 2.0 * max(share, 1.0 / len(sample)))
            raw_p.append(p_value)
            staged.append({
                "reference_model": reference, "compared_model": other, "metric": name,
                f"{reference}_value": float(fn(y, a)),
                f"{other}_value": float(fn(y, b)),
                "difference": float(observed),
                "ci95_low": float(low), "ci95_high": float(high),
                "ci_excludes_zero": bool(low > 0 or high < 0),
                "bootstrap_p_raw": float(p_value),
                "n_bootstrap_effective": len(sample),
            })
        for record, adjusted in zip(staged, _holm(raw_p), strict=True):
            record["bootstrap_p_holm"] = float(adjusted)
            record["significant_holm_0p05"] = bool(adjusted < 0.05)
            rows.append(record)

        # ---- 2. McNemar exact --------------------------------------------------
        correct_a, correct_b = (a == y), (b == y)
        only_a = int((correct_a & ~correct_b).sum())
        only_b = int((~correct_a & correct_b).sum())
        discordant = only_a + only_b
        if discordant == 0:
            p_mcnemar = 1.0
        else:
            p_mcnemar = float(
                binomtest(only_a, discordant, 0.5, alternative="two-sided").pvalue
            )
        mcnemar_rows.append({
            "reference_model": reference, "compared_model": other,
            "reference_only_correct": only_a, "compared_only_correct": only_b,
            "discordant_pairs": discordant,
            "both_correct": int((correct_a & correct_b).sum()),
            "both_wrong": int((~correct_a & ~correct_b).sum()),
            "mcnemar_exact_p": p_mcnemar,
            "significant_0p05": bool(p_mcnemar < 0.05),
            "test": "exact two-sided binomial on discordant pairs",
        })

        # ---- 3. per-fold paired t / Wilcoxon (SUPPLEMENTARY, low power) --------
        for name, fn in METRICS.items():
            va = np.array([fn(y[fold == f], a[fold == f]) for f in range(n_folds)])
            vb = np.array([fn(y[fold == f], b[fold == f]) for f in range(n_folds)])
            difference = va - vb
            if np.allclose(difference, 0.0):
                t_p = w_p = 1.0
            else:
                t_p = float(ttest_rel(va, vb).pvalue)
                try:
                    w_p = float(wilcoxon(va, vb).pvalue)
                except ValueError:
                    w_p = float("nan")
            fold_rows.append({
                "reference_model": reference, "compared_model": other, "metric": name,
                "mean_fold_difference": float(difference.mean()),
                "sd_fold_difference": float(difference.std(ddof=1)),
                "folds_won": int((difference > 0).sum()), "n_folds": n_folds,
                "paired_t_p": t_p, "wilcoxon_p": w_p,
                "interpretation": "SUPPLEMENTARY ONLY - n=5 and folds share training data",
            })

    bootstrap = pd.DataFrame(rows)
    mcnemar = pd.DataFrame(mcnemar_rows)
    per_fold = pd.DataFrame(fold_rows)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".cv5sig.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "paired_bootstrap_ci_and_p.csv", bootstrap)
        _write_csv(tmp / "mcnemar_exact.csv", mcnemar)
        _write_csv(tmp / "per_fold_ttest_supplementary.csv", per_fold)

        lines = [
            f"# Significance: {reference} vs the others (5-fold D1-train CV, n=324)",
            "",
            "Primary evidence: paired bootstrap over compounds (effect sizes, any metric)",
            "and McNemar's exact test (decisions/accuracy). The per-fold paired t-test is",
            "reported but is **supplementary only** - n=5 and CV folds share training data,",
            "which violates its independence assumption and makes it anti-conservative.",
            "",
            f"Bootstrap: {resamples} resamples, seed {seed}. Holm-Bonferroni adjustment is",
            "applied across the six metrics within each comparison.",
            "",
            "## McNemar exact test (paired decisions)",
            "",
            "| Compared model | TabFM-only correct | Other-only correct | "
            "Discordant | Exact p | Significant |",
            "|---|---|---|---|---|---|",
        ]
        for row in mcnemar.itertuples(index=False):
            lines.append(
                f"| {row.compared_model} | {row.reference_only_correct} | "
                f"{row.compared_only_correct} | {row.discordant_pairs} | "
                f"{row.mcnemar_exact_p:.4g} | {'YES' if row.significant_0p05 else 'no'} |"
            )
        lines += ["", "## Paired bootstrap: difference, 95% CI, p", ""]
        for other in others:
            sub = bootstrap[bootstrap.compared_model == other]
            lines += [f"### vs `{other}`", "",
                      "| Metric | Difference | 95% CI | CI excludes 0 | "
                      "p (raw) | p (Holm) | Significant |",
                      "|---|---|---|---|---|---|---|"]
            for row in sub.itertuples(index=False):
                lines.append(
                    f"| {row.metric} | {row.difference:+.4f} | "
                    f"[{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] | "
                    f"{'yes' if row.ci_excludes_zero else 'no'} | "
                    f"{row.bootstrap_p_raw:.4g} | {row.bootstrap_p_holm:.4g} | "
                    f"{'YES' if row.significant_holm_0p05 else 'no'} |"
                )
            lines.append("")
        (tmp / "summary.md").write_text("\n".join(lines), encoding="utf-8")

        manifest = {
            "schema_version": f"{SCHEMA}.run_manifest.v1",
            "run_id": run_id, "protocol_sha256": protocol_sha256,
            "reference_model": reference, "compared_against": others,
            "metrics": list(METRICS),
            "bootstrap": {"resamples": resamples, "seed": seed,
                          "unit": "compound (row)", "type": "paired percentile"},
            "mcnemar": "exact two-sided binomial on discordant pairs",
            "multiplicity_correction": "Holm-Bonferroni across the six metrics per comparison",
            "per_fold_ttest_role": (
                "supplementary only; n=5 and CV folds share training data, violating "
                "independence and making the naive CV t-test anti-conservative "
                "(Dietterich 1998; Nadeau & Bengio 2003)"
            ),
            "known_limitation": (
                "the 324 rows are cross-fitted OOF predictions, so rows in different folds "
                "come from models trained on overlapping data; bootstrapping over compounds "
                "treats them as exchangeable, which is standard but mildly optimistic"
            ),
            "nothing_is_refitted": True,
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
    parser.add_argument("--run-id", required=True)
    a = parser.parse_args(argv)
    run(root=a.root, config_path=a.config, run_id=a.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
