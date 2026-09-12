"""Sealed re-derivation of the 15-subset core ablation used by manuscript Figure 2.

Figure 2 previously read `outputs/gb4_ablation_eq_quarters_20260822/`, which carries
no `RUN_MANIFEST.json` or `COMPLETED.json` and is therefore structurally unsealed
relative to every other source in the figure batch. This module rebuilds the same
table as a properly sealed, immutable derived run.

NO MODEL IS REFITTED. Every number comes from the sealed component-probability
streams in `outputs/quad_blend_20260822/`:

  * `predictions_d1_test.csv`      -- 81 held-out compounds;
  * `cv5_train_oof_predictions.csv` -- 324 cross-fitted training compounds.

All 15 non-empty equal-weight subsets of the four components (paper_svm,
tanimoto_svc, tabpfn_v2, tabfm) are enumerated. Row identity is carried by
`paper_row_index`, the label column is taken from the stream itself, and the OOF
fold column is preserved so fold alignment can be audited. The decision threshold
is fixed at 0.5 for every threshold-dependent metric; nothing is tuned, selected
or chosen by outcome.

Statistics reuse `ablation_candidate`'s validated estimators verbatim, so the two
tables are directly comparable: the same compound-level weighted bootstrap
(`rng.multinomial`, 10,000 resamples, seed 20260823), the same percentile
intervals, the same two-sided bootstrap p value against the full four-component
blend, and the same Holm-Bonferroni correction across the 14 reduced subsets
within each cohort and metric.
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

from geroprotector.ablation_candidate import (
    BASE,
    BOOTSTRAP,
    LOWER_BETTER,
    METRICS,
    SEED,
    THRESHOLD,
    bundle,
    holm,
    point,
)
from geroprotector.hashing import atomic_write_json, sha256_file


class CoreAblationError(RuntimeError):
    """Raised when a sealed input or an internal consistency check fails."""


SCHEMA = "geroprotector.gb4_core_ablation_sealed"
SOURCE_RUN = "quad_blend_20260822"
COHORT_FILES = {
    "d1_test": ("predictions_d1_test.csv", 81),
    "cv5_train_oof": ("cv5_train_oof_predictions.csv", 324),
}
COMPONENT_COLUMN = {
    "paper_svm": "probability_paper_svm",
    "tanimoto_svc": "probability_tanimoto_svc",
    "tabpfn_v2": "probability_tabpfn_v2",
    "tabfm": "probability_tabfm",
}
FULL = (0, 1, 2, 3)
# Long-form component names, matching the labels the predecessor table and manuscript
# Figure 2 already use, so the sealed replacement is a drop-in for both.
LONG = {"paper_svm": "SVM", "tanimoto_svc": "Tanimoto",
        "tabpfn_v2": "TabPFN-v2", "tabfm": "TabFM"}
# The unsealed predecessor, read only to confirm the point estimates agree.
LEGACY_TABLE = "gb4_ablation_eq_quarters_20260822/ablation_metrics_ci_pvalues.csv"
LEGACY_POINT_ATOL = 1e-9


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def _load_cohort(outputs: Path, cohort: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    relative, expected_rows = COHORT_FILES[cohort]
    path = outputs / SOURCE_RUN / relative
    if not path.is_file():
        raise CoreAblationError(f"Sealed component stream missing: {path}")
    frame = pd.read_csv(path)
    if len(frame) != expected_rows:
        raise CoreAblationError(
            f"{cohort}: {len(frame)} rows, expected {expected_rows}")
    for required in ["paper_row_index", "label"] + list(COMPONENT_COLUMN.values()):
        if required not in frame.columns:
            raise CoreAblationError(f"{cohort}: column {required!r} absent from {path.name}")
    if frame["paper_row_index"].duplicated().any():
        raise CoreAblationError(f"{cohort}: duplicated paper_row_index values")
    # Deterministic row order by compound identity, never by file order.
    frame = frame.sort_values("paper_row_index").reset_index(drop=True)
    labels = frame["label"].to_numpy(int)
    if set(np.unique(labels)) != {0, 1}:
        raise CoreAblationError(f"{cohort}: labels are not binary 0/1")
    panel = np.column_stack([frame[COMPONENT_COLUMN[c]].to_numpy(float) for c in BASE])
    if not np.isfinite(panel).all():
        raise CoreAblationError(f"{cohort}: component probabilities contain non-finite values")
    if (panel < 0).any() or (panel > 1).any():
        raise CoreAblationError(f"{cohort}: component probabilities outside [0, 1]")
    audit = {
        "path": str(path), "sha256": sha256_file(path), "rows": int(len(frame)),
        "positive": int(labels.sum()), "negative": int((labels == 0).sum()),
        "identity_column": "paper_row_index",
        "row_order": "ascending paper_row_index",
        "fold_column_present": bool("fold" in frame.columns),
        "fold_counts": (frame["fold"].value_counts().sort_index().to_dict()
                        if "fold" in frame.columns else None),
        "component_columns": {c: COMPONENT_COLUMN[c] for c in BASE},
    }
    return labels, panel, frame, audit


def run(*, root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"gb4_core_ablation_sealed_[a-z0-9_.-]+", run_id):
        raise CoreAblationError("RUN_ID must start with gb4_core_ablation_sealed_")
    root = root.resolve()
    outputs = root / "outputs"
    destination = outputs / run_id
    if destination.exists() or destination.is_symlink():
        raise CoreAblationError(f"Run directory already exists: {destination}")

    subsets = [tuple(c) for k in range(1, 5) for c in itertools.combinations(range(4), k)]
    if len(subsets) != 15:
        raise CoreAblationError(f"Expected 15 non-empty subsets, enumerated {len(subsets)}")
    label_of = lambda s: "+".join(LONG[BASE[i]] for i in s)  # noqa: E731

    rng = np.random.default_rng(SEED)
    grid_rows, source_audit = [], {}
    for cohort in COHORT_FILES:
        y, panel, frame, audit = _load_cohort(outputs, cohort)
        source_audit[cohort] = audit
        n = len(y)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=BOOTSTRAP).astype(float)
        # A resample with only one class present cannot support AUROC/AUPRC.
        weights = weights[(weights @ (y == 1).astype(float) > 0) &
                          (weights @ (y == 0).astype(float) > 0)]
        audit["bootstrap_resamples_retained"] = int(len(weights))

        score = {s: panel[:, list(s)] @ np.full(len(s), 1 / len(s)) for s in subsets}
        boot = {s: bundle(y, score[s], weights) for s in subsets}
        estimate = {s: point(y, score[s]) for s in subsets}

        # The vectorized weighted estimators must agree with the direct ones.
        unit = np.ones((1, n))
        for s in subsets:
            check = bundle(y, score[s], unit)
            for metric in METRICS:
                if abs(check[metric][0] - estimate[s][metric]) > 1e-9:
                    raise CoreAblationError(
                        f"{cohort}/{label_of(s)}/{metric}: weighted estimator disagrees "
                        f"with the direct point estimate")

        for metric in METRICS:
            raw = {}
            for s in subsets:
                values = boot[s][metric]
                low, high = np.percentile(values, [2.5, 97.5])
                delta = boot[FULL][metric] - values
                if metric in LOWER_BETTER:
                    delta = -delta
                raw[s] = min(2 * min((delta <= 0).mean(), (delta >= 0).mean()), 1.0)
                grid_rows.append({
                    "cohort": cohort, "n": n, "metric": metric, "subset": label_of(s),
                    "k": len(s),
                    **{f"has_{BASE[i]}": int(i in s) for i in range(4)},
                    "value": estimate[s][metric], "ci_low": low, "ci_high": high,
                    "delta_full_minus_subset": (estimate[FULL][metric] - estimate[s][metric]),
                    "p_vs_full": raw[s]})
            reduced = [s for s in subsets if s != FULL]
            for s, q in zip(reduced, holm([raw[s] for s in reduced])):
                for row in grid_rows:
                    if (row["cohort"] == cohort and row["metric"] == metric
                            and row["subset"] == label_of(s)):
                        row["q_vs_full"] = q
        print(f"  {cohort}: 15 subsets x {len(METRICS)} metrics, "
              f"{len(weights)} usable resamples", flush=True)

    grid = pd.DataFrame(grid_rows)
    expected_rows = len(COHORT_FILES) * len(subsets) * len(METRICS)
    if len(grid) != expected_rows:
        raise CoreAblationError(f"Grid has {len(grid)} rows, expected {expected_rows}")
    if grid.duplicated(["cohort", "metric", "subset"]).any():
        raise CoreAblationError("Grid contains duplicated cohort/metric/subset rows")
    if not np.isfinite(grid[["value", "ci_low", "ci_high"]].to_numpy(float)).all():
        raise CoreAblationError("Grid contains non-finite estimates or intervals")
    outside = grid[(grid["value"] < grid["ci_low"] - 1e-12) |
                   (grid["value"] > grid["ci_high"] + 1e-12)]
    if len(outside):
        raise CoreAblationError(f"{len(outside)} point estimates fall outside their interval")

    # ---- agreement with the unsealed predecessor, point estimates only ---------
    legacy_path = outputs / LEGACY_TABLE
    legacy_report = {"available": legacy_path.is_file(), "path": str(legacy_path)}
    if legacy_path.is_file():
        legacy = pd.read_csv(legacy_path)
        legacy_report["sha256"] = sha256_file(legacy_path)
        merged = grid.merge(legacy, on=["cohort", "metric", "subset"],
                            suffixes=("_new", "_legacy"), how="inner")
        if len(merged) != len(grid):
            raise CoreAblationError(
                f"Legacy comparison covered {len(merged)} of {len(grid)} rows")
        point_deviation = float(np.max(np.abs(merged["value_new"] - merged["value_legacy"])))
        legacy_report.update({
            "rows_compared": int(len(merged)),
            "max_abs_point_estimate_difference": point_deviation,
            "point_estimate_tolerance": LEGACY_POINT_ATOL,
            "point_estimates_reproduce": bool(point_deviation <= LEGACY_POINT_ATOL),
            "max_abs_ci_low_difference": float(np.max(np.abs(
                merged["ci_low_new"] - merged["ci_low_legacy"]))),
            "max_abs_ci_high_difference": float(np.max(np.abs(
                merged["ci_high_new"] - merged["ci_high_legacy"]))),
            "interval_note": (
                "The predecessor directory records no manifest and therefore no bootstrap "
                "seed, so its resampling draw cannot be reproduced. Point estimates are "
                "deterministic and must agree exactly; interval endpoints and bootstrap p "
                "values may differ by Monte-Carlo error from a different draw. This run "
                "records seed "
                f"{SEED} and {BOOTSTRAP} resamples so it is reproducible going forward.")})
        if not legacy_report["point_estimates_reproduce"]:
            raise CoreAblationError(
                f"Point estimates disagree with the predecessor table by "
                f"{point_deviation:.3e} (> {LEGACY_POINT_ATOL:.0e}); refusing to seal")
        print(f"  legacy point estimates reproduce to {point_deviation:.3e}", flush=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".coreabl.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "ablation_metrics_ci_pvalues.csv", grid)
        _write_csv(tmp / "legacy_point_estimate_comparison.csv", pd.DataFrame([legacy_report]))
        atomic_write_json(tmp / "RUN_MANIFEST.json", {
            "schema_version": f"{SCHEMA}.run_manifest.v1", "run_id": run_id,
            "purpose": "sealed re-derivation of the 15-subset core ablation for Figure 2",
            "any_model_is_refitted": False,
            "derived_from_sealed_streams_only": True,
            "source_run": SOURCE_RUN,
            "sources": source_audit,
            "components": list(BASE),
            "component_order": list(BASE),
            "component_columns": dict(COMPONENT_COLUMN),
            "component_display_names": dict(LONG),
            "subset_enumeration": "all 15 non-empty subsets of the four components",
            "subsets": [label_of(s) for s in subsets],
            "combination_rule": "equal weights, arithmetic mean of member probabilities",
            "fixed_threshold": THRESHOLD, "threshold_is_tuned": False,
            "metrics": list(METRICS), "lower_is_better": sorted(LOWER_BETTER),
            "bootstrap": {"kind": "compound-level weighted multinomial resampling",
                          "resamples": BOOTSTRAP, "seed": SEED,
                          "interval": "percentile 2.5 / 97.5",
                          "resamples_lacking_a_class_discarded": True},
            "multiplicity": {"method": "Holm-Bonferroni",
                             "family": "the 14 reduced subsets within each cohort and metric",
                             "reference": "full four-component blend"},
            "estimators_reused_from": "geroprotector.ablation_candidate "
                                      "(bundle, point, holm) -- identical methodology to the "
                                      "five-candidate ablation runs",
            "legacy_comparison": legacy_report,
            "held_out_labels_used_for_selection": False,
            "threshold_or_subset_selected_by_outcome": False,
            "existing_runs_modified": False,
            "runtime": {"python": platform.python_version(),
                        "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__},
        })
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
    print(json.dumps({"run": str(destination), "status": "COMPLETE"}, indent=2))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    a = parser.parse_args(argv)
    run(root=a.root, run_id=a.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
