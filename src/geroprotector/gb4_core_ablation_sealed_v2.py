"""Fully provenance-pinned successor to `gb4_core_ablation_sealed_20260828`.

The predecessor is numerically correct and its input hashes do agree with
`quad_blend_20260822/COMPLETED.json`, but it never checked that agreement at run
time and it did not record the hashes of the code that produced it. Those are
provenance gaps, not numerical ones, so this successor recomputes the identical
table under identical settings while adding:

  * fail-fast verification of the upstream completion record and of every input
    stream's SHA-256 *before* any probability is read;
  * hashes of the producing module, the reused estimator module, and the hashing
    helper, so the code path is pinned as tightly as the data;
  * a canonical hash over the full statistical contract;
  * an exact-parity assertion against the predecessor grid.

Nothing is refitted, no seed or threshold changes, and the subset enumeration,
bootstrap procedure and Holm family are inherited unchanged from the predecessor
module so the two runs cannot silently diverge.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from geroprotector import ablation_candidate as _estimators
from geroprotector import gb4_core_ablation_sealed as _v1
from geroprotector import hashing as _hashing
from geroprotector.ablation_candidate import (
    BOOTSTRAP,
    LOWER_BETTER,
    METRICS,
    SEED,
    THRESHOLD,
    bundle,
    holm,
    point,
)
from geroprotector.gb4_core_ablation_sealed import (
    COHORT_FILES,
    COMPONENT_COLUMN,
    FULL,
    LONG,
    SOURCE_RUN,
    CoreAblationError,
    _load_cohort,
)
from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file

SCHEMA = "geroprotector.gb4_core_ablation_sealed_v2"
PREDECESSOR = "gb4_core_ablation_sealed_20260828"
PARITY_ATOL = 1e-15
# Every local module that materially implements hashing, metrics, bootstrap
# sampling, Holm adjustment or output sealing.
CODE_MODULES = {
    "gb4_core_ablation_sealed_v2.py": __file__,
    "gb4_core_ablation_sealed.py": _v1.__file__,
    "ablation_candidate.py": _estimators.__file__,
    "hashing.py": _hashing.__file__,
}
# Files the upstream completion record must declare and whose hashes must match.
REQUIRED_UPSTREAM = ("RUN_MANIFEST.json", "predictions_d1_test.csv",
                     "cv5_train_oof_predictions.csv")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def verify_upstream(outputs: Path) -> dict:
    """Refuse to read a single probability until the upstream run is proven intact."""
    source = outputs / SOURCE_RUN
    completed_path = source / "COMPLETED.json"
    if not completed_path.is_file():
        raise CoreAblationError(f"Upstream completion record absent: {completed_path}")
    completed = json.loads(completed_path.read_text())
    if completed.get("status") != "COMPLETE":
        raise CoreAblationError(
            f"Upstream run {SOURCE_RUN} status is {completed.get('status')!r}, "
            f"expected 'COMPLETE'")
    declared = completed.get("artifact_hashes") or {}
    verified = {}
    for relative in REQUIRED_UPSTREAM:
        if relative not in declared:
            raise CoreAblationError(
                f"Upstream completion record does not declare {relative!r}; "
                f"it declares {sorted(declared)}")
        target = source / relative
        if not target.is_file():
            raise CoreAblationError(f"Upstream artifact absent on disk: {target}")
        observed = sha256_file(target)
        if observed != declared[relative]:
            raise CoreAblationError(
                f"Upstream SHA-256 mismatch for {relative}: completion record says "
                f"{declared[relative]}, file on disk is {observed}")
        verified[relative] = observed
    return {
        "upstream_run": SOURCE_RUN,
        "completed_json": {"path": str(completed_path),
                           "sha256": sha256_file(completed_path),
                           "status": completed.get("status")},
        "run_manifest_sha256_recorded_upstream": completed.get("run_manifest_sha256"),
        "verified_input_hashes": verified,
        "files_required_and_checked": list(REQUIRED_UPSTREAM),
        "checked_before_reading_probabilities": True,
    }


def statistical_contract(subsets, label_of) -> dict:
    """The full statistical contract, hashed canonically so it cannot drift silently."""
    return {
        "components": list(COMPONENT_COLUMN),
        "component_order": list(COMPONENT_COLUMN),
        "component_columns": dict(COMPONENT_COLUMN),
        "component_display_names": dict(LONG),
        "subsets": [label_of(s) for s in subsets],
        "n_subsets": len(subsets),
        "combination_rule": "equal weights, arithmetic mean of member probabilities",
        "fixed_threshold": THRESHOLD,
        "threshold_is_tuned": False,
        "metrics": list(METRICS),
        "lower_is_better": sorted(LOWER_BETTER),
        "bootstrap_unit": "compound",
        "bootstrap_kind": "weighted multinomial resampling of compound multiplicities",
        "bootstrap_resamples": BOOTSTRAP,
        "bootstrap_seed": SEED,
        "confidence_interval_rule": "percentile 2.5 / 97.5",
        "discarded_resample_rule": "resamples containing only one class are discarded "
                                   "because AUROC and AUPRC are undefined there",
        "reference_subset": "full four-component blend",
        "p_value_rule": "two-sided bootstrap: min(1, 2 * min(P(d<=0), P(d>=0))) on the "
                        "full-minus-subset difference, sign-flipped for lower-is-better "
                        "metrics",
        "multiplicity_method": "Holm-Bonferroni",
        "multiplicity_family": "the 14 reduced subsets within each cohort and metric",
        "cohorts": {c: COHORT_FILES[c][1] for c in COHORT_FILES},
        "identity_column": "paper_row_index",
        "row_order": "ascending paper_row_index",
        "any_model_is_refitted": False,
        "held_out_outcome_used_for_selection": False,
    }


def run(*, root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"gb4_core_ablation_sealed_v2_[a-z0-9_.-]+", run_id):
        raise CoreAblationError("RUN_ID must start with gb4_core_ablation_sealed_v2_")
    root = root.resolve()
    outputs = root / "outputs"
    destination = outputs / run_id
    if destination.exists() or destination.is_symlink():
        raise CoreAblationError(f"Run directory already exists: {destination}")

    # ---- fail fast on upstream provenance, before touching any probability ----
    upstream = verify_upstream(outputs)
    print(f"  upstream {SOURCE_RUN} verified: "
          f"{len(upstream['verified_input_hashes'])} input hashes match", flush=True)

    import itertools
    subsets = [tuple(c) for k in range(1, 5) for c in itertools.combinations(range(4), k)]
    if len(subsets) != 15:
        raise CoreAblationError(f"Expected 15 non-empty subsets, enumerated {len(subsets)}")
    base = list(COMPONENT_COLUMN)
    label_of = lambda s: "+".join(LONG[base[i]] for i in s)  # noqa: E731

    rng = np.random.default_rng(SEED)
    grid_rows, source_audit = [], {}
    for cohort in COHORT_FILES:
        y, panel, frame, audit = _load_cohort(outputs, cohort)
        source_audit[cohort] = audit
        n = len(y)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=BOOTSTRAP).astype(float)
        weights = weights[(weights @ (y == 1).astype(float) > 0) &
                          (weights @ (y == 0).astype(float) > 0)]
        audit["bootstrap_resamples_retained"] = int(len(weights))

        score = {s: panel[:, list(s)] @ np.full(len(s), 1 / len(s)) for s in subsets}
        boot = {s: bundle(y, score[s], weights) for s in subsets}
        estimate = {s: point(y, score[s]) for s in subsets}
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
                    **{f"has_{base[i]}": int(i in s) for i in range(4)},
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
    expected = len(COHORT_FILES) * len(subsets) * len(METRICS)
    if len(grid) != expected:
        raise CoreAblationError(f"Grid has {len(grid)} rows, expected {expected}")
    if grid.duplicated(["cohort", "metric", "subset"]).any():
        raise CoreAblationError("Grid contains duplicated cohort/metric/subset rows")
    if not np.isfinite(grid[["value", "ci_low", "ci_high"]].to_numpy(float)).all():
        raise CoreAblationError("Grid contains non-finite estimates or intervals")
    outside = grid[(grid["value"] < grid["ci_low"] - 1e-12) |
                   (grid["value"] > grid["ci_high"] + 1e-12)]
    if len(outside):
        raise CoreAblationError(f"{len(outside)} point estimates fall outside their interval")

    # ---- exact parity with the predecessor -----------------------------------
    predecessor_path = outputs / PREDECESSOR / "ablation_metrics_ci_pvalues.csv"
    if not predecessor_path.is_file():
        raise CoreAblationError(f"Predecessor grid absent: {predecessor_path}")
    predecessor = pd.read_csv(predecessor_path)
    keys = ["cohort", "metric", "subset"]
    merged = grid.merge(predecessor, on=keys, suffixes=("_new", "_old"), how="inner")
    if len(merged) != len(grid):
        raise CoreAblationError(
            f"Predecessor parity covered {len(merged)} of {len(grid)} rows")
    numeric_fields = ["value", "ci_low", "ci_high", "delta_full_minus_subset",
                      "p_vs_full", "q_vs_full", "n", "k"]
    deviations, exact, missing_pattern = {}, {}, {}
    for field in numeric_fields:
        new = merged[f"{field}_new"].to_numpy(float)
        old = merged[f"{field}_old"].to_numpy(float)
        # `q_vs_full` is undefined for the reference subset itself, so NaN is the
        # correct value there. Require the NaN pattern to match exactly, then
        # compare only the defined entries.
        new_missing, old_missing = np.isnan(new), np.isnan(old)
        if not np.array_equal(new_missing, old_missing):
            raise CoreAblationError(
                f"Predecessor parity failed on {field}: the pattern of undefined "
                f"values differs ({int(new_missing.sum())} vs {int(old_missing.sum())})")
        missing_pattern[field] = int(new_missing.sum())
        defined = ~new_missing
        difference = np.abs(new[defined] - old[defined])
        deviations[field] = float(difference.max()) if difference.size else 0.0
        exact[field] = bool(np.array_equal(new[defined], old[defined]))
        if deviations[field] > PARITY_ATOL:
            raise CoreAblationError(
                f"Predecessor parity failed on {field}: max |diff| "
                f"{deviations[field]:.3e} exceeds {PARITY_ATOL:.0e}")
    parity = {"predecessor_run": PREDECESSOR,
              "predecessor_grid_sha256": sha256_file(predecessor_path),
              "rows_compared": int(len(merged)),
              "join_keys": keys,
              "numeric_fields_compared": numeric_fields,
              "max_abs_difference": deviations,
              "bitwise_equal": exact,
              "undefined_entries_per_field": missing_pattern,
              "undefined_value_note": "q_vs_full is undefined for the reference "
                                      "(full four-component) subset; the NaN pattern is "
                                      "required to match and defined entries are compared",
              "tolerance": PARITY_ATOL,
              "passed": True}
    print(f"  predecessor parity: max |diff| over all numeric fields = "
          f"{max(deviations.values()):.3e}", flush=True)

    contract = statistical_contract(subsets, label_of)
    code_hashes = {name: sha256_file(Path(path)) for name, path in CODE_MODULES.items()}

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".coreablv2.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "ablation_metrics_ci_pvalues.csv", grid)
        _write_csv(tmp / "predecessor_parity.csv", pd.DataFrame([
            {"field": f, "max_abs_difference": deviations[f], "bitwise_equal": exact[f],
             "undefined_entries": missing_pattern[f],
             "tolerance": PARITY_ATOL, "passed": deviations[f] <= PARITY_ATOL}
            for f in numeric_fields]))
        atomic_write_json(tmp / "RUN_MANIFEST.json", {
            "schema_version": f"{SCHEMA}.run_manifest.v1", "run_id": run_id,
            "purpose": "provenance-pinned successor to " + PREDECESSOR,
            "supersedes": PREDECESSOR,
            "any_model_is_refitted": False,
            "derived_from_sealed_streams_only": True,
            "upstream_verification": upstream,
            "sources": source_audit,
            "code_provenance": {
                "modules": code_hashes,
                "execution_command": " ".join([sys.executable, "-m",
                                               "geroprotector.gb4_core_ablation_sealed_v2",
                                               "--root", str(root), "--run-id", run_id]),
                "python_executable": sys.executable},
            "statistical_contract": contract,
            "statistical_contract_sha256": canonical_sha256(contract),
            "predecessor_parity": parity,
            "row_counts": {"grid_rows": int(len(grid)),
                           **{c: source_audit[c]["rows"] for c in source_audit}},
            "class_counts": {c: {"positive": source_audit[c]["positive"],
                                 "negative": source_audit[c]["negative"]}
                             for c in source_audit},
            "fold_counts": {c: source_audit[c]["fold_counts"] for c in source_audit},
            "identity_column": "paper_row_index",
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
