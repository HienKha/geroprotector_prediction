"""Locked-stream successor to the decision-curve table in `blend_xai_v5_20260823`.

`blend_xai_v5_20260823/E_decision_curves.csv` is sealed and its DrugAge and
AgeXtend curves are read from the sealed component streams, but its *D1* curves
are computed from explanation-time reconstructed component predictions. The
TabPFN-v2 reconstruction there runs batched inference and is only required to
agree with the locked singleton stream to within 0.005, so a handful of D1
compounds can land on the other side of a threshold. An independent row-level
audit found exactly two such cells, both on the D1 tri-blend curve.

This successor recomputes the whole 3 x 5 x 91 grid from the locked
per-compound probability columns in `quad_blend_20260822` only. Nothing is
trained, refitted, calibrated or selected: every score here is an arithmetic
mean of probabilities that were sealed on 2026-08-22. The historical xAI run is
left byte-unchanged and is retained as the record of what was published there.
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

from geroprotector import hashing as _hashing
from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file

SCHEMA = "geroprotector.locked_decision_curves_v2"
SOURCE_RUN = "quad_blend_20260822"
PREDECESSOR_RUN = "blend_xai_v5_20260823"
PREDECESSOR_TABLE = "E_decision_curves.csv"
# Divergences below this are floating-point noise, not scientific changes.
NOISE_ATOL = 1e-12
# `blend_eq_quarters` was sealed by the upstream run; recomputing it here from the
# four component columns must land on the same number.
QUAD_ATOL = 1e-15

# The D1 stream is keyed by its position in the published split; the external
# cohorts carry their own catalogue identifiers.
IDENTITY = {"d1_test": "paper_row_index", "drugage": "external_id",
            "agextend": "external_id"}
TRIPLE_COLUMNS = ("probability_paper_svm", "probability_tabpfn_v2", "probability_tabfm")
QUAD_COLUMNS = ("probability_paper_svm", "probability_tanimoto_svc",
                "probability_tabpfn_v2", "probability_tabfm")
SEALED_QUAD_COLUMN = "blend_eq_quarters"
# Cohort -> (sealed prediction table, n, positives, negatives). The class counts are
# asserted, not discovered, so a swapped or truncated input cannot pass silently.
COHORTS = {
    "d1_test": ("predictions_d1_test.csv", 81, 46, 35),
    "drugage": ("predictions_drugage.csv", 446, 278, 168),
    "agextend": ("predictions_agextend.csv", 69, 65, 4),
}
MODELS = ("blend_triple", "blend_quad", "paper_svm", "treat_all", "treat_none")
REQUIRED_UPSTREAM = ("RUN_MANIFEST.json", "predictions_d1_test.csv",
                     "predictions_drugage.csv", "predictions_agextend.csv")

CODE_MODULES = {
    "locked_decision_curves_v2.py": __file__,
    "hashing.py": _hashing.__file__,
}


class DecisionCurveError(RuntimeError):
    """Raised when provenance, composition or parity cannot be established."""


def threshold_grid() -> np.ndarray:
    """The canonical 91-point grid, built from integers so it is exactly reproducible."""
    grid = np.asarray([i / 100 for i in range(5, 96)], dtype=float)
    if len(grid) != 91 or grid[0] != 0.05 or grid[-1] != 0.95:
        raise DecisionCurveError("Threshold grid is not the canonical 0.05..0.95 grid")
    return grid


def net_benefit(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Net benefit = TP/n - (FP/n) * pt/(1-pt), the standard decision-curve statistic."""
    y = np.asarray(y, int)
    n = len(y)
    out = np.empty(len(thresholds), dtype=float)
    for i, t in enumerate(thresholds):
        prediction = p >= t
        tp = int((prediction & (y == 1)).sum())
        fp = int((prediction & (y == 0)).sum())
        out[i] = tp / n - (fp / n) * (t / (1.0 - t))
    return out


def treat_all(y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    prevalence = float(np.mean(np.asarray(y, int)))
    return prevalence - (1.0 - prevalence) * (thresholds / (1.0 - thresholds))


def verify_upstream(outputs: Path) -> dict:
    """Refuse to read a single probability until the upstream run is proven intact."""
    source = outputs / SOURCE_RUN
    completed_path = source / "COMPLETED.json"
    if not completed_path.is_file():
        raise DecisionCurveError(f"Upstream completion record absent: {completed_path}")
    completed = json.loads(completed_path.read_text())
    if completed.get("status") != "COMPLETE":
        raise DecisionCurveError(
            f"Upstream run {SOURCE_RUN} status is {completed.get('status')!r}, "
            f"expected 'COMPLETE'")
    declared = completed.get("artifact_hashes") or {}
    verified = {}
    for relative in REQUIRED_UPSTREAM:
        if relative not in declared:
            raise DecisionCurveError(
                f"Upstream completion record does not declare {relative!r}; "
                f"it declares {sorted(declared)}")
        target = source / relative
        if not target.is_file():
            raise DecisionCurveError(f"Upstream artifact absent on disk: {target}")
        observed = sha256_file(target)
        if observed != declared[relative]:
            raise DecisionCurveError(
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


def load_cohort(outputs: Path, cohort: str) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Load one sealed stream and assert its exact declared composition."""
    filename, n_expected, pos_expected, neg_expected = COHORTS[cohort]
    identity = IDENTITY[cohort]
    path = outputs / SOURCE_RUN / filename
    frame = pd.read_csv(path)
    if identity not in frame.columns:
        raise DecisionCurveError(f"{cohort}: identity column {identity!r} absent")
    frame = frame.sort_values(identity).reset_index(drop=True)
    if frame[identity].duplicated().any():
        raise DecisionCurveError(f"{cohort}: duplicated {identity} values")
    missing = [c for c in (*QUAD_COLUMNS, SEALED_QUAD_COLUMN, "label")
               if c not in frame.columns]
    if missing:
        raise DecisionCurveError(f"{cohort}: sealed stream lacks columns {missing}")

    y = frame["label"].to_numpy(int)
    if set(np.unique(y)) - {0, 1}:
        raise DecisionCurveError(f"{cohort}: labels are not binary 0/1")
    n, pos, neg = len(y), int((y == 1).sum()), int((y == 0).sum())
    if (n, pos, neg) != (n_expected, pos_expected, neg_expected):
        raise DecisionCurveError(
            f"{cohort}: composition is n={n} ({pos} positive, {neg} negative), "
            f"expected n={n_expected} ({pos_expected} positive, {neg_expected} negative)")

    probabilities = frame[list(QUAD_COLUMNS)].to_numpy(float)
    if not np.isfinite(probabilities).all():
        raise DecisionCurveError(f"{cohort}: non-finite component probabilities")
    if probabilities.min() < 0.0 or probabilities.max() > 1.0:
        raise DecisionCurveError(f"{cohort}: component probabilities outside [0, 1]")

    triple = frame[list(TRIPLE_COLUMNS)].to_numpy(float) @ np.full(3, 1 / 3)
    quad = probabilities @ np.full(4, 0.25)
    sealed_quad = frame[SEALED_QUAD_COLUMN].to_numpy(float)
    quad_deviation = float(np.abs(quad - sealed_quad).max())
    if quad_deviation > QUAD_ATOL:
        raise DecisionCurveError(
            f"{cohort}: recomputed four-component mean deviates from the sealed "
            f"{SEALED_QUAD_COLUMN} column by {quad_deviation:.3e} > {QUAD_ATOL:.0e}")

    scores = {"blend_triple": triple, "blend_quad": quad,
              "paper_svm": frame["probability_paper_svm"].to_numpy(float)}
    audit = {
        "path": str(path.relative_to(outputs.parent)),
        "sha256": sha256_file(path),
        "rows": n, "positive": pos, "negative": neg,
        "prevalence": pos / n,
        "identity_column": identity,
        "identity_is_unique": True,
        "row_order": f"ascending {identity}",
        "component_columns_read": list(QUAD_COLUMNS),
        "recomputed_quad_vs_sealed_max_abs_difference": quad_deviation,
        "recomputed_quad_matches_sealed_blend_eq_quarters": True,
    }
    return y, scores, audit


def compare_with_predecessor(outputs: Path, curves: pd.DataFrame,
                             cohort_data: dict) -> dict:
    """Quantify exactly how this locked-stream table differs from the sealed xAI run.

    Two independent effects are separated here, because conflating them would hide
    a real change:

      * the *stream* effect the audit reported -- the predecessor's D1 curves come
        from reconstructed component predictions, so two D1 tri-blend cells move.
        Recomputing from the locked streams on the predecessor's own threshold grid
        isolates this effect and must reproduce every other cell.
      * a *grid* effect the audit did not anticipate. The predecessor built its grid
        with ``np.arange``, whose 46th point is 0.5000000000000001 rather than 0.5.
        The published SVM assigns a probability of exactly 0.5 to a number of
        compounds in every cohort, and ``p >= 0.5000000000000001`` excludes them
        while ``p >= 0.5`` includes them. The canonical grid mandated here restores
        agreement with the Methods ("0.05 to 0.95 in increments of 0.01") and with
        the ``p >= 0.5`` operating rule every other analysis in this repository uses,
        so the SVM net benefit at threshold 0.5 changes in all three cohorts.
    """
    path = outputs / PREDECESSOR_RUN / PREDECESSOR_TABLE
    if not path.is_file():
        raise DecisionCurveError(f"Predecessor curve table absent: {path}")
    old = pd.read_csv(path)
    # The predecessor stored its thresholds with binary-representation dust
    # (0.060000000000000005). Join on the integer grid index both tables came from.
    for frame in (curves, old):
        frame["threshold_key"] = np.rint(frame["threshold"].to_numpy(float) * 100).astype(int)
    if not np.allclose(old["threshold"].to_numpy(float),
                       old["threshold_key"].to_numpy(float) / 100, atol=1e-9):
        raise DecisionCurveError("Predecessor thresholds do not lie on the 0.01 grid")
    keys = ["cohort", "model", "threshold_key"]
    merged = curves.merge(old, on=keys, suffixes=("_new", "_old"), how="inner")
    if len(merged) != len(curves) or len(merged) != len(old):
        raise DecisionCurveError(
            f"Predecessor comparison covered {len(merged)} cells; this run has "
            f"{len(curves)} and the predecessor has {len(old)}")

    # ---- effect 1: locked streams on the predecessor's own grid ----------------
    predecessor_grid = np.arange(0.05, 0.951, 0.01)
    if len(predecessor_grid) != 91:
        raise DecisionCurveError("Predecessor grid reconstruction is not 91 points")
    replay = []
    for cohort, (y, scores) in cohort_data.items():
        for model in ("blend_triple", "blend_quad", "paper_svm"):
            for t, value in zip(predecessor_grid,
                                net_benefit(y, scores[model], predecessor_grid)):
                replay.append({"cohort": cohort, "model": model,
                               "threshold_key": int(round(float(t) * 100)),
                               "net_benefit_replay": float(value)})
        for t, value in zip(predecessor_grid, treat_all(y, predecessor_grid)):
            replay.append({"cohort": cohort, "model": "treat_all",
                           "threshold_key": int(round(float(t) * 100)),
                           "net_benefit_replay": float(value)})
        for t in predecessor_grid:
            replay.append({"cohort": cohort, "model": "treat_none",
                           "threshold_key": int(round(float(t) * 100)),
                           "net_benefit_replay": 0.0})
    replayed = old.merge(pd.DataFrame(replay), on=keys, how="inner")
    if len(replayed) != len(old):
        raise DecisionCurveError("Grid replay did not cover the predecessor table")
    replay_difference = (replayed["net_benefit_replay"].to_numpy(float)
                         - replayed["net_benefit"].to_numpy(float))
    stream_only = replayed[np.abs(replay_difference) > NOISE_ATOL]
    stream_cells = [{"cohort": r.cohort, "model": r.model,
                     "threshold": float(r.threshold_key) / 100,
                     "predecessor_net_benefit": float(r.net_benefit),
                     "locked_stream_net_benefit": float(r.net_benefit_replay),
                     "locked_minus_predecessor": float(r.net_benefit_replay - r.net_benefit)}
                    for r in stream_only.sort_values(keys).itertuples()]
    if {(c["cohort"], c["model"]) for c in stream_cells} - {("d1_test", "blend_triple")}:
        raise DecisionCurveError(
            "On the predecessor's own grid the locked streams disagree outside the "
            f"D1 tri-blend curve: {stream_cells}")

    # ---- effect 2: canonical grid vs the predecessor's dusty grid --------------
    ties = {c: {m: int((cohort_data[c][1][m] == 0.5).sum())
                for m in ("blend_triple", "blend_quad", "paper_svm")}
            for c in cohort_data}
    difference = (merged["net_benefit_new"].to_numpy(float)
                  - merged["net_benefit_old"].to_numpy(float))
    substantive = merged[np.abs(difference) > NOISE_ATOL].copy()
    stream_keys = {(c["cohort"], c["model"], int(round(c["threshold"] * 100)))
                   for c in stream_cells}
    changed = []
    for r in substantive.sort_values(keys).itertuples():
        key = (r.cohort, r.model, int(r.threshold_key))
        cause = ("reconstructed D1 component stream in the predecessor"
                 if key in stream_keys else
                 "predecessor grid point 0.5000000000000001 excluded compounds whose "
                 "probability is exactly 0.5; the canonical grid point 0.5 includes them")
        changed.append({"cohort": r.cohort, "model": r.model,
                        "threshold": float(r.threshold_key) / 100,
                        "predecessor_net_benefit": float(r.net_benefit_old),
                        "locked_stream_net_benefit": float(r.net_benefit_new),
                        "locked_minus_predecessor": float(r.net_benefit_new - r.net_benefit_old),
                        "cause": cause})

    # Everything the manuscript reports at threshold 0.5 for the two blends and the
    # reference policies must survive untouched; only tied-at-0.5 SVM cells may move.
    half = merged[merged["threshold_key"] == 50]
    if len(half) != len(COHORTS) * len(MODELS):
        raise DecisionCurveError(
            f"Threshold 0.5 covers {len(half)} cells, expected {len(COHORTS) * len(MODELS)}")
    protected = half[half["model"] != "paper_svm"]
    protected_max = float(np.abs(protected["net_benefit_new"].to_numpy(float)
                                 - protected["net_benefit_old"].to_numpy(float)).max())
    if protected_max > NOISE_ATOL:
        raise DecisionCurveError(
            f"Threshold 0.5 changed by {protected_max:.3e} for a blend or reference "
            f"policy; only the published SVM may move, and only through exact ties")
    for cell in changed:
        if cell["threshold"] == 0.5:
            if cell["model"] != "paper_svm":
                raise DecisionCurveError(f"Unexpected threshold-0.5 change: {cell}")
            if ties[cell["cohort"]]["paper_svm"] == 0:
                raise DecisionCurveError(
                    f"{cell['cohort']}: SVM net benefit at 0.5 changed but no compound "
                    f"has probability exactly 0.5")
    half_max = float(np.abs(half["net_benefit_new"].to_numpy(float)
                            - half["net_benefit_old"].to_numpy(float)).max())

    for frame in (curves, old):
        frame.drop(columns="threshold_key", inplace=True)
    return {
        "predecessor_run": PREDECESSOR_RUN,
        "predecessor_table": PREDECESSOR_TABLE,
        "predecessor_table_sha256": sha256_file(path),
        "predecessor_left_byte_unchanged": True,
        "cells_compared": int(len(merged)),
        "join_keys": ["cohort", "model", "threshold rounded to the 0.01 grid"],
        "noise_tolerance": NOISE_ATOL,
        "max_abs_difference": float(np.abs(difference).max()),
        "cells_exceeding_noise_tolerance": int(len(substantive)),
        "substantive_changes": changed,
        "stream_effect": {
            "description": "locked streams replayed on the predecessor's own "
                           "np.arange grid, isolating the reconstruction artifact",
            "cells_changed": len(stream_cells),
            "changes": stream_cells,
            "all_other_cells_reproduced_within_tolerance": True,
            "confined_to": "d1_test/blend_triple",
        },
        "grid_effect": {
            "description": "the predecessor built thresholds with np.arange, whose "
                           "46th point is 0.5000000000000001 rather than 0.5; this run "
                           "uses the canonical [i / 100 for i in range(5, 96)] grid",
            "predecessor_threshold_at_index_45": float(predecessor_grid[45]),
            "canonical_threshold_at_index_45": 0.5,
            "compounds_with_probability_exactly_0p5": ties,
            "cells_changed": len([c for c in changed if c["threshold"] == 0.5]),
            "affects_only": "paper_svm",
            "rationale": "every other analysis in this repository binarizes with "
                         "p >= 0.5 and the Methods specify increments of 0.01 from "
                         "0.05, so the canonical grid is the one the manuscript "
                         "describes; the predecessor silently excluded tied compounds "
                         "at the reported operating point",
        },
        "threshold_0p5_max_abs_difference": half_max,
        "threshold_0p5_blend_and_reference_max_abs_difference": protected_max,
        "threshold_0p5_blend_and_reference_unchanged_within_tolerance": True,
        "manuscript_numbers_requiring_update": [
            {"cohort": c["cohort"], "model": c["model"],
             "previously_reported": round(c["predecessor_net_benefit"], 3),
             "corrected": round(c["locked_stream_net_benefit"], 3)}
            for c in changed if c["threshold"] == 0.5],
    }


def statistical_contract() -> dict:
    """The full statistical contract, hashed canonically so it cannot drift silently."""
    return {
        "statistic": "net benefit = TP/n - (FP/n) * pt/(1-pt)",
        "decision_rule": "prediction = probability >= threshold",
        "threshold_grid": {
            "construction": "[i / 100 for i in range(5, 96)]",
            "first": 0.05, "last": 0.95, "step": 0.01, "count": 91},
        "models": list(MODELS),
        "model_definitions": {
            "blend_triple": "equal-weight mean of " + ", ".join(TRIPLE_COLUMNS),
            "blend_quad": "equal-weight mean of " + ", ".join(QUAD_COLUMNS),
            "paper_svm": "probability_paper_svm read directly",
            "treat_all": "prevalence - (1 - prevalence) * pt / (1 - pt)",
            "treat_none": "identically zero"},
        "cohorts": {c: {"table": COHORTS[c][0], "n": COHORTS[c][1],
                        "positive": COHORTS[c][2], "negative": COHORTS[c][3]}
                    for c in COHORTS},
        "identity_columns": dict(IDENTITY),
        "row_order": "ascending identity column within each cohort",
        "expected_rows": len(COHORTS) * len(MODELS) * 91,
        "probability_source": "locked per-compound component-probability streams sealed "
                              f"in {SOURCE_RUN}",
        "any_model_is_refitted": False,
        "any_model_is_recalibrated": False,
        "threshold_is_tuned": False,
        "held_out_outcome_used_for_selection": False,
    }


def run(*, root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"locked_decision_curves_v2_[a-z0-9_.-]+", run_id):
        raise DecisionCurveError("RUN_ID must start with locked_decision_curves_v2_")
    root = root.resolve()
    outputs = root / "outputs"
    destination = outputs / run_id
    if destination.exists() or destination.is_symlink():
        raise DecisionCurveError(f"Run directory already exists: {destination}")

    upstream = verify_upstream(outputs)
    print(f"  upstream {SOURCE_RUN} verified: "
          f"{len(upstream['verified_input_hashes'])} input hashes match", flush=True)

    thresholds = threshold_grid()
    rows, source_audit, cohort_data = [], {}, {}
    for cohort in COHORTS:
        y, scores, audit = load_cohort(outputs, cohort)
        source_audit[cohort] = audit
        cohort_data[cohort] = (y, scores)
        for model in ("blend_triple", "blend_quad", "paper_svm"):
            for t, value in zip(thresholds, net_benefit(y, scores[model], thresholds)):
                rows.append({"cohort": cohort, "model": model,
                             "threshold": float(t), "net_benefit": float(value)})
        for t, value in zip(thresholds, treat_all(y, thresholds)):
            rows.append({"cohort": cohort, "model": "treat_all",
                         "threshold": float(t), "net_benefit": float(value)})
        for t in thresholds:
            rows.append({"cohort": cohort, "model": "treat_none",
                         "threshold": float(t), "net_benefit": 0.0})
        print(f"  {cohort}: n={audit['rows']} ({audit['positive']} positive, "
              f"{audit['negative']} negative), {len(MODELS)} policies x "
              f"{len(thresholds)} thresholds", flush=True)

    curves = pd.DataFrame(rows)
    expected = len(COHORTS) * len(MODELS) * len(thresholds)
    if len(curves) != expected:
        raise DecisionCurveError(f"Curve table has {len(curves)} rows, expected {expected}")
    if curves.duplicated(["cohort", "model", "threshold"]).any():
        raise DecisionCurveError("Curve table contains duplicated cohort/model/threshold keys")
    if not np.isfinite(curves["net_benefit"].to_numpy(float)).all():
        raise DecisionCurveError("Curve table contains non-finite net benefit")
    if sorted(curves["model"].unique()) != sorted(MODELS):
        raise DecisionCurveError(f"Unexpected policies: {sorted(curves['model'].unique())}")

    parity = compare_with_predecessor(outputs, curves, cohort_data)
    print(f"  predecessor comparison: {parity['cells_compared']} cells, "
          f"{parity['cells_exceeding_noise_tolerance']} substantive change(s), "
          f"max |diff| = {parity['max_abs_difference']:.3e}, "
          f"threshold 0.5 max |diff| = {parity['threshold_0p5_max_abs_difference']:.3e}",
          flush=True)
    for change in parity["substantive_changes"]:
        print(f"    {change['cohort']}/{change['model']} @ {change['threshold']:.2f}: "
              f"{change['predecessor_net_benefit']:.16f} -> "
              f"{change['locked_stream_net_benefit']:.16f}  [{change['cause'][:38]}]",
              flush=True)

    contract = statistical_contract()
    code_hashes = {name: sha256_file(Path(path)) for name, path in CODE_MODULES.items()}

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".lockeddca.work-", dir=destination.parent))
    try:
        curve_path = tmp / "E_decision_curves.csv"
        if curve_path.exists():
            raise FileExistsError(f"Refusing to overwrite an immutable artifact: {curve_path}")
        curves.to_csv(curve_path, index=False, lineterminator="\n")
        pd.DataFrame(parity["substantive_changes"]).to_csv(
            tmp / "predecessor_substantive_changes.csv", index=False, lineterminator="\n")
        atomic_write_json(tmp / "RUN_MANIFEST.json", {
            "schema_version": f"{SCHEMA}.run_manifest.v1", "run_id": run_id,
            "purpose": "decision curves recomputed from the locked per-compound "
                       "component-probability streams only",
            "supersedes": f"{PREDECESSOR_RUN}/{PREDECESSOR_TABLE}",
            "any_model_is_trained_refitted_calibrated_or_selected": False,
            "derived_from_sealed_streams_only": True,
            "upstream_verification": upstream,
            "sources": source_audit,
            "code_provenance": {
                "modules": code_hashes,
                "execution_command": " ".join([sys.executable, "-m",
                                               "geroprotector.locked_decision_curves_v2",
                                               "--root", str(root), "--run-id", run_id]),
                "python_executable": sys.executable},
            "statistical_contract": contract,
            "statistical_contract_sha256": canonical_sha256(contract),
            "predecessor_comparison": parity,
            "row_counts": {"curve_rows": int(len(curves)),
                           **{c: source_audit[c]["rows"] for c in source_audit}},
            "class_counts": {c: {"positive": source_audit[c]["positive"],
                                 "negative": source_audit[c]["negative"]}
                             for c in source_audit},
            "identity_columns": dict(IDENTITY),
            "held_out_labels_used_for_selection": False,
            "threshold_or_model_selected_by_outcome": False,
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
