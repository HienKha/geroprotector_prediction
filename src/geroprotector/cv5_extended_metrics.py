"""F1 and other metrics BEYOND the source paper's set, for the 5-fold D1-train CV.

Companion to `cv5_paper_metrics.py`, kept separate on purpose.  That module is
contractually restricted to the four metrics the source paper actually reports
(accuracy, confusion matrix, specificity, Cohen's kappa).  F1 is NOT one of them:
the paper's official notebook computes no F1, no precision/recall as named
metrics, and no ROC/AUC.

So F1 lives here instead, clearly labelled as an ADDED metric, so that a reader
comparing against the published paper is never misled into thinking there is a
published F1 to compare against.

Nothing is refitted and no model is re-run.  This module reads the sealed
per-row cross-validation predictions written by `cv5_paper_metrics.py` -- same
324 D1-training rows, same StratifiedKFold(5, shuffle=True, random_state=42)
folds, same decisions -- and derives F1 from them.  The 81-row held-out test set
and both external cohorts are untouched.

Three F1 variants are reported because "F1 score" is ambiguous:
  * f1_positive : binary F1 for the positive (geroprotector) class -- what a
                  binary-classification paper usually means;
  * f1_macro    : unweighted mean over both classes -- what the rest of this
                  project's comparison tables use;
  * f1_weighted : support-weighted mean over both classes.
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
from sklearn.metrics import f1_score

from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file


class CV5ExtendedMetricsError(RuntimeError):
    """Raised when a sealed input or a protocol invariant fails."""


SCHEMA = "geroprotector.cv5_extended_metrics"
VARIANTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("f1_positive", {"pos_label": 1, "average": "binary"}),
    ("f1_macro", {"average": "macro"}),
    ("f1_weighted", {"average": "weighted"}),
)


def _regular_file(path: Path, role: str, expected: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise CV5ExtendedMetricsError(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected is not None and sha256_file(resolved) != expected:
        raise CV5ExtendedMetricsError(f"{role} SHA256 differs from the protocol lock")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    text = _regular_file(path, "CV5 extended protocol").read_text(encoding="utf-8")
    protocol = yaml.safe_load(text)
    expected_schema = f"{SCHEMA}.protocol.v1"
    if not isinstance(protocol, dict) or protocol.get("schema_version") != expected_schema:
        raise CV5ExtendedMetricsError("Unknown CV5 extended protocol schema")
    contract = protocol.get("contract", {})
    if contract.get("metrics_are_beyond_the_source_paper_set") is not True:
        raise CV5ExtendedMetricsError("Contract must declare these metrics as paper-external")
    for key in ("nothing_is_refitted", "train_rows_only"):
        if contract.get(key) is not True:
            raise CV5ExtendedMetricsError(f"Contract differs at {key}")
    for key in ("test_rows_used", "external_rows_used"):
        if contract.get(key) is not False:
            raise CV5ExtendedMetricsError(f"Contract differs at {key}")
    immutability = protocol.get("immutability", {})
    if immutability.get("existing_run_directories_are_read_only") is not True:
        raise CV5ExtendedMetricsError("Immutability contract differs")
    for record in protocol.get("sealed_inputs", {}).values():
        _regular_file(root / record["path"], f"sealed input {record['path']}", record["sha256"])
    return protocol, canonical_sha256(protocol)


def run(*, root: Path, config_path: Path, run_id: str) -> Path:
    if not re.fullmatch(r"cv5_extended_metrics_[a-z0-9_.-]+", run_id):
        raise CV5ExtendedMetricsError("RUN_ID must start with cv5_extended_metrics_")
    root = root.resolve()
    protocol, protocol_sha256 = load_protocol(root, config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise CV5ExtendedMetricsError(f"Run directory already exists: {destination}")
    sealed = protocol["sealed_inputs"]

    predictions = pd.read_csv(root / sealed["cv5_per_row_predictions"]["path"])
    expected_rows = int(protocol["expected"]["n_rows"])
    expected_folds = int(protocol["expected"]["n_folds"])
    if len(predictions) != expected_rows:
        raise CV5ExtendedMetricsError(f"Expected {expected_rows} CV rows")
    if sorted(predictions.fold.unique().tolist()) != list(range(expected_folds)):
        raise CV5ExtendedMetricsError(f"Expected folds 0..{expected_folds - 1}")

    y = predictions.label.to_numpy(dtype=int)
    fold = predictions.fold.to_numpy(dtype=int)
    # Identical decision rules to cv5_paper_metrics.py, rederived from the same columns.
    rules: list[tuple[str, str, np.ndarray]] = [
        ("paper_svm", "fixed_0p5",
         (predictions.probability_paper_svm.to_numpy(float) >= 0.5).astype(int)),
        ("paper_svm", "paper_svm_native",
         predictions.paper_svm_native_decision.to_numpy(int)),
        ("blend_eq_thirds_tabpfnv2", "fixed_0p5",
         (predictions.blend_eq_thirds_tabpfnv2.to_numpy(float) >= 0.5).astype(int)),
        ("blend_eq_thirds_tabfm", "fixed_0p5",
         (predictions.blend_eq_thirds_tabfm.to_numpy(float) >= 0.5).astype(int)),
    ]

    per_fold_rows, pooled_rows = [], []
    for model, rule, decision in rules:
        for index in range(expected_folds):
            mask = fold == index
            row = {"model": model, "threshold_rule": rule, "fold": index, "n": int(mask.sum())}
            for name, kwargs in VARIANTS:
                row[name] = float(f1_score(y[mask], decision[mask], zero_division=0, **kwargs))
            per_fold_rows.append(row)
        row = {"model": model, "threshold_rule": rule,
               "aggregation": f"pooled_oof_{expected_rows}_rows", "n": len(y)}
        for name, kwargs in VARIANTS:
            row[name] = float(f1_score(y, decision, zero_division=0, **kwargs))
        pooled_rows.append(row)
    per_fold = pd.DataFrame(per_fold_rows)
    pooled = pd.DataFrame(pooled_rows)

    summary_rows = []
    for (model, rule), group in per_fold.groupby(["model", "threshold_rule"], sort=False):
        row = {"model": model, "threshold_rule": rule,
               "aggregation": f"mean_sd_across_{expected_folds}_folds"}
        for name, _kwargs in VARIANTS:
            row[f"{name}_mean"] = float(group[name].mean())
            row[f"{name}_sd"] = float(group[name].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    # Paired per-fold deltas vs the paper SVM at the same common 0.5 rule.
    baseline = per_fold[
        (per_fold.model == "paper_svm") & (per_fold.threshold_rule == "fixed_0p5")
    ].sort_values("fold")
    delta_rows = []
    for model in ("blend_eq_thirds_tabpfnv2", "blend_eq_thirds_tabfm"):
        challenger = per_fold[
            (per_fold.model == model) & (per_fold.threshold_rule == "fixed_0p5")
        ].sort_values("fold")
        for name, _kwargs in VARIANTS:
            difference = challenger[name].to_numpy() - baseline[name].to_numpy()
            delta_rows.append({
                "model": model, "versus": "paper_svm", "threshold_rule": "fixed_0p5",
                "metric": name,
                "mean_delta": float(difference.mean()),
                "sd_delta": float(difference.std(ddof=1)),
                "folds_won": int((difference > 0).sum()),
                "n_folds": len(difference),
                "per_fold_delta": json.dumps([round(float(v), 6) for v in difference]),
            })
    deltas = pd.DataFrame(delta_rows)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".cv5ext.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "cv5_f1_per_fold.csv", per_fold)
        _write_csv(tmp / "cv5_f1_mean_sd_across_folds.csv", summary)
        _write_csv(tmp / "cv5_f1_pooled.csv", pooled)
        _write_csv(tmp / "cv5_f1_paired_deltas_vs_paper_svm.csv", deltas)

        lines = [
            "# F1 for the 5-fold D1-train cross-validation",
            "",
            "**F1 is NOT one of the metrics the source paper reports.** Its official notebook",
            "computes accuracy, confusion matrix, specificity and Cohen's kappa only. Present",
            "F1 as an added metric; there is no published F1 to compare against.",
            "",
            "Derived from the sealed per-row CV predictions of `cv5_paper_metrics_20260821`:",
            "same 324 D1-training rows, same StratifiedKFold(5, shuffle=True, random_state=42)",
            "folds, same decisions. Nothing was refitted. Test and external cohorts untouched.",
            "",
            "## Mean +/- SD across the 5 folds",
            "",
            "| Model | Rule | F1 (positive) | F1 macro | F1 weighted |",
            "|---|---|---|---|---|",
        ]
        for row in summary.itertuples(index=False):
            lines.append(
                f"| {row.model} | {row.threshold_rule} | "
                f"{row.f1_positive_mean:.4f} ± {row.f1_positive_sd:.4f} | "
                f"{row.f1_macro_mean:.4f} ± {row.f1_macro_sd:.4f} | "
                f"{row.f1_weighted_mean:.4f} ± {row.f1_weighted_sd:.4f} |"
            )
        lines += ["", "## Pooled over all 324 OOF rows", "",
                  "| Model | Rule | F1 (positive) | F1 macro | F1 weighted |",
                  "|---|---|---|---|---|"]
        for row in pooled.itertuples(index=False):
            lines.append(
                f"| {row.model} | {row.threshold_rule} | {row.f1_positive:.4f} | "
                f"{row.f1_macro:.4f} | {row.f1_weighted:.4f} |"
            )
        lines += ["", "## Paired per-fold deltas vs the paper SVM (both at fixed 0.5)", "",
                  "| Model | Metric | Mean delta | Folds won |", "|---|---|---|---|"]
        for row in deltas.itertuples(index=False):
            lines.append(
                f"| {row.model} | {row.metric} | {row.mean_delta:+.4f} | "
                f"{row.folds_won}/{row.n_folds} |"
            )
        lines.append("")
        (tmp / "summary.md").write_text("\n".join(lines), encoding="utf-8")

        manifest = {
            "schema_version": f"{SCHEMA}.run_manifest.v1",
            "run_id": run_id,
            "protocol_sha256": protocol_sha256,
            "companion_run": str(sealed["cv5_per_row_predictions"]["path"]),
            "metrics_reported": [name for name, _ in VARIANTS],
            "metrics_are_beyond_the_source_paper_set": True,
            "source_paper_reports": ["accuracy", "confusion_matrix", "specificity",
                                     "cohen_kappa"],
            "source_paper_reports_f1": False,
            "nothing_is_refitted": True,
            "train_rows_only": True,
            "test_rows_used": False,
            "external_rows_used": False,
            "n_rows": len(y),
            "n_folds": expected_folds,
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
