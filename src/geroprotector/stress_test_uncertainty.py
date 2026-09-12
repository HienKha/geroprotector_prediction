"""Bootstrap uncertainty for the locked D1-model stress-test score streams.

This analysis does not fit or recalibrate a model. It verifies the sealed prediction
artifacts from quad_blend_20260822, reconstructs the two focal equal-weight scores, and
resamples compounds to obtain descriptive percentile intervals for AP, AP lift over
prevalence, and AUROC.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(os.environ.get("GERO_PROJECT_ROOT", Path.cwd())).resolve()
UPSTREAM = ROOT / "outputs" / "quad_blend_20260822"
OUTPUT = ROOT / "outputs" / "stress_test_uncertainty_20260910"
SEED = 20260910
N_BOOTSTRAP = 10_000

COHORTS = {
    "drugage": ("DrugAge stress test", "predictions_drugage.csv"),
    "agextend": ("AgeXtend stress test", "predictions_agextend.csv"),
}
MODELS = {
    "Source-SVM reproduction": lambda d: d["probability_paper_svm"].to_numpy(float),
    "SVM_PFN_FM": lambda d: (
        d["probability_paper_svm"].to_numpy(float)
        + d["probability_tabpfn_v2"].to_numpy(float)
        + d["probability_tabfm"].to_numpy(float)
    ) / 3.0,
    "SVM_PFN_FM_Tani": lambda d: d["blend_eq_quarters"].to_numpy(float),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_upstream() -> dict[str, str]:
    completed_path = UPSTREAM / "COMPLETED.json"
    completed = json.loads(completed_path.read_text())
    if completed.get("status") != "COMPLETE":
        raise RuntimeError(f"Upstream status is {completed.get('status')!r}")
    declared = completed.get("artifact_hashes") or {}
    required = ["RUN_MANIFEST.json"] + [artifact for _, artifact in COHORTS.values()]
    observed: dict[str, str] = {"COMPLETED.json": sha256(completed_path)}
    for name in required:
        if name not in declared:
            raise RuntimeError(f"Upstream completion record does not declare {name}")
        path = UPSTREAM / name
        value = sha256(path)
        if value != declared[name]:
            raise RuntimeError(f"Upstream hash mismatch for {name}")
        observed[name] = value
    return observed


def metric_values(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    prevalence = float(y.mean())
    ap = float(average_precision_score(y, score))
    return {
        "AP": ap,
        "AP lift": ap - prevalence,
        "AUROC": float(roc_auc_score(y, score)),
    }


def bootstrap_cohort(frame: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    y = frame["label"].to_numpy(int)
    if set(np.unique(y)) != {0, 1}:
        raise RuntimeError("Each stress-test cohort must contain both classes")
    n = len(y)
    indices = rng.integers(0, n, size=(N_BOOTSTRAP, n), endpoint=False)
    score_by_model = {name: getter(frame) for name, getter in MODELS.items()}
    draws = {
        model: {"AP": [], "AP lift": [], "AUROC": []}
        for model in MODELS
    }
    for sample in indices:
        yb = y[sample]
        prevalence = float(yb.mean())
        both_classes = len(np.unique(yb)) == 2
        for model, score in score_by_model.items():
            sb = score[sample]
            ap = float(average_precision_score(yb, sb))
            draws[model]["AP"].append(ap)
            draws[model]["AP lift"].append(ap - prevalence)
            if both_classes:
                draws[model]["AUROC"].append(float(roc_auc_score(yb, sb)))

    rows = []
    for model, score in score_by_model.items():
        point = metric_values(y, score)
        for metric in ("AP", "AP lift", "AUROC"):
            values = np.asarray(draws[model][metric], dtype=float)
            if len(values) == 0:
                raise RuntimeError(f"No valid bootstrap values for {model}/{metric}")
            lo, hi = np.quantile(values, [0.025, 0.975])
            rows.append({
                "model": model,
                "metric": metric,
                "n": n,
                "n_positive": int(y.sum()),
                "n_negative": int(n - y.sum()),
                "positive_prevalence": float(y.mean()),
                "value": point[metric],
                "ci_low": float(lo),
                "ci_high": float(hi),
                "bootstrap_resamples_requested": N_BOOTSTRAP,
                "bootstrap_resamples_valid": int(len(values)),
                "bootstrap_resamples_discarded": int(N_BOOTSTRAP - len(values)),
                "bootstrap_unit": "compound",
                "confidence_interval": "2.5th and 97.5th percentiles",
                "seed": SEED,
            })
    return rows


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing output: {OUTPUT}")
    upstream_hashes = verify_upstream()
    rng = np.random.default_rng(SEED)
    rows = []
    cohort_audit = {}
    for cohort, (display, artifact) in COHORTS.items():
        frame = pd.read_csv(UPSTREAM / artifact)
        required = {
            "label", "probability_paper_svm", "probability_tabpfn_v2",
            "probability_tabfm", "blend_eq_quarters",
        }
        if not required.issubset(frame.columns):
            raise RuntimeError(f"{artifact} lacks {sorted(required - set(frame.columns))}")
        if frame[list(required)].isna().any().any():
            raise RuntimeError(f"{artifact} contains missing labels or scores")
        cohort_rows = bootstrap_cohort(frame, rng)
        for row in cohort_rows:
            row["cohort"] = cohort
            row["cohort_display"] = display
            row["source_artifact"] = artifact
        rows.extend(cohort_rows)
        cohort_audit[cohort] = {
            "rows": int(len(frame)),
            "positive": int(frame["label"].sum()),
            "negative": int(len(frame) - frame["label"].sum()),
        }

    result = pd.DataFrame(rows)
    expected_rows = len(COHORTS) * len(MODELS) * 3
    if len(result) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, obtained {len(result)}")
    if not (
        (result["ci_low"] <= result["value"])
        & (result["value"] <= result["ci_high"])
    ).all():
        raise RuntimeError("A point estimate lies outside its percentile interval")

    producer = Path(__file__).resolve()
    contract = {
        "analysis": "descriptive uncertainty for locked-model stress tests",
        "model_refitting": False,
        "threshold_used": False,
        "bootstrap_unit": "compound",
        "bootstrap_resamples": N_BOOTSTRAP,
        "seed": SEED,
        "confidence_interval": "percentile, 2.5th to 97.5th",
        "auroc_single_class_resamples": "discarded",
        "ap_lift": "AP minus resampled positive prevalence within each bootstrap draw",
        "multiplicity_adjustment": "none; descriptive intervals",
    }
    manifest = {
        "schema_version": "geroprotector.stress_test_uncertainty.v1",
        "run_id": OUTPUT.name,
        "command": " ".join(sys.argv),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "producer": str(producer),
        "producer_sha256": sha256(producer),
        "upstream_run": str(UPSTREAM),
        "upstream_hashes_verified_before_read": upstream_hashes,
        "cohorts": cohort_audit,
        "models": list(MODELS),
        "statistical_contract": contract,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent) as tmp:
        stage = Path(tmp)
        result.to_csv(stage / "stress_test_bootstrap_intervals.csv", index=False)
        (stage / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        completed = {
            "schema_version": "geroprotector.stress_test_uncertainty.completed.v1",
            "run_id": OUTPUT.name,
            "status": "COMPLETE",
            "artifact_hashes": {
                name: sha256(stage / name)
                for name in ("stress_test_bootstrap_intervals.csv", "RUN_MANIFEST.json")
            },
        }
        (stage / "COMPLETED.json").write_text(json.dumps(completed, indent=2) + "\n")
        os.rename(stage, OUTPUT)

    print(f"Complete: {OUTPUT}")
    print(result[["cohort", "model", "metric", "value", "ci_low", "ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
