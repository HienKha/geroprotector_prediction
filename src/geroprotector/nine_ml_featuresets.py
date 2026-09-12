"""The nine ML algorithms on two feature sets: the paper's seven, and all descriptors.

The sealed `traditional405_20260817` run evaluated all nine algorithms on the
paper's SEVEN DataWarrior descriptors only.  This module reproduces that arm and
adds the missing one -- the same nine algorithms on the full RDKit2D descriptor
panel -- so the two feature sets are directly comparable, produced by identical
code, identical split, and identical hyperparameters.

  --feature-set paper7  -> the 7 published DataWarrior descriptors, used verbatim
  --feature-set all     -> the 217 RDKit2D descriptors computed from SMILES,
                           reduced by a TRAIN-ONLY median imputation and
                           zero-variance filter

LEAKAGE.  The imputation medians and the variance mask are fitted on the 324
training rows alone and then applied unchanged to the 81 test rows; the scalers
inside the KNN and logistic-regression pipelines are likewise fitted on training
folds only.  Nothing about the test rows -- values, labels or distribution --
enters any fit.  The threshold is fixed at 0.5 for every algorithm, so no
operating point is tuned either.

The algorithms, their hyperparameters, the 405 source rows and the 80/20 split
are all read from configs/traditional_paper405.yaml, the same contract the
sealed run used, so the paper7 arm here must reproduce the sealed numbers
exactly.  That reproduction is asserted, not assumed: if the paper7 arm does not
match the sealed metrics to 1e-12, the run fails.
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

from geroprotector.fixed_blend_paper405 import _features
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.screening_blend_altmodels import _apply_context, _imputer_context
from geroprotector.traditional_paper405 import (
    _load_protocol,
    _models,
    _positive_score,
    _read_sources,
    paper_split_indices,
)


class NineMLError(RuntimeError):
    """Raised when a contract, a parity proof or a leakage guard fails."""


SCHEMA = "geroprotector.nine_ml_featuresets"
FIXED_THRESHOLD = 0.5
MODEL_ORDER = (
    "svm_original_paper", "logistic_regression", "linear_regression", "knn",
    "random_forest", "extra_trees", "xgboost", "lightgbm", "catboost",
)
SEALED_PAPER7_METRICS = Path("outputs/traditional405_20260817/metrics.csv")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def _metrics(y: np.ndarray, score: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, int)
    decision = (probability >= FIXED_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, decision, labels=[0, 1]).ravel()
    both = len(set(y)) == 2
    return {
        "n_test": len(y), "positive_prevalence": float(y.mean()),
        "auprc_average_precision_positive": float(average_precision_score(y, score)),
        "auroc": float(roc_auc_score(y, score)) if both else float("nan"),
        "accuracy": float(accuracy_score(y, decision)),
        "balanced_accuracy": float(balanced_accuracy_score(y, decision)),
        "mcc": float(matthews_corrcoef(y, decision)),
        "macro_f1": float(f1_score(y, decision, average="macro", zero_division=0)),
        "f1_positive": float(f1_score(y, decision, pos_label=1, zero_division=0)),
        "precision_positive": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "recall_sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "npv": float(tn / (tn + fn)) if (tn + fn) else float("nan"),
        "cohen_kappa": float(cohen_kappa_score(y, decision)),
        "brier": float(brier_score_loss(y, np.clip(probability, 0.0, 1.0))),
        "log_loss": float(log_loss(y, np.clip(probability, 1e-9, 1 - 1e-9))),
        "threshold": FIXED_THRESHOLD,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def run(
    *, root: Path, config_path: Path, positive_path: Path, negative_path: Path,
    feature_set: str, run_id: str,
) -> Path:
    if feature_set not in {"paper7", "all"}:
        raise NineMLError("feature-set must be 'paper7' or 'all'")
    if not re.fullmatch(rf"nineml_{feature_set}_[a-z0-9_.-]+", run_id):
        raise NineMLError(f"RUN_ID must start with nineml_{feature_set}_")
    root = root.resolve()
    protocol, protocol_sha256 = _load_protocol(config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise NineMLError(f"Run directory already exists: {destination}")

    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    frame, audit = _read_sources(positive_path.resolve(), negative_path.resolve(), traditional)
    train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    labels = frame["label"].to_numpy(dtype=int)
    y_train, y_test = labels[train_indices], labels[test_indices]

    smiles, _ = _validated_raw_smiles(frame)
    features = _features(frame, smiles, fixed_protocol)

    if feature_set == "paper7":
        panel = features["paper"].astype(float)
        names = list(protocol["features"])
        context = None
    else:
        raw = features["rdkit2d"]
        # imputation medians and the variance mask come from TRAIN ROWS ONLY
        context = _imputer_context(raw[train_indices])
        panel = _apply_context(context, raw).astype(float)
        keep = np.flatnonzero(np.asarray(context["finite_any_mask"], dtype=bool))
        keep = keep[np.asarray(context["varying_after_imputation_mask"], dtype=bool)]
        # the same ordered RDKit descriptor list _features() builds the panel from
        from rdkit.Chem import Descriptors

        all_names = [name for name, _function in Descriptors._descList]
        if len(all_names) != raw.shape[1]:
            raise NineMLError("RDKit descriptor name list does not match the panel width")
        names = [all_names[i] for i in keep]
    x_train, x_test = panel[train_indices], panel[test_indices]
    print(f"[{feature_set}] panel {panel.shape} -> train {x_train.shape}, "
          f"test {x_test.shape}", flush=True)

    models = _models(protocol)
    missing = set(MODEL_ORDER) - set(models)
    if missing:
        raise NineMLError(f"Model factory is missing: {sorted(missing)}")

    metric_rows, prediction_rows = [], []
    for model_id in MODEL_ORDER:
        model = models[model_id]
        model.fit(x_train, y_train)
        score, probability = _positive_score(model, x_test, model_id=model_id)
        metric_rows.append({"model_id": model_id, "feature_set": feature_set,
                            "n_features": int(x_train.shape[1]), **_metrics(
                                y_test, score, probability)})
        for position, row in enumerate(test_indices):
            prediction_rows.append({
                "model_id": model_id, "feature_set": feature_set,
                "paper_row_index": int(row),
                "compound_name": str(frame.set_index("paper_row_index")
                                     .loc[row, "compound_name"]),
                "label": int(y_test[position]),
                "ranking_score": float(score[position]),
                "probability": float(probability[position]),
                "decision": int(probability[position] >= FIXED_THRESHOLD),
                "threshold": FIXED_THRESHOLD,
            })
        print(f"  {model_id:22s} AP+ {metric_rows[-1]['auprc_average_precision_positive']:.4f}"
              f"  MCC {metric_rows[-1]['mcc']:.4f}"
              f"  Acc {metric_rows[-1]['accuracy']:.4f}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)

    # ---- parity: the paper7 arm must reproduce the sealed run exactly ---------
    parity: dict[str, Any] = {"checked": False}
    sealed_path = root / SEALED_PAPER7_METRICS
    if feature_set == "paper7" and sealed_path.is_file():
        sealed = pd.read_csv(sealed_path).set_index("model_id")
        columns = [c for c in ("auroc", "accuracy", "mcc", "macro_f1", "brier",
                               "auprc_average_precision_positive")
                   if c in sealed.columns]
        drift = {}
        for model_id in MODEL_ORDER:
            if model_id not in sealed.index:
                continue
            mine = metrics.set_index("model_id").loc[model_id]
            drift[model_id] = float(max(
                abs(float(mine[c]) - float(sealed.loc[model_id, c])) for c in columns))
        worst = max(drift.values()) if drift else 0.0
        parity = {"checked": True, "sealed_run": str(SEALED_PAPER7_METRICS),
                  "max_abs_metric_difference": worst, "per_model": drift,
                  "tolerance": 1e-12}
        if worst > 1e-12:
            raise NineMLError(
                f"paper7 arm does not reproduce the sealed traditional405 metrics "
                f"(max drift {worst!r}); refusing to publish a divergent rerun"
            )
        print(f"parity with sealed traditional405: max drift {worst:.2e}", flush=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".nineml.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "metrics.csv", metrics)
        _write_csv(tmp / "predictions.csv", predictions)
        _write_csv(tmp / "feature_list.csv", pd.DataFrame(
            {"index": range(len(names)), "feature": names}))
        atomic_write_json(tmp / "RUN_MANIFEST.json", {
            "schema_version": f"{SCHEMA}.run_manifest.v1",
            "run_id": run_id, "feature_set": feature_set,
            "n_features": int(x_train.shape[1]),
            "feature_source": ("paper 7 DataWarrior descriptors, verbatim"
                               if feature_set == "paper7"
                               else "RDKit2D descriptors from SMILES, train-only median "
                                    "imputation and zero-variance filter"),
            "protocol_sha256": protocol_sha256,
            "paper_split_sha256": split_sha256,
            "train_rows": len(train_indices), "test_rows": len(test_indices),
            "models": list(MODEL_ORDER),
            "model_settings": protocol["models"],
            "fixed_threshold": FIXED_THRESHOLD,
            "threshold_is_tuned": False,
            "test_rows_used_in_any_fit_or_transform": False,
            "imputation_and_variance_mask_fitted_on": (
                "train rows only" if feature_set == "all" else "not applicable"),
            "parity_with_sealed_traditional405": parity,
            "data_audit": audit,
            "runtime": {"python": platform.python_version(),
                        "platform": platform.platform()},
            "existing_runs_modified": False,
        })
        atomic_write_json(tmp / "COMPLETED.json", {
            "schema_version": f"{SCHEMA}.completed.v1", "status": "COMPLETE",
            "run_id": run_id,
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
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--feature-set", required=True, choices=["paper7", "all"])
    parser.add_argument("--run-id", required=True)
    a = parser.parse_args(argv)
    run(root=a.root, config_path=a.config, positive_path=a.positive,
        negative_path=a.negative, feature_set=a.feature_set, run_id=a.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
