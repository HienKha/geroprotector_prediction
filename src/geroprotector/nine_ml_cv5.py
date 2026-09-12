"""Five-fold cross-validation of the nine ML algorithms on the paper's seven descriptors.

The sealed `traditional405_20260817` run evaluated the nine algorithms on a single
80/20 split only.  This module adds the missing arm: the same nine algorithms,
the same seven DataWarrior descriptors and the same hyperparameters, cross-
validated on the 324 D1 training rows with

    StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

which is the identical fold structure used by every other cross-validated run in
this project, so the resulting out-of-fold predictions are row-aligned with
`cv5_paper_metrics_20260821`, `blend7feat_20260821` and `tabnet7feat_20260822`
and can be compared or pooled with them directly.

The 81 held-out test rows are never loaded into any fit.  Each fold fits on its
four training folds and predicts the fifth; scalers inside the KNN and logistic-
regression pipelines are therefore fitted per fold on fit rows only.  The
decision threshold is fixed at 0.5 for every algorithm, so no operating point is
selected from the data being scored.
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
from sklearn.model_selection import StratifiedKFold

from geroprotector.fixed_blend_paper405 import _features
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.nine_ml_featuresets import FIXED_THRESHOLD, MODEL_ORDER, _metrics
from geroprotector.traditional_paper405 import (
    _load_protocol,
    _models,
    _positive_score,
    _read_sources,
    paper_split_indices,
)


class NineMLCV5Error(RuntimeError):
    """Raised when a contract or a leakage guard fails."""


SCHEMA = "geroprotector.nine_ml_cv5"


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def run(
    *, root: Path, config_path: Path, positive_path: Path, negative_path: Path, run_id: str
) -> Path:
    if not re.fullmatch(r"nineml_cv5_[a-z0-9_.-]+", run_id):
        raise NineMLCV5Error("RUN_ID must start with nineml_cv5_")
    root = root.resolve()
    protocol, protocol_sha256 = _load_protocol(config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise NineMLCV5Error(f"Run directory already exists: {destination}")

    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    frame, audit = _read_sources(positive_path.resolve(), negative_path.resolve(), traditional)
    train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    labels = frame["label"].to_numpy(dtype=int)
    y_train = labels[train_indices]
    smiles, _ = _validated_raw_smiles(frame)
    panel = _features(frame, smiles, fixed_protocol)["paper"].astype(float)
    if panel.shape[1] != 7:
        raise NineMLCV5Error(f"Expected 7 paper descriptors, got {panel.shape[1]}")
    x_train = panel[train_indices]
    print(f"panel {panel.shape}; CV on {len(train_indices)} training rows "
          f"({int(y_train.sum())} positive); test rows untouched", flush=True)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_split = list(folds.split(train_indices, y_train))
    fold_id = np.full(len(train_indices), -1, dtype=int)
    for fold, (_fit, validation) in enumerate(fold_split):
        fold_id[validation] = fold
    if (fold_id < 0).any():
        raise NineMLCV5Error("Fold assignment is incomplete")

    oof_score = {m: np.full(len(train_indices), np.nan) for m in MODEL_ORDER}
    oof_probability = {m: np.full(len(train_indices), np.nan) for m in MODEL_ORDER}
    for fold, (relative_fit, relative_validation) in enumerate(fold_split):
        models = _models(protocol)          # fresh, unfitted estimators every fold
        for model_id in MODEL_ORDER:
            model = models[model_id]
            model.fit(x_train[relative_fit], y_train[relative_fit])
            score, probability = _positive_score(
                model, x_train[relative_validation], model_id=model_id)
            oof_score[model_id][relative_validation] = score
            oof_probability[model_id][relative_validation] = probability
        print(f"fold {fold + 1}/5 complete", flush=True)
    for model_id in MODEL_ORDER:
        if not np.isfinite(oof_score[model_id]).all():
            raise NineMLCV5Error(f"{model_id} OOF stream is incomplete")

    pooled = pd.DataFrame([
        {"model_id": m, "aggregation": "pooled_oof", "n_features": 7,
         **_metrics(y_train, oof_score[m], oof_probability[m])}
        for m in MODEL_ORDER
    ])
    per_fold = pd.DataFrame([
        {"model_id": m, "fold": fold,
         **_metrics(y_train[fold_id == fold], oof_score[m][fold_id == fold],
                    oof_probability[m][fold_id == fold])}
        for m in MODEL_ORDER for fold in range(5)
    ])
    summary_metrics = ("auprc_average_precision_positive", "auroc", "accuracy",
                       "cohen_kappa", "mcc", "macro_f1", "brier",
                       "recall_sensitivity", "specificity")
    mean_sd = per_fold.groupby("model_id").agg(
        **{f"{m}_{s}": (m, s) for m in summary_metrics for s in ("mean", "std")}
    ).reindex(MODEL_ORDER).reset_index()

    for row in pooled.itertuples(index=False):
        print(f"  {row.model_id:22s} AP+ {row.auprc_average_precision_positive:.4f}"
              f"  MCC {row.mcc:.4f}  Acc {row.accuracy:.4f}"
              f"  kappa {row.cohen_kappa:.4f}", flush=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".ninecv5.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "cv5_pooled_oof_metrics.csv", pooled)
        _write_csv(tmp / "cv5_per_fold_metrics.csv", per_fold)
        _write_csv(tmp / "cv5_mean_sd_across_folds.csv", mean_sd)
        _write_csv(tmp / "cv5_train_oof_predictions.csv", pd.DataFrame({
            "paper_row_index": train_indices, "fold": fold_id, "label": y_train,
            **{f"score_{m}": oof_score[m] for m in MODEL_ORDER},
            **{f"probability_{m}": oof_probability[m] for m in MODEL_ORDER},
        }))
        atomic_write_json(tmp / "RUN_MANIFEST.json", {
            "schema_version": f"{SCHEMA}.run_manifest.v1", "run_id": run_id,
            "protocol_sha256": protocol_sha256, "paper_split_sha256": split_sha256,
            "feature_panel": "paper 7 DataWarrior descriptors", "n_features": 7,
            "models": list(MODEL_ORDER), "model_settings": protocol["models"],
            "folds": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "fold_structure_matches": ["cv5_paper_metrics_20260821",
                                       "blend7feat_20260821", "tabnet7feat_20260822"],
            "rows": "D1 train only (324)",
            "fixed_threshold": FIXED_THRESHOLD, "threshold_is_tuned": False,
            "test_rows_used_in_any_fit": False,
            "estimators_are_reconstructed_per_fold": True,
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
    parser.add_argument("--run-id", required=True)
    a = parser.parse_args(argv)
    run(root=a.root, config_path=a.config, positive_path=a.positive,
        negative_path=a.negative, run_id=a.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
