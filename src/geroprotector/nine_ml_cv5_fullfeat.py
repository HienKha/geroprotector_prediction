"""Five-fold cross-validation on the FULL RDKit2D panel for the conventional ML models.

This fills the one genuine gap in the full-feature evidence. `nineml_full_scaled_20260823`
already provides held-out test predictions on the full panel, and
`screeningblend_altmodels_20260819` already provides cross-fitted train-OOF predictions
for BiSHop, TabM and TabNet on the same panel. What does not exist anywhere is
cross-fitted train-OOF predictions for the conventional algorithms on the full panel --
`nineml_cv5_20260822` covers only the paper's seven descriptors.

Folds are `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on `train_indices`,
identical to every other cross-validated run in this project, so the resulting streams are
row-aligned with `cv5_paper_metrics_20260821` and the alt-model OOF file and can be pooled
with them directly.

PREPROCESSING.  The feature panel is built with a median imputation and zero-variance
filter fitted on training rows only, and refitted inside each fold on that fold's fit rows
so that no validation row influences its own transform. The scale-sensitive models (SVM,
k-NN, logistic regression, linear regression) carry a StandardScaler fitted per fold; the
tree ensembles are scale-invariant and are left unscaled. This mirrors
`nine_ml_fullfeat`, where unscaled fitting fails outright on this panel.

The 81 held-out test rows are never loaded.

Default model set is the three gradient-boosting algorithms, which are the ones needed for
the ablation grids; pass --all-models for the complete set of nine.
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

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from geroprotector.fixed_blend_paper405 import _features
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.nine_ml_featuresets import FIXED_THRESHOLD, MODEL_ORDER, _metrics
from geroprotector.screening_blend_altmodels import _apply_context, _imputer_context
from geroprotector.traditional_paper405 import (
    _load_protocol,
    _models,
    _positive_score,
    _read_sources,
    paper_split_indices,
)


class NineMLCV5FullError(RuntimeError):
    """Raised when a contract or a leakage guard fails."""


SCHEMA = "geroprotector.nine_ml_cv5_fullfeat"
GRADIENT_BOOSTING = ("xgboost", "lightgbm", "catboost")
SCALE_SENSITIVE = ("svm_original_paper", "knn", "logistic_regression", "linear_regression")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def _build_models(protocol) -> dict:
    """Fresh estimators, with scalers added to the scale-sensitive ones."""
    settings = protocol["models"]["svm_original_paper"]
    models = _models(protocol)
    models["svm_original_paper"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel=settings["kernel"], C=float(settings["C"]),
                      gamma=float(settings["gamma"]),
                      probability=bool(settings["probability"]),
                      random_state=int(settings["random_state"])))])
    models["linear_regression"] = Pipeline([
        ("scaler", StandardScaler()), ("model", LinearRegression())])
    return models


def run(*, root: Path, config_path: Path, positive_path: Path, negative_path: Path,
        run_id: str, model_ids: list[str]) -> Path:
    if not re.fullmatch(r"nineml_cv5_full_[a-z0-9_.-]+", run_id):
        raise NineMLCV5FullError("RUN_ID must start with nineml_cv5_full_")
    root = root.resolve()
    protocol, protocol_sha256 = _load_protocol(config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise NineMLCV5FullError(f"Run directory already exists: {destination}")
    unknown = set(model_ids) - set(MODEL_ORDER)
    if unknown:
        raise NineMLCV5FullError(f"Unknown model ids: {sorted(unknown)}")

    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    frame, audit = _read_sources(positive_path.resolve(), negative_path.resolve(), traditional)
    train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    labels = frame["label"].to_numpy(int)
    y_train = labels[train_indices]
    smiles, _ = _validated_raw_smiles(frame)
    raw = _features(frame, smiles, fixed_protocol)["rdkit2d"]
    raw_train = raw[train_indices]
    print(f"raw RDKit2D panel {raw.shape}; CV on {len(train_indices)} training rows "
          f"({int(y_train.sum())} positive); test rows untouched", flush=True)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_split = list(folds.split(train_indices, y_train))
    fold_id = np.full(len(train_indices), -1, dtype=int)
    for fold, (_fit, validation) in enumerate(fold_split):
        fold_id[validation] = fold
    if (fold_id < 0).any():
        raise NineMLCV5FullError("Fold assignment is incomplete")

    oof_score = {m: np.full(len(train_indices), np.nan) for m in model_ids}
    oof_probability = {m: np.full(len(train_indices), np.nan) for m in model_ids}
    widths = []
    for fold, (relative_fit, relative_validation) in enumerate(fold_split):
        # transform refitted inside the fold: no validation row informs its own panel
        context = _imputer_context(raw_train[relative_fit])
        x_fit = _apply_context(context, raw_train[relative_fit]).astype(np.float64)
        x_val = _apply_context(context, raw_train[relative_validation]).astype(np.float64)
        widths.append(int(x_fit.shape[1]))
        models = _build_models(protocol)
        for model_id in model_ids:
            model = models[model_id]
            model.fit(x_fit, y_train[relative_fit])
            score, probability = _positive_score(model, x_val, model_id=model_id)
            oof_score[model_id][relative_validation] = score
            oof_probability[model_id][relative_validation] = probability
        print(f"fold {fold + 1}/5 complete ({x_fit.shape[1]} descriptors retained)",
              flush=True)
    for model_id in model_ids:
        if not np.isfinite(oof_score[model_id]).all():
            raise NineMLCV5FullError(f"{model_id} OOF stream is incomplete")

    pooled = pd.DataFrame([
        {"model_id": m, "aggregation": "pooled_oof", "feature_set": "full_rdkit2d",
         **_metrics(y_train, oof_score[m], oof_probability[m])} for m in model_ids])
    per_fold = pd.DataFrame([
        {"model_id": m, "fold": fold,
         **_metrics(y_train[fold_id == fold], oof_score[m][fold_id == fold],
                    oof_probability[m][fold_id == fold])}
        for m in model_ids for fold in range(5)])
    for row in pooled.itertuples(index=False):
        print(f"  {row.model_id:22s} AP+ {row.auprc_average_precision_positive:.4f}"
              f"  MCC {row.mcc:.4f}  Acc {row.accuracy:.4f}", flush=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".ninecv5full.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "cv5_pooled_oof_metrics.csv", pooled)
        _write_csv(tmp / "cv5_per_fold_metrics.csv", per_fold)
        _write_csv(tmp / "cv5_train_oof_predictions.csv", pd.DataFrame({
            "paper_row_index": train_indices, "fold": fold_id, "label": y_train,
            **{f"score_{m}": oof_score[m] for m in model_ids},
            **{f"probability_{m}": oof_probability[m] for m in model_ids}}))
        atomic_write_json(tmp / "RUN_MANIFEST.json", {
            "schema_version": f"{SCHEMA}.run_manifest.v1", "run_id": run_id,
            "protocol_sha256": protocol_sha256, "paper_split_sha256": split_sha256,
            "feature_set": "full RDKit2D panel, transform refitted inside each fold",
            "descriptors_retained_per_fold": widths,
            "models": list(model_ids), "model_settings": protocol["models"],
            "scaled_models": [m for m in model_ids if m in SCALE_SENSITIVE],
            "folds": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "fold_structure_matches": ["cv5_paper_metrics_20260821",
                                       "screeningblend_altmodels_20260819",
                                       "nineml_cv5_20260822"],
            "rows": "D1 train only (324)",
            "fixed_threshold": FIXED_THRESHOLD, "threshold_is_tuned": False,
            "test_rows_used_in_any_fit": False,
            "data_audit": audit,
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
    print(json.dumps({"run": str(destination), "status": "COMPLETE"}, indent=2))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--all-models", action="store_true",
                        help="run all nine algorithms instead of the three GBMs")
    a = parser.parse_args(argv)
    run(root=a.root, config_path=a.config, positive_path=a.positive,
        negative_path=a.negative, run_id=a.run_id,
        model_ids=list(MODEL_ORDER) if a.all_models else list(GRADIENT_BOOSTING))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
