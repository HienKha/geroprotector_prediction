"""Standalone TabNet on the paper's seven DataWarrior descriptors.

`blend7feat_20260821` evaluated four slot-3 models on the seven published
descriptors -- tabpfn_v2, tabfm, bishop, tabm -- and left TabNet out.  This
module fills that single gap using the identical protocol, so the resulting
numbers drop straight into that run's tables:

  * feature panel : the 7 DataWarrior descriptors, verbatim, no transformation
                    beyond the alt-model pipeline's own imputer + quantile
                    transform (exactly what bishop/tabm received there);
  * folds         : StratifiedKFold(5, shuffle=True, random_state=42) on the 324
                    training rows, seeds 42+fold, matching blend7feat;
  * lock/test     : refit on all 324 training rows, score the 81 held-out rows
                    once;
  * threshold     : fixed 0.5, no selection.

No test row -- value or label -- enters any fit or any transform, and the split
is hash-verified against the sealed assignment before anything is trained.
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from geroprotector.fixed_blend_paper405 import _features
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.screening_blend_altmodels import _alt_probabilities
from geroprotector.traditional_paper405 import _read_sources, paper_split_indices


class TabNet7FeatError(RuntimeError):
    """Raised when a contract, a sealed input or a leakage guard fails."""


SCHEMA = "geroprotector.tabnet7feat"
FIXED_THRESHOLD = 0.5
MODEL = "tabnet"


def _regular_file(path: Path, role: str, expected: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise TabNet7FeatError(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected is not None and sha256_file(resolved) != expected:
        raise TabNet7FeatError(f"{role} SHA256 differs from the protocol lock")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = yaml.safe_load(_regular_file(path, "tabnet7feat protocol").read_text("utf-8"))
    if not isinstance(protocol, dict) or protocol.get("schema_version") != f"{SCHEMA}.protocol.v1":
        raise TabNet7FeatError("Unknown tabnet7feat protocol schema")
    design = protocol["design"]
    if design.get("test_rows_used_in_any_fit") is not False:
        raise TabNet7FeatError("Leakage contract differs")
    if design.get("slot3_feature_panel") != "paper 7 DataWarrior descriptors":
        raise TabNet7FeatError("Feature-panel contract differs")
    if MODEL not in protocol["altmodels"]["alt_models"]:
        raise TabNet7FeatError("TabNet settings are missing from the protocol")
    if protocol.get("immutability", {}).get("existing_run_directories_are_read_only") is not True:
        raise TabNet7FeatError("Immutability contract differs")
    for record in protocol["sealed_inputs"].values():
        _regular_file(root / record["path"], f"sealed input {record['path']}", record["sha256"])
    return protocol, canonical_sha256(protocol)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    d = (p >= FIXED_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, d, labels=[0, 1]).ravel()
    both = len(set(y)) == 2
    return {
        "n": len(y), "n_positive": int(y.sum()),
        "accuracy": float(accuracy_score(y, d)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "cohen_kappa": float(cohen_kappa_score(y, d)),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)) if both else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-9, 1 - 1e-9))),
        "mcc": float(matthews_corrcoef(y, d)),
        "f1_macro": float(f1_score(y, d, average="macro", zero_division=0)),
        "f1_positive": float(f1_score(y, d, pos_label=1, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": FIXED_THRESHOLD,
    }


def run(
    *, root: Path, config_path: Path, positive_path: Path, negative_path: Path, run_id: str
) -> Path:
    if not re.fullmatch(r"tabnet7feat_[a-z0-9_.-]+", run_id):
        raise TabNet7FeatError("RUN_ID must start with tabnet7feat_")
    root = root.resolve()
    protocol, protocol_sha256 = load_protocol(root, config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise TabNet7FeatError(f"Run directory already exists: {destination}")

    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    frame, _audit = _read_sources(
        _regular_file(positive_path, "positive source", protocol["sources"]["positive_sha256"]),
        _regular_file(negative_path, "negative source", protocol["sources"]["negative_sha256"]),
        traditional,
    )
    train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    if split_sha256 != protocol["paper_split"]["expected_assignment_sha256"]:
        raise TabNet7FeatError("Paper split differs from the sealed assignment")
    labels = frame["label"].to_numpy(dtype=int)
    smiles, _ = _validated_raw_smiles(frame)
    panel = _features(frame, smiles, fixed_protocol)["paper"].astype(np.float64)
    if panel.shape[1] != 7:
        raise TabNet7FeatError(f"Expected 7 paper descriptors, got {panel.shape[1]}")
    y_train, y_test = labels[train_indices], labels[test_indices]
    print(f"panel {panel.shape}; train {len(train_indices)}, test {len(test_indices)}", flush=True)

    # row alignment against the sealed component files (labels must agree)
    # the sealed train-OOF file carries no label column, so alignment is proven on
    # its row set; the sealed test file does carry labels and is checked on those
    sealed_oof = pd.read_csv(root / protocol["sealed_inputs"]["weighted_train_oof"]["path"])
    if not np.array_equal(np.sort(sealed_oof.paper_row_index.to_numpy()),
                          np.sort(np.asarray(train_indices))):
        raise TabNet7FeatError("Sealed OOF row set differs from the paper train split")
    sealed_test = pd.read_csv(root / protocol["sealed_inputs"]["weighted_test_components"]["path"])
    sealed_test = sealed_test.set_index("paper_row_index").loc[test_indices]
    if not np.array_equal(sealed_test.label.to_numpy(int), y_test):
        raise TabNet7FeatError("Test labels differ from the sealed component file")

    settings = protocol["altmodels"]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_split = list(folds.split(train_indices, y_train))
    fold_id = np.full(len(train_indices), -1, dtype=int)

    oof = np.full(len(train_indices), np.nan)
    for fold, (relative_fit, relative_validation) in enumerate(fold_split):
        fold_id[relative_validation] = fold
        probabilities, _extra = _alt_probabilities(
            root, MODEL, settings,
            panel[train_indices[relative_fit]], labels[train_indices[relative_fit]],
            {"validation": panel[train_indices[relative_validation]]},
        )
        oof[relative_validation] = probabilities["validation"]
        print(f"[tabnet] OOF fold {fold + 1}/5", flush=True)
    if not np.isfinite(oof).all():
        raise TabNet7FeatError("TabNet OOF stream is incomplete")

    # lock on all 324 training rows, score the 81 held-out rows once
    scored, _extra = _alt_probabilities(
        root, MODEL, settings, panel[train_indices], y_train,
        {"d1_test": panel[test_indices]},
    )
    test_probability = scored["d1_test"]
    print(f"[tabnet] locked fit done; scored {len(test_probability)} test rows", flush=True)

    metrics = pd.DataFrame([
        {"cohort": "cv5_train_oof", "model": "slot3_tabnet",
         "feature_panel": "paper_7_datawarrior", **_metrics(y_train, oof)},
        {"cohort": "d1_test_80_20", "model": "slot3_tabnet",
         "feature_panel": "paper_7_datawarrior", **_metrics(y_test, test_probability)},
    ])
    per_fold = pd.DataFrame([
        {"model": "slot3_tabnet", "fold": fold,
         **_metrics(y_train[fold_id == fold], oof[fold_id == fold])}
        for fold in range(5)
    ])

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".tabnet7.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "metrics_all.csv", metrics)
        _write_csv(tmp / "cv5_per_fold_metrics.csv", per_fold)
        _write_csv(tmp / "cv5_train_oof_predictions.csv", pd.DataFrame({
            "paper_row_index": train_indices, "fold": fold_id, "label": y_train,
            "probability_slot3_tabnet": oof}))
        _write_csv(tmp / "d1_test_predictions.csv", pd.DataFrame({
            "paper_row_index": test_indices, "label": y_test,
            "probability_slot3_tabnet": test_probability}))
        atomic_write_json(tmp / "RUN_MANIFEST.json", {
            "schema_version": f"{SCHEMA}.run_manifest.v1", "run_id": run_id,
            "protocol_sha256": protocol_sha256, "paper_split_sha256": split_sha256,
            "model": MODEL, "feature_panel": "paper 7 DataWarrior descriptors",
            "n_features": int(panel.shape[1]),
            "completes": ("blend7feat_20260821, which covered tabpfn_v2/tabfm/bishop/tabm "
                          "on the same panel but not tabnet"),
            "protocol_matches": "blend7feat_20260821",
            "folds": "StratifiedKFold(5, shuffle=True, random_state=42) on train_indices",
            "fixed_threshold": FIXED_THRESHOLD, "threshold_is_tuned": False,
            "test_rows_used_in_any_fit_or_transform": False,
            "external_cohorts_used": False,
            "hyperparameters": settings["alt_models"][MODEL],
            "training": settings["alt_model_training"],
            "runtime": {"python": platform.python_version(),
                        "platform": platform.platform()},
            "sealed_inputs_sha256": {k: sha256_file(root / v["path"])
                                     for k, v in protocol["sealed_inputs"].items()},
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
