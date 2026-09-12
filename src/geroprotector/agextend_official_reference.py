"""Official-source resolution and released-model parity for AgeXtend.

This stage has a narrower role than ``agextend_endpoint_benchmark``.  It does
not refit a model.  It inventories the released training cohort, selected
features, published ten-fold/LOOCV streams and 84-compound challenge, then uses
the authors' pinned fitted model and Signaturizer implementation to verify the
published challenge probabilities when the historical environment is present.

The released model makes inference reproducible.  It does not reveal the
historical split identities, fold registry, SMOTE/Boruta random states or the
complete training program, so an exact historical refit remains
``BLOCKED_PARTIAL``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.official_sources import require_pinned_file, xlsx_table


SCHEMA = "geroprotector.agextend_official_reference"
EXPECTED = {"training_rows": 972, "positive": 583, "negative": 389,
            "features": 71, "challenge_rows": 84, "folds": 10,
            "loocv_rows": 972}


class AgeXtendReferenceError(RuntimeError):
    """Raised when an official-source or parity contract is violated."""


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != f"{SCHEMA}.protocol.v1":
        raise AgeXtendReferenceError("Unknown AgeXtend official-reference protocol")
    if protocol.get("expected") != EXPECTED:
        raise AgeXtendReferenceError("AgeXtend expected-count contract differs")
    if protocol["historical_refit"]["status"] != "BLOCKED_PARTIAL":
        raise AgeXtendReferenceError("Exact historical refit must remain BLOCKED_PARTIAL")
    return protocol, sha256_file(path)


def _load_official_model(path: Path):
    # A joblib/pickle may execute code.  This is allowed only after matching the
    # official model's pinned SHA256 and is recorded in the manifest.
    import joblib
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(path)
    return model, [str(item.message) for item in caught]


def extract_sources(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, pd.DataFrame], Any]:
    sources = protocol["sources"]
    paths: dict[str, Path] = {}
    for name, source in sources.items():
        paths[name] = require_pinned_file(
            source["path"], source["sha256"], role=f"AgeXtend {name}")

    training = pd.read_csv(paths["training_dataset"], sep="\t")
    if ({"smiles", "status"} - set(training.columns) or
            len(training) != EXPECTED["training_rows"] or
            set(training["status"].astype(int)) != {0, 1}):
        raise AgeXtendReferenceError("Official AgeXtend training table differs")
    counts = training["status"].astype(int).value_counts().to_dict()
    if counts != {1: EXPECTED["positive"], 0: EXPECTED["negative"]}:
        raise AgeXtendReferenceError(f"Official AgeXtend label counts differ: {counts}")

    challenge = xlsx_table(
        paths["supplementary_workbook"], "Supplementary Table 6",
        required_columns=("S.No.", "Compound Name", "Isomeric_SMILES", "Label",
                          "Canonical_SMILES", "Anti_Aging_Status",
                          "Anti_Aging_Prob", "Source"))
    challenge["Label"] = challenge["Label"].astype(int)
    challenge["Anti_Aging_Status"] = challenge["Anti_Aging_Status"].astype(int)
    challenge["Anti_Aging_Prob"] = challenge["Anti_Aging_Prob"].astype(float)
    if len(challenge) != EXPECTED["challenge_rows"]:
        raise AgeXtendReferenceError("Official AgeXtend challenge row count differs")

    tenfold = xlsx_table(
        paths["source_data_workbook"], "Fig 1f",
        required_columns=("Fold", "Testing Accuracy:", "Testing MCC Score:",
                          "Testing F1 Score:", "Testing AUC VALUE:",
                          "Testing kappa Score:", "Testing Precision:",
                          "Testing Recall:"))
    if len(tenfold) != EXPECTED["folds"]:
        raise AgeXtendReferenceError("Published AgeXtend ten-fold table differs")

    loocv = xlsx_table(
        paths["source_data_workbook"], "Fig 1h",
        required_columns=("Actual Status", "Prediction Status", "Probability 1",
                          "Probability 0"))
    if len(loocv) != EXPECTED["loocv_rows"]:
        raise AgeXtendReferenceError("Published AgeXtend LOOCV row count differs")
    actual = loocv["Actual Status"].astype(int).to_numpy()
    if not np.array_equal(actual, training["status"].astype(int).to_numpy()):
        raise AgeXtendReferenceError(
            "Published LOOCV label order does not align to the official training table")
    loocv.insert(0, "official_training_row", np.arange(len(loocv), dtype=int))
    loocv.insert(1, "SMILES", training["smiles"].astype(str).to_numpy())

    model, model_warnings = _load_official_model(paths["fitted_model"])
    feature_names = [str(value) for value in model.feature_names_in_]
    if len(feature_names) != EXPECTED["features"]:
        raise AgeXtendReferenceError("Official fitted model does not expose 71 features")
    feature_hash = canonical_sha256(feature_names)
    if feature_hash != protocol["official_model"]["feature_names_sha256"]:
        raise AgeXtendReferenceError("Official selected-feature identity hash differs")
    if (float(model.C) != 1.5 or float(model.gamma) != 2.5 or
            bool(model.probability) is not True or list(model.classes_) != [0, 1]):
        raise AgeXtendReferenceError("Official fitted SVC settings differ")
    features = pd.DataFrame({"feature_order": np.arange(len(feature_names), dtype=int),
                             "feature_name": feature_names})

    resolution = {
        "citations": protocol["citations"],
        "source_files": {name: {"path": str(paths[name]),
                                 "sha256": sha256_file(paths[name])}
                         for name in sorted(paths)},
        "official_training_cohort": {"rows": len(training), "positive": counts[1],
                                     "negative": counts[0]},
        "released_model": {"class": type(model).__name__, "C": float(model.C),
                           "gamma": float(model.gamma), "probability": True,
                           "classes": [int(value) for value in model.classes_],
                           "selected_features": len(feature_names),
                           "feature_names_sha256": feature_hash,
                           "load_warnings": model_warnings,
                           "pickle_execution_allowed_only_after_hash_check": True},
        "published_results": {"tenfold_rows": len(tenfold),
                              "loocv_rows": len(loocv),
                              "loocv_compound_alignment": "exact_ordered_label_match",
                              "challenge_rows": len(challenge)},
        "availability": {
            "published_result_extraction": "REPRODUCIBLE",
            "official_model_inference": "REPRODUCIBLE_IN_PINNED_ENVIRONMENT",
            "official_challenge_parity": "RUNNABLE_IN_PINNED_ENVIRONMENT",
            "exact_historical_refit": "BLOCKED_PARTIAL",
            "new_fixed_panel_protocol": "PROTOCOL_RECONSTRUCTION"},
        "exact_historical_refit_blockers": protocol["historical_refit"]["blockers"],
    }
    return resolution, {"training": training, "challenge": challenge,
                        "tenfold": tenfold, "loocv": loocv,
                        "features": features}, model


def _signaturizer_features(smiles: list[str]) -> pd.DataFrame:
    try:
        from signaturizer import Signaturizer
    except ImportError as exc:
        raise ModuleNotFoundError(
            "signaturizer==1.1.11 is required for official AgeXtend parity") from exc
    blocks = []
    for family in "ABCDE":
        for index in range(1, 6):
            signature = f"{family}{index}"
            result = Signaturizer(signature).predict(smiles)
            values = np.asarray(result.signature, dtype=float)
            if values.shape != (len(smiles), 128):
                raise AgeXtendReferenceError(
                    f"Signaturizer {signature} returned shape {values.shape}")
            blocks.append(pd.DataFrame(
                values, columns=[f"{signature}_{column}" for column in range(128)]))
    frame = pd.concat(blocks, axis=1)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    # This intentionally reproduces Predictor.py.  It is query-batch mean
    # imputation, not a corrected or inferred training-set imputation.
    frame = frame.fillna(frame.mean(axis=0))
    if frame.isna().any().any():
        raise AgeXtendReferenceError("Official batch-mean imputation left missing values")
    return frame


def run_parity(challenge: pd.DataFrame, model: Any,
               *, probability_atol: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    smiles = challenge["Canonical_SMILES"].astype(str).tolist()
    features = _signaturizer_features(smiles)
    selected = features[[str(value) for value in model.feature_names_in_]]
    probability = np.asarray(model.predict_proba(selected), dtype=float)
    positive_index = list(model.classes_).index(1)
    p_positive = probability[:, positive_index]
    decision = np.asarray(model.classes_)[np.argmax(probability, axis=1)].astype(int)
    published_probability = challenge["Anti_Aging_Prob"].to_numpy(float)
    published_decision = challenge["Anti_Aging_Status"].to_numpy(int)
    difference = np.abs(p_positive - published_probability)
    frame = pd.DataFrame({
        "official_challenge_row": np.arange(len(challenge), dtype=int),
        "compound_name": challenge["Compound Name"].astype(str),
        "canonical_smiles": smiles,
        "label": challenge["Label"].astype(int),
        "published_probability": published_probability,
        "recomputed_probability": p_positive,
        "absolute_probability_difference": difference,
        "published_decision": published_decision,
        "recomputed_decision": decision,
        "decision_match": decision == published_decision,
    })
    status = {
        "status": "PASS" if (difference.max() <= probability_atol and
                                 np.array_equal(decision, published_decision)) else "FAIL",
        "rows": len(frame), "probability_atol": probability_atol,
        "maximum_absolute_probability_difference": float(difference.max()),
        "decision_matches": int((decision == published_decision).sum()),
        "batch_mean_imputation_reproduced_verbatim": True,
    }
    if status["status"] != "PASS":
        raise AgeXtendReferenceError(f"Official challenge inference parity failed: {status}")
    return frame, status


def run(*, root: Path, config_path: Path, run_id: str,
        require_parity: bool = False) -> Path:
    if not re.fullmatch(r"agextend_official_reference_[a-z0-9_.-]+", run_id):
        raise AgeXtendReferenceError(
            "RUN_ID must start with agextend_official_reference_")
    root = root.resolve()
    protocol, protocol_sha = load_protocol(config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise AgeXtendReferenceError(f"Run directory already exists: {destination}")
    resolution, tables, model = extract_sources(protocol)

    parity: pd.DataFrame
    try:
        parity, parity_status = run_parity(
            tables["challenge"], model,
            probability_atol=float(protocol["parity"]["probability_atol"]))
    except ModuleNotFoundError as exc:
        if require_parity:
            raise
        parity = pd.DataFrame(columns=(
            "official_challenge_row", "compound_name", "canonical_smiles", "label",
            "published_probability", "recomputed_probability",
            "absolute_probability_difference", "published_decision",
            "recomputed_decision", "decision_match"))
        parity_status = {"status": "BLOCKED_ENVIRONMENT", "reason": str(exc),
                         "required_environment": protocol["parity"]["environment"]}
    resolution["availability"]["official_challenge_parity"] = parity_status["status"]

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".agextend-reference-", dir=destination.parent))
    try:
        atomic_write_json(tmp / "SOURCE_RESOLUTION.json", resolution)
        _write_csv(tmp / "official_training_cohort.csv", tables["training"])
        _write_csv(tmp / "official_selected_features.csv", tables["features"])
        _write_csv(tmp / "published_tenfold_metrics.csv", tables["tenfold"])
        _write_csv(tmp / "published_loocv_predictions.csv", tables["loocv"])
        _write_csv(tmp / "official_challenge_reference.csv", tables["challenge"])
        _write_csv(tmp / "official_challenge_inference_parity.csv", parity)
        atomic_write_json(tmp / "official_challenge_parity_status.json", parity_status)
        summary = ["# AgeXtend official-source and released-model audit", "",
                   f"- official training cohort: {len(tables['training'])} rows "
                   f"({int(tables['training'].status.astype(int).sum())} positive)",
                   f"- exact selected features: {len(tables['features'])}",
                   f"- published ten-fold rows: {len(tables['tenfold'])}",
                   f"- compound-aligned published LOOCV rows: {len(tables['loocv'])}",
                   f"- official 84-compound inference parity: **{parity_status['status']}**",
                   "- exact historical refit: **BLOCKED_PARTIAL** because split/fold "
                   "identities, seeds, training code, and SMOTE/Boruta states are absent.", ""]
        (tmp / "agextend_official_reference_summary.md").write_text(
            "\n".join(summary), encoding="utf-8")
        atomic_write_json(tmp / "RUN_MANIFEST.json", {
            "schema_version": f"{SCHEMA}.run_manifest.v1", "run_id": run_id,
            "protocol_sha256": protocol_sha, "source_resolution": resolution,
            "parity_status": parity_status,
            "historical_model_refitted": False,
            "challenge_labels_used_for_model_or_threshold_selection": False,
            "official_pickle_executed_after_sha256_verification": True,
            "runtime": {"python": platform.python_version(),
                        "platform": platform.platform()},
            "existing_runs_modified": False})
        atomic_write_json(tmp / "COMPLETED.json", {
            "schema_version": f"{SCHEMA}.completed.v1",
            "status": "COMPLETE_SOURCE_AUDIT", "run_id": run_id,
            "run_manifest_sha256": sha256_file(tmp / "RUN_MANIFEST.json"),
            "artifact_hashes": {str(path.relative_to(tmp)): sha256_file(path)
                                for path in sorted(tmp.rglob("*"))
                                if path.is_file() and path.name != "COMPLETED.json"}})
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    print(json.dumps({"run": str(destination), "parity": parity_status}, indent=2))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--require-parity", action="store_true")
    args = parser.parse_args(argv)
    run(root=args.root, config_path=args.config, run_id=args.run_id,
        require_parity=args.require_parity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
