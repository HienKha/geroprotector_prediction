"""Fixed V3 blend and SVM variants on the exact raw paper-405 split."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from geroprotector.audit import runtime_environment, source_tree_sha256, validate_core_runtime
from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.hypermoltab_paper405 import (
    _paper_contract,
    _validated_raw_smiles,
    select_mcc_threshold,
)
from geroprotector.traditional_paper405 import (
    _cross_split_audit,
    _read_sources,
    evaluation_metrics,
    paper_split_indices,
)


class FixedBlendPaper405Error(RuntimeError):
    pass


COMPONENTS = ("extra_trees", "tanimoto_svc", "tabpfn_v2", "paper_svm")
BLENDS = (
    "sota_fixed_blend_original",
    "sota_fixed_blend_plus_svm",
    "sota_fixed_blend_svm_replaces_extratrees",
)


def _regular_file(path: Path, role: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FixedBlendPaper405Error(f"{role} must be a regular non-symlink file: {path}")
    return path.resolve()


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = yaml.safe_load(_regular_file(path, "protocol").read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "geroprotector.fixed_blend_paper405.protocol.v1":
        raise FixedBlendPaper405Error("Unknown fixed-blend protocol schema")
    expected = {
        "sota_fixed_blend_original": {
            "components": ["extra_trees", "tanimoto_svc", "tabpfn_v2"],
            "weights": [1 / 3, 1 / 3, 1 / 3],
        },
        "sota_fixed_blend_plus_svm": {
            "components": ["extra_trees", "tanimoto_svc", "tabpfn_v2", "paper_svm"],
            "weights": [0.25, 0.25, 0.25, 0.25],
        },
        "sota_fixed_blend_svm_replaces_extratrees": {
            "components": ["paper_svm", "tanimoto_svc", "tabpfn_v2"],
            "weights": [1 / 3, 1 / 3, 1 / 3],
        },
    }
    observed = protocol.get("blends", {})
    if tuple(observed) != BLENDS:
        raise FixedBlendPaper405Error("Blend set/order differs from the lock")
    for name in BLENDS:
        if observed[name]["components"] != expected[name]["components"] or not np.allclose(
            observed[name]["weights"], expected[name]["weights"], rtol=0.0, atol=1e-15
        ):
            raise FixedBlendPaper405Error(f"{name} components/weights differ from lock")
    evaluation = protocol.get("evaluation", {})
    if evaluation != {
        "threshold_selection": "cross_fitted_oof_on_324_paper_train",
        "threshold_metric": "mcc",
        "calibration": "none",
        "outer_test_used_for_components_or_weights": False,
        "outer_test_used_for_threshold": False,
        "hagr_loaded": False,
    }:
        raise FixedBlendPaper405Error("Fixed-blend evaluation/firewall differs from lock")
    return protocol, canonical_sha256(protocol)


def _features(
    frame: pd.DataFrame, smiles: list[str], protocol: dict[str, Any]
) -> dict[str, np.ndarray]:
    paper = frame[protocol["features"]["paper_descriptors"]].to_numpy(dtype=np.float64)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(protocol["features"]["morgan_radius"]),
        fpSize=int(protocol["features"]["morgan_bits"]),
        includeChirality=False,
    )
    bits = np.zeros((len(smiles), int(protocol["features"]["morgan_bits"])), dtype=np.uint8)
    descriptor_names = tuple(name for name, _function in Descriptors._descList)
    if len(descriptor_names) > int(protocol["features"]["rdkit2d_max_features"]):
        raise FixedBlendPaper405Error("RDKit2D descriptor count exceeds TabPFN lock")
    rdkit2d = np.full((len(smiles), len(descriptor_names)), np.nan, dtype=np.float64)
    for row_index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        fingerprint = generator.GetFingerprint(molecule)
        bits[row_index] = np.asarray(fingerprint, dtype=np.uint8)
        for column_index, (_name, function) in enumerate(Descriptors._descList):
            try:
                result = float(function(molecule))
            except Exception:
                result = np.nan
            rdkit2d[row_index, column_index] = (
                result
                if np.isfinite(result) and abs(result) <= np.finfo(np.float32).max
                else np.nan
            )
    if not np.isfinite(paper).all() or not np.isin(bits, (0, 1)).all():
        raise FixedBlendPaper405Error("Paper descriptors or Morgan bits are invalid")
    return {"paper": paper, "morgan": bits, "rdkit2d": rdkit2d}


def _fit_imputer(train: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    finite = np.isfinite(train)
    keep = finite.any(axis=0)
    train_kept = train[:, keep]
    target_kept = target[:, keep]
    medians = np.nanmedian(np.where(np.isfinite(train_kept), train_kept, np.nan), axis=0)
    train_filled = np.where(np.isfinite(train_kept), train_kept, medians)
    target_filled = np.where(np.isfinite(target_kept), target_kept, medians)
    varying = np.ptp(train_filled, axis=0) > 0
    train_final = train_filled[:, varying].astype(np.float32)
    target_final = target_filled[:, varying].astype(np.float32)
    if not np.isfinite(train_final).all() or not np.isfinite(target_final).all():
        raise FixedBlendPaper405Error("Fold-local descriptor imputation failed")
    return (
        train_final,
        target_final,
        {
            "input_columns": int(train.shape[1]),
            "retained_columns": int(varying.sum()),
            "fit_rows": len(train),
        },
    )


def _tanimoto(query: np.ndarray, train: np.ndarray) -> np.ndarray:
    query_float, train_float = query.astype(np.float32), train.astype(np.float32)
    intersection = query_float @ train_float.T
    union = query_float.sum(axis=1)[:, None] + train_float.sum(axis=1)[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )


def _fit_tanimoto(
    train_bits: np.ndarray, y_train: np.ndarray, target_bits: np.ndarray, settings: dict
) -> tuple[np.ndarray, Any]:
    model = SVC(
        C=float(settings["C"]),
        kernel="precomputed",
        probability=False,
        random_state=42,
    )
    model.fit(_tanimoto(train_bits, train_bits), y_train)
    probability = expit(model.decision_function(_tanimoto(target_bits, train_bits)))
    return np.asarray(probability, dtype=np.float64), model


def _tabpfn_probability(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_target: np.ndarray,
    settings: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    os.environ["TABPFN_DISABLE_TELEMETRY"] = "true"
    if os.environ.get("TABPFN_DISABLE_TELEMETRY") != "true":
        raise FixedBlendPaper405Error("TabPFN telemetry is not disabled")
    observed_version = importlib.metadata.version("tabpfn")
    if observed_version != str(settings["required_package_version"]):
        raise FixedBlendPaper405Error("Installed TabPFN version differs from the lock")
    checkpoint = _regular_file(Path(settings["checkpoint_path"]), "TabPFN v2 checkpoint")
    if sha256_file(checkpoint) != settings["checkpoint_sha256"]:
        raise FixedBlendPaper405Error("TabPFN v2 checkpoint SHA256 differs from lock")
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    if settings["model_version"] != "v2":
        raise FixedBlendPaper405Error("Only the locked TabPFN v2 model is allowed")
    estimator = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        model_path=str(checkpoint),
        device=str(settings["device"]),
        random_state=int(seed),
    )
    estimator.fit(X_train, y_train)
    classes = np.asarray(estimator.classes_)
    if classes.shape != (2,) or set(map(int, classes)) != {0, 1}:
        raise FixedBlendPaper405Error("TabPFN v2 returned invalid classes")
    positive_column = int(np.flatnonzero(classes == 1)[0])
    canonical_order = sorted(
        range(len(X_target)), key=lambda index: X_target[index].astype("<f8").tobytes()
    )
    probability = np.empty(len(X_target), dtype=np.float64)
    for index in canonical_order:
        raw = np.asarray(estimator.predict_proba(X_target[index : index + 1]), dtype=float)
        if raw.shape != (1, 2) or not np.isfinite(raw).all():
            raise FixedBlendPaper405Error("TabPFN singleton prediction is invalid")
        probability[index] = raw[0, positive_column]
    sentinels = tuple(
        dict.fromkeys(
            [
                canonical_order[0],
                canonical_order[len(canonical_order) // 2],
                canonical_order[-1],
            ]
        )
    )
    repeated = np.asarray(
        [
            estimator.predict_proba(X_target[index : index + 1])[0, positive_column]
            for index in sentinels
        ],
        dtype=float,
    )
    maximum_difference = float(np.max(np.abs(repeated - probability[list(sentinels)])))
    if maximum_difference > float(settings["repeat_atol"]):
        raise FixedBlendPaper405Error("TabPFN isolated singleton repeatability failed")
    del estimator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.clip(probability, 1e-7, 1 - 1e-7), {
        "inference_mode": "canonical_isolated_singleton",
        "repeat_sentinels": len(sentinels),
        "repeat_max_abs_difference": maximum_difference,
        "checkpoint_sha256": settings["checkpoint_sha256"],
        "package_version": observed_version,
        "telemetry_disabled": True,
    }


def _component_predictions(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    fit: np.ndarray,
    target: np.ndarray,
    settings: dict[str, Any],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    return _selected_component_predictions(
        features,
        labels,
        fit,
        target,
        settings,
        seed,
        requested=COMPONENTS,
    )


def _selected_component_predictions(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    fit: np.ndarray,
    target: np.ndarray,
    settings: dict[str, Any],
    seed: int,
    *,
    requested: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Fit only requested components without changing the locked legacy wrapper."""
    unknown = set(requested) - set(COMPONENTS)
    if unknown or len(requested) != len(set(requested)) or not requested:
        raise FixedBlendPaper405Error(f"Invalid requested blend components: {requested}")
    probabilities: dict[str, np.ndarray] = {}
    models: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    train_desc: np.ndarray | None = None
    target_desc: np.ndarray | None = None
    if {"extra_trees", "tabpfn_v2"} & set(requested):
        train_desc, target_desc, descriptor_audit = _fit_imputer(
            features["rdkit2d"][fit], features["rdkit2d"][target]
        )
        audit["descriptors"] = descriptor_audit
    if "extra_trees" in requested:
        assert train_desc is not None and target_desc is not None
        combined_train = np.column_stack([train_desc, features["morgan"][fit]])
        combined_target = np.column_stack([target_desc, features["morgan"][target]])
        extra_trees = ExtraTreesClassifier(**settings["extra_trees"])
        extra_trees.fit(combined_train, labels[fit])
        probabilities["extra_trees"] = np.asarray(
            extra_trees.predict_proba(combined_target)[:, 1], dtype=float
        )
        models["extra_trees"] = extra_trees
    if "tanimoto_svc" in requested:
        tanimoto_probability, tanimoto_model = _fit_tanimoto(
            features["morgan"][fit],
            labels[fit],
            features["morgan"][target],
            settings["tanimoto_svc"],
        )
        probabilities["tanimoto_svc"] = tanimoto_probability
        models["tanimoto_svc"] = tanimoto_model
    if "paper_svm" in requested:
        paper_svm = SVC(
            kernel=settings["paper_svm"]["kernel"],
            C=float(settings["paper_svm"]["C"]),
            gamma=float(settings["paper_svm"]["gamma"]),
            probability=bool(settings["paper_svm"]["probability"]),
            random_state=int(settings["paper_svm"]["random_state"]),
        )
        paper_svm.fit(features["paper"][fit], labels[fit])
        probabilities["paper_svm"] = np.asarray(
            paper_svm.predict_proba(features["paper"][target])[:, 1], dtype=float
        )
        models["paper_svm"] = paper_svm
    if "tabpfn_v2" in requested:
        assert train_desc is not None and target_desc is not None
        tabpfn_probability, tabpfn_audit = _tabpfn_probability(
            train_desc,
            labels[fit],
            target_desc,
            settings["tabpfn_v2"],
            seed,
        )
        probabilities["tabpfn_v2"] = tabpfn_probability
        audit["tabpfn"] = tabpfn_audit
    if any(
        len(values) != len(target)
        or not np.isfinite(values).all()
        or ((values < 0) | (values > 1)).any()
        for values in probabilities.values()
    ):
        raise FixedBlendPaper405Error("A blend component returned invalid probabilities")
    return (
        probabilities,
        models,
        audit,
    )


def _blends(
    component_probability: dict[str, np.ndarray], protocol: dict[str, Any]
) -> dict[str, np.ndarray]:
    output = {}
    for name, specification in protocol["blends"].items():
        matrix = np.stack(
            [component_probability[component] for component in specification["components"]]
        )
        output[name] = np.average(
            matrix, axis=0, weights=np.asarray(specification["weights"], dtype=float)
        )
    return output


def run(
    *,
    root: Path,
    config_path: Path,
    positive_path: Path,
    negative_path: Path,
    run_id: str,
) -> Path:
    if not re.fullmatch(r"fixedblend405_[a-z0-9_.-]+", run_id):
        raise FixedBlendPaper405Error("RUN_ID must start with fixedblend405_")
    root = root.resolve()
    validate_core_runtime(root / "requirements-lock.txt")
    protocol, protocol_hash = load_protocol(config_path)
    source_hash = source_tree_sha256(root)
    traditional = _paper_contract(root, protocol)
    frame, source_audit = _read_sources(positive_path, negative_path, traditional)
    train_indices, test_indices, split_hash = paper_split_indices(traditional)
    smiles, smiles_audit = _validated_raw_smiles(frame)
    features = _features(frame, smiles, protocol)
    labels = frame["label"].to_numpy(dtype=int)
    settings = protocol["components"]
    folds = StratifiedKFold(
        n_splits=int(settings["cross_fitted_oof_folds"]),
        shuffle=True,
        random_state=int(settings["cross_fitted_oof_seed"]),
    )
    oof = {name: np.full(len(train_indices), np.nan) for name in COMPONENTS}
    fold_audits = []
    for fold, (relative_fit, relative_validation) in enumerate(
        folds.split(train_indices, labels[train_indices])
    ):
        fit = train_indices[relative_fit]
        validation = train_indices[relative_validation]
        print(f"TabPFN/blend OOF fold {fold + 1}/5", flush=True)
        probabilities, _models, audit = _component_predictions(
            features, labels, fit, validation, settings, seed=42 + fold
        )
        for name in COMPONENTS:
            oof[name][relative_validation] = probabilities[name]
        fold_audits.append(
            {
                "fold": fold,
                "fit_paper_indices": sorted(map(int, fit)),
                "validation_paper_indices": sorted(map(int, validation)),
                "audit": audit,
            }
        )
        del _models
    if any(not np.isfinite(values).all() for values in oof.values()):
        raise FixedBlendPaper405Error("Cross-fitted component OOF predictions are incomplete")
    oof_all = {**oof, **_blends(oof, protocol)}
    thresholds = {
        name: select_mcc_threshold(labels[train_indices], probability)[0]
        for name, probability in oof_all.items()
    }
    print("Final component refit on all 324 paper-training rows", flush=True)
    test_components, final_models, final_audit = _component_predictions(
        features, labels, train_indices, test_indices, settings, seed=42
    )
    test_all = {**test_components, **_blends(test_components, protocol)}
    output_root = root / "outputs"
    output_root.mkdir(exist_ok=True)
    destination = output_root / run_id
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing fixed-blend run: {destination}")
    work = Path(tempfile.mkdtemp(prefix=f".{run_id}.work-", dir=output_root))
    (work / "models").mkdir()
    metric_rows, prediction_frames, oof_frames = [], [], []
    for name in (*COMPONENTS, *BLENDS):
        probability = test_all[name]
        threshold = thresholds[name]
        metric_rows.append(
            {
                "model_id": name,
                **evaluation_metrics(
                    labels[test_indices], probability, probability, threshold=threshold
                ),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "model_id": name,
                    "paper_row_index": test_indices,
                    "compound_name": frame.iloc[test_indices]["compound_name"].to_numpy(),
                    "label": labels[test_indices],
                    "probability": probability,
                    "threshold": threshold,
                    "decision": (probability >= threshold).astype(int),
                }
            )
        )
        oof_frames.append(
            pd.DataFrame(
                {
                    "model_id": name,
                    "paper_row_index": train_indices,
                    "label": labels[train_indices],
                    "oof_probability": oof_all[name],
                    "selected_threshold": threshold,
                }
            )
        )
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    oof_predictions = pd.concat(oof_frames, ignore_index=True)
    metrics.to_csv(work / "metrics.csv", index=False, lineterminator="\n")
    predictions.to_csv(work / "predictions.csv", index=False, lineterminator="\n")
    oof_predictions.to_csv(work / "train_oof_predictions.csv", index=False, lineterminator="\n")
    model_hashes = {}
    for name, model in final_models.items():
        path = work / "models" / f"{name}.joblib"
        joblib.dump(model, path, compress=3)
        model_hashes[name] = sha256_file(path)
    split_frame = frame[
        ["paper_row_index", "compound_name", "smiles", "label", "source_role"]
    ].copy()
    split_frame["role"] = "paper_train"
    split_frame.loc[split_frame.paper_row_index.isin(test_indices), "role"] = "paper_test"
    split_frame.to_csv(work / "split_registry.csv", index=False, lineterminator="\n")
    audit = {
        **source_audit,
        **smiles_audit,
        "paper_split_sha256": split_hash,
        "paper_train_rows": len(train_indices),
        "paper_test_rows": len(test_indices),
        "cross_split_overlap_audit": _cross_split_audit(frame, train_indices, test_indices),
        "oof_fold_lineage_sha256": canonical_sha256(fold_audits),
        "oof_fold_audits": fold_audits,
        "final_component_audit": final_audit,
        "blend_weights_learned_from_test": False,
        "thresholds_selected_from_train_oof_only": True,
        "hagr_loaded": False,
    }
    atomic_write_json(work / "audit.json", audit)
    lines = [
        f"# Fixed blend paper-405 — {run_id}",
        "",
        "Exact raw 405-row paper split; all blend weights fixed before this run.",
        "Thresholds use five-fold OOF predictions from the 324 training rows only.",
        "",
        "| Model | AUPRC+ | AUROC | MCC | Macro F1 | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.model_id} | {row.auprc_average_precision_positive:.4f} | "
            f"{row.auroc:.4f} | {row.mcc:.4f} | {row.macro_f1:.4f} | {row.accuracy:.4f} |"
        )
    (work / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if source_tree_sha256(root) != source_hash:
        raise FixedBlendPaper405Error("Source tree changed during fixed-blend run")
    manifest = {
        "schema_version": "geroprotector.fixed_blend_paper405.run.v1",
        "run_id": run_id,
        "protocol_sha256": protocol_hash,
        "source_tree_sha256": source_hash,
        "runtime_environment": runtime_environment(),
        "model_hashes": model_hashes,
        "tabpfn_checkpoint_sha256": settings["tabpfn_v2"]["checkpoint_sha256"],
        "metrics_sha256": sha256_file(work / "metrics.csv"),
        "predictions_sha256": sha256_file(work / "predictions.csv"),
        "train_oof_predictions_sha256": sha256_file(work / "train_oof_predictions.csv"),
        "audit_sha256": sha256_file(work / "audit.json"),
        "outer_test_used_for_model_weight_or_threshold_selection": False,
    }
    atomic_write_json(work / "run_manifest.json", manifest)
    atomic_write_json(
        work / "COMPLETED.json",
        {
            "status": "COMPLETE",
            "run_id": run_id,
            "run_manifest_sha256": sha256_file(work / "run_manifest.json"),
            "metrics_sha256": manifest["metrics_sha256"],
            "predictions_sha256": manifest["predictions_sha256"],
        },
    )
    os.rename(work, destination)
    print(f"Complete: {destination}", flush=True)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--positive", required=True)
    parser.add_argument("--negative", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    run(
        root=Path(args.root),
        config_path=_regular_file(Path(args.config), "protocol"),
        positive_path=_regular_file(Path(args.positive), "positive source"),
        negative_path=_regular_file(Path(args.negative), "negative source"),
        run_id=args.run_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
