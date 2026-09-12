"""HyperMolTab variants on the publication's exact raw 405-row holdout."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from geroprotector.audit import runtime_environment, source_tree_sha256, validate_core_runtime
from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.hypermoltab_core import (
    HyperMolTabArtifact,
    fit_model,
    predict_model,
    variant_config,
)
from geroprotector.traditional_paper405 import (
    _cross_split_audit,
    _read_sources,
    evaluation_metrics,
    paper_split_indices,
)


class HyperMolTabPaper405Error(RuntimeError):
    pass


VARIANTS = (
    "hyper_moltab",
    "hyper_moltab_no_graph",
    "hyper_moltab_no_tabm",
    "hyper_moltab_no_kan",
    "hyper_moltab_no_tree",
    "hyper_moltab_no_cl",
    "hyper_moltab_no_graph_no_cl",
    "hyper_moltab_distill_rank",
)


def _regular_file(path: Path, role: str) -> Path:
    lexical = Path(path)
    if lexical.is_symlink() or not lexical.is_file():
        raise HyperMolTabPaper405Error(f"{role} must be a regular non-symlink file")
    return lexical.resolve()


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = yaml.safe_load(_regular_file(path, "protocol").read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise HyperMolTabPaper405Error("HyperMolTab protocol must be a mapping")
    if tuple(protocol.get("variants", ())) != VARIANTS:
        raise HyperMolTabPaper405Error("HyperMolTab variant set/order differs from the lock")
    training = protocol.get("training", {})
    schema = protocol.get("schema_version")
    if schema == "geroprotector.hypermoltab_paper405.protocol.v1":
        locked = {
            "loss": "focal",
            "focal_alpha": 1.0,
            "focal_gamma": 1.0,
            "learning_rate": 4e-4,
            "epochs": 180,
            "patience": 25,
        }
    elif schema == "geroprotector.hypermoltab_paper405.tuned_protocol.v2":
        locked = {
            "loss": "focal",
            "focal_alpha": 1.0,
            "focal_gamma": 1.0,
            "learning_rate": 1e-3,
            "epochs": 100,
            "patience": 20,
            "lr_step_size": 10,
            "lr_scheduler_gamma": 0.5,
        }
        search = protocol.get("focal_search", {})
        if search != {
            "enabled": True,
            "alpha_positive_grid": [0.75, 1.0, 1.25],
            "gamma_grid": [0.5, 1.0, 2.0],
            "selection_metric": "auprc_average_precision_positive",
            "tie_break": "mcc_then_closest_to_alpha1_gamma1",
            "outer_test_consulted": False,
        }:
            raise HyperMolTabPaper405Error("Focal alpha/gamma search differs from v2 lock")
    else:
        raise HyperMolTabPaper405Error("Unknown HyperMolTab paper405 protocol schema")
    for key, expected in locked.items():
        if training.get(key) != expected:
            raise HyperMolTabPaper405Error(
                f"Training setting {key}={training.get(key)!r} differs from {expected!r}"
            )
    observed_torch = str(torch.__version__).split("+", maxsplit=1)[0]
    if observed_torch != str(training.get("required_torch_version")):
        raise HyperMolTabPaper405Error(
            f"Imported torch {observed_torch} differs from the locked "
            f"{training.get('required_torch_version')}"
        )
    evaluation = protocol.get("evaluation", {})
    if any(
        evaluation.get(key) is not False
        for key in (
            "outer_test_used_for_epoch_selection",
            "outer_test_used_for_threshold_selection",
            "outer_test_used_for_variant_selection",
            "hagr_loaded",
        )
    ):
        raise HyperMolTabPaper405Error("Outer-test/HAGR firewall is not locked false")
    return protocol, canonical_sha256(protocol)


def _focal_candidates(protocol: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    search = protocol.get("focal_search", {})
    if not search.get("enabled", False):
        training = protocol["training"]
        return ((float(training["focal_alpha"]), float(training["focal_gamma"])),)
    return tuple(
        (float(alpha), float(gamma))
        for alpha in search["alpha_positive_grid"]
        for gamma in search["gamma_grid"]
    )


def _paper_contract(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    traditional = yaml.safe_load(
        (root / "configs" / "traditional_paper405.yaml").read_text(encoding="utf-8")
    )
    sources = protocol["sources"]
    if (
        traditional["sources"]["positive"]["expected_sha256"] != sources["positive_sha256"]
        or traditional["sources"]["negative"]["expected_sha256"] != sources["negative_sha256"]
        or traditional["split"]["expected_assignment_sha256"]
        != protocol["paper_split"]["expected_assignment_sha256"]
        or traditional["features"] != protocol["features"]["paper_descriptors"]
    ):
        raise HyperMolTabPaper405Error(
            "HyperMolTab raw-data/split contract differs from the sealed traditional benchmark"
        )
    return traditional


def _validated_raw_smiles(frame: pd.DataFrame) -> tuple[list[str], dict]:
    output = []
    for row in frame.itertuples(index=False):
        value = str(row.smiles).strip()
        if Chem.MolFromSmiles(value) is None:
            raise HyperMolTabPaper405Error(
                f"RDKit cannot parse paper row {row.paper_row_index}: {row.compound_name}"
            )
        output.append(value)
    return output, {
        "smiles_repairs_applied": 0,
        "all_405_raw_smiles_rdkit_parseable_after_boundary_trim": True,
        "raw_smiles_preserved_in_split_registry": True,
    }


def _vectors(frame: pd.DataFrame, smiles: list[str], protocol: dict[str, Any]) -> np.ndarray:
    descriptor_names = protocol["features"]["paper_descriptors"]
    descriptors = frame[descriptor_names].to_numpy(dtype=np.float64)
    settings = protocol["features"]
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(settings["morgan_radius"]),
        fpSize=int(settings["morgan_bits"]),
        includeChirality=bool(settings["morgan_use_chirality"]),
    )
    fingerprints = np.zeros((len(smiles), int(settings["morgan_bits"])), dtype=np.float32)
    for index, value in enumerate(smiles):
        fingerprint = generator.GetFingerprint(Chem.MolFromSmiles(value))
        fingerprints[index] = np.asarray(fingerprint, dtype=np.float32)
    vectors = np.column_stack([descriptors, fingerprints]).astype(np.float32)
    if not np.isfinite(vectors).all():
        raise HyperMolTabPaper405Error("HyperMolTab feature vectors contain non-finite values")
    return vectors


def _teacher_estimator(settings: dict[str, Any], kind: str, seed: int):
    if kind == "extra_trees":
        return ExtraTreesClassifier(random_state=seed, **settings["extra_trees"])
    return XGBClassifier(random_state=seed, **settings["xgboost"])


def _teacher_columns(probability_one: np.ndarray, probability_two: np.ndarray) -> np.ndarray:
    first = np.clip(probability_one, 1e-6, 1 - 1e-6)
    second = np.clip(probability_two, 1e-6, 1 - 1e-6)
    return np.column_stack(
        [first, np.log(first / (1 - first)), second, np.log(second / (1 - second))]
    ).astype(np.float32)


def cross_fitted_teachers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_target: np.ndarray,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    folds = int(settings["folds"])
    seed = int(settings["random_state"])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    first_oof = np.full(len(y_train), np.nan, dtype=np.float64)
    second_oof = np.full(len(y_train), np.nan, dtype=np.float64)
    lineage = []
    for fold, (fit, validation) in enumerate(splitter.split(X_train, y_train)):
        first = _teacher_estimator(settings, "xgboost", seed + fold)
        second = _teacher_estimator(settings, "extra_trees", seed + fold)
        first.fit(X_train[fit], y_train[fit])
        second.fit(X_train[fit], y_train[fit])
        first_oof[validation] = first.predict_proba(X_train[validation])[:, 1]
        second_oof[validation] = second.predict_proba(X_train[validation])[:, 1]
        lineage.append(
            {
                "fold": fold,
                "fit_positions": sorted(map(int, fit)),
                "validation_positions": sorted(map(int, validation)),
            }
        )
    if not np.isfinite(first_oof).all() or not np.isfinite(second_oof).all():
        raise HyperMolTabPaper405Error("Tree-teacher OOF predictions are incomplete")
    first = _teacher_estimator(settings, "xgboost", seed)
    second = _teacher_estimator(settings, "extra_trees", seed)
    first.fit(X_train, y_train)
    second.fit(X_train, y_train)
    target_first = first.predict_proba(X_target)[:, 1]
    target_second = second.predict_proba(X_target)[:, 1]
    return (
        _teacher_columns(first_oof, second_oof),
        _teacher_columns(target_first, target_second),
        {
            "cross_fitted_training_predictions": True,
            "folds": folds,
            "lineage_sha256": canonical_sha256(lineage),
            "target_labels_read": False,
        },
    )


def select_mcc_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.r_[0.0, 0.5, 1.0, probabilities])
    rows = []
    for threshold in candidates:
        score = float(matthews_corrcoef(labels, probabilities >= threshold))
        rows.append((score, -abs(float(threshold) - 0.5), -float(threshold), float(threshold)))
    winner = max(rows)
    return winner[3], winner[0]


def _save_artifact(path: Path, artifact: HyperMolTabArtifact) -> None:
    torch.save(
        {
            "schema_version": "geroprotector.hypermoltab_paper405.model.v1",
            "config": artifact.config,
            "atom_dim": artifact.atom_dim,
            "vector_dim": artifact.vector_dim,
            "teacher_dim": artifact.teacher_dim,
            "state_dict": artifact.state_dict,
            "selected_epoch": artifact.selected_epoch,
            "seed": artifact.seed,
        },
        path,
    )


def _valid_completed_variant(directory: Path, protocol_hash: str) -> bool:
    marker = directory / "COMPLETED.json"
    if not marker.is_file() or marker.is_symlink():
        return False
    record = json.loads(marker.read_text(encoding="utf-8"))
    if record.get("protocol_sha256") != protocol_hash:
        raise HyperMolTabPaper405Error(f"Existing {directory.name} uses another protocol")
    for name in ("model.pt", "predictions.csv", "metrics.json", "audit.json"):
        path = directory / name
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != record[f"{name}_sha256"]
        ):
            raise HyperMolTabPaper405Error(f"Existing {directory.name}/{name} failed integrity")
    if "focal_search.csv_sha256" in record:
        search = directory / "focal_search.csv"
        if (
            not search.is_file()
            or search.is_symlink()
            or sha256_file(search) != record["focal_search.csv_sha256"]
        ):
            raise HyperMolTabPaper405Error(
                f"Existing {directory.name}/focal_search.csv failed integrity"
            )
    return True


def run(
    *,
    root: Path,
    config_path: Path,
    positive_path: Path,
    negative_path: Path,
    run_id: str,
) -> Path:
    if not re.fullmatch(r"hypermoltab405_[a-z0-9_.-]+", run_id):
        raise HyperMolTabPaper405Error("RUN_ID must start with hypermoltab405_")
    root = root.resolve()
    validate_core_runtime(root / "requirements-lock.txt")
    protocol, protocol_hash = load_protocol(config_path)
    traditional = _paper_contract(root, protocol)
    frame, source_audit = _read_sources(positive_path, negative_path, traditional)
    train_indices, test_indices, assignment_hash = paper_split_indices(traditional)
    if assignment_hash != protocol["paper_split"]["expected_assignment_sha256"]:
        raise HyperMolTabPaper405Error("Paper split hash differs from HyperMolTab protocol")
    smiles, repair_audit = _validated_raw_smiles(frame)
    raw_vectors = _vectors(frame, smiles, protocol)
    labels = frame["label"].to_numpy(dtype=int)
    selection = protocol["selection_split"]
    fit_indices, validation_indices = train_test_split(
        train_indices,
        test_size=float(selection["validation_fraction_of_paper_train"]),
        random_state=int(selection["random_state"]),
        stratify=labels[train_indices],
    )
    if set(fit_indices) & set(validation_indices) or set(train_indices) != set(
        fit_indices
    ) | set(validation_indices):
        raise HyperMolTabPaper405Error(
            "Internal selection split is not an exact train partition"
        )
    selection_scaler = StandardScaler().fit(raw_vectors[fit_indices])
    selection_fit = selection_scaler.transform(raw_vectors[fit_indices]).astype(np.float32)
    selection_validation = selection_scaler.transform(raw_vectors[validation_indices]).astype(
        np.float32
    )
    final_scaler = StandardScaler().fit(raw_vectors[train_indices])
    final_train = final_scaler.transform(raw_vectors[train_indices]).astype(np.float32)
    final_test = final_scaler.transform(raw_vectors[test_indices]).astype(np.float32)
    teacher_settings = protocol["teachers"]
    selection_teacher_fit, selection_teacher_validation, selection_teacher_audit = (
        cross_fitted_teachers(
            selection_fit, labels[fit_indices], selection_validation, teacher_settings
        )
    )
    final_teacher_train, final_teacher_test, final_teacher_audit = cross_fitted_teachers(
        final_train, labels[train_indices], final_test, teacher_settings
    )
    source_hash = source_tree_sha256(root)
    output = root / "outputs" / run_id
    output.mkdir(parents=True, exist_ok=True)
    running_path = output / "RUNNING.json"
    binding = {
        "schema_version": "geroprotector.hypermoltab_paper405.running.v1",
        "run_id": run_id,
        "protocol_sha256": protocol_hash,
        "source_tree_sha256": source_hash,
        "paper_split_sha256": assignment_hash,
    }
    if running_path.exists():
        if json.loads(running_path.read_text(encoding="utf-8")) != binding:
            raise HyperMolTabPaper405Error("Existing resumable run binding differs")
    else:
        atomic_write_json(running_path, binding)
    if (output / "COMPLETED.json").exists():
        print(f"Already complete: {output}", flush=True)
        return output

    settings = protocol["training"]
    variants_root = output / "variants"
    variants_root.mkdir(exist_ok=True)
    for position, variant in enumerate(VARIANTS, start=1):
        directory = variants_root / variant
        if _valid_completed_variant(directory, protocol_hash):
            print(f"[{position}/{len(VARIANTS)}] resume verified: {variant}", flush=True)
            continue
        if directory.exists():
            raise HyperMolTabPaper405Error(
                f"Partial unsealed variant directory needs inspection: {directory}"
            )
        variant_work = Path(tempfile.mkdtemp(prefix=f".{variant}.work-", dir=variants_root))
        base_config = variant_config(variant, settings)
        search_rows = []
        candidates = _focal_candidates(protocol)
        for candidate_index, (alpha, gamma) in enumerate(candidates, start=1):
            print(
                f"[{position}/{len(VARIANTS)}] {variant} focal search "
                f"{candidate_index}/{len(candidates)}: alpha={alpha}, gamma={gamma}",
                flush=True,
            )
            candidate_config = replace(
                base_config, focal_alpha=float(alpha), focal_gamma=float(gamma)
            )
            selection_artifact, validation_probability = fit_model(
                selection_fit,
                selection_teacher_fit,
                [smiles[index] for index in fit_indices],
                labels[fit_indices],
                X_validation=selection_validation,
                teacher_validation=selection_teacher_validation,
                smiles_validation=[smiles[index] for index in validation_indices],
                y_validation=labels[validation_indices],
                config=candidate_config,
                seed=int(settings["seed"]),
                requested_device=str(settings["device"]),
            )
            candidate_threshold, candidate_mcc = select_mcc_threshold(
                labels[validation_indices], validation_probability
            )
            search_rows.append(
                {
                    "alpha_positive": float(alpha),
                    "gamma": float(gamma),
                    "validation_auprc": float(
                        average_precision_score(
                            labels[validation_indices], validation_probability
                        )
                    ),
                    "validation_mcc": float(candidate_mcc),
                    "threshold": float(candidate_threshold),
                    "selected_epoch": int(selection_artifact.selected_epoch),
                    "fit_seed": int(settings["seed"]),
                    "outer_test_consulted": False,
                }
            )
            del selection_artifact
        winner = max(
            search_rows,
            key=lambda row: (
                row["validation_auprc"],
                row["validation_mcc"],
                -abs(row["alpha_positive"] - 1.0) - abs(row["gamma"] - 1.0),
                -row["alpha_positive"],
                -row["gamma"],
            ),
        )
        config = replace(
            base_config,
            focal_alpha=winner["alpha_positive"],
            focal_gamma=winner["gamma"],
        )
        threshold = float(winner["threshold"])
        validation_mcc = float(winner["validation_mcc"])
        selected_epoch = int(winner["selected_epoch"])
        pd.DataFrame(search_rows).to_csv(
            variant_work / "focal_search.csv", index=False, lineterminator="\n"
        )
        print(
            f"[{position}/{len(VARIANTS)}] final refit: {variant}; "
            f"epoch={selected_epoch}, threshold={threshold:.6f}",
            flush=True,
        )
        final_artifact, _ = fit_model(
            final_train,
            final_teacher_train,
            [smiles[index] for index in train_indices],
            labels[train_indices],
            config=config,
            seed=int(settings["seed"]),
            requested_device=str(settings["device"]),
            fixed_epochs=selected_epoch,
        )
        probability = predict_model(
            final_artifact,
            final_test,
            final_teacher_test,
            [smiles[index] for index in test_indices],
            requested_device=str(settings["device"]),
        )
        metrics = evaluation_metrics(
            labels[test_indices], probability, probability, threshold=threshold
        )
        predictions = pd.DataFrame(
            {
                "model_id": variant,
                "paper_row_index": test_indices.astype(int),
                "compound_name": frame.iloc[test_indices]["compound_name"].to_numpy(),
                "source_role": frame.iloc[test_indices]["source_role"].to_numpy(),
                "label": labels[test_indices],
                "probability": probability,
                "decision": (probability >= threshold).astype(int),
                "threshold_selected_on_internal_validation": threshold,
            }
        )
        _save_artifact(variant_work / "model.pt", final_artifact)
        predictions.to_csv(variant_work / "predictions.csv", index=False, lineterminator="\n")
        atomic_write_json(variant_work / "metrics.json", {"model_id": variant, **metrics})
        atomic_write_json(
            variant_work / "audit.json",
            {
                "model_id": variant,
                "selected_epoch": selected_epoch,
                "threshold": threshold,
                "internal_validation_mcc": validation_mcc,
                "loss": "focal",
                "focal_alpha": config.focal_alpha,
                "focal_gamma": config.focal_gamma,
                "learning_rate": config.learning_rate,
                "lr_step_size": config.lr_step_size,
                "lr_scheduler_gamma": config.lr_scheduler_gamma,
                "focal_search_candidates": len(search_rows),
                "focal_search_selected_on_internal_validation_only": True,
                "selection_fit_rows": len(fit_indices),
                "selection_validation_rows": len(validation_indices),
                "final_train_rows": len(train_indices),
                "outer_test_rows": len(test_indices),
                "outer_test_labels_read_before_final_prediction": False,
                "hagr_loaded": False,
            },
        )
        marker = {"protocol_sha256": protocol_hash}
        for name in (
            "model.pt",
            "predictions.csv",
            "metrics.json",
            "audit.json",
            "focal_search.csv",
        ):
            marker[f"{name}_sha256"] = sha256_file(variant_work / name)
        atomic_write_json(variant_work / "COMPLETED.json", marker)
        os.rename(variant_work, directory)
        print(f"[{position}/{len(VARIANTS)}] complete: {variant}", flush=True)

    if source_tree_sha256(root) != source_hash:
        raise HyperMolTabPaper405Error("Source tree changed during the resumable run")
    metric_rows, prediction_frames = [], []
    for variant in VARIANTS:
        directory = output / "variants" / variant
        if not _valid_completed_variant(directory, protocol_hash):
            raise HyperMolTabPaper405Error(f"Variant did not seal: {variant}")
        metric_rows.append(json.loads((directory / "metrics.json").read_text(encoding="utf-8")))
        prediction_frames.append(pd.read_csv(directory / "predictions.csv"))
    metrics_frame = pd.DataFrame(metric_rows)
    predictions_frame = pd.concat(prediction_frames, ignore_index=True)
    metrics_frame.to_csv(output / "metrics.csv", index=False, lineterminator="\n")
    predictions_frame.to_csv(output / "predictions.csv", index=False, lineterminator="\n")
    split_frame = frame[
        ["paper_row_index", "compound_name", "smiles", "label", "source_role"]
    ].copy()
    split_frame["role"] = "paper_train"
    split_frame.loc[split_frame.paper_row_index.isin(test_indices), "role"] = "paper_test"
    split_frame.loc[split_frame.paper_row_index.isin(validation_indices), "selection_role"] = (
        "validation"
    )
    split_frame.loc[split_frame.paper_row_index.isin(fit_indices), "selection_role"] = "fit"
    split_frame.to_csv(output / "split_registry.csv", index=False, lineterminator="\n")
    data_audit = {
        **source_audit,
        **repair_audit,
        "paper_split_sha256": assignment_hash,
        "paper_train_rows": len(train_indices),
        "paper_test_rows": len(test_indices),
        "cross_split_overlap_audit": _cross_split_audit(frame, train_indices, test_indices),
        "selection_teacher_audit": selection_teacher_audit,
        "final_teacher_audit": final_teacher_audit,
        "prior_predicted_candidates_loaded": False,
        "traditional_models_refit": False,
        "v5bis_models_refit": False,
        "hagr_loaded": False,
    }
    atomic_write_json(output / "data_audit.json", data_audit)
    lines = [
        f"# HyperMolTab paper-405 benchmark — {run_id}",
        "",
        "Exact raw 405-row paper split (324 train / 81 test); contextual, not external.",
        "All epoch and threshold choices used only an internal split of the 324 training rows.",
        "",
        "| Variant | AUPRC+ | AUROC | MCC | Macro F1 | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_frame.itertuples(index=False):
        lines.append(
            f"| {row.model_id} | {row.auprc_average_precision_positive:.4f} | "
            f"{row.auroc:.4f} | {row.mcc:.4f} | {row.macro_f1:.4f} | {row.accuracy:.4f} |"
        )
    lines.extend(
        [
            "",
            "Focal alpha/gamma and learning-rate settings are recorded in each sealed "
            "variant audit.",
            "Tree teachers are cross-fitted for training rows; test labels never enter "
            "fitting.",
            "Source role remains perfectly confounded with label; weak negatives are not "
            "confirmed negatives.",
            "Single training seed (42); no uncertainty claim is made.",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "schema_version": "geroprotector.hypermoltab_paper405.run.v1",
        "run_id": run_id,
        "protocol_sha256": protocol_hash,
        "source_tree_sha256": source_hash,
        "runtime_environment": runtime_environment(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "variants": list(VARIANTS),
        "loss": {
            "name": "focal",
            "alpha_semantics": "positive_class_weight; negative_class_weight_is_one",
            "search": protocol.get("focal_search", {"enabled": False}),
        },
        "learning_rate": float(settings["learning_rate"]),
        "maximum_epochs": int(settings["epochs"]),
        "early_stopping_patience": int(settings["patience"]),
        "lr_step_size": int(settings.get("lr_step_size", 0)),
        "lr_scheduler_gamma": float(settings.get("lr_scheduler_gamma", 1.0)),
        "traditional_output_modified": False,
        "v5bis_output_modified": False,
        "metrics_sha256": sha256_file(output / "metrics.csv"),
        "predictions_sha256": sha256_file(output / "predictions.csv"),
        "data_audit_sha256": sha256_file(output / "data_audit.json"),
    }
    atomic_write_json(output / "run_manifest.json", manifest)
    atomic_write_json(
        output / "COMPLETED.json",
        {
            "status": "COMPLETE",
            "run_id": run_id,
            "run_manifest_sha256": sha256_file(output / "run_manifest.json"),
            "metrics_sha256": manifest["metrics_sha256"],
            "predictions_sha256": manifest["predictions_sha256"],
        },
    )
    print(f"Complete: {output}", flush=True)
    return output


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
