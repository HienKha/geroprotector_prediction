"""Traditional-ML benchmark on the publication's exact 405-row D1 split."""

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
import yaml
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from geroprotector.audit import runtime_environment, source_tree_sha256, validate_core_runtime
from geroprotector.hashing import (
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)


class TraditionalPaper405Error(RuntimeError):
    """Raised when the exact-paper benchmark contract is violated."""


MODEL_ORDER = (
    "extra_trees",
    "random_forest",
    "xgboost",
    "catboost",
    "lightgbm",
    "logistic_regression",
    "linear_regression",
    "knn",
    "svm_original_paper",
)


def _regular_file(path: str | Path, *, role: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise TraditionalPaper405Error(
            f"{role} must be a regular non-symlink file: {candidate}"
        )
    if not candidate.is_absolute():
        candidate = candidate.resolve()
    if candidate.is_symlink() or not candidate.is_file():
        raise TraditionalPaper405Error(
            f"{role} must be a regular non-symlink file: {candidate}"
        )
    return candidate


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol_path = _regular_file(path, role="traditional protocol")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise TraditionalPaper405Error("Traditional protocol must be a mapping")
    required = {"schema_version", "sources", "features", "split", "evaluation", "models"}
    if required - set(protocol):
        raise TraditionalPaper405Error(
            f"Traditional protocol is missing keys: {sorted(required - set(protocol))}"
        )
    if tuple(protocol["models"]) != MODEL_ORDER:
        raise TraditionalPaper405Error(
            "Traditional model order/set differs from the locked suite"
        )
    if protocol["evaluation"] != {
        "decision_threshold": 0.5,
        "threshold_selected_on_test": False,
        "model_selection_on_test": False,
        "calibration": "none",
        "positive_class": 1,
        "auprc_definition": "sklearn_average_precision",
    }:
        raise TraditionalPaper405Error(
            "Evaluation contract differs from the locked test-only protocol"
        )
    return protocol, canonical_sha256(protocol)


def _read_sources(
    positive_path: Path,
    negative_path: Path,
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sources = protocol["sources"]
    observed_hashes = {
        "positive": sha256_file(positive_path),
        "negative": sha256_file(negative_path),
    }
    for role in ("positive", "negative"):
        if observed_hashes[role] != sources[role]["expected_sha256"]:
            raise TraditionalPaper405Error(
                f"{role} source SHA256 differs from the paper-pinned file"
            )
    positive = pd.read_csv(positive_path, sep="\t", encoding="latin-1")
    negative = pd.read_csv(negative_path, encoding="latin-1")
    if len(positive) != int(sources["positive"]["expected_rows"]):
        raise TraditionalPaper405Error("Positive source row count differs from 206")
    if len(negative) != int(sources["negative"]["expected_rows"]):
        raise TraditionalPaper405Error("Negative source row count differs from 199")
    positive_required = {"Compound Name", "Smiles", "Geroprotectors", *protocol["features"]}
    negative_required = {
        "compound_name",
        "canonical_smiles",
        "Geroprotectors",
        *protocol["features"],
    }
    if positive_required - set(positive) or negative_required - set(negative):
        raise TraditionalPaper405Error(
            "A paper source is missing required identity/descriptor columns"
        )
    if set(pd.to_numeric(positive["Geroprotectors"], errors="raise")) != {1}:
        raise TraditionalPaper405Error("Positive source labels are not exactly 1")
    if set(pd.to_numeric(negative["Geroprotectors"], errors="raise")) != {0}:
        raise TraditionalPaper405Error("Negative source labels are not exactly 0")

    metadata = pd.DataFrame(
        {
            "paper_row_index": np.arange(len(positive) + len(negative), dtype=int),
            "compound_name": pd.concat(
                [positive["Compound Name"], negative["compound_name"]], ignore_index=True
            ).astype(str),
            "smiles": pd.concat(
                [positive["Smiles"], negative["canonical_smiles"]], ignore_index=True
            ).astype(str),
            "label": np.r_[
                np.ones(len(positive), dtype=int), np.zeros(len(negative), dtype=int)
            ],
            "source_role": ["reported_positive"] * len(positive)
            + ["weak_reference_negative"] * len(negative),
            "paper_dataset": ["Geroprotectors"] * len(positive)
            + ["NoGeroprotectors"] * len(negative),
        }
    )
    feature_frame = pd.concat(
        [positive[protocol["features"]], negative[protocol["features"]]],
        ignore_index=True,
    ).apply(pd.to_numeric, errors="raise")
    values = feature_frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise TraditionalPaper405Error(
            "The seven paper descriptors contain missing/non-finite values"
        )
    frame = pd.concat([metadata, feature_frame], axis=1)
    expected_counts = {
        "total": int(sources["expected_total_rows"]),
        "positive": int(sources["expected_positive_rows"]),
        "negative": int(sources["expected_negative_rows"]),
    }
    observed_counts = {
        "total": len(frame),
        "positive": int(frame["label"].sum()),
        "negative": int((frame["label"] == 0).sum()),
    }
    if observed_counts != expected_counts:
        raise TraditionalPaper405Error(
            f"Paper D1 counts differ: observed {observed_counts}, expected {expected_counts}"
        )

    canonical_rows: list[dict[str, Any]] = []
    official_names = [
        "Name",
        "Total Molweight",
        "cLogP",
        "H-Acceptors",
        "H-Donors",
        "Total Surface Area",
        "Relative PSA",
        "Rotatable Bonds",
        "Geroprotectors",
        "DataSet",
        "Smile",
    ]
    for _, row in frame.iterrows():
        record: dict[str, Any] = {
            "row_index": int(row["paper_row_index"]),
            "Name": str(row["compound_name"]),
            "Geroprotectors": int(row["label"]),
            "DataSet": str(row["paper_dataset"]),
            "Smile": str(row["smiles"]),
        }
        for feature in protocol["features"]:
            record[feature] = float(row[feature])
        canonical_rows.append({key: record[key] for key in ["row_index", *official_names]})
    dataset_hash = canonical_sha256(
        {"schema": "paper405.seven_descriptor_rows.v1", "rows": canonical_rows}
    )
    if dataset_hash != sources["canonical_seven_descriptor_dataset_sha256"]:
        raise TraditionalPaper405Error(
            "Reconstructed seven-descriptor D1 differs from official Concatenadas.csv"
        )
    audit = {
        "source_sha256": observed_hashes,
        "observed_counts": observed_counts,
        "canonical_seven_descriptor_dataset_sha256": dataset_hash,
        "official_concatenated_csv_sha256_reference": protocol["source_paper"][
            "official_concatenated_csv_sha256"
        ],
    }
    return frame, audit


def paper_split_indices(protocol: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    split = protocol["split"]
    indices = np.arange(int(protocol["sources"]["expected_total_rows"]), dtype=int)
    train_indices, test_indices = train_test_split(
        indices,
        train_size=float(split["train_size"]),
        random_state=int(split["random_state"]),
        stratify=None,
    )
    assignment_hash = canonical_sha256(
        {"train": sorted(map(int, train_indices)), "test": sorted(map(int, test_indices))}
    )
    if assignment_hash != split["expected_assignment_sha256"]:
        raise TraditionalPaper405Error(
            "Generated sklearn split differs from the official split"
        )
    return np.asarray(train_indices), np.asarray(test_indices), assignment_hash


def _cross_split_audit(
    frame: pd.DataFrame, train: np.ndarray, test: np.ndarray
) -> dict[str, Any]:
    train_frame = frame.iloc[train]
    test_frame = frame.iloc[test]

    def normalize(value: object) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).casefold())

    train_names = {normalize(value) for value in train_frame["compound_name"]}
    test_names = {normalize(value) for value in test_frame["compound_name"]}
    train_smiles = {str(value).strip() for value in train_frame["smiles"]}
    test_smiles = {str(value).strip() for value in test_frame["smiles"]}
    descriptor_columns = [column for column in frame if column in set(frame.columns[6:])]
    train_descriptors = {
        tuple(float(value) for value in row)
        for row in train_frame[descriptor_columns].to_numpy()
    }
    test_descriptors = {
        tuple(float(value) for value in row)
        for row in test_frame[descriptor_columns].to_numpy()
    }
    return {
        "normalized_name_overlap_count": len(train_names & test_names),
        "raw_trimmed_smiles_overlap_count": len(train_smiles & test_smiles),
        "identical_seven_descriptor_vector_overlap_count": len(
            train_descriptors & test_descriptors
        ),
        "identity_resolution_before_split": False,
        "chemical_component_grouping": False,
        "interpretation": (
            "Original paper row split; overlaps are audited but intentionally retained."
        ),
    }


def _require_extension_version(package: str, expected: str) -> str:
    try:
        observed = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as exc:
        raise TraditionalPaper405Error(
            f"Missing {package}=={expected}; install requirements-traditional-lock.txt"
        ) from exc
    if observed != expected:
        raise TraditionalPaper405Error(f"{package} version {observed} != locked {expected}")
    return observed


def _models(protocol: dict[str, Any]) -> dict[str, Any]:
    settings = protocol["models"]
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "extra_trees": ExtraTreesClassifier(**settings["extra_trees"]),
        "random_forest": RandomForestClassifier(**settings["random_forest"]),
        "xgboost": XGBClassifier(**settings["xgboost"]),
        "catboost": CatBoostClassifier(
            **{
                key: value
                for key, value in settings["catboost"].items()
                if key != "required_version"
            },
            verbose=False,
            allow_writing_files=False,
        ),
        "lightgbm": LGBMClassifier(
            **{
                key: value
                for key, value in settings["lightgbm"].items()
                if key != "required_version"
            },
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        ),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(settings["logistic_regression"]["C"]),
                        solver=settings["logistic_regression"]["solver"],
                        max_iter=int(settings["logistic_regression"]["max_iter"]),
                        random_state=int(settings["logistic_regression"]["random_state"]),
                    ),
                ),
            ]
        ),
        "linear_regression": LinearRegression(),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=int(settings["knn"]["n_neighbors"]),
                        weights=settings["knn"]["weights"],
                        metric=settings["knn"]["metric"],
                        p=int(settings["knn"]["p"]),
                    ),
                ),
            ]
        ),
        "svm_original_paper": SVC(
            kernel=settings["svm_original_paper"]["kernel"],
            C=float(settings["svm_original_paper"]["C"]),
            gamma=float(settings["svm_original_paper"]["gamma"]),
            probability=bool(settings["svm_original_paper"]["probability"]),
            random_state=int(settings["svm_original_paper"]["random_state"]),
        ),
    }


def _positive_score(
    model: Any, X: np.ndarray, *, model_id: str
) -> tuple[np.ndarray, np.ndarray]:
    if model_id == "linear_regression":
        raw = np.asarray(model.predict(X), dtype=float)
        return raw, np.clip(raw, 0.0, 1.0)
    if not hasattr(model, "predict_proba"):
        raise TraditionalPaper405Error(f"{model_id} does not expose predict_proba")
    classes = np.asarray(model.classes_)
    if classes.shape != (2,) or set(map(int, classes)) != {0, 1}:
        raise TraditionalPaper405Error(f"{model_id} has invalid class order {classes.tolist()}")
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.shape != (len(X), 2) or not np.isfinite(probabilities).all():
        raise TraditionalPaper405Error(f"{model_id} returned invalid probabilities")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise TraditionalPaper405Error(f"{model_id} returned out-of-range probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-7):
        raise TraditionalPaper405Error(f"{model_id} probability rows do not sum to one")
    positive_column = int(np.flatnonzero(classes == 1)[0])
    score = probabilities[:, positive_column]
    return score, score.copy()


def evaluation_metrics(
    labels: np.ndarray,
    ranking_score: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    score = np.asarray(ranking_score, dtype=float)
    p = np.asarray(probability, dtype=float)
    if len(y) != len(score) or len(y) != len(p) or set(y) != {0, 1}:
        raise TraditionalPaper405Error("Evaluation requires aligned two-class test data")
    if (
        not np.isfinite(score).all()
        or not np.isfinite(p).all()
        or (p < 0).any()
        or (p > 1).any()
    ):
        raise TraditionalPaper405Error("Evaluation scores are invalid")
    decision = (score >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, decision, labels=[0, 1]).ravel()
    precision_curve, recall_curve, _ = precision_recall_curve(y, score)
    ap_positive = float(average_precision_score(y, score))
    ap_negative = float(average_precision_score(1 - y, -score))
    return {
        "n_test": len(y),
        "positive_prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, score)),
        "auprc_average_precision_positive": ap_positive,
        "auprc_average_precision_negative": ap_negative,
        "auprc_macro": float((ap_positive + ap_negative) / 2.0),
        "pr_auc_trapezoidal_positive": float(auc(recall_curve, precision_curve)),
        "accuracy": float(accuracy_score(y, decision)),
        "balanced_accuracy": float(balanced_accuracy_score(y, decision)),
        "mcc": float(matthews_corrcoef(y, decision)),
        "macro_f1": float(f1_score(y, decision, average="macro", zero_division=0)),
        "f1_positive": float(f1_score(y, decision, pos_label=1, zero_division=0)),
        "precision_positive": float(precision_score(y, decision, pos_label=1, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, decision, pos_label=1, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "npv": float(tn / (tn + fn)) if tn + fn else None,
        "cohen_kappa": float(cohen_kappa_score(y, decision)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1])),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def _summary_markdown(run_id: str, metrics: pd.DataFrame, data_audit: dict[str, Any]) -> str:
    lines = [
        f"# Traditional paper-405 benchmark — {run_id}",
        "",
        "Contextual internal benchmark on the publication's exact 405 raw rows and "
        "one-time random 80/20 split. Not external or headline evidence.",
        "",
        "| Model | AUPRC+ | AUROC | MCC | Macro F1 | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.model_id} | {row.auprc_average_precision_positive:.4f} | "
            f"{row.auroc:.4f} | {row.mcc:.4f} | {row.macro_f1:.4f} | "
            f"{row.accuracy:.4f} |"
        )
    overlap = data_audit["cross_split_overlap_audit"]
    lines.extend(
        [
            "",
            "## Locked design",
            "",
            "- D1: 206 reported-positive + 199 weak-reference-negative raw rows.",
            "- Split: 324 train / 81 test, random_state=42, no stratification.",
            "- Seven DataWarrior descriptors; threshold fixed at 0.5; no test tuning.",
            f"- Exact trimmed-SMILES train/test overlaps: "
            f"{overlap['raw_trimmed_smiles_overlap_count']}.",
            "- Labels remain perfectly source-confounded; weak references are not "
            "confirmed negatives.",
            "- SVM follows the executable public notebook (unscaled linear SVC); KNN uses "
            "the manuscript-fixed k=27 with train-only scaling.",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark(
    *,
    root: Path,
    protocol_path: Path,
    positive_path: Path,
    negative_path: Path,
    run_id: str,
) -> Path:
    if not re.fullmatch(r"traditional405_[a-z0-9_.-]+", run_id):
        raise TraditionalPaper405Error(
            "RUN_ID must start with traditional405_ and contain lowercase safe characters"
        )
    project = root.resolve()
    protocol_path = _regular_file(protocol_path, role="traditional protocol")
    positive_path = _regular_file(positive_path, role="positive source")
    negative_path = _regular_file(negative_path, role="negative source")
    validate_core_runtime(project / "requirements-lock.txt")
    protocol, protocol_hash = _load_protocol(protocol_path)
    extension_versions = {
        "catboost": _require_extension_version(
            "catboost", str(protocol["models"]["catboost"]["required_version"])
        ),
        "lightgbm": _require_extension_version(
            "lightgbm", str(protocol["models"]["lightgbm"]["required_version"])
        ),
    }
    source_hash_at_start = source_tree_sha256(project)
    frame, source_audit = _read_sources(positive_path, negative_path, protocol)
    train_indices, test_indices, assignment_hash = paper_split_indices(protocol)
    split = protocol["split"]
    observed_split_counts = {
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "train_class_0": int((frame.iloc[train_indices]["label"] == 0).sum()),
        "train_class_1": int((frame.iloc[train_indices]["label"] == 1).sum()),
        "test_class_0": int((frame.iloc[test_indices]["label"] == 0).sum()),
        "test_class_1": int((frame.iloc[test_indices]["label"] == 1).sum()),
    }
    expected_split_counts = {
        key: int(split[f"expected_{key}"]) for key in observed_split_counts
    }
    if observed_split_counts != expected_split_counts:
        raise TraditionalPaper405Error(
            f"Official split class counts differ: {observed_split_counts} != "
            f"{expected_split_counts}"
        )

    output_root = project / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / run_id
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {destination}")
    work = Path(tempfile.mkdtemp(prefix=f".{run_id}.work-", dir=output_root))
    (work / "models").mkdir()
    features = list(protocol["features"])
    X = frame[features].to_numpy(dtype=np.float64)
    y = frame["label"].to_numpy(dtype=int)
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    model_objects = _models(protocol)
    if tuple(model_objects) != MODEL_ORDER:
        raise TraditionalPaper405Error("Constructed model suite differs from locked order")
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    model_artifacts: dict[str, Any] = {}
    threshold = float(protocol["evaluation"]["decision_threshold"])
    for model_id, model in model_objects.items():
        model.fit(X_train, y_train)
        ranking_score, probability = _positive_score(model, X_test, model_id=model_id)
        metrics = evaluation_metrics(y_test, ranking_score, probability, threshold=threshold)
        metric_rows.append({"model_id": model_id, **metrics})
        decision = (ranking_score >= threshold).astype(int)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "model_id": model_id,
                    "paper_row_index": test_indices.astype(int),
                    "compound_name": frame.iloc[test_indices]["compound_name"].to_numpy(),
                    "source_role": frame.iloc[test_indices]["source_role"].to_numpy(),
                    "label": y_test,
                    "ranking_score": ranking_score,
                    "probability_for_proper_scores": probability,
                    "decision": decision,
                    "threshold": threshold,
                }
            )
        )
        artifact = work / "models" / f"{model_id}.joblib"
        joblib.dump(model, artifact, compress=3)
        model_artifacts[model_id] = {
            "path": artifact.relative_to(work).as_posix(),
            "sha256": sha256_file(artifact),
        }

    metrics_frame = pd.DataFrame(metric_rows)
    predictions_frame = pd.concat(prediction_frames, ignore_index=True)
    split_frame = frame[
        ["paper_row_index", "compound_name", "smiles", "label", "source_role"]
    ].copy()
    split_frame["role"] = "train"
    split_frame.loc[split_frame["paper_row_index"].isin(test_indices), "role"] = "test"
    split_frame["random_state"] = int(split["random_state"])
    _write_dataframe(work / "metrics.csv", metrics_frame)
    _write_dataframe(work / "predictions.csv", predictions_frame)
    _write_dataframe(work / "split_registry.csv", split_frame)
    data_audit = {
        **source_audit,
        "split_assignment_sha256": assignment_hash,
        "split_counts": observed_split_counts,
        "cross_split_overlap_audit": _cross_split_audit(frame, train_indices, test_indices),
        "prior_predicted_candidates_loaded": False,
        "hagr_loaded": False,
        "outer_test_used_for_model_or_threshold_selection": False,
    }
    atomic_write_json(work / "data_audit.json", data_audit)
    atomic_write_json(work / "model_settings.json", protocol["models"])
    (work / "summary.md").write_text(
        _summary_markdown(run_id, metrics_frame, data_audit), encoding="utf-8"
    )
    if source_tree_sha256(project) != source_hash_at_start:
        raise TraditionalPaper405Error("Source tree changed while the benchmark was running")
    artifact_entries = []
    for path in sorted(work.rglob("*")):
        if path.is_file() and path.name not in {"run_manifest.json", "COMPLETED.json"}:
            artifact_entries.append(
                {
                    "path": path.relative_to(work).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    run_manifest = {
        "schema_version": "geroprotector.traditional_paper405.run.v1",
        "run_id": run_id,
        "protocol_sha256": protocol_hash,
        "source_tree_sha256": source_hash_at_start,
        "runtime_environment": runtime_environment(),
        "extension_package_versions": extension_versions,
        "model_order": list(MODEL_ORDER),
        "model_artifacts": model_artifacts,
        "data_audit_sha256": sha256_file(work / "data_audit.json"),
        "metrics_sha256": sha256_file(work / "metrics.csv"),
        "predictions_sha256": sha256_file(work / "predictions.csv"),
        "split_registry_sha256": sha256_file(work / "split_registry.csv"),
        "artifact_inventory": artifact_entries,
        "hagr_loaded": False,
        "outer_test_used_for_selection": False,
        "headline_eligible": False,
    }
    atomic_write_json(work / "run_manifest.json", run_manifest)
    completed = {
        "schema_version": "geroprotector.traditional_paper405.completed.v1",
        "run_id": run_id,
        "run_manifest_sha256": sha256_file(work / "run_manifest.json"),
        "metrics_sha256": run_manifest["metrics_sha256"],
        "predictions_sha256": run_manifest["predictions_sha256"],
        "status": "COMPLETE",
    }
    atomic_write_json(work / "COMPLETED.json", completed)
    os.rename(work, destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--positive", required=True)
    parser.add_argument("--negative", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    output = run_benchmark(
        root=root,
        protocol_path=_regular_file(Path(args.config).resolve(), role="protocol"),
        positive_path=_regular_file(Path(args.positive).resolve(), role="positive source"),
        negative_path=_regular_file(Path(args.negative).resolve(), role="negative source"),
        run_id=args.run_id,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
