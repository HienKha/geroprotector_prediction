"""Post-lock internal reporting; no model is fitted in this module."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

from .audit import source_tree_sha256, validate_core_runtime
from .chemistry.descriptors import descriptor_frame
from .chemistry.fingerprints import morgan_bitvectors
from .config import resolve_config, resolved_config_sha256, validate_protocol_lock
from .data.curate import load_curated_cohort
from .hashing import atomic_write_bytes, atomic_write_json, canonical_sha256, sha256_file
from .logging import utc_now
from .validation.bootstrap import paired_component_bootstrap
from .validation.metrics import aggregate_repeated_oof, metric_bundle
from .validation.nested_cv import verify_completed_run


class ReportingError(RuntimeError):
    pass


def _schema(root: Path, name: str) -> dict:
    path = root / "schemas" / name
    if path.is_symlink() or not path.is_file():
        raise ReportingError(f"Schema is unavailable: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _threshold_metrics(labels: np.ndarray, decisions: np.ndarray) -> dict[str, float | None]:
    tn, fp, fn, tp = confusion_matrix(labels, decisions, labels=[0, 1]).ravel()
    return {
        "mcc": float(matthews_corrcoef(labels, decisions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, decisions)),
        "macro_f1": float(f1_score(labels, decisions, average="macro", zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "npv": float(tn / (tn + fn)) if tn + fn else None,
    }


def _screening_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    from rdkit.ML.Scoring.Scoring import CalcBEDROC

    order = np.argsort(-scores, kind="stable")
    prevalence = float(labels.mean())
    output: dict[str, float | None] = {}
    for k in (10, 20, 50):
        selected = labels[order[: min(k, len(labels))]]
        output[f"precision_at_{k}"] = float(selected.mean()) if len(selected) else None
        output[f"recall_at_{k}"] = (
            float(selected.sum() / labels.sum()) if labels.sum() else None
        )
    for fraction, label in ((0.01, "1pct"), (0.05, "5pct"), (0.10, "10pct")):
        count = max(1, int(np.ceil(len(labels) * fraction)))
        precision = float(labels[order[:count]].mean())
        output[f"enrichment_top_{label}"] = (
            float(precision / prevalence) if prevalence > 0 else None
        )
    ranked = [[int(labels[position])] for position in order]
    output["bedroc_alpha_20"] = float(CalcBEDROC(ranked, 0, 20.0))
    return output


def _reliability_bins(
    labels: np.ndarray, scores: np.ndarray, *, adaptive: bool, bins: int = 10
) -> list[dict[str, Any]]:
    if adaptive:
        partitions = np.array_split(np.argsort(scores, kind="stable"), min(bins, len(scores)))
    else:
        edges = np.linspace(0.0, 1.0, bins + 1)
        membership = np.clip(np.digitize(scores, edges[1:-1]), 0, bins - 1)
        partitions = [np.flatnonzero(membership == index) for index in range(bins)]
    rows = []
    for index, positions in enumerate(partitions):
        if not len(positions):
            continue
        rows.append(
            {
                "bin": index,
                "n": len(positions),
                "mean_probability": float(scores[positions].mean()),
                "observed_fraction": float(labels[positions].mean()),
            }
        )
    return rows


def _reliability_svg(curves: Mapping[str, list[dict[str, Any]]]) -> bytes:
    width, height, margin = 720, 560, 70
    plot_width, plot_height = width - 2 * margin, height - 2 * margin
    palette = ("#006BA4", "#FF800E", "#ABABAB", "#595959", "#5F9ED1", "#C85200")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" '
        f'y2="{margin}" stroke="#888" stroke-dasharray="5,5"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" '
        'stroke="black"/>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" '
        f'y2="{height - margin}" stroke="black"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" '
        'font-family="sans-serif">Mean predicted probability</text>',
        f'<text x="18" y="{height / 2}" text-anchor="middle" font-family="sans-serif" '
        'transform="rotate(-90 18,280)">Observed positive fraction</text>',
    ]
    for index, (model_id, points) in enumerate(sorted(curves.items())):
        color = palette[index % len(palette)]
        coordinates = " ".join(
            f"{margin + point['mean_probability'] * plot_width:.2f},"
            f"{height - margin - point['observed_fraction'] * plot_height:.2f}"
            for point in points
        )
        if coordinates:
            lines.append(
                f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
                'stroke-width="2"/>'
            )
        legend_y = 24 + index * 18
        lines.append(
            f'<text x="{margin}" y="{legend_y}" fill="{color}" '
            f'font-family="sans-serif" font-size="12">{escape(model_id)}</text>'
        )
    lines.append("</svg>")
    return ("\n".join(lines) + "\n").encode()


def _risk_coverage(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
    confidence = np.abs(scores - 0.5)
    order = np.argsort(-confidence, kind="stable")
    rows = []
    for coverage in (0.25, 0.50, 0.75, 0.90, 1.0):
        count = max(2, int(np.ceil(len(labels) * coverage)))
        selected = order[:count]
        if set(labels[selected]) != {0, 1}:
            rows.append({"coverage": coverage, "n": count, "metrics": None})
        else:
            rows.append(
                {
                    "coverage": coverage,
                    "n": count,
                    "metrics": metric_bundle(labels[selected], scores[selected]),
                }
            )
    return rows


def _source_confounding_audit(
    root: Path,
    run_manifest: Mapping[str, Any],
    aggregated: pd.DataFrame,
    metric_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Quantify source chemistry after locking predictions; never select a model."""

    from rdkit import DataStructs

    config = resolve_config(root / "configs" / "data.yaml")
    validate_protocol_lock(root=root, contract_name="data", config=config)
    outputs = config["outputs"]
    curated_path = root / str(outputs["curated_table"])
    provenance_path = root / str(outputs["provenance_table"])
    manifest_path = root / str(outputs["manifest"])
    curated, provenance, _manifest = load_curated_cohort(
        curated_table=curated_path,
        provenance_table=provenance_path,
        manifest_path=manifest_path,
        resolved_config_sha256=resolved_config_sha256(config),
    )
    if sha256_file(manifest_path) != run_manifest["data"]["curation_manifest_sha256"]:
        raise ReportingError("Source audit data differs from the training curation manifest")
    source_label = curated.groupby("source_role", sort=True)["label"].agg(["min", "max"])
    if len(source_label) != 2 or not source_label["min"].eq(source_label["max"]).all():
        raise ReportingError("Source/label relation is no longer the locked perfect bijection")
    if set(source_label["min"].astype(int)) != {0, 1}:
        raise ReportingError("Source roles no longer map one-to-one onto the binary endpoint")

    descriptors = descriptor_frame(
        curated["standardized_parent_smiles"],
        panel="chemistry_32",
        index=curated["compound_id"],
    )
    descriptor_names = (
        "MolWt",
        "MolLogP",
        "TPSA",
        "NumHDonors",
        "NumHAcceptors",
        "RingCount",
        "FractionCSP3",
    )
    labels_by_id = curated.set_index("compound_id")["label"].astype(int)
    distribution_rows: list[dict[str, Any]] = []
    for name in descriptor_names:
        negative = descriptors.loc[labels_by_id.index[labels_by_id.eq(0)], name].dropna()
        positive = descriptors.loc[labels_by_id.index[labels_by_id.eq(1)], name].dropna()
        pooled_sd = float(np.sqrt((negative.var(ddof=1) + positive.var(ddof=1)) / 2.0))
        ks = ks_2samp(positive, negative, alternative="two-sided", method="auto")
        distribution_rows.append(
            {
                "descriptor": name,
                "role": (
                    "natural_product_likeness_proxy" if name == "FractionCSP3" else "property"
                ),
                "positive_n": len(positive),
                "weak_reference_n": len(negative),
                "positive_median": float(positive.median()),
                "weak_reference_median": float(negative.median()),
                "positive_q1": float(positive.quantile(0.25)),
                "positive_q3": float(positive.quantile(0.75)),
                "weak_reference_q1": float(negative.quantile(0.25)),
                "weak_reference_q3": float(negative.quantile(0.75)),
                "standardized_mean_difference": (
                    float((positive.mean() - negative.mean()) / pooled_sd)
                    if pooled_sd > 0
                    else None
                ),
                "ks_statistic": float(ks.statistic),
                "ks_p_value_descriptive_only": float(ks.pvalue),
            }
        )
    distributions = pd.DataFrame(distribution_rows)

    ids = tuple(curated["compound_id"].astype(str))
    sources = tuple(curated["source_role"].astype(str))
    vectors = morgan_bitvectors(curated["standardized_parent_smiles"])
    same_source: list[bool] = []
    nearest_similarity: list[float] = []
    expected_same_source: list[float] = []
    source_counts = pd.Series(sources).value_counts().to_dict()
    for index, vector in enumerate(vectors):
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(vector, vectors))
        similarities[index] = -1.0
        nearest = int(np.argmax(similarities))
        same_source.append(sources[nearest] == sources[index])
        nearest_similarity.append(float(similarities[nearest]))
        expected_same_source.append((int(source_counts[sources[index]]) - 1) / (len(ids) - 1))

    positive_indices = np.flatnonzero(curated["label"].to_numpy(dtype=int) == 1)
    negative_indices = np.flatnonzero(curated["label"].to_numpy(dtype=int) == 0)
    pair_candidates: list[tuple[float, str, str]] = []
    for positive_index in positive_indices:
        similarities = DataStructs.BulkTanimotoSimilarity(
            vectors[int(positive_index)], [vectors[int(value)] for value in negative_indices]
        )
        pair_candidates.extend(
            (
                float(similarity),
                ids[int(positive_index)],
                ids[int(negative_index)],
            )
            for negative_index, similarity in zip(negative_indices, similarities, strict=True)
        )
    used_positive: set[str] = set()
    used_negative: set[str] = set()
    matched_rows: list[dict[str, Any]] = []
    for similarity, positive_id, negative_id in sorted(
        pair_candidates, key=lambda value: (-value[0], value[1], value[2])
    ):
        if positive_id in used_positive or negative_id in used_negative:
            continue
        used_positive.add(positive_id)
        used_negative.add(negative_id)
        matched_rows.append(
            {
                "positive_compound_id": positive_id,
                "weak_reference_compound_id": negative_id,
                "tanimoto": similarity,
            }
        )
    matches = pd.DataFrame(matched_rows)
    matched_sensitivity: dict[str, Any] = {}
    for threshold in (0.20, 0.30, 0.40):
        retained = matches.loc[matches["tanimoto"].ge(threshold)]
        retained_ids = set(retained["positive_compound_id"]) | set(
            retained["weak_reference_compound_id"]
        )
        model_values: dict[str, Any] = {}
        for model_id, rows in aggregated.groupby("model_id", sort=True):
            subset = rows.loc[rows["compound_id"].isin(retained_ids)]
            if len(subset) != 2 * len(retained) or not len(retained):
                model_values[str(model_id)] = None
                continue
            values = metric_bundle(
                subset["label"].to_numpy(dtype=int),
                subset["probability_calibrated"].to_numpy(dtype=float),
            )
            model_values[str(model_id)] = {
                key: values[key] for key in ("ap_positive", "ap_negative", "auroc", "brier")
            }
        matched_sensitivity[f"tanimoto_at_least_{threshold:.2f}"] = {
            "n_matched_pairs": len(retained),
            "model_metrics_exploratory": model_values,
        }

    qc_by_source: dict[str, Any] = {}
    descriptor_missing = descriptors.isna()
    for source, rows in curated.groupby("source_role", sort=True):
        positions = rows.index.to_numpy(dtype=int)
        flag_counts: dict[str, int] = {}
        for raw_flags in rows["qc_flags"].astype(str):
            for flag in json.loads(raw_flags):
                flag_counts[str(flag)] = flag_counts.get(str(flag), 0) + 1
        qc_by_source[str(source)] = {
            "n_compounds": len(rows),
            "smiles_curated_count": int(rows["smiles_curated"].astype(bool).sum()),
            "metal_sensitive_representation_count": int(
                rows["metal_sensitive_representation"].astype(bool).sum()
            ),
            "multi_component_input_count": int(rows["component_count"].astype(int).gt(1).sum()),
            "qc_flag_counts": dict(sorted(flag_counts.items())),
            "descriptor_missing_cells": int(
                descriptor_missing.iloc[positions].to_numpy().sum()
            ),
        }

    metric_by_model = {record["model_id"]: record["metrics"] for record in metric_records}
    source_control_id = "R2_extra_trees_v3_compatible"
    audit = {
        "schema_version": "geroprotector.source_confounding_audit.v1",
        "post_lock_exploratory_only": True,
        "used_for_model_selection": False,
        "source_target_is_bijective_with_observed_label": True,
        "source_label_mapping": {
            str(source): int(row["min"]) for source, row in source_label.iterrows()
        },
        "source_classifier": {
            "model_id": source_control_id,
            "method": (
                "existing sealed outer-OOF R2 predictions reused because source target "
                "is exactly bijective with the observed label; no report-time model fit"
            ),
            "metrics": metric_by_model.get(source_control_id),
        },
        "descriptor_distribution_table": "descriptor_distributions.csv",
        "fraction_csp3_is_only_a_structural_proxy_not_a_validated_np_likeness_score": True,
        "nearest_neighbor_source_enrichment": {
            "n_compounds": len(ids),
            "same_source_fraction": float(np.mean(same_source)),
            "chance_expectation_from_source_sizes": float(np.mean(expected_same_source)),
            "enrichment_ratio": float(np.mean(same_source) / np.mean(expected_same_source)),
            "nearest_tanimoto_median": float(np.median(nearest_similarity)),
        },
        "source_specific_qc_and_missingness": qc_by_source,
        "chemistry_matching": {
            "method": "deterministic_greedy_one_to_one_cross_source_morgan_r2_2048",
            "n_total_pairs": len(matches),
            "pair_ledger": "chemistry_matches.csv",
            "sensitivities": matched_sensitivity,
        },
        "n_raw_provenance_rows_verified": len(provenance),
        "interpretation": (
            "These diagnostics quantify source chemistry but cannot identify a biological "
            "geroprotection signal because source and label are perfectly aligned."
        ),
    }
    return audit, distributions, matches


def _holm(raw_p_values: list[float]) -> list[float]:
    count = len(raw_p_values)
    order = np.argsort(raw_p_values, kind="stable")
    adjusted = np.zeros(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * float(raw_p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def _validate_prediction_rows(root: Path, predictions: pd.DataFrame) -> None:
    schema = _schema(root, "prediction.schema.json")
    for record in predictions.to_dict(orient="records"):
        clean = {
            key: (None if pd.isna(value) else _jsonable(value)) for key, value in record.items()
        }
        jsonschema.Draft202012Validator(schema).validate(clean)


def _report_manifest(directory: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(directory.rglob("*")):
        if path.is_dir() or path.name == "report_manifest.json":
            continue
        if path.is_symlink() or not path.is_file():
            raise ReportingError(f"Unsafe report artifact: {path}")
        entries.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {"schema_version": "geroprotector.internal_report.v1", "entries": entries}
    manifest["canonical_sha256"] = canonical_sha256(manifest)
    return manifest


def _verify_existing_report(directory: Path) -> Path:
    manifest_path = directory / "report_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ReportingError("Existing report lacks a safe manifest")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = dict(value)
    claimed = payload.pop("canonical_sha256", None)
    if claimed != canonical_sha256(payload):
        raise ReportingError("Existing report manifest integrity failed")
    if value != _report_manifest(directory):
        raise ReportingError("Existing report artifacts changed")
    return directory


def report_internal(
    run_directory: str | Path,
    *,
    reference_model_id: str | None = None,
) -> Path:
    """Create a hash-bound report from a completed run without fitting estimators."""

    run = Path(run_directory).resolve()
    _, run_manifest, _ = verify_completed_run(run)
    split_strategy = str(run_manifest["splits"]["primary_strategy"])
    paper_contextual = split_strategy == "paper_random_80_20"
    root = run.parent.parent.resolve()
    validate_core_runtime(root / "requirements-lock.txt")
    if source_tree_sha256(root) != run_manifest["code"]["source_tree_sha256"]:
        raise ReportingError("Current reporting source differs from the training snapshot")
    report = run / "report"
    if report.exists():
        return _verify_existing_report(report)
    predictions = pd.read_parquet(run / "predictions" / "outer_long.parquet")
    _validate_prediction_rows(root, predictions)
    model_ids = tuple(sorted(predictions["model_id"].astype(str).unique()))
    runtime_table = pd.read_parquet(run / "audits" / "runtime_jobs.parquet")
    required_runtime = {
        "model_id",
        "repeat",
        "outer_fold",
        "calibration_crossfit_seconds",
        "outer_train_final_fit_seconds",
        "outer_prediction_and_audit_seconds",
        "job_total_seconds_before_serialization",
    }
    if required_runtime - set(runtime_table):
        raise ReportingError("Runtime audit has an incomplete schema")
    expected_jobs = int(run_manifest["outer_repeats"]) * int(run_manifest["outer_folds"])
    if runtime_table.duplicated(["model_id", "repeat", "outer_fold"]).any():
        raise ReportingError("Runtime audit contains duplicate outer jobs")
    if set(runtime_table["model_id"].astype(str)) != set(model_ids):
        raise ReportingError("Runtime audit model coverage differs from predictions")
    runtime_summary: dict[str, Any] = {}
    for model_id, rows in runtime_table.groupby("model_id", sort=True):
        if len(rows) != expected_jobs:
            raise ReportingError(f"Runtime audit coverage is incomplete for {model_id}")
        runtime_summary[str(model_id)] = {
            "n_outer_jobs": len(rows),
            "calibration_crossfit_seconds_total": float(
                rows["calibration_crossfit_seconds"].sum()
            ),
            "outer_train_final_fit_seconds_total": float(
                rows["outer_train_final_fit_seconds"].sum()
            ),
            "outer_prediction_and_audit_seconds_total": float(
                rows["outer_prediction_and_audit_seconds"].sum()
            ),
            "job_total_seconds_before_serialization": float(
                rows["job_total_seconds_before_serialization"].sum()
            ),
        }
    inference_collection = json.loads(
        (run / "audits" / "v6_inference_audits.json").read_text(encoding="utf-8")
    )
    if inference_collection.get("pipeline_id") != run_manifest["pipeline_id"]:
        raise ReportingError("V6 inference-audit collection pipeline binding changed")
    if int(inference_collection.get("n_audits", -1)) != len(
        inference_collection.get("audits", [])
    ):
        raise ReportingError("V6 inference-audit collection count changed")
    inference_summary: dict[str, Any] = {}
    for record in inference_collection.get("audits", []):
        inference_summary.setdefault(record["model_id"], []).append(record["audit"])
    inference_summary = {
        model_id: {
            "n_outer_jobs": len(audits),
            "all_canonical_hard_checks_passed": bool(
                all(bool(value["passed"]) for value in audits)
            ),
            "max_whole_batch_vs_singleton_difference": float(
                max(value["max_abs_difference"]["whole_batch_vs_singleton"] for value in audits)
            ),
            "max_random_composition_vs_singleton_difference": float(
                max(
                    value["max_abs_difference"]["random_composition_vs_singleton"]
                    for value in audits
                )
            ),
            "inference_audit_wall_clock_seconds_total": float(
                sum(value["runtime"]["wall_clock_seconds"] for value in audits)
            ),
            "estimated_forward_passes_total": int(
                sum(value["runtime"]["total_estimated_forward_passes"] for value in audits)
            ),
            "gpu_peak_memory_bytes_max": (
                max(
                    (
                        value["runtime"]["gpu_peak_memory_bytes"]
                        for value in audits
                        if value["runtime"]["gpu_peak_memory_bytes"] is not None
                    ),
                    default=None,
                )
            ),
        }
        for model_id, audits in inference_summary.items()
    }
    for model_id in model_ids:
        if (
            model_id.startswith("v6_foundation_")
            and inference_summary.get(model_id, {}).get("n_outer_jobs") != expected_jobs
        ):
            raise ReportingError(f"V6 inference-audit coverage is incomplete for {model_id}")
    selection_table = pd.read_parquet(run / "selections" / "all_attempts.parquet")
    if "model_id" not in selection_table or "status" not in selection_table:
        raise ReportingError("Selection trace lacks model/status audit fields")
    selection_audit: dict[str, Any] = {}
    for model_id in model_ids:
        attempts = selection_table.loc[
            selection_table["model_id"].astype(str).eq(model_id)
            & selection_table["status"].isin(["success", "failed"])
        ]
        failures = attempts.loc[attempts["status"].eq("failed")]
        failure_types = (
            pd.Series("unknown", index=failures.index, dtype="string")
            if "failure_type" not in failures
            else failures["failure_type"].fillna("unknown").astype(str)
        )
        selection_audit[model_id] = {
            "attempted_candidates_across_all_fit_scopes": len(attempts),
            "successful_candidates_across_all_fit_scopes": int(
                attempts["status"].eq("success").sum()
            ),
            "failed_candidates_across_all_fit_scopes": len(failures),
            "failure_types": {
                str(key): int(value)
                for key, value in failure_types.value_counts().sort_index().items()
            },
        }
    manifest_collection = json.loads(
        (run / "audits" / "fitted_model_manifests.json").read_text(encoding="utf-8")
    )
    manifest_rows = manifest_collection.get("manifests", [])
    if (
        manifest_collection.get("pipeline_id") != run_manifest["pipeline_id"]
        or int(manifest_collection.get("n_manifests", -1)) != len(manifest_rows)
        or len(manifest_rows) != expected_jobs * len(model_ids)
    ):
        raise ReportingError("Fitted-model manifest collection coverage changed")
    model_diagnostics: dict[str, Any] = {}
    for model_id in model_ids:
        manifests = [
            row["manifest"] for row in manifest_rows if str(row.get("model_id")) == model_id
        ]
        if len(manifests) != expected_jobs:
            raise ReportingError(f"Fitted-model manifest coverage incomplete for {model_id}")
        summary_row: dict[str, Any] = {}
        stability_values: list[float] = []
        fallback_count = 0
        representation_dimensions: list[int] = []
        checkpoint_records: dict[str, dict[str, Any]] = {}
        panel_counts: dict[str, int] = {}
        for manifest in manifests:
            importance = manifest.get("importance", {})
            stability_values.extend(
                float(value)
                for value in importance.get("stability_spearman", {}).values()
                if np.isfinite(float(value))
            )
            representation = manifest.get("representation", {})
            fallback_count += int(bool(representation.get("stability_fallback_applied")))
            embedding = representation.get("embedding", {})
            dimension = embedding.get("n_components", embedding.get("effective_components"))
            if dimension is not None:
                representation_dimensions.append(int(dimension))
            panel = manifest.get("panel", {})
            panel_name = panel.get("panel") or manifest.get("winner", {}).get("panel", {}).get(
                "panel"
            )
            if panel_name:
                panel_counts[str(panel_name)] = panel_counts.get(str(panel_name), 0) + 1
            model = manifest.get("model", {})
            if model.get("checkpoint_sha256"):
                checkpoint_records[str(model["model_id"])] = {
                    key: model.get(key)
                    for key in (
                        "package",
                        "package_version",
                        "explicit_model_version",
                        "checkpoint_sha256",
                        "license_sha256",
                        "checkpoint_source",
                        "access_date_utc",
                    )
                }
        if stability_values:
            summary_row["importance_stability_spearman"] = {
                "median": float(np.median(stability_values)),
                "minimum": float(np.min(stability_values)),
                "n_values": len(stability_values),
                "stability_fallback_outer_jobs": fallback_count,
            }
        if representation_dimensions:
            summary_row["selected_representation_dimensions"] = {
                str(value): int(representation_dimensions.count(value))
                for value in sorted(set(representation_dimensions))
            }
        if panel_counts:
            summary_row["selected_panel_outer_jobs"] = dict(sorted(panel_counts.items()))
        if checkpoint_records:
            summary_row["verified_checkpoint_license_records"] = checkpoint_records
        model_diagnostics[model_id] = summary_row
    if reference_model_id is None:
        reference_model_id = (
            "R2_extra_trees_v3_compatible"
            if "R2_extra_trees_v3_compatible" in model_ids
            else model_ids[0]
        )
    if reference_model_id not in model_ids:
        raise ReportingError(f"Reference model is absent: {reference_model_id}")
    staging = Path(tempfile.mkdtemp(prefix=f".{run.name}.report-work-", dir=run.parent))
    aggregated_frames: list[pd.DataFrame] = []
    metric_records: list[dict[str, Any]] = []
    repeat_records: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    reliability_curves: dict[str, list[dict[str, Any]]] = {}
    created = utc_now()
    for model_id in model_ids:
        table = predictions.loc[predictions["model_id"].eq(model_id)].copy()
        table = table.rename(
            columns={"y_true": "label", "primary_component_id": "component_id"}
        )
        aggregated = aggregate_repeated_oof(table)
        metadata = (
            table.sort_values(["compound_id", "repeat"], kind="stable")
            .groupby("compound_id", as_index=False)
            .agg(
                murcko_scaffold_id=("murcko_scaffold_id", "first"),
                nearest_train_tanimoto=("nearest_train_tanimoto", "mean"),
                robust_descriptor_distance=("robust_descriptor_distance", "mean"),
                inside_applicability_domain=("inside_applicability_domain", "mean"),
                majority_decision=("predicted_class", lambda values: int(values.mean() >= 0.5)),
            )
        )
        aggregated = aggregated.merge(metadata, on="compound_id", validate="one_to_one")
        aggregated.insert(0, "model_id", model_id)
        aggregated_frames.append(aggregated)
        y = aggregated["label"].to_numpy(dtype=int)
        p = aggregated["probability_calibrated"].to_numpy(dtype=float)
        metrics = metric_bundle(y, p)
        metrics.update(
            _threshold_metrics(y, aggregated["majority_decision"].to_numpy(dtype=int))
        )
        metrics.update(_screening_metrics(y, p))
        metrics["coverage"] = float(aggregated["inside_applicability_domain"].mean())
        record = {
            "run_id": run_manifest["run_id"],
            "pipeline_id": run_manifest["pipeline_id"],
            "model_id": model_id,
            "reference_model_id": reference_model_id,
            "split_strategy": split_strategy,
            "evaluation_role": (
                "paper_holdout_contextual"
                if paper_contextual
                else "outer_test_repeated_consensus"
            ),
            "aggregation_level": "one_final_prediction_per_compound",
            "n_compounds": len(aggregated),
            "n_positive": int(y.sum()),
            "n_negative": int((1 - y).sum()),
            "n_primary_components": int(aggregated["component_id"].nunique()),
            "prevalence_positive": float(y.mean()),
            "threshold": None,
            "metrics": metrics,
            "paired_comparison": None,
            "created_utc": created,
        }
        metric_records.append(record)
        for repeat, repeated in table.groupby("repeat", sort=True):
            repeated_y = repeated["label"].to_numpy(dtype=int)
            repeated_p = repeated["probability_calibrated"].to_numpy(dtype=float)
            repeat_records.append(
                {
                    "model_id": model_id,
                    "repeat": int(repeat),
                    **metric_bundle(repeated_y, repeated_p),
                }
            )
        similarity_rows = []
        bins = pd.cut(
            aggregated["nearest_train_tanimoto"],
            bins=[-np.inf, 0.1, 0.2, 0.3, np.inf],
            labels=["<=0.10", "0.10-0.20", "0.20-0.30", ">0.30"],
        )
        for bin_name, subset in aggregated.groupby(bins, observed=False):
            values = None
            if len(subset) >= 2 and set(subset["label"].astype(int)) == {0, 1}:
                values = metric_bundle(
                    subset["label"].to_numpy(dtype=int),
                    subset["probability_calibrated"].to_numpy(dtype=float),
                )
            similarity_rows.append({"bin": str(bin_name), "n": len(subset), "metrics": values})
        scaffold_rows = []
        scaffold_type = (
            aggregated["murcko_scaffold_id"]
            .astype(str)
            .map(lambda value: "acyclic" if value == "ACYCLIC" else "cyclic")
        )
        for kind, subset in aggregated.groupby(scaffold_type):
            values = None
            if len(subset) >= 2 and set(subset["label"].astype(int)) == {0, 1}:
                values = metric_bundle(
                    subset["label"].to_numpy(dtype=int),
                    subset["probability_calibrated"].to_numpy(dtype=float),
                )
            scaffold_rows.append({"scaffold_type": kind, "n": len(subset), "metrics": values})
        reliability_fixed = _reliability_bins(y, p, adaptive=False)
        reliability_adaptive = _reliability_bins(y, p, adaptive=True)
        reliability_curves[model_id] = reliability_fixed
        diagnostics[model_id] = {
            "risk_coverage": _risk_coverage(y, p),
            "reliability_fixed_width": reliability_fixed,
            "reliability_adaptive_count": reliability_adaptive,
            "nearest_train_tanimoto_bins": similarity_rows,
            "scaffold_type": scaffold_rows,
            "calibration_sensitivity": {
                "raw": metric_bundle(y, aggregated["probability_raw"].to_numpy(dtype=float)),
                "platt": metric_bundle(y, p),
                "beta": metric_bundle(
                    y,
                    table.groupby("compound_id")["probability_beta_sensitivity"]
                    .mean()
                    .reindex(aggregated["compound_id"])
                    .to_numpy(dtype=float),
                ),
            },
        }
    all_aggregated = pd.concat(aggregated_frames, ignore_index=True)
    reference = all_aggregated.loc[
        all_aggregated["model_id"].eq(reference_model_id),
        ["compound_id", "label", "component_id", "probability_calibrated"],
    ].rename(columns={"probability_calibrated": "reference_probability"})
    comparisons: list[dict[str, Any]] = []
    comparison_record_indices: list[int] = []
    for record_index, record in enumerate(metric_records):
        model_id = record["model_id"]
        if model_id == reference_model_id:
            continue
        candidate = all_aggregated.loc[
            all_aggregated["model_id"].eq(model_id),
            ["compound_id", "label", "component_id", "probability_calibrated"],
        ].rename(columns={"probability_calibrated": "candidate_probability"})
        paired = candidate.merge(
            reference,
            on=["compound_id", "label", "component_id"],
            validate="one_to_one",
        )
        comparison = paired_component_bootstrap(
            paired,
            candidate_column="candidate_probability",
            reference_column="reference_probability",
            n_resamples=10_000,
            seed=20260408,
        )
        comparison["candidate_model_id"] = model_id
        comparison["reference_model_id"] = reference_model_id
        comparison["raw_p_two_sided"] = comparison["p_two_sided"]
        comparisons.append(comparison)
        comparison_record_indices.append(record_index)
    if comparisons:
        adjusted = _holm([value["raw_p_two_sided"] for value in comparisons])
        for comparison, adjusted_p, record_index in zip(
            comparisons, adjusted, comparison_record_indices, strict=True
        ):
            comparison["holm_adjusted_p"] = adjusted_p
            schema_comparison = {
                key: value
                for key, value in comparison.items()
                if key
                not in {
                    "candidate_model_id",
                    "reference_model_id",
                    "raw_p_two_sided",
                    "holm_adjusted_p",
                }
            }
            schema_comparison["p_two_sided"] = adjusted_p
            schema_comparison["multiple_testing_adjustment"] = "holm"
            metric_records[record_index]["paired_comparison"] = schema_comparison
    metric_schema = _schema(root, "metrics.schema.json")
    for record in metric_records:
        jsonschema.Draft202012Validator(metric_schema).validate(_jsonable(record))
    source_audit, source_distributions, chemistry_matches = _source_confounding_audit(
        root, run_manifest, all_aggregated, metric_records
    )
    claims = {
        "pipeline_id": run_manifest["pipeline_id"],
        "evidence_level": (
            "internal_paper_holdout_contextual"
            if paper_contextual
            else "internal_grouped_validation"
        ),
        "allowed_claims": (
            [
                "Performance was estimated on the locked identity-safe paper-style holdout.",
                "Any model difference is contextual for the source-defined benchmark endpoint.",
            ]
            if paper_contextual
            else [
                "Performance was estimated under repeated chemical-component nested CV.",
                "Any model difference is predictive for the source-defined benchmark endpoint.",
            ]
        ),
        "forbidden_claims": [
            "The model proves geroprotective efficacy.",
            "The model identifies causal anti-aging mechanisms.",
            "Internal performance alone establishes state of the art or clinical utility.",
        ],
        "required_qualifiers": [
            "Positive and weak-reference labels come from different sources.",
            "Weak references are not experimentally confirmed negatives.",
            (
                "The random 80/20 holdout permits chemical-component proximity "
                "across train/test."
                if paper_contextual
                else "The bootstrap conditions on repeated-CV consensus predictions."
            ),
        ],
        "causal_language_allowed": False,
        "human_efficacy_language_allowed": False,
        "source_label_confounding_must_be_disclosed": True,
        "external_dataset_name": None,
        "locked_utc": created,
    }
    jsonschema.Draft202012Validator(_schema(root, "claims_allowed.schema.json")).validate(
        claims
    )
    summary = {
        "schema_version": "geroprotector.internal_summary.v1",
        "run_id": run_manifest["run_id"],
        "pipeline_id": run_manifest["pipeline_id"],
        "scientific_role": claims["evidence_level"],
        "split_strategy": split_strategy,
        "headline_eligible": not paper_contextual,
        "reference_model_id": reference_model_id,
        "estimand": (
            "one locked identity-safe paper-style holdout prediction per test compound"
            if paper_contextual
            else "five-repeat OOF consensus prediction per compound"
        ),
        "confidence_interval_estimand": (
            "paired test-set component bootstrap conditional on one trained paper-holdout model"
            if paper_contextual
            else (
                "paired primary-component bootstrap conditional on trained repeated-CV "
                "predictions"
            )
        ),
        "source_label_confounding_disclosed": True,
        "external_validation_included": False,
        "model_metrics": metric_records,
        "paired_comparisons": comparisons,
        "diagnostics": diagnostics,
        "runtime": runtime_summary,
        "v6_inference_audit": inference_summary,
        "selection_audit": selection_audit,
        "fitted_model_diagnostics": model_diagnostics,
        "source_confounding_audit": source_audit,
        "created_utc": created,
    }
    (staging / "metrics").mkdir(parents=True)
    (staging / "reports").mkdir(parents=True)
    (staging / "source_confounding_audit").mkdir(parents=True)
    atomic_write_json(staging / "summary.json", _jsonable(summary))
    atomic_write_json(staging / "metrics" / "model_metrics.json", _jsonable(metric_records))
    atomic_write_json(staging / "metrics" / "paired_comparisons.json", _jsonable(comparisons))
    atomic_write_json(staging / "metrics" / "diagnostics.json", _jsonable(diagnostics))
    atomic_write_json(staging / "claims_allowed.json", claims)
    atomic_write_json(
        staging / "source_confounding_audit" / "audit.json", _jsonable(source_audit)
    )
    source_distributions.to_csv(
        staging / "source_confounding_audit" / "descriptor_distributions.csv", index=False
    )
    chemistry_matches.to_csv(
        staging / "source_confounding_audit" / "chemistry_matches.csv", index=False
    )
    atomic_write_bytes(
        staging / "reports" / "reliability.svg", _reliability_svg(reliability_curves)
    )
    all_aggregated.to_parquet(staging / "compound_aggregated.parquet", index=False)
    pd.DataFrame(repeat_records).to_parquet(
        staging / "metrics" / "per_repeat.parquet", index=False
    )
    lines = [
        f"# Internal report — {run_manifest['run_id']}",
        "",
        (
            "This is an internal identity-safe paper-style random holdout, contextual only; "
            "it is not external or headline evidence."
            if paper_contextual
            else (
                "This is internal repeated chemical-component validation, not external "
                "evidence."
            )
        ),
        "The endpoint is source-defined and perfectly source-confounded; weak references "
        "are not confirmed negatives.",
        "",
        f"Reference: `{reference_model_id}`",
        "",
        (
            "| Model | AP+ | AUROC | Brier | MCC | AD coverage |"
            if paper_contextual
            else "| Model | AP+ | AUROC | Brier | MCC (majority across repeats) | AD coverage |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in metric_records:
        values = record["metrics"]
        lines.append(
            f"| {record['model_id']} | {values['ap_positive']:.4f} | "
            f"{values['auroc']:.4f} | {values['brier']:.4f} | "
            f"{values['mcc']:.4f} | {values['coverage']:.4f} |"
        )
    lines.extend(
        [
            "",
            (
                "Intervals/p-values are conditional on this single locked random holdout and "
                "do not include split or model-training uncertainty. Holm correction is "
                "applied across the reported comparisons."
                if paper_contextual
                else (
                    "Intervals/p-values are conditional on the trained repeated-CV consensus "
                    "and do not include model-training uncertainty. Holm correction is applied "
                    "across the reported comparisons."
                )
            ),
        ]
    )
    lines.extend(["", "## Runtime and V6 inference audit", ""])
    for model_id in model_ids:
        runtime = runtime_summary[model_id]
        lines.append(
            f"- `{model_id}`: {runtime['job_total_seconds_before_serialization']:.1f}s "
            f"across {runtime['n_outer_jobs']} outer jobs."
        )
        if model_id in inference_summary:
            audit = inference_summary[model_id]
            lines.append(
                "  Canonical hard checks: "
                f"{audit['all_canonical_hard_checks_passed']}; max whole-batch delta "
                f"{audit['max_whole_batch_vs_singleton_difference']:.6g}; max random-batch "
                f"delta {audit['max_random_composition_vs_singleton_difference']:.6g}; "
                f"estimated audit forwards {audit['estimated_forward_passes_total']}."
            )
    lines.extend(["", "## Selection and fitted-model diagnostics", ""])
    for model_id in model_ids:
        audit = selection_audit[model_id]
        lines.append(
            f"- `{model_id}`: {audit['successful_candidates_across_all_fit_scopes']}/"
            f"{audit['attempted_candidates_across_all_fit_scopes']} candidate attempts "
            f"succeeded; {audit['failed_candidates_across_all_fit_scopes']} failed."
        )
        stability = model_diagnostics[model_id].get("importance_stability_spearman")
        if stability:
            lines.append(
                f"  Median V5 importance Spearman={stability['median']:.3f}; "
                f"stability fallback used in {stability['stability_fallback_outer_jobs']} "
                "outer jobs."
            )
        if model_id in inference_summary:
            lines.append(
                "  GPU memory and forward-pass counters cover the label-free inference "
                "semantics audit only, not model selection or fitting."
            )
    lines.extend(
        [
            "",
            "## Source-confounding audit",
            "",
            "Source role is exactly bijective with the observed label. Descriptor, "
            "nearest-neighbor and chemistry-matched diagnostics are exploratory and "
            "cannot distinguish source chemistry from biological geroprotection.",
            "",
            f"Nearest-neighbor same-source enrichment ratio: "
            f"{source_audit['nearest_neighbor_source_enrichment']['enrichment_ratio']:.3f}.",
        ]
    )
    atomic_write_bytes(staging / "reports" / "summary.md", ("\n".join(lines) + "\n").encode())
    model_card = """# Model card

## Intended use

Research ranking under a source-defined geroprotector-versus-weak-reference endpoint.

## Invalid uses

Clinical decisions, causal/mechanistic claims, or efficacy claims.

## Validation boundary

All architecture selection, feature fitting, calibration and thresholds were confined to
outer-training partitions. The outer predictions were produced once and metrics were
computed only after the training run was sealed. HAGR outcomes were not accessed.

## Principal limitation

The positive and weak-reference classes originate from different sources, so source-label
confounding cannot be resolved by grouped cross-validation alone.
"""
    atomic_write_bytes(staging / "reports" / "model_card.md", model_card.encode())
    report_manifest = _report_manifest(staging)
    atomic_write_json(staging / "report_manifest.json", report_manifest)
    os.rename(staging, report)
    return report
