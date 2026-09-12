"""Finalize Experiment E from hash-bound checkpoints after a prespecified
publication-grouped sensitivity analysis proves infeasible.

This module never fits a model.  It reconstructs the exact curated cohort and
checkpoint binding, verifies every cached OOF stream, records why the grouped
analysis is not estimable, completes the downstream bias analyses, and seals a
new immutable run directory.
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
from sklearn.model_selection import StratifiedGroupKFold

from geroprotector.study_common import (bind_checkpoint_directory, md_table,
                                        morgan_generator, morgan_matrix)
from geroprotector.model_panel import PANEL, Panel
from geroprotector.drugage_celegans_benchmark import (
    LABEL_VARIANTS,
    N_SPLITS,
    REPEATED,
    SCHEMA,
    SEED,
    _load_endpoint_protocol,
    _paper_contract,
    _publication_components,
    _qed_audit,
    _read_sources,
    _validated_raw_smiles,
    build_features,
    class_shifts,
    compound_endpoints,
    curate,
    d1_connectivity_keys,
    fold_local_baselines,
    load_fixed_protocol,
    observation_ledger,
    paper_split_indices,
    resolve_sources,
)
from geroprotector.endpoint_benchmark import (
    EndpointBenchmarkError,
    PRIMARY_METRICS,
    attach_molecular_variables,
    cross_validate,
    metric_table,
)
from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file


COMPLETION_STATUS = (
    "COMPLETE_TRACK_E1_BLOCKED_PARTIAL_PUBLICATION_GROUPED_INFEASIBLE"
)
PUBLICATION_STATUS = "BLOCKED_INFEASIBLE_VALIDATION_FOLD_SINGLE_CLASS"


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def publication_fold_feasibility(labels: np.ndarray, groups: np.ndarray, *,
                                 folds: int, seed: int
                                 ) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Build the prespecified registry and report, but never repair, infeasibility."""
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups, dtype=int)
    if len(labels) != len(groups) or len(labels) == 0:
        raise EndpointBenchmarkError("Publication labels/groups are empty or misaligned")
    splitter = StratifiedGroupKFold(n_splits=int(folds), shuffle=True,
                                    random_state=int(seed))
    assignment = np.full(len(labels), -1, dtype=int)
    for fold, (_fit, validation) in enumerate(
            splitter.split(np.arange(len(labels)), labels, groups=groups)):
        assignment[validation] = fold
    if (assignment < 0).any():
        raise EndpointBenchmarkError("Publication-grouped fold assignment is incomplete")
    if pd.DataFrame({"group": groups, "fold": assignment}).groupby("group")[
            "fold"].nunique().max() != 1:
        raise EndpointBenchmarkError("A publication component crosses validation folds")
    rows = []
    for fold in range(int(folds)):
        active = labels[assignment == fold]
        positive = int((active == 1).sum())
        negative = int((active == 0).sum())
        rows.append({"fold": fold, "validation_compounds": int(len(active)),
                     "positive": positive, "negative": negative,
                     "classes_present": int(len(np.unique(active))),
                     "estimable": bool(positive > 0 and negative > 0)})
    audit = pd.DataFrame(rows)
    feasible = bool(audit["estimable"].all())
    status = "FEASIBLE" if feasible else PUBLICATION_STATUS
    return assignment, audit, {
        "status": status,
        "feasible": feasible,
        "folds": int(folds),
        "random_state": int(seed),
        "reason": ("all validation folds contain both classes" if feasible else
                   "at least one prespecified publication-grouped validation fold "
                   "contains only one endpoint class; AUROC and comparative metrics "
                   "are therefore not estimable without a post hoc split change"),
        "post_hoc_seed_or_fold_search_performed": False,
        "model_fitting_performed_for_grouped_analysis": False,
    }


def _validate_checkpoint(frame: pd.DataFrame, labels: np.ndarray, *, folds: int,
                         role: str) -> None:
    required = {"fold", "label", "maximum_tanimoto_to_active_fit_fold"}
    required.update(f"p_{model}" for model in PANEL)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise EndpointBenchmarkError(f"{role}: checkpoint columns missing: {missing}")
    if len(frame) != len(labels):
        raise EndpointBenchmarkError(
            f"{role}: checkpoint has {len(frame)} rows, expected {len(labels)}")
    observed_labels = frame["label"].to_numpy(int)
    if not np.array_equal(observed_labels, np.asarray(labels, dtype=int)):
        raise EndpointBenchmarkError(f"{role}: checkpoint labels/order differ")
    observed_folds = set(frame["fold"].astype(int).unique())
    if observed_folds != set(range(int(folds))):
        raise EndpointBenchmarkError(f"{role}: checkpoint fold registry differs")
    for model in PANEL:
        probability = frame[f"p_{model}"].to_numpy(float)
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise EndpointBenchmarkError(f"{role}: invalid probabilities for {model}")
    similarity = frame["maximum_tanimoto_to_active_fit_fold"].to_numpy(float)
    if not np.isfinite(similarity).all() or ((similarity < 0) | (similarity > 1)).any():
        raise EndpointBenchmarkError(f"{role}: invalid fold-local similarities")


def _checkpoint_contract(root: Path, run_id: str, protocol_sha: str,
                         resolution: dict[str, Any], curated: pd.DataFrame
                         ) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "protocol_sha256": protocol_sha,
        "source_sha256": resolution["source_files"],
        "curated_compound_ids_sha256": canonical_sha256(
            curated["compound_id"].astype(str).tolist()),
        "endpoint_labels_sha256": {
            name: canonical_sha256(curated[f"label_{name}"].astype(int).tolist())
            for name in LABEL_VARIANTS
        },
        "code_sha256": {
            name: sha256_file(root / "src" / "geroprotector" / name)
            for name in ("drugage_celegans_benchmark.py", "endpoint_benchmark.py",
                         "model_panel.py")
        },
    }


def run(*, root: Path, config_path: Path, positive: Path, negative: Path,
        run_id: str) -> Path:
    if not re.fullmatch(r"drugage_celegans_benchmark_[a-z0-9_.-]+", run_id):
        raise EndpointBenchmarkError("RUN_ID must start with drugage_celegans_benchmark_")
    root = root.resolve()
    destination = root / "outputs" / run_id
    checkpoint_dir = root / "outputs" / f".{run_id}.work"
    if destination.exists() or destination.is_symlink():
        raise EndpointBenchmarkError(f"Run directory already exists: {destination}")
    if not checkpoint_dir.is_dir() or checkpoint_dir.is_symlink():
        raise EndpointBenchmarkError(f"Bound checkpoint directory is absent: {checkpoint_dir}")

    protocol, protocol_sha = _load_endpoint_protocol(config_path)
    observations, mapping, resolution = resolve_sources(protocol)
    ledger = observation_ledger(observations, mapping)
    endpoints = compound_endpoints(ledger)
    d1_keys, d1_audit = d1_connectivity_keys(root, positive, negative)
    resolution["d1_overlap_reference"] = d1_audit

    primary = endpoints[["connectivity_inchikey", "compound_name", "canonical_smiles",
                         "label_significant_positive"]].copy()
    raw = pd.DataFrame({
        "provider_record_id": primary["connectivity_inchikey"],
        "compound_name": primary["compound_name"],
        "source_smiles": primary["canonical_smiles"].astype(str),
        "label": primary["label_significant_positive"].astype(int),
    })
    curated, curation_ledger, conflicts = curate(
        raw, cohort="drugage_celegans", d1_connectivity=d1_keys)
    curated = curated.merge(
        endpoints[["connectivity_inchikey", "publications", "publication_ids_json",
                   "label_significant_positive", "label_consistent_positive",
                   "label_conflict_excluded", "conflict_status", "observations"]],
        on="connectivity_inchikey", how="left", validate="one_to_one")
    if curated["publications"].isna().any():
        raise EndpointBenchmarkError("Publication annotation did not cover every identity")

    features, feature_audit = build_features(
        root, curated, verify_d1_parity=True, nonfinite_descriptor_policy="exclude")
    eligibility = feature_audit["descriptor_eligibility"]
    eligible_indices = np.asarray(eligibility["eligible_input_indices"], dtype=int)
    descriptor_excluded = curated.iloc[
        np.asarray(eligibility["excluded_input_indices"], dtype=int)].copy()
    if len(eligible_indices) != len(features["paper"]):
        raise EndpointBenchmarkError("Descriptor eligibility and feature rows differ")
    if not descriptor_excluded.empty:
        excluded_keys = set(descriptor_excluded["connectivity_inchikey"].astype(str))
        excluded_mask = curation_ledger["connectivity_inchikey"].astype(str).isin(excluded_keys)
        curation_ledger.loc[excluded_mask, "included"] = False
        curation_ledger.loc[excluded_mask, "exclusion_reason"] = \
            "required_datawarrior_descriptors_nonfinite"
    curated = curated.iloc[eligible_indices].reset_index(drop=True)
    resolution["track_e2_protocol"]["fixed_panel_descriptor_eligibility"] = {
        "input_parent_identities": int(eligibility["input_rows"]),
        "eligible_parent_identities": len(curated),
        "excluded_parent_identities": int(eligibility["excluded_rows"]),
        "excluded_compound_ids": eligibility["excluded_compound_ids"],
        "rule_uses_endpoint_labels": False,
        "positive_after_exclusion": int(curated["label_significant_positive"].sum()),
        "no_recorded_significant_extension_after_exclusion": int(
            (curated["label_significant_positive"] == 0).sum()),
    }

    contract = _checkpoint_contract(root, run_id, protocol_sha, resolution, curated)
    binding_sha = bind_checkpoint_directory(
        checkpoint_dir, contract, error_cls=EndpointBenchmarkError)
    labels = curated["label_significant_positive"].to_numpy(int)

    variant_predictions, sensitivity_rows = [], []
    oof: pd.DataFrame | None = None
    for variant in LABEL_VARIANTS:
        active_labels = curated[f"label_{variant}"].to_numpy(int)
        mask = active_labels >= 0
        checkpoint = checkpoint_dir / f"variant_{variant}_oof.csv"
        active_oof = cross_validate(
            object(), {}, active_labels[mask], n_splits=N_SPLITS, seed=SEED,
            checkpoint=checkpoint, checkpoint_binding_sha256=binding_sha,
            tag=f"DrugAge finalize {variant}")
        _validate_checkpoint(active_oof, active_labels[mask], folds=N_SPLITS,
                             role=f"endpoint variant {variant}")
        decorated = active_oof.copy()
        decorated.insert(0, "compound_id", curated.loc[mask, "compound_id"].to_numpy())
        decorated.insert(0, "endpoint_variant", variant)
        variant_predictions.append(decorated)
        piece = metric_table(active_oof)
        piece.insert(0, "endpoint_variant", variant)
        piece.insert(1, "n_compounds", int(mask.sum()))
        piece.insert(2, "primary", bool(LABEL_VARIANTS[variant]["primary"]))
        sensitivity_rows.append(piece)
        if LABEL_VARIANTS[variant]["primary"]:
            oof = active_oof.copy()
    if oof is None:
        raise EndpointBenchmarkError("Primary endpoint checkpoint is absent")
    pooled = metric_table(oof)
    per_fold = metric_table(oof, group="fold")
    endpoint_sensitivity = pd.concat(sensitivity_rows, ignore_index=True)
    variant_predictions_frame = pd.concat(variant_predictions, ignore_index=True)
    fold_mean_sd = per_fold.groupby("model_id")[list(PRIMARY_METRICS)].agg(["mean", "std"])
    fold_mean_sd.columns = [f"{metric}_{stat}" for metric, stat in fold_mean_sd.columns]
    fold_mean_sd = fold_mean_sd.reset_index()

    repeated_predictions = []
    for repeat_seed in REPEATED["seeds"]:
        checkpoint = checkpoint_dir / f"repeat5x5_seed{repeat_seed}.csv"
        repeat_oof = cross_validate(
            object(), {}, labels, n_splits=REPEATED["folds"], seed=repeat_seed,
            checkpoint=checkpoint, checkpoint_binding_sha256=binding_sha,
            tag=f"DrugAge finalize repeated seed {repeat_seed}")
        _validate_checkpoint(repeat_oof, labels, folds=REPEATED["folds"],
                             role=f"repeated seed {repeat_seed}")
        repeat_oof.insert(0, "repeat_seed", int(repeat_seed))
        repeat_oof.insert(0, "compound_id", curated["compound_id"].to_numpy())
        repeated_predictions.append(repeat_oof)
    repeated_predictions_frame = pd.concat(repeated_predictions, ignore_index=True)
    repeated_metrics = []
    for repeat_seed, block in repeated_predictions_frame.groupby("repeat_seed"):
        piece = metric_table(block)
        piece.insert(0, "repeat_seed", int(repeat_seed))
        repeated_metrics.append(piece)
    repeated_metrics_frame = pd.concat(repeated_metrics, ignore_index=True)

    publication_groups, component_audit = _publication_components(
        curated["publication_ids_json"])
    publication_fold, publication_fold_audit, publication_feasibility = \
        publication_fold_feasibility(labels, publication_groups, folds=5, seed=20260825)
    publication_feasibility["component_audit"] = component_audit
    if publication_feasibility["feasible"]:
        raise EndpointBenchmarkError(
            "Publication-grouped analysis is now feasible; use the primary runner instead")
    publication_registry = pd.DataFrame({
        "compound_id": curated["compound_id"], "fold": publication_fold,
        "label": labels, "publication_component": publication_groups,
    })
    prediction_columns = ["compound_id", "fold", "label", "publication_component"] + \
        [f"p_{model}" for model in PANEL] + ["maximum_tanimoto_to_active_fit_fold"]
    publication_predictions = pd.DataFrame(columns=prediction_columns)
    publication_metrics = pd.DataFrame(columns=["model_id", *PRIMARY_METRICS])
    print("publication-grouped sensitivity: " + PUBLICATION_STATUS, flush=True)
    print(publication_fold_audit.to_string(index=False), flush=True)

    publication_rows = []
    for fold in sorted(oof["fold"].unique()):
        mask = (oof["fold"] == fold).to_numpy()
        inside = set().union(*[set(json.loads(v)) for v in
                               curated.loc[mask, "publication_ids_json"]]) or set()
        outside = set().union(*[set(json.loads(v)) for v in
                                curated.loc[~mask, "publication_ids_json"]]) or set()
        publication_rows.append({
            "fold": int(fold), "validation_compounds": int(mask.sum()),
            "validation_publications": int(len(inside)),
            "publications_also_in_fitting_folds": int(len(inside & outside)),
            "fraction_shared": float(len(inside & outside) / max(1, len(inside))),
            "publication_component_group_crossing": False,
        })
    publication_audit = pd.DataFrame(publication_rows)

    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    d1_frame, _audit = _read_sources(positive.resolve(), negative.resolve(), traditional)
    train_indices, _test_indices, _split = paper_split_indices(traditional)
    d1_smiles, _smiles_audit = _validated_raw_smiles(d1_frame)
    d1_bits = morgan_matrix([d1_smiles[i] for i in train_indices], morgan_generator())
    oof_table = attach_molecular_variables(
        curated, oof, cohort="drugage_celegans_oof", d1_train_bits=d1_bits)
    nonoverlap = oof_table[~oof_table["overlaps_d1_connectivity"]].copy()
    nonoverlap["cohort"] = "drugage_celegans_oof_no_d1_overlap"
    bias_table = pd.concat([oof_table, nonoverlap], ignore_index=True, sort=False)
    bias = _qed_audit(bias_table)
    control_rng = np.random.default_rng(20260825)
    control_baselines, control_permutations = fold_local_baselines(
        bias_table, control_rng, cohort="drugage_celegans_oof", permutations=100)
    property_shifts = class_shifts(
        bias_table, cohorts=("drugage_celegans_oof",
                             "drugage_celegans_oof_no_d1_overlap"))
    print("QED and basic-property analyses complete", flush=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".drugce-finalize-", dir=destination.parent))
    try:
        atomic_write_json(tmp / "SOURCE_RESOLUTION.json", resolution)
        atomic_write_json(tmp / "publication_group_feasibility.json",
                          publication_feasibility)
        (tmp / "historical_reproduction_status.md").write_text(
            "# Track E1 -- exact historical RF refit: BLOCKED_PARTIAL\n\n"
            "Target: Kapsiani and Howlin (2021), *Scientific Reports*, "
            "doi 10.1038/s41598-021-93070-6.\n\n"
            "The official supplement provides the exact 1,430 structures and labels "
            "plus all 69 selected MOE feature identities. The following inputs needed "
            "for an exact RF refit were not released and were not guessed:\n\n"
            + "\n".join(f"- {item}" for item in
                        resolution["track_e1_historical_reproduction"]["what_is_missing"])
            + "\n", encoding="utf-8")
        reference_status = pd.DataFrame([{
            "status": resolution["track_e1_historical_reproduction"]["status"],
            "target": resolution["track_e1_historical_reproduction"]["target"],
            "consequence": resolution["track_e1_historical_reproduction"]["consequence"],
            "what_is_missing": json.dumps(
                resolution["track_e1_historical_reproduction"]["what_is_missing"]),
        }])
        for name, frame in (
            ("reference_reproduction_metrics.csv", reference_status),
            ("observation_level_ledger.csv", ledger),
            ("compound_level_endpoint_ledger.csv", endpoints),
            ("identity_mapping_ledger.csv", curation_ledger),
            ("identity_conflicts.csv", conflicts),
            ("publication_group_audit.csv", publication_audit),
            ("publication_grouped_fold_audit.csv", publication_fold_audit),
            ("publication_grouped_fold_registry.csv", publication_registry),
            ("publication_grouped_oof_predictions.csv", publication_predictions),
            ("publication_grouped_metrics.csv", publication_metrics),
            ("fixed_panel_fold_metrics.csv", per_fold),
            ("fixed_panel_fold_mean_sd.csv", fold_mean_sd),
            ("fixed_panel_pooled_metrics.csv", pooled),
            ("endpoint_sensitivity_metrics.csv", endpoint_sensitivity),
            ("endpoint_variant_oof_predictions.csv", variant_predictions_frame),
            ("repeated5x5_oof_predictions.csv", repeated_predictions_frame),
            ("repeated5x5_metrics.csv", repeated_metrics_frame),
            ("drugage_qed_bias.csv", bias["associations"]),
            ("drugage_qed_stratum_recall.csv", bias["recall"]),
            ("drugage_qed_trend_tests.csv", bias["trends"]),
            ("drugage_qed_logistic_models.csv", bias["logistic"]),
            ("drugage_qed_adjusted_models.csv", bias["adjusted"]),
            ("drugage_bias_compound_table.csv", bias_table),
            ("drugage_simple_baseline_metrics.csv", control_baselines),
            ("drugage_label_permutation_controls.csv", control_permutations),
            ("drugage_basic_property_class_shifts.csv", property_shifts),
        ):
            _write_csv(tmp / name, frame)
        registry = oof[["fold", "label"]].copy()
        registry.insert(0, "compound_id", curated["compound_id"].to_numpy())
        _write_csv(tmp / "drugage_split_registry.csv", registry)
        predictions = oof.copy()
        predictions.insert(0, "compound_id", curated["compound_id"].to_numpy())
        predictions["overlaps_d1_connectivity"] = \
            curated["overlaps_d1_connectivity"].to_numpy()
        _write_csv(tmp / "fixed_panel_oof_predictions.csv", predictions)

        lines = [
            "# Experiment E -- DrugAge *C. elegans* endpoint benchmark", "",
            f"run_id: `{run_id}`", "",
            "**Scientific role.** Organism-specific endpoint-aligned retraining; "
            "not external validation of a D1-fitted model.", "",
            "## Completion status", "",
            f"- primary ten-fold CV: complete ({len(curated)} identities)",
            "- repeated five-by-five CV: complete",
            f"- publication-grouped sensitivity: **{PUBLICATION_STATUS}**",
            "- no grouped model was fitted and no grouped metric was fabricated",
            "- downstream QED and basic-property analyses: complete", "",
            "## Pooled ten-fold OOF metrics", "",
            md_table(pooled[["model_id", *PRIMARY_METRICS]], "{:.4f}"), "",
            "## Endpoint-variant sensitivity", "",
            md_table(endpoint_sensitivity[
                ["endpoint_variant", "n_compounds", "primary", "model_id",
                 "auprc_average_precision_positive", "auroc", "mcc",
                 "recall_sensitivity", "specificity"]], "{:.4f}"), "",
            "## Publication-grouped feasibility", "",
            md_table(publication_fold_audit, "{:.3f}"), "",
            "## QED bias replication", "",
            md_table(bias["associations"][
                ["cohort", "model_id", "n_positive", "spearman_rho", "rho_ci_low",
                 "rho_ci_high", "q_holm_within_cohort"]], "{:.3f}"), "",
        ]
        (tmp / "drugage_celegans_summary.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")

        panel = Panel(root)
        manifest = {
            "schema_version": f"{SCHEMA}.run_manifest.v1",
            "run_id": run_id,
            "protocol_sha256": protocol_sha,
            "checkpoint_binding_sha256": binding_sha,
            "scientific_role": "organism_specific_endpoint_aligned_benchmark",
            "is_external_validation_of_the_d1_model": False,
            "conflated_with_all_organism_stress_test": False,
            "source_resolution": resolution,
            "label_variants": LABEL_VARIANTS,
            "primary_variant": "significant_positive",
            "variant_selected_by_model_performance": False,
            "curation": {
                "compound_endpoints": int(len(endpoints)),
                "curated_identities": int(len(curated)),
                "excluded_rows": int((~curation_ledger.included).sum()),
                "conflict_groups": int(len(conflicts)),
                "d1_overlap_identities": int(curated.overlaps_d1_connectivity.sum()),
            },
            "features": feature_audit,
            "panel": list(PANEL),
            "resolved_model_settings": panel.resolved_settings,
            "folds": (f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, "
                      f"random_state={SEED}) on curated parent identities"),
            "fixed_threshold": 0.5,
            "threshold_is_tuned": False,
            "publication_group_audit": publication_rows,
            "publication_component_audit": component_audit,
            "publication_grouped_cv_executed": False,
            "publication_grouped_cv_status": PUBLICATION_STATUS,
            "publication_grouped_cv_feasibility": publication_feasibility,
            "endpoint_variants_refitted_separately": True,
            "repeated_5x5_executed": True,
            "finalization": {
                "mode": "hash_verified_checkpoint_recovery",
                "models_refitted_during_finalization": False,
                "checkpoint_files_retained": True,
                "finalizer_source_sha256": sha256_file(
                    root / "src" / "geroprotector" / "drugage_celegans_finalize.py"),
                "original_runner_source_sha256": contract["code_sha256"][
                    "drugage_celegans_benchmark.py"],
            },
            "runtime": {"python": platform.python_version(),
                        "platform": platform.platform()},
            "existing_runs_modified": False,
        }
        atomic_write_json(tmp / "RUN_MANIFEST.json", manifest)
        atomic_write_json(tmp / "COMPLETED.json", {
            "schema_version": f"{SCHEMA}.completed.v1",
            "status": COMPLETION_STATUS,
            "run_id": run_id,
            "run_manifest_sha256": sha256_file(tmp / "RUN_MANIFEST.json"),
            "artifact_hashes": {
                str(path.relative_to(tmp)): sha256_file(path)
                for path in sorted(tmp.rglob("*"))
                if path.is_file() and path.name != "COMPLETED.json"
            },
        })
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    print(json.dumps({"run": str(destination), "status": COMPLETION_STATUS}, indent=2))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    run(root=args.root, config_path=args.config, positive=args.positive,
        negative=args.negative, run_id=args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
