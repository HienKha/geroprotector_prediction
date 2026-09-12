"""Fixed-panel reconstruction on the official Kapsiani-Howlin cohort.

The official supplement exposes the exact 1,430 structures and binary endpoint
(304 DrugAge Build 3 positives and 1,126 literature negatives) as well as the 69
selected MOE feature identities.  This runner starts from that exact source
cohort, applies the same prespecified structure curation used by the insight
suite, and evaluates the fixed ten-model panel in a new deterministic ten-fold
registry.

It is a protocol reconstruction, not a reproduction of the article's random
80/20 split or its MOE random forest.  Those remain partially blocked because
the split membership, 354-value MOE matrix, complete software environment and
RF random state were not released.
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

from geroprotector.study_common import bind_checkpoint_directory, md_table
from geroprotector.model_panel import PANEL, Panel
from geroprotector.endpoint_benchmark import (
    EndpointBenchmarkError,
    PRIMARY_METRICS,
    build_features,
    cross_validate,
    curate,
    d1_connectivity_keys,
    metric_table,
)
from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.official_sources import require_pinned_file, xlsx_table


SCHEMA = "geroprotector.kapsiani_historical_benchmark"
N_SPLITS = 10
SEED = 42
EXPECTED = {"rows": 1430, "positive": 304, "literature_negative": 1126,
            "selected_moe_features": 69}


class KapsianiBenchmarkError(RuntimeError):
    """Raised when the official cohort or reconstruction contract differs."""


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != f"{SCHEMA}.protocol.v1":
        raise KapsianiBenchmarkError("Unknown Kapsiani historical protocol")
    if protocol.get("expected") != EXPECTED:
        raise KapsianiBenchmarkError("Kapsiani source-count contract differs")
    evaluation = protocol["evaluation"]
    if (evaluation != {"splitter": "StratifiedKFold", "n_splits": N_SPLITS,
                       "shuffle": True, "random_state": SEED,
                       "fixed_threshold": 0.5,
                       "threshold_tuning": False}):
        raise KapsianiBenchmarkError("Kapsiani evaluation contract differs")
    if protocol.get("panel") != list(PANEL):
        raise KapsianiBenchmarkError("Kapsiani fixed model panel differs")
    return protocol, sha256_file(path)


def resolve_sources(protocol: dict[str, Any]
                    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source = protocol["source"]
    workbook = require_pinned_file(
        source["path"], source["sha256"], role="Kapsiani official supplement")
    cohort = xlsx_table(
        workbook, "DrugAge Database",
        required_columns=("Chemical Name", "Canonical SMILES", "Target"))
    feature_ranking = xlsx_table(
        workbook, "Feature Ranking",
        required_columns=("Descriptor", '"Gini" index'))
    cohort["Target"] = cohort["Target"].astype(int)
    counts = cohort["Target"].value_counts().to_dict()
    if (len(cohort) != EXPECTED["rows"] or
            counts != {0: EXPECTED["literature_negative"],
                       1: EXPECTED["positive"]}):
        raise KapsianiBenchmarkError(
            f"Official Kapsiani cohort differs: rows={len(cohort)}, counts={counts}")
    if len(feature_ranking) != EXPECTED["selected_moe_features"]:
        raise KapsianiBenchmarkError("Official selected MOE feature count differs")
    if canonical_sha256(cohort.to_dict(orient="records")) != \
            source["cohort_records_sha256"]:
        raise KapsianiBenchmarkError("Official ordered cohort record hash differs")
    if canonical_sha256(feature_ranking["Descriptor"].astype(str).tolist()) != \
            source["selected_feature_names_sha256"]:
        raise KapsianiBenchmarkError("Official MOE feature identity hash differs")

    resolution = {
        "citations": protocol["citations"],
        "official_supplement": {"path": str(workbook),
                                "sha256": sha256_file(workbook)},
        "official_source_cohort": {
            "rows": len(cohort), "positive": counts[1],
            "negative": counts[0],
            "negative_class_name": "curated_literature_negative",
            "negative_source": "Barardo_et_al_2017_literature_negative_panel",
            "drugage_build": "Build_3_for_positive_class"},
        "published_model": {
            "initial_moe_descriptors": 354,
            "selected_moe_features": len(feature_ranking),
            "classifier": "RandomForestClassifier",
            "n_estimators": 100, "class_weight": "balanced",
            "max_features": "sqrt",
            "published_test": protocol["published_test"]},
        "availability": {
            "official_cohort_and_labels": "REPRODUCIBLE",
            "selected_moe_feature_identities": "REPRODUCIBLE",
            "exact_historical_rf_refit": "BLOCKED_PARTIAL",
            "fixed_panel_evaluation": "PROTOCOL_RECONSTRUCTION"},
        "exact_historical_rf_refit_blockers": protocol["historical_rf"]["blockers"],
        "selection_description": {
            "paper_procedure": "variance_filter_then_adjusted_mutual_information_ranking_on_the_80_percent_training_partition",
            "nested_inside_each_cv_fold": False,
            "fixed_panel_reconstruction_reuses_paper_feature_selection": False},
        "not_drugage_build5": True,
    }
    return cohort, feature_ranking, resolution


def run(*, root: Path, config_path: Path, positive: Path, negative: Path,
        run_id: str) -> Path:
    if not re.fullmatch(r"kapsiani_historical_benchmark_[a-z0-9_.-]+", run_id):
        raise KapsianiBenchmarkError(
            "RUN_ID must start with kapsiani_historical_benchmark_")
    root = root.resolve()
    protocol, protocol_sha = load_protocol(config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise KapsianiBenchmarkError(f"Run directory already exists: {destination}")

    source_cohort, selected_features, resolution = resolve_sources(protocol)
    d1_keys, d1_audit = d1_connectivity_keys(root, positive, negative)
    resolution["d1_overlap_reference"] = d1_audit
    raw = pd.DataFrame({
        "provider_record_id": [f"official_row_{index}" for index in range(len(source_cohort))],
        "compound_name": source_cohort["Chemical Name"].astype(str),
        "source_smiles": source_cohort["Canonical SMILES"].astype(str),
        "label": source_cohort["Target"].astype(int),
    })
    curated, ledger, conflicts = curate(
        raw, cohort="kapsiani_historical", d1_connectivity=d1_keys)
    if set(curated["label"].astype(int)) != {0, 1}:
        raise KapsianiBenchmarkError("Kapsiani analysis cohort lost one endpoint class")
    resolution["structure_curated_cohort_before_descriptor_eligibility"] = {
        "source_rows": len(source_cohort),
        "curated_parent_identities": len(curated),
        "positive": int(curated["label"].sum()),
        "curated_literature_negative": int((curated["label"] == 0).sum()),
        "excluded_provider_rows": int((~ledger["included"]).sum()),
        "conflicting_parent_identities": len(conflicts),
        "metrics_apply_to": "curated_structure_eligible_parent_identity_subset",
    }

    features, feature_audit = build_features(
        root, curated, verify_d1_parity=True,
        nonfinite_descriptor_policy="exclude")
    eligibility = feature_audit["descriptor_eligibility"]
    eligible_indices = np.asarray(eligibility["eligible_input_indices"], dtype=int)
    descriptor_excluded = curated.iloc[
        np.asarray(eligibility["excluded_input_indices"], dtype=int)].copy()
    if len(eligible_indices) != len(features["paper"]):
        raise KapsianiBenchmarkError(
            "Descriptor eligibility mask and feature rows differ")
    if not descriptor_excluded.empty:
        excluded_keys = set(descriptor_excluded["connectivity_inchikey"].astype(str))
        mask = ledger["connectivity_inchikey"].astype(str).isin(excluded_keys)
        ledger.loc[mask, "included"] = False
        ledger.loc[mask, "exclusion_reason"] = \
            "required_datawarrior_descriptors_nonfinite"
    curated = curated.iloc[eligible_indices].reset_index(drop=True)
    resolution["fixed_panel_analysis_cohort"] = {
        "source_rows": len(source_cohort),
        "curated_parent_identities_before_descriptor_eligibility":
            int(eligibility["input_rows"]),
        "descriptor_eligible_parent_identities": len(curated),
        "positive": int(curated["label"].sum()),
        "curated_literature_negative": int((curated["label"] == 0).sum()),
        "excluded_provider_rows_total": int((~ledger["included"]).sum()),
        "descriptor_ineligible_parent_identities": int(eligibility["excluded_rows"]),
        "descriptor_ineligible_compound_ids": eligibility["excluded_compound_ids"],
        "conflicting_parent_identities": len(conflicts),
        "descriptor_exclusion_rule_uses_endpoint_labels": False,
        "metrics_apply_to": "curated_structure_and_fixed_panel_descriptor_eligible_parent_identity_subset",
    }
    checkpoint_dir = root / "outputs" / f".{run_id}.work"
    checkpoint_contract = {
        "run_id": run_id, "protocol_sha256": protocol_sha,
        "official_supplement_sha256": resolution["official_supplement"]["sha256"],
        "curated_compound_ids_sha256": canonical_sha256(
            curated["compound_id"].astype(str).tolist()),
        "curated_labels_sha256": canonical_sha256(
            curated["label"].astype(int).tolist()),
        "code_sha256": {name: sha256_file(root / "src" / "geroprotector" / name)
                         for name in ("kapsiani_historical_benchmark.py",
                                      "endpoint_benchmark.py", "model_panel.py",
                                      "official_sources.py")},
    }
    binding_sha = bind_checkpoint_directory(
        checkpoint_dir, checkpoint_contract, error_cls=KapsianiBenchmarkError)
    panel = Panel(root)
    labels = curated["label"].to_numpy(int)
    oof = cross_validate(
        panel, features, labels, n_splits=N_SPLITS, seed=SEED,
        checkpoint=checkpoint_dir / "primary_oof.csv",
        checkpoint_binding_sha256=binding_sha, tag="kapsiani historical")
    pooled = metric_table(oof)
    per_fold = metric_table(oof, group="fold")
    mean_sd = per_fold.groupby("model_id")[list(PRIMARY_METRICS)].agg(["mean", "std"])
    mean_sd.columns = [f"{metric}_{stat}" for metric, stat in mean_sd.columns]
    mean_sd = mean_sd.reset_index()
    no_d1 = ~curated["overlaps_d1_connectivity"].to_numpy(bool)
    nonoverlap = metric_table(oof[no_d1])

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".kapsiani-historical-", dir=destination.parent))
    try:
        atomic_write_json(tmp / "SOURCE_RESOLUTION.json", resolution)
        _write_csv(tmp / "official_1430_source_cohort.csv", source_cohort)
        _write_csv(tmp / "official_69_selected_moe_features.csv", selected_features)
        _write_csv(tmp / "raw_to_curated_ledger.csv", ledger)
        _write_csv(tmp / "identity_conflicts.csv", conflicts)
        overlap = curated[["compound_id", "compound_name", "connectivity_inchikey",
                           "label", "overlaps_d1_connectivity"]].copy()
        overlap["negative_class_name"] = np.where(
            overlap["label"] == 0, "curated_literature_negative", "not_applicable")
        _write_csv(tmp / "d1_overlap_ledger.csv", overlap)
        registry = oof[["fold", "label"]].copy()
        registry.insert(0, "compound_id", curated["compound_id"].to_numpy())
        _write_csv(tmp / "kapsiani_split_registry.csv", registry)
        predictions = oof.copy()
        predictions.insert(0, "compound_id", curated["compound_id"].to_numpy())
        _write_csv(tmp / "fixed_panel_oof_predictions.csv", predictions)
        _write_csv(tmp / "fixed_panel_fold_metrics.csv", per_fold)
        _write_csv(tmp / "fixed_panel_fold_mean_sd.csv", mean_sd)
        _write_csv(tmp / "fixed_panel_pooled_metrics.csv", pooled)
        _write_csv(tmp / "d1_nonoverlap_sensitivity_metrics.csv", nonoverlap)
        historical_status = pd.DataFrame([{
            "status": "BLOCKED_PARTIAL",
            "target": "exact historical MOE Random Forest refit",
            "available": json.dumps([
                "official 1430-row structures and labels",
                "official 69 selected MOE feature identities",
                "published model family, settings, test counts and metrics"]),
            "blockers": json.dumps(protocol["historical_rf"]["blockers"]),
        }])
        _write_csv(tmp / "historical_rf_reproduction_status.csv", historical_status)
        lines = ["# Kapsiani-Howlin historical-endpoint fixed-panel reconstruction", "",
                 "**Scientific role.** A new deterministic fixed-panel evaluation on "
                 "the article's official historical endpoint. It is not an exact "
                 "refit of the MOE Random Forest or its hidden 80/20 split.", "",
                 f"- source cohort: {len(source_cohort)} compounds, "
                 f"{int(source_cohort.Target.sum())} DrugAge Build 3 positives and "
                 f"{int((source_cohort.Target == 0).sum())} curated literature negatives",
                 f"- structure-eligible analysis cohort: {len(curated)} parent identities",
                 f"- D1-overlapping parent identities: "
                 f"{int(curated.overlaps_d1_connectivity.sum())}",
                 f"- evaluation: {N_SPLITS}-fold stratified CV, seed {SEED}, "
                 "fixed threshold 0.5", "",
                 "## Pooled out-of-fold metrics", "",
                 md_table(pooled[["model_id"] + list(PRIMARY_METRICS)], "{:.4f}"), "",
                 "The Build 5 C. elegans task is separate. Its background label is "
                 "`no_recorded_significant_extension` and is not this literature-negative class.", ""]
        (tmp / "kapsiani_historical_summary.md").write_text(
            "\n".join(lines), encoding="utf-8")
        atomic_write_json(tmp / "RUN_MANIFEST.json", {
            "schema_version": f"{SCHEMA}.run_manifest.v1", "run_id": run_id,
            "protocol_sha256": protocol_sha,
            "checkpoint_binding_sha256": binding_sha,
            "scientific_role": "fixed_panel_protocol_reconstruction_on_historical_endpoint",
            "exact_historical_rf_reproduction": False,
            "drugage_build5_conflated": False,
            "source_resolution": resolution,
            "features": feature_audit, "panel": list(PANEL),
            "resolved_model_settings": panel.resolved_settings,
            "folds": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={SEED})",
            "fixed_threshold": 0.5, "threshold_is_tuned": False,
            "outcomes_used_for_model_or_feature_selection": False,
            "runtime": {"python": platform.python_version(),
                        "platform": platform.platform()},
            "existing_runs_modified": False})
        atomic_write_json(tmp / "COMPLETED.json", {
            "schema_version": f"{SCHEMA}.completed.v1",
            "status": "COMPLETE_PROTOCOL_RECONSTRUCTION", "run_id": run_id,
            "run_manifest_sha256": sha256_file(tmp / "RUN_MANIFEST.json"),
            "artifact_hashes": {str(path.relative_to(tmp)): sha256_file(path)
                                for path in sorted(tmp.rglob("*"))
                                if path.is_file() and path.name != "COMPLETED.json"}})
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    print(json.dumps({"run": str(destination), "status": "COMPLETE"}, indent=2))
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
