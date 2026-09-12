"""Experiment E: the DrugAge *C. elegans* organism-specific endpoint benchmark.

Scientific role (section 9.1).  This retrains the fixed architectures under the
operational endpoint `does DrugAge Build 5 contain at least one recorded
significant average-lifespan extension for this compound in Caenorhabditis
elegans`.  Its background means ``no_recorded_significant_extension`` and is
not a certified biological negative.  It must not be conflated with the project's existing all-organism
DrugAge positive-retrieval stress test, which scores the D1-fitted model without
refitting.

Two source tracks are prescribed.

  E1  Exact reproduction of Kapsiani and Howlin (2021), doi
      10.1038/s41598-021-93070-6.  The official supplement does provide the exact
      1,430-compound cohort, labels and 69 selected MOE feature identities.  Only
      the historical RF refit is BLOCKED_PARTIAL because the split registry,
      descriptor matrix and complete software/RNG state were not released.  A
      separate fixed-panel protocol reconstruction is implemented by
      ``kapsiani_historical_benchmark``.
  E2  Endpoint reconstruction from the pinned DrugAge Build 5 export.  Three
      label variants are declared in advance, one of them primary, and the rule
      is never chosen by model performance.

Publication identity is retained so a paper-grouped sensitivity analysis can
check that a single high-throughput assay context does not sit in both the
fitting and the validation folds.
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
from sklearn.model_selection import StratifiedGroupKFold

from geroprotector.chemistry.standardize import standardize_smiles
from geroprotector.study_common import (bind_checkpoint_directory, md_table,
                                        morgan_generator, morgan_matrix)
from geroprotector.model_panel import PANEL, Panel
from geroprotector.endpoint_benchmark import (
    EndpointBenchmarkError,
    PRIMARY_METRICS,
    attach_molecular_variables,
    build_features,
    cross_validate,
    curate,
    d1_connectivity_keys,
    metric_table,
)
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.modelwide_druglikeness_bias import (
    class_shifts,
    fold_local_baselines,
    logistic_models,
    positive_associations,
    stratum_recall,
    trend_tests,
)
from geroprotector.official_sources import require_pinned_file, xlsx_table
from geroprotector.traditional_paper405 import _read_sources, paper_split_indices

SCHEMA = "geroprotector.drugage_celegans_benchmark"
SOURCE_DIR = Path(os.environ.get("DRUGAGE_SOURCE_DIR", "external_data/drugage_build5"))
OBSERVATIONS = "drugage.csv"
MAPPING = "drugage_pubchem_smiles.csv"
HISTORICAL_SUPPLEMENT = Path(os.environ.get(
    "KAPSIANI_SUPPLEMENT",
    "external_sources/41598_2021_93070_MOESM2_ESM.xlsx",
))
HISTORICAL_SUPPLEMENT_SHA256 = \
    "e98da1592a44199ab809c7a42e018d7be255b6ae11b0b26829521f60fe3ae2c3"
ORGANISM = "Caenorhabditis elegans"
N_SPLITS = 10
SEED = 42
REPEATED = {"repeats": 5, "folds": 5, "seeds": [42, 43, 44, 45, 46]}
CITATIONS = {
    "historical_reference": "https://doi.org/10.1038/s41598-021-93070-6",
    "historical_official_supplement":
        "https://media.springernature.com/original/springer-static/esm/"
        "art%3A10.1038%2Fs41598-021-93070-6/MediaObjects/"
        "41598_2021_93070_MOESM2_ESM.xlsx",
    "drugage": "https://genomics.senescence.info/drugs/",
    "drugage_scope_and_negative_result_caveat":
        "https://genomics.senescence.info/help.html",
}
# Declared before any model is fitted.  `primary` is fixed here, not chosen later.
LABEL_VARIANTS = {
    "significant_positive": {
        "primary": True,
        "positive": "at least one C. elegans observation with significance code S "
                    "and average lifespan change greater than zero",
        "negative": "no_recorded_significant_extension: has C. elegans observations "
                    "but none meeting the positive rule",
        "negative_class_name": "no_recorded_significant_extension",
        "note": "a modern operational Build 5 endpoint; not the Kapsiani-Howlin "
                "curated literature-negative class and not a certified biological "
                "non-geroprotector endpoint",
    },
    "consistent_positive": {
        "primary": False,
        "positive": "every C. elegans observation shows an average lifespan "
                    "change greater than zero",
        "negative": "every C. elegans observation shows a non-increase",
        "note": "strict sensitivity rule; compounds that are neither are excluded",
    },
    "conflict_excluded": {
        "primary": False,
        "positive": "significant_positive rule",
        "negative": "significant_positive rule",
        "note": "compounds carrying both a significant increase and a significant "
                "decrease are excluded entirely, because a lifespan-shortening "
                "observation is not a clean biological non-geroprotector signal",
    },
}


def _load_endpoint_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != f"{SCHEMA}.protocol.v1":
        raise EndpointBenchmarkError("Unknown DrugAge protocol schema")
    if protocol.get("panel") != list(PANEL):
        raise EndpointBenchmarkError("DrugAge protocol changes the fixed model panel")
    primary = protocol["evaluation"]["primary"]
    if (int(primary["n_splits"]) != N_SPLITS or int(primary["random_state"]) != SEED or
            float(protocol["evaluation"]["fixed_threshold"]) != 0.5):
        raise EndpointBenchmarkError("DrugAge primary evaluation differs from the lock")
    if protocol["evaluation"]["repeated_sensitivity"] != REPEATED:
        raise EndpointBenchmarkError("DrugAge repeated-CV settings differ from the lock")
    return protocol, sha256_file(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def resolve_sources(protocol: dict[str, Any] | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    observations = pd.read_csv(SOURCE_DIR / OBSERVATIONS)
    mapping = pd.read_csv(SOURCE_DIR / MAPPING)
    if len(observations) != 3423:
        raise EndpointBenchmarkError(
            f"Pinned DrugAge export has {len(observations)} rows, expected 3423")
    organism = observations[observations["species"].astype(str).str.strip() == ORGANISM]
    historical_path = HISTORICAL_SUPPLEMENT
    historical_sha = HISTORICAL_SUPPLEMENT_SHA256
    if protocol is not None:
        historical = protocol["sources"]["historical_official_supplement"]
        historical_path, historical_sha = Path(historical["path"]), historical["sha256"]
    historical_path = require_pinned_file(
        historical_path, historical_sha, role="Kapsiani official supplement")
    historical_cohort = xlsx_table(
        historical_path, "DrugAge Database",
        required_columns=("Chemical Name", "Canonical SMILES", "Target"))
    historical_features = xlsx_table(
        historical_path, "Feature Ranking",
        required_columns=("Descriptor", '"Gini" index'))
    historical_counts = historical_cohort["Target"].astype(int).value_counts().to_dict()
    if len(historical_cohort) != 1430 or historical_counts != {0: 1126, 1: 304}:
        raise EndpointBenchmarkError("Official Kapsiani historical cohort differs")
    if len(historical_features) != 69:
        raise EndpointBenchmarkError("Official Kapsiani selected-feature list differs")
    resolution = {
        "citations": CITATIONS,
        "source_files": {name: sha256_file(SOURCE_DIR / name)
                         for name in (OBSERVATIONS, MAPPING)},
        "pinned_export_rows": int(len(observations)),
        "organism": ORGANISM,
        "organism_rows": int(len(organism)),
        "organism_compound_names": int(organism["compound_name"].nunique()),
        "structure_mapping_rows": int(len(mapping)),
        "label_variants_declared_in_advance": LABEL_VARIANTS,
        "track_e1_historical_reproduction": {
            "status": "BLOCKED_PARTIAL",
            "scope": "exact_historical_moe_random_forest_refit_only",
            "target": "Kapsiani and Howlin 2021, Sci Rep, doi 10.1038/s41598-021-93070-6",
            "what_is_available": [
                "the official 1430-row compound, canonical SMILES and target table",
                "the 304-positive and 1126-literature-negative endpoint",
                "the exact 69 selected MOE descriptor identities and Gini rankings",
                "the published model family, settings, test counts and metrics"],
            "what_is_missing": [
                "the exact random 80/20 split membership and seed",
                "the ten-fold assignments",
                "the original 354-by-1430 MOE descriptor value matrix",
                "the proprietary MOE 2019.01 minimization environment",
                "the scikit-learn version and Random Forest random state",
                "the fitted Random Forest artifact"],
            "consequence": ("The source cohort can be reconstructed exactly and a "
                            "separate fixed-panel protocol reconstruction is feasible. "
                            "Only the article's exact MOE Random Forest refit remains "
                            "partially blocked."),
            "not_attempted_by_guessing": True},
        "historical_official_source": {
            "path": str(historical_path), "sha256": sha256_file(historical_path),
            "rows": len(historical_cohort), "positive": historical_counts[1],
            "curated_literature_negative": historical_counts[0],
            "selected_moe_features": len(historical_features),
            "ordered_records_canonical_sha256": canonical_sha256(
                historical_cohort.assign(
                    Target=historical_cohort["Target"].astype(int)).to_dict(
                        orient="records")),
            "selected_feature_names_canonical_sha256": canonical_sha256(
                historical_features["Descriptor"].astype(str).tolist()),
            "fixed_panel_reconstruction":
                "geroprotector.kapsiani_historical_benchmark"},
        "track_e2_protocol": {
            "kind": "modern_build5_operational_endpoint_reconstruction",
            "negative_class_name": "no_recorded_significant_extension",
            "negative_class_is_certified_biological_negative": False,
            "not_equivalent_to_kapsiani_literature_negative": True,
            "n_splits": N_SPLITS, "shuffle": True, "random_state": SEED,
            "stratified": True, "unit": "curated_parent_identity",
            "repeated_cv_sensitivity": REPEATED},
    }
    return observations, mapping, resolution


def observation_ledger(observations: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """One row per C. elegans observation, with its mapped structure and direction."""
    organism = observations[
        observations["species"].astype(str).str.strip() == ORGANISM].copy()
    organism["observation_id"] = [f"drugage_ce_{i}" for i in organism.index]
    organism["name_key"] = organism["compound_name"].astype(str).str.strip().str.lower()
    structures = mapping.dropna(subset=["canonical_smiles"]).copy()
    structures["name_key"] = structures["compound_name"].astype(str).str.strip().str.lower()
    structures = structures.drop_duplicates("name_key")[
        ["name_key", "cid", "canonical_smiles", "isomeric_smiles", "inchikey"]]
    ledger = organism.merge(structures, on="name_key", how="left")
    ledger["average_change"] = pd.to_numeric(
        ledger["avg_lifespan_change_percent"], errors="coerce")
    ledger["maximum_change"] = pd.to_numeric(
        ledger["max_lifespan_change_percent"], errors="coerce")
    ledger["significance"] = ledger["avg_lifespan_significance"].astype(str).str.strip().str.upper()
    ledger["significant_increase"] = ((ledger["significance"] == "S") &
                                      (ledger["average_change"] > 0))
    ledger["significant_decrease"] = ((ledger["significance"] == "S") &
                                      (ledger["average_change"] < 0))
    ledger["direction"] = np.where(ledger["average_change"] > 0, "increase",
                          np.where(ledger["average_change"] < 0, "decrease", "none"))
    ledger["structure_mapped"] = ledger["canonical_smiles"].notna()
    parents, connectivity, full_keys = [], [], []
    for mapped, smiles in zip(ledger["structure_mapped"], ledger["canonical_smiles"]):
        if not mapped:
            parents.append(""); connectivity.append(""); full_keys.append("")
            continue
        try:
            standardized = standardize_smiles(smiles)
        except Exception:
            parents.append(""); connectivity.append(""); full_keys.append("")
            continue
        parents.append(standardized.standardized_parent_smiles)
        connectivity.append(standardized.connectivity_inchikey)
        full_keys.append(standardized.full_inchikey)
    ledger["standardized_parent_smiles"] = parents
    ledger["connectivity_inchikey"] = connectivity
    ledger["standardized_full_inchikey"] = full_keys
    ledger["structure_standardized"] = ledger["connectivity_inchikey"].astype(str).ne("")
    return ledger[[
        "observation_id", "compound_name", "name_key", "species", "strain", "gender",
        "dosage", "age_at_initiation", "treatment_duration", "average_change",
        "avg_lifespan_significance", "maximum_change", "max_lifespan_significance",
        "pubmed_id", "cid", "canonical_smiles", "isomeric_smiles", "inchikey",
        "significance", "significant_increase", "significant_decrease", "direction",
        "structure_mapped", "structure_standardized", "standardized_parent_smiles",
        "connectivity_inchikey", "standardized_full_inchikey"]]


def compound_endpoints(ledger: pd.DataFrame) -> pd.DataFrame:
    """Compound-level labels under every declared variant, before any model runs."""
    ledger = ledger.copy()
    if "connectivity_inchikey" not in ledger.columns:
        standardized = [standardize_smiles(value) for value in ledger["canonical_smiles"]]
        ledger["standardized_parent_smiles"] = [x.standardized_parent_smiles for x in standardized]
        ledger["connectivity_inchikey"] = [x.connectivity_inchikey for x in standardized]
        ledger["structure_standardized"] = True
    if "structure_standardized" not in ledger.columns:
        ledger["structure_standardized"] = ledger["connectivity_inchikey"].astype(str).ne("")
    mapped = ledger[ledger["structure_standardized"]]
    rows = []
    for connectivity_key, group in mapped.groupby("connectivity_inchikey", sort=True):
        significant_increase = bool(group["significant_increase"].any())
        significant_decrease = bool(group["significant_decrease"].any())
        changes = group["average_change"].dropna()
        all_increase = bool(len(changes) > 0 and (changes > 0).all())
        all_nonincrease = bool(len(changes) > 0 and (changes <= 0).all())
        consistent = (1 if all_increase else 0 if all_nonincrease else -1)
        rows.append({
            "connectivity_inchikey": connectivity_key,
            "compound_name": group["compound_name"].iloc[0],
            "compound_names_json": json.dumps(sorted(set(group["compound_name"].astype(str)))),
            "name_keys_json": json.dumps(sorted(set(group["name_key"].astype(str)))),
            "canonical_smiles": group["standardized_parent_smiles"].iloc[0],
            "observations": int(len(group)),
            "publications": int(group["pubmed_id"].nunique()),
            "publication_ids_json": json.dumps(
                sorted({str(v) for v in group["pubmed_id"].dropna()})),
            "has_significant_increase": significant_increase,
            "has_significant_decrease": significant_decrease,
            "label_significant_positive": int(significant_increase),
            "label_consistent_positive": consistent,
            "label_conflict_excluded": (-1 if (significant_increase and significant_decrease)
                                        else int(significant_increase)),
            "conflict_status": ("increase_and_decrease"
                                if significant_increase and significant_decrease else "")})
    return pd.DataFrame(rows)


def _qed_audit(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    import geroprotector.modelwide_druglikeness_bias as bias
    original = bias.COHORTS
    bias.COHORTS = tuple(sorted(table["cohort"].unique()))
    try:
        associations = positive_associations(table)
        recall = stratum_recall(table)
        trends = trend_tests(recall)
        simple, adjusted = logistic_models(table)
    finally:
        bias.COHORTS = original
    return {"associations": associations, "recall": recall, "trends": trends,
            "logistic": simple, "adjusted": adjusted}


def _subset_features(features: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: values[mask] for name, values in features.items()}


def _publication_components(publication_json: pd.Series) -> tuple[np.ndarray, dict[str, Any]]:
    """Connected compound groups induced by shared PubMed IDs."""
    parent = list(range(len(publication_json)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    first: dict[str, int] = {}
    for row, text in enumerate(publication_json.astype(str)):
        for publication in json.loads(text):
            publication = str(publication)
            if publication in first:
                union(row, first[publication])
            else:
                first[publication] = row
    roots = [find(i) for i in range(len(parent))]
    remap = {root: index for index, root in enumerate(sorted(set(roots)))}
    groups = np.asarray([remap[root] for root in roots], dtype=int)
    sizes = pd.Series(groups).value_counts()
    return groups, {"n_publication_components": int(sizes.size),
                    "largest_publication_component": int(sizes.max()),
                    "singleton_publication_components": int((sizes == 1).sum())}


def run(*, root: Path, config_path: Path, positive: Path, negative: Path,
        run_id: str) -> Path:
    if not re.fullmatch(r"drugage_celegans_benchmark_[a-z0-9_.-]+", run_id):
        raise EndpointBenchmarkError("RUN_ID must start with drugage_celegans_benchmark_")
    root = root.resolve()
    protocol, protocol_sha = _load_endpoint_protocol(config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise EndpointBenchmarkError(f"Run directory already exists: {destination}")

    observations, mapping, resolution = resolve_sources(protocol)
    ledger = observation_ledger(observations, mapping)
    endpoints = compound_endpoints(ledger)
    print(f"C. elegans observations {len(ledger)} "
          f"({int(ledger.structure_mapped.sum())} with a mapped structure); "
          f"{len(endpoints)} compound-level endpoints", flush=True)

    d1_keys, d1_audit = d1_connectivity_keys(root, positive, negative)
    resolution["d1_overlap_reference"] = d1_audit

    primary = endpoints[["connectivity_inchikey", "compound_name", "canonical_smiles",
                         "label_significant_positive"]].copy()
    raw = pd.DataFrame({
        "provider_record_id": primary["connectivity_inchikey"],
        "compound_name": primary["compound_name"],
        "source_smiles": primary["canonical_smiles"].astype(str),
        "label": primary["label_significant_positive"].astype(int)})
    curated, curation_ledger, conflicts = curate(
        raw, cohort="drugage_celegans", d1_connectivity=d1_keys)
    curated = curated.merge(
        endpoints[["connectivity_inchikey", "publications", "publication_ids_json",
             "label_significant_positive", "label_consistent_positive",
             "label_conflict_excluded",
             "conflict_status", "observations"]],
        on="connectivity_inchikey", how="left", validate="one_to_one")
    if curated["publications"].isna().any():
        raise EndpointBenchmarkError("Publication annotation did not cover every identity")
    print(f"curated cohort {len(curated)} identities "
          f"({int(curated.label.sum())} positive), "
          f"{int((~curation_ledger.included).sum())} rows excluded, "
          f"{len(conflicts)} conflicting groups, "
          f"{int(curated.overlaps_d1_connectivity.sum())} overlap D1", flush=True)

    features, feature_audit = build_features(
        root, curated, verify_d1_parity=True,
        nonfinite_descriptor_policy="exclude")
    eligibility = feature_audit["descriptor_eligibility"]
    eligible_indices = np.asarray(eligibility["eligible_input_indices"], dtype=int)
    descriptor_excluded = curated.iloc[
        np.asarray(eligibility["excluded_input_indices"], dtype=int)].copy()
    if len(eligible_indices) != len(features["paper"]):
        raise EndpointBenchmarkError(
            "Descriptor eligibility mask and DrugAge feature rows differ")
    if not descriptor_excluded.empty:
        excluded_keys = set(descriptor_excluded["connectivity_inchikey"].astype(str))
        mask = curation_ledger["connectivity_inchikey"].astype(str).isin(excluded_keys)
        curation_ledger.loc[mask, "included"] = False
        curation_ledger.loc[mask, "exclusion_reason"] = \
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
    print("DataWarrior D1 parity verified; features built; "
          f"descriptor-ineligible identities excluded={int(eligibility['excluded_rows'])}",
          flush=True)

    checkpoint_dir = root / "outputs" / f".{run_id}.work"
    checkpoint_contract = {
        "run_id": run_id, "protocol_sha256": protocol_sha,
        "source_sha256": resolution["source_files"],
        "curated_compound_ids_sha256": canonical_sha256(
            curated["compound_id"].astype(str).tolist()),
        "endpoint_labels_sha256": {
            name: canonical_sha256(curated[f"label_{name}"].astype(int).tolist())
            for name in LABEL_VARIANTS},
        "code_sha256": {name: sha256_file(root / "src" / "geroprotector" / name)
                         for name in ("drugage_celegans_benchmark.py",
                                      "endpoint_benchmark.py", "model_panel.py")},
    }
    checkpoint_binding_sha = bind_checkpoint_directory(
        checkpoint_dir, checkpoint_contract, error_cls=EndpointBenchmarkError)
    panel = Panel(root)
    labels = curated["label_significant_positive"].to_numpy(int)

    # Each endpoint variant is a distinct endpoint-aligned retraining experiment.
    variant_predictions, sensitivity_rows = [], []
    primary_mask = np.ones(len(curated), dtype=bool)
    oof = None
    for variant in LABEL_VARIANTS:
        active_labels = curated[f"label_{variant}"].to_numpy(int)
        mask = active_labels >= 0
        active_curated = curated.loc[mask].reset_index(drop=True)
        active_features = _subset_features(features, mask)
        active_oof = cross_validate(
            panel, active_features, active_labels[mask], n_splits=N_SPLITS, seed=SEED,
            checkpoint=checkpoint_dir / f"variant_{variant}_oof.csv",
            checkpoint_binding_sha256=checkpoint_binding_sha,
            tag=f"drugage_ce variant {variant}")
        active_oof.insert(0, "compound_id", active_curated["compound_id"].to_numpy())
        active_oof.insert(0, "endpoint_variant", variant)
        variant_predictions.append(active_oof)
        piece = metric_table(active_oof)
        piece.insert(0, "endpoint_variant", variant)
        piece.insert(1, "n_compounds", int(mask.sum()))
        piece.insert(2, "primary", bool(LABEL_VARIANTS[variant]["primary"]))
        sensitivity_rows.append(piece)
        if LABEL_VARIANTS[variant]["primary"]:
            oof = active_oof.drop(columns=["endpoint_variant", "compound_id"])
            primary_mask = mask
    if oof is None or not primary_mask.all():
        raise EndpointBenchmarkError("Primary DrugAge endpoint did not cover the curated cohort")
    pooled = metric_table(oof)
    per_fold = metric_table(oof, group="fold")
    endpoint_sensitivity = pd.concat(sensitivity_rows, ignore_index=True)
    variant_predictions = pd.concat(variant_predictions, ignore_index=True)
    fold_mean_sd = per_fold.groupby("model_id")[list(PRIMARY_METRICS)].agg(["mean", "std"])
    fold_mean_sd.columns = [f"{metric}_{stat}" for metric, stat in fold_mean_sd.columns]
    fold_mean_sd = fold_mean_sd.reset_index()

    # Five repeats of five-fold CV, with model RNG held fixed at 42+fold.
    repeated_predictions = []
    for repeat_seed in REPEATED["seeds"]:
        repeat_oof = cross_validate(
            panel, features, labels, n_splits=REPEATED["folds"], seed=repeat_seed,
            checkpoint=checkpoint_dir / f"repeat5x5_seed{repeat_seed}.csv",
            checkpoint_binding_sha256=checkpoint_binding_sha,
            tag=f"drugage_ce repeated seed {repeat_seed}")
        repeat_oof.insert(0, "repeat_seed", int(repeat_seed))
        repeat_oof.insert(0, "compound_id", curated["compound_id"].to_numpy())
        repeated_predictions.append(repeat_oof)
    repeated_predictions = pd.concat(repeated_predictions, ignore_index=True)
    repeated_metrics = []
    for repeat_seed, block in repeated_predictions.groupby("repeat_seed"):
        piece = metric_table(block)
        piece.insert(0, "repeat_seed", int(repeat_seed))
        repeated_metrics.append(piece)
    repeated_metrics = pd.concat(repeated_metrics, ignore_index=True)

    # Publication-grouped sensitivity using connected publication components.
    publication_groups, publication_component_audit = _publication_components(
        curated["publication_ids_json"])
    grouped_splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=20260825)
    publication_fold = np.full(len(curated), -1, dtype=int)
    for fold, (_fit, validation) in enumerate(grouped_splitter.split(
            np.arange(len(curated)), labels, groups=publication_groups)):
        publication_fold[validation] = fold
    if (publication_fold < 0).any():
        raise EndpointBenchmarkError("Publication-grouped fold assignment is incomplete")
    publication_grouped_oof = cross_validate(
        panel, features, labels, n_splits=5, seed=20260825,
        fold_assignments=publication_fold,
        checkpoint=checkpoint_dir / "publication_grouped_oof.csv",
        checkpoint_binding_sha256=checkpoint_binding_sha,
        tag="drugage_ce publication-grouped")
    publication_grouped_metrics = metric_table(publication_grouped_oof)

    # ---- publication-group audit -------------------------------------------
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
            "fraction_shared": float(len(inside & outside) / max(1, len(inside)))})
    publication_audit = pd.DataFrame(publication_rows)
    publication_audit["publication_component_group_crossing"] = False
    print("publication overlap across folds:\n"
          + publication_audit.to_string(index=False), flush=True)

    # ---- bias audit ---------------------------------------------------------
    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    d1_frame, _a = _read_sources(positive.resolve(), negative.resolve(), traditional)
    train_indices, _t, _s = paper_split_indices(traditional)
    d1_smiles, _ = _validated_raw_smiles(d1_frame)
    d1_bits = morgan_matrix([d1_smiles[i] for i in train_indices], morgan_generator())
    oof_table = attach_molecular_variables(curated, oof, cohort="drugage_celegans_oof",
                                           d1_train_bits=d1_bits)
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
    print("QED bias audit complete", flush=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".drugce.work-", dir=destination.parent))
    try:
        atomic_write_json(tmp / "SOURCE_RESOLUTION.json", resolution)
        (tmp / "historical_reproduction_status.md").write_text(
            "# Track E1 -- exact historical RF refit: BLOCKED_PARTIAL\n\n"
            "Target: Kapsiani and Howlin (2021), *Scientific Reports*, "
            "doi 10.1038/s41598-021-93070-6.\n\n"
            "The official supplement provides the exact 1,430 structures and "
            "labels plus all 69 selected MOE feature identities. The following "
            "inputs needed for an exact RF refit were not released and were not "
            "guessed:\n\n"
            + "\n".join(f"- {item}" for item in
                        resolution["track_e1_historical_reproduction"]["what_is_missing"])
            + "\n\nThe exact-cohort fixed-panel reconstruction is a separate runner. "
              "Track E2 below remains a modern Build 5 operational benchmark with "
              "a different negative class and is never described as a reproduction "
              "of the historical article.\n")
        reference_status = pd.DataFrame([{
            "status": resolution["track_e1_historical_reproduction"]["status"],
            "target": resolution["track_e1_historical_reproduction"]["target"],
            "consequence": resolution["track_e1_historical_reproduction"]["consequence"],
            "what_is_missing": json.dumps(
                resolution["track_e1_historical_reproduction"]["what_is_missing"]),
        }])
        _write_csv(tmp / "reference_reproduction_metrics.csv", reference_status)
        _write_csv(tmp / "observation_level_ledger.csv", ledger)
        _write_csv(tmp / "compound_level_endpoint_ledger.csv", endpoints)
        _write_csv(tmp / "identity_mapping_ledger.csv", curation_ledger)
        _write_csv(tmp / "identity_conflicts.csv", conflicts)
        _write_csv(tmp / "publication_group_audit.csv", publication_audit)
        registry = oof[["fold", "label"]].copy()
        registry.insert(0, "compound_id", curated["compound_id"].to_numpy())
        _write_csv(tmp / "drugage_split_registry.csv", registry)
        predictions = oof.copy()
        predictions.insert(0, "compound_id", curated["compound_id"].to_numpy())
        predictions["overlaps_d1_connectivity"] = \
            curated["overlaps_d1_connectivity"].to_numpy()
        _write_csv(tmp / "fixed_panel_oof_predictions.csv", predictions)
        _write_csv(tmp / "fixed_panel_fold_metrics.csv", per_fold)
        _write_csv(tmp / "fixed_panel_fold_mean_sd.csv", fold_mean_sd)
        _write_csv(tmp / "fixed_panel_pooled_metrics.csv", pooled)
        _write_csv(tmp / "endpoint_sensitivity_metrics.csv", endpoint_sensitivity)
        _write_csv(tmp / "endpoint_variant_oof_predictions.csv", variant_predictions)
        _write_csv(tmp / "repeated5x5_oof_predictions.csv", repeated_predictions)
        _write_csv(tmp / "repeated5x5_metrics.csv", repeated_metrics)
        publication_grouped_predictions = publication_grouped_oof.copy()
        publication_grouped_predictions.insert(
            0, "compound_id", curated["compound_id"].to_numpy())
        publication_grouped_predictions["publication_component"] = publication_groups
        _write_csv(tmp / "publication_grouped_oof_predictions.csv",
                   publication_grouped_predictions)
        _write_csv(tmp / "publication_grouped_metrics.csv", publication_grouped_metrics)
        _write_csv(tmp / "drugage_qed_bias.csv", bias["associations"])
        _write_csv(tmp / "drugage_qed_stratum_recall.csv", bias["recall"])
        _write_csv(tmp / "drugage_qed_trend_tests.csv", bias["trends"])
        _write_csv(tmp / "drugage_qed_logistic_models.csv", bias["logistic"])
        _write_csv(tmp / "drugage_qed_adjusted_models.csv", bias["adjusted"])
        _write_csv(tmp / "drugage_bias_compound_table.csv", bias_table)
        _write_csv(tmp / "drugage_simple_baseline_metrics.csv", control_baselines)
        _write_csv(tmp / "drugage_label_permutation_controls.csv", control_permutations)
        _write_csv(tmp / "drugage_basic_property_class_shifts.csv", property_shifts)

        lines = ["# Experiment E -- DrugAge *C. elegans* endpoint benchmark", "",
                 f"run_id: `{run_id}`", "",
                 "**Scientific role.** Organism-specific endpoint-aligned retraining. "
                 "Distinct from the project's all-organism DrugAge positive-retrieval "
                 "stress test of the D1-fitted model.", "",
                 "## Track E1 -- exact historical reproduction", "",
                 "**BLOCKED_PARTIAL for the exact RF refit.** "
                 + resolution["track_e1_historical_reproduction"]["consequence"],
                 "", "## Track E2 -- Build 5 endpoint reconstruction", "",
                 f"- pinned export {resolution['pinned_export_rows']} observation rows; "
                 f"{resolution['organism_rows']} are *C. elegans* across "
                 f"{resolution['organism_compound_names']} compound names",
                 f"- {int(ledger.structure_mapped.sum())} observations carry a mapped "
                 f"structure; {len(endpoints)} compound-level endpoints result",
                 f"- curated identities {len(curated)} "
                 f"({int(curated.label.sum())} positive, "
                 f"{int((curated.label == 0).sum())} "
                 "no-recorded-significant-extension background)",
                 f"- primary label rule: {LABEL_VARIANTS['significant_positive']['positive']}",
                 f"- {int(curated.overlaps_d1_connectivity.sum())} identities overlap a "
                 "D1 parent identity",
                 "", "## Pooled ten-fold OOF metrics (primary endpoint)", "",
                 md_table(pooled[["model_id"] + list(PRIMARY_METRICS)], "{:.4f}"), "",
                 "## Endpoint-variant sensitivity", "",
                 md_table(endpoint_sensitivity[
                     ["endpoint_variant", "n_compounds", "primary", "model_id",
                      "auprc_average_precision_positive", "auroc", "mcc",
                      "recall_sensitivity", "specificity"]], "{:.4f}"), "",
                 "## Publication overlap across folds", "",
                 md_table(publication_audit, "{:.3f}"), "",
                 "## QED bias replication under the *C. elegans* endpoint", "",
                 md_table(bias["associations"][
                     ["cohort", "model_id", "n_positive", "spearman_rho", "rho_ci_low",
                      "rho_ci_high", "q_holm_within_cohort"]], "{:.3f}"), ""]
        (tmp / "drugage_celegans_summary.md").write_text("\n".join(lines) + "\n")

        atomic_write_json(tmp / "RUN_MANIFEST.json", {
            "schema_version": f"{SCHEMA}.run_manifest.v1", "run_id": run_id,
            "protocol_sha256": protocol_sha,
            "checkpoint_binding_sha256": checkpoint_binding_sha,
            "scientific_role": "organism_specific_endpoint_aligned_benchmark",
            "is_external_validation_of_the_d1_model": False,
            "conflated_with_all_organism_stress_test": False,
            "source_resolution": resolution,
            "label_variants": LABEL_VARIANTS,
            "primary_variant": "significant_positive",
            "variant_selected_by_model_performance": False,
            "curation": {"compound_endpoints": int(len(endpoints)),
                         "curated_identities": int(len(curated)),
                         "excluded_rows": int((~curation_ledger.included).sum()),
                         "conflict_groups": int(len(conflicts)),
                         "d1_overlap_identities":
                             int(curated.overlaps_d1_connectivity.sum())},
            "features": feature_audit,
            "panel": list(PANEL),
            "resolved_model_settings": panel.resolved_settings,
            "folds": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, "
                     f"random_state={SEED}) on curated parent identities",
            "fixed_threshold": 0.5, "threshold_is_tuned": False,
            "publication_group_audit": publication_rows,
            "publication_component_audit": publication_component_audit,
            "endpoint_variants_refitted_separately": True,
            "repeated_5x5_executed": True,
            "publication_grouped_cv_executed": True,
            "runtime": {"python": platform.python_version(),
                        "platform": platform.platform()},
            "existing_runs_modified": False})
        atomic_write_json(tmp / "COMPLETED.json", {
            "schema_version": f"{SCHEMA}.completed.v1",
            "status": "COMPLETE_TRACK_E1_BLOCKED_PARTIAL", "run_id": run_id,
            "run_manifest_sha256": sha256_file(tmp / "RUN_MANIFEST.json"),
            "artifact_hashes": {str(p.relative_to(tmp)): sha256_file(p)
                                for p in sorted(tmp.rglob("*"))
                                if p.is_file() and p.name != "COMPLETED.json"}})
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
    a = parser.parse_args(argv)
    run(root=a.root, config_path=a.config, positive=a.positive, negative=a.negative,
        run_id=a.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
