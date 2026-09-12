"""Experiment D: the AgeXtend original-endpoint benchmark.

Scientific role (section 8.1).  This retrains the fixed architectures under
AgeXtend's own operational geroprediction label.  It tests whether the
architecture transports to another benchmark task.  It is NOT external validation
of the D1-fitted model, and the ensembles here are the `blend3 architecture` and
the `GB4 architecture`, not the locked D1 model.

Two analyses are kept strictly apart:

  1. Resolution of the authors' released geroprediction model.  Official model
     inference and published-result extraction are reproducible; only an exact
     historical *refit* remains partially blocked because split/fold identities,
     seeds and complete training state were not released.  Released-model
     challenge parity is implemented separately by
     ``agextend_official_reference``.
  2. The fixed ten-model panel trained inside a ten-fold registry over the
     curated AgeXtend training cohort, plus one scoring of the curated
     Supplementary Table 6 challenge.

The challenge analysis is retrospective: its outcomes are public and this project
has already inspected them.  With only a handful of curated negatives, positive
retrieval and ranking are primary and binary metrics are unstable secondary
descriptions.
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
    score_once,
)
from geroprotector.fixed_blend_paper405 import _features
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
from geroprotector.screening_blend_external import _xlsx_sheet
from geroprotector.traditional_paper405 import _read_sources, paper_split_indices

SCHEMA = "geroprotector.agextend_endpoint_benchmark"
SOURCE_DIR = Path(os.environ.get("AGEXTEND_SOURCE_DIR", "external_data/agextend_2024"))
TRAINING_WORKBOOK = "43587_2024_763_MOESM3_ESM.xlsx"
SUPPLEMENTARY_WORKBOOK = "43587_2024_763_MOESM2_ESM.xlsx"
N_SPLITS = 10
SEED = 42
CITATIONS = {
    "article": "https://doi.org/10.1038/s43587-024-00763-4",
    "zenodo": "https://doi.org/10.5281/zenodo.10034994",
    "repository": "https://github.com/the-ahuja-lab/AgeXtend",
}
EXPECTED = {"rows": 972, "geroprotector": 583, "neutral": 389}
OFFICIAL_REFERENCE_PROTOCOL = "configs/agextend_official_reference_protocol.yaml"


def _load_endpoint_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != f"{SCHEMA}.protocol.v1":
        raise EndpointBenchmarkError("Unknown AgeXtend protocol schema")
    evaluation = protocol["evaluation"]
    if (int(evaluation["n_splits"]) != N_SPLITS or
            int(evaluation["random_state"]) != SEED or
            float(evaluation["fixed_threshold"]) != 0.5):
        raise EndpointBenchmarkError("AgeXtend protocol differs from the locked evaluation")
    if protocol.get("panel") != list(PANEL):
        raise EndpointBenchmarkError("AgeXtend protocol changes the fixed model panel")
    return protocol, sha256_file(path)


_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _raw_rows(path: Path, sheet_name: str) -> list[dict[str, str]]:
    """Cell-addressed rows for sheets whose header is not the first row.

    Supplementary Tables 1 and 4 open with a title line, so the project's
    header-first XLSX reader cannot address them. This returns raw column-letter
    cells instead and leaves interpretation to the caller.
    """
    import xml.etree.ElementTree as ET
    import zipfile

    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = {r.get("Id"): r.get("Target") for r in relations}
        identifier = None
        for sheet in workbook.iter(f"{_NS}sheet"):
            if sheet.get("name") == sheet_name:
                identifier = sheet.get(f"{_REL}id")
        if identifier is None:
            raise EndpointBenchmarkError(f"Sheet not found: {sheet_name}")
        location = target[identifier]
        location = location if location.startswith("xl/") else "xl/" + location.lstrip("/")
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).iter(f"{_NS}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{_NS}t")))
        rows = []
        for row in ET.fromstring(archive.read(location)).iter(f"{_NS}row"):
            cells = {}
            for cell in row.iter(f"{_NS}c"):
                column = re.match(r"([A-Z]+)", cell.get("r")).group(1)
                value = cell.find(f"{_NS}v")
                kind = cell.get("t")
                if kind == "s" and value is not None:
                    text = shared[int(value.text)]
                elif kind == "inlineStr":
                    text = "".join(node.text or "" for node in cell.iter(f"{_NS}t"))
                else:
                    text = value.text if value is not None else ""
                cells[column] = "" if text is None else str(text)
            rows.append(cells)
    return rows


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def resolve_sources() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Read the official tables and settle the label orientation from documents."""
    training = _xlsx_sheet(SOURCE_DIR / TRAINING_WORKBOOK, "Fig 1b", ("SMILES", "status"))
    challenge = _xlsx_sheet(
        SOURCE_DIR / SUPPLEMENTARY_WORKBOOK, "Supplementary Table 6",
        ("S.No.", "Compound Name", "Isomeric_SMILES", "Label", "Canonical_SMILES",
         "Anti_Aging_Status", "Anti_Aging_Prob", "Source"))
    counts = training["status"].astype(str).value_counts().to_dict()

    # Label orientation is NOT inferred from the numeric value.  Supplementary
    # Table 1 states the geroprediction dataset holds 583 geroprotectors and 389
    # neutral compounds; the Fig 1b status column holds exactly 583 ones and 389
    # zeros, which fixes status=1 as geroprotector.
    table1 = _raw_rows(SOURCE_DIR / SUPPLEMENTARY_WORKBOOK, "Supplementary Table 1")
    stated = {row.get("A", "").strip(): row.get("B", "").strip() for row in table1
              if row.get("A", "").strip() in ("Geroprotector", "Neutral")}
    orientation_ok = (counts.get("1") == EXPECTED["geroprotector"] and
                      counts.get("0") == EXPECTED["neutral"] and
                      stated.get("Geroprotector") == str(EXPECTED["geroprotector"]) and
                      stated.get("Neutral") == str(EXPECTED["neutral"]))
    if not orientation_ok:
        raise EndpointBenchmarkError(
            "AgeXtend label orientation is unresolved: Fig 1b counts "
            f"{counts} do not reconcile with Supplementary Table 1 {stated}")

    table4 = _raw_rows(SOURCE_DIR / SUPPLEMENTARY_WORKBOOK, "Supplementary Table 4")
    reference = {}
    for row in table4:
        if row.get("A", "").strip() == "Geroprediction":
            reference = {"module": row.get("A", ""), "property": row.get("B", ""),
                         "ml_model": row.get("C", ""),
                         "class_balancing": row.get("D", ""),
                         "parameters": row.get("E", "")}
            break

    resolution = {
        "citations": CITATIONS,
        "source_files": {name: sha256_file(SOURCE_DIR / name)
                         for name in (TRAINING_WORKBOOK, SUPPLEMENTARY_WORKBOOK)},
        "training_table": {"workbook": TRAINING_WORKBOOK, "sheet": "Fig 1b",
                           "rows": int(len(training)), "status_counts": counts},
        "official_supplementary_counts": stated,
        "expected_source_resolution": EXPECTED,
        "source_counts_match_official_supplementary": bool(orientation_ok),
        "label_orientation": {
            "resolved": "status=1 is geroprotector, status=0 is neutral",
            "evidence": ("Supplementary Table 1 states 583 geroprotectors and 389 "
                         "neutral compounds for the geroprediction module; the Fig 1b "
                         "status column contains exactly 583 ones and 389 zeros."),
            "inferred_from_numeric_value_alone": False},
        "challenge_table": {"workbook": SUPPLEMENTARY_WORKBOOK,
                            "sheet": "Supplementary Table 6",
                            "rows": int(len(challenge)),
                            "label_counts": challenge["Label"].astype(str)
                                            .value_counts().to_dict()},
        "reference_model_published_specification": reference,
        "reference_reproduction_status": {
            "status": "BLOCKED_PARTIAL",
            "scope": "exact_historical_refit_only",
            "what_is_available": [
                "the classifier family and hyperparameters (Supplementary Table 4)",
                "the class-balancing ratio (Supplementary Table 4)",
                "the training SMILES and labels (Fig 1b)",
                "the exact 71 selected feature identities in the official fitted model",
                "the official Signaturizer feature-generation code",
                "the official fitted SVC model artifact",
                "the published ten-fold metrics and all 972 LOOCV predictions",
                "the 84-compound challenge predictions and probabilities"],
            "what_is_missing": [
                "the random 75/25 split membership",
                "the cross-validation fold identities and random seeds",
                "the complete historical training script",
                "the exact SMOTE and Boruta random states"],
            "consequence": ("Official released-model inference is reproducible, but "
                            "an exact historical refit cannot be claimed. The fixed "
                            "ten-fold panel below is a prespecified protocol "
                            "reconstruction, not the hidden historical registry."),
            "not_attempted_by_guessing": True},
        "official_model_inference_status": {
            "status": "REPRODUCIBLE_IN_PINNED_ENVIRONMENT",
            "parity_target": "official 84-compound Supplementary Table 6",
            "implementation": "geroprotector.agextend_official_reference",
            "protocol": OFFICIAL_REFERENCE_PROTOCOL},
        "fixed_panel_protocol": {
            "kind": "protocol_reconstruction",
            "reason": ("ten-fold evaluation is published, but fold identities, "
                       "shuffle rule and seed are not"),
            "n_splits": N_SPLITS, "shuffle": True, "random_state": SEED,
            "stratified": True, "unit": "curated_parent_identity"},
    }
    return training, challenge, resolution


def _qed_audit(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Reuse the Experiment A analyses without changing a stratum or a covariate."""
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


def run(*, root: Path, config_path: Path, positive: Path, negative: Path, run_id: str) -> Path:
    if not re.fullmatch(r"agextend_endpoint_benchmark_[a-z0-9_.-]+", run_id):
        raise EndpointBenchmarkError("RUN_ID must start with agextend_endpoint_benchmark_")
    root = root.resolve()
    protocol, protocol_sha = _load_endpoint_protocol(config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise EndpointBenchmarkError(f"Run directory already exists: {destination}")

    training, challenge, resolution = resolve_sources()
    print(f"AgeXtend Fig 1b: {len(training)} rows, "
          f"{resolution['training_table']['status_counts']}", flush=True)

    d1_keys, d1_audit = d1_connectivity_keys(root, positive, negative)
    resolution["d1_overlap_reference"] = d1_audit

    raw = pd.DataFrame({
        "provider_record_id": [f"fig1b_{i}" for i in range(len(training))],
        "compound_name": [f"agextend_row_{i}" for i in range(len(training))],
        "source_smiles": training["SMILES"].astype(str),
        "label": training["status"].astype(int)})
    curated, ledger, conflicts = curate(raw, cohort="agextend_train",
                                        d1_connectivity=d1_keys)
    print(f"curated training cohort: {len(curated)} identities "
          f"({int(curated.label.sum())} geroprotector), "
          f"{int((~ledger.included).sum())} provider rows excluded, "
          f"{len(conflicts)} conflicting identity groups, "
          f"{int(curated.overlaps_d1_connectivity.sum())} overlap D1", flush=True)

    challenge_raw = pd.DataFrame({
        "provider_record_id": [f"table6_{i}" for i in range(len(challenge))],
        "compound_name": challenge["Compound Name"].astype(str),
        "source_smiles": challenge["Canonical_SMILES"].astype(str),
        "label": challenge["Label"].astype(int)})
    challenge_curated, challenge_ledger, challenge_conflicts = curate(
        challenge_raw, cohort="agextend_challenge", d1_connectivity=d1_keys)
    training_keys = set(curated["connectivity_inchikey"])
    challenge_curated["overlaps_agextend_training"] = \
        challenge_curated["connectivity_inchikey"].isin(training_keys)
    scored_challenge = challenge_curated[
        ~challenge_curated["overlaps_agextend_training"]].reset_index(drop=True)
    print(f"curated challenge: {len(challenge_curated)} identities, "
          f"{int(challenge_curated.overlaps_agextend_training.sum())} overlap the "
          f"training cohort -> {len(scored_challenge)} scored "
          f"({int(scored_challenge.label.sum())} positive)", flush=True)

    features, feature_audit = build_features(root, curated, verify_d1_parity=True)
    print("DataWarrior D1 parity verified; training features built", flush=True)
    challenge_features, _ = build_features(root, scored_challenge, verify_d1_parity=False)

    checkpoint_dir = root / "outputs" / f".{run_id}.work"
    checkpoint_contract = {
        "run_id": run_id, "protocol_sha256": protocol_sha,
        "source_sha256": resolution["source_files"],
        "curated_compound_ids_sha256": canonical_sha256(
            curated["compound_id"].astype(str).tolist()),
        "curated_labels_sha256": canonical_sha256(curated["label"].astype(int).tolist()),
        "code_sha256": {name: sha256_file(root / "src" / "geroprotector" / name)
                         for name in ("agextend_endpoint_benchmark.py",
                                      "endpoint_benchmark.py", "model_panel.py")},
    }
    checkpoint_binding_sha = bind_checkpoint_directory(
        checkpoint_dir, checkpoint_contract, error_cls=EndpointBenchmarkError)
    panel = Panel(root)
    labels = curated["label"].to_numpy(int)
    oof = cross_validate(panel, features, labels, n_splits=N_SPLITS, seed=SEED,
                         checkpoint=checkpoint_dir / "primary_oof.csv",
                         checkpoint_binding_sha256=checkpoint_binding_sha,
                         tag="agextend")
    pooled = metric_table(oof)
    per_fold = metric_table(oof, group="fold")
    fold_mean_sd = per_fold.groupby("model_id")[list(PRIMARY_METRICS)].agg(["mean", "std"])
    fold_mean_sd.columns = [f"{metric}_{stat}" for metric, stat in fold_mean_sd.columns]
    fold_mean_sd = fold_mean_sd.reset_index()

    # D1-nonoverlap sensitivity cohort: the same OOF predictions, restricted.
    nonoverlap_mask = ~curated["overlaps_d1_connectivity"].to_numpy(bool)
    sensitivity = metric_table(oof[nonoverlap_mask])
    sensitivity.insert(0, "cohort", "agextend_oof_no_d1_overlap")

    challenge_predictions = score_once(panel, features, labels, challenge_features,
                                       seed=SEED)
    challenge_predictions.insert(0, "compound_id", scored_challenge["compound_id"])
    challenge_predictions["label"] = scored_challenge["label"].to_numpy(int)
    challenge_metrics = metric_table(challenge_predictions)
    challenge_metrics.insert(0, "cohort", "agextend_challenge")
    # ranking is primary when the negative class is tiny
    ranking_rows = []
    for model in PANEL:
        order = np.argsort(-challenge_predictions[f"p_{model}"].to_numpy(float))
        y = challenge_predictions["label"].to_numpy(int)[order]
        for k in (10, 20, 50):
            k = min(k, len(y))
            ranking_rows.append({"model_id": model, "k": int(k),
                                 "precision_at_k": float(y[:k].mean()),
                                 "recall_at_k": float(y[:k].sum() / max(1, y.sum()))})
    ranking = pd.DataFrame(ranking_rows)

    # ---- bias audit on the endpoint-aligned cohorts -------------------------
    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    d1_frame, _audit = _read_sources(positive.resolve(), negative.resolve(), traditional)
    train_indices, _test, _sha = paper_split_indices(traditional)
    d1_smiles, _ = _validated_raw_smiles(d1_frame)
    d1_bits = morgan_matrix([d1_smiles[i] for i in train_indices], morgan_generator())

    oof_table = attach_molecular_variables(curated, oof, cohort="agextend_oof",
                                           d1_train_bits=d1_bits)
    nonoverlap_table = oof_table[~oof_table["overlaps_d1_connectivity"]].copy()
    nonoverlap_table["cohort"] = "agextend_oof_no_d1_overlap"
    challenge_table = attach_molecular_variables(
        scored_challenge, challenge_predictions.drop(columns=["compound_id"]),
        cohort="agextend_challenge", d1_train_bits=d1_bits)
    bias_table = pd.concat([oof_table, nonoverlap_table, challenge_table],
                           ignore_index=True, sort=False)
    bias = _qed_audit(bias_table)
    control_baselines, control_permutations = [], []
    control_rng = np.random.default_rng(20260825)
    for cohort_name in ("agextend_oof", "agextend_oof_no_d1_overlap"):
        baseline, permutation = fold_local_baselines(
            bias_table, control_rng, cohort=cohort_name, permutations=100)
        control_baselines.append(baseline)
        control_permutations.append(permutation)
    control_baselines = pd.concat(control_baselines, ignore_index=True)
    control_permutations = pd.concat(control_permutations, ignore_index=True)
    property_shifts = class_shifts(
        bias_table, cohorts=("agextend_oof", "agextend_oof_no_d1_overlap"))
    print("QED bias audit complete", flush=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".agext.work-", dir=destination.parent))
    try:
        atomic_write_json(tmp / "SOURCE_RESOLUTION.json", resolution)
        _write_csv(tmp / "raw_to_curated_ledger.csv", ledger)
        _write_csv(tmp / "identity_conflicts.csv", conflicts)
        _write_csv(tmp / "d1_overlap_ledger.csv", curated[
            ["compound_id", "connectivity_inchikey", "compound_name", "label",
             "overlaps_d1_connectivity"]])
        registry = oof[["fold", "label"]].copy()
        registry.insert(0, "compound_id", curated["compound_id"].to_numpy())
        _write_csv(tmp / "agextend_split_registry.csv", registry)
        reference_status = pd.DataFrame([
            resolution["reference_reproduction_status"] |
            {"what_is_available": json.dumps(
                resolution["reference_reproduction_status"]["what_is_available"]),
             "what_is_missing": json.dumps(
                     resolution["reference_reproduction_status"]["what_is_missing"])}])
        _write_csv(tmp / "reference_reproduction_status.csv", reference_status)
        _write_csv(tmp / "reference_reproduction_metrics.csv", reference_status)
        fixed_predictions = oof.copy()
        fixed_predictions.insert(0, "compound_id", curated["compound_id"].to_numpy())
        fixed_predictions["overlaps_d1_connectivity"] = \
            curated["overlaps_d1_connectivity"].to_numpy()
        _write_csv(tmp / "fixed_panel_oof_predictions.csv", fixed_predictions)
        _write_csv(tmp / "fixed_panel_fold_metrics.csv", per_fold)
        _write_csv(tmp / "fixed_panel_fold_mean_sd.csv", fold_mean_sd)
        _write_csv(tmp / "fixed_panel_pooled_metrics.csv", pooled)
        _write_csv(tmp / "d1_nonoverlap_sensitivity_metrics.csv", sensitivity)
        _write_csv(tmp / "challenge_predictions.csv", challenge_predictions)
        _write_csv(tmp / "challenge_metrics.csv", challenge_metrics)
        _write_csv(tmp / "challenge_ranking.csv", ranking)
        _write_csv(tmp / "challenge_curation_ledger.csv", challenge_ledger)
        _write_csv(tmp / "agextend_qed_bias.csv", bias["associations"])
        _write_csv(tmp / "agextend_qed_stratum_recall.csv", bias["recall"])
        _write_csv(tmp / "agextend_qed_trend_tests.csv", bias["trends"])
        _write_csv(tmp / "agextend_qed_logistic_models.csv", bias["logistic"])
        _write_csv(tmp / "agextend_qed_adjusted_models.csv", bias["adjusted"])
        _write_csv(tmp / "agextend_bias_compound_table.csv", bias_table)
        _write_csv(tmp / "agextend_simple_baseline_metrics.csv", control_baselines)
        _write_csv(tmp / "agextend_label_permutation_controls.csv", control_permutations)
        _write_csv(tmp / "agextend_basic_property_class_shifts.csv", property_shifts)

        lines = ["# Experiment D -- AgeXtend original-endpoint benchmark", "",
                 f"run_id: `{run_id}`", "",
                 "**Scientific role.** Endpoint-aligned cross-benchmark retraining. "
                 "Not external validation of the D1-fitted model. The ensembles here "
                 "are the blend3 and GB4 *architectures*.", "",
                 "## Source resolution", "",
                 f"- Fig 1b: {len(training)} rows, "
                 f"{resolution['training_table']['status_counts']} "
                 f"(official Supplementary Table 1: {resolution['official_supplementary_counts']})",
                 f"- label orientation: {resolution['label_orientation']['resolved']}",
                 f"- reference reproduction: "
                 f"**{resolution['reference_reproduction_status']['status']}** -- "
                 + "; ".join(resolution["reference_reproduction_status"]["what_is_missing"]),
                 f"- official released-model inference: "
                 f"**{resolution['official_model_inference_status']['status']}** "
                 "(separate 84-compound parity stage)",
                 f"- fixed-panel protocol: {N_SPLITS}-fold stratified, seed {SEED} "
                 "(protocol reconstruction: official fold identities and seed absent)",
                 "",
                 "## Curation", "",
                 f"- provider rows {len(ledger)}; excluded {int((~ledger.included).sum())}; "
                 f"conflicting identity groups {len(conflicts)}",
                 f"- curated identities {len(curated)} "
                 f"({int(curated.label.sum())} geroprotector, "
                 f"{int((curated.label == 0).sum())} neutral)",
                 f"- overlapping a D1 parent identity: "
                 f"{int(curated.overlaps_d1_connectivity.sum())}",
                 "", "## Pooled ten-fold OOF metrics", "",
                 md_table(pooled[["model_id"] + list(PRIMARY_METRICS)], "{:.4f}"), "",
                 "## D1-nonoverlap sensitivity cohort", "",
                 md_table(sensitivity[["model_id"] + list(PRIMARY_METRICS)], "{:.4f}"), "",
                 "## Supplementary Table 6 challenge (retrospective)", "",
                 f"{len(scored_challenge)} scored identities, "
                 f"{int(scored_challenge.label.sum())} positive, "
                 f"{int((scored_challenge.label == 0).sum())} negative. With so few "
                 "negatives, specificity, MCC and kappa are unstable and are reported "
                 "only as secondary descriptions.", "",
                 md_table(challenge_metrics[["model_id"] + list(PRIMARY_METRICS)], "{:.4f}"),
                 "", "## QED bias replication under the AgeXtend endpoint", "",
                 md_table(bias["associations"][
                     ["cohort", "model_id", "n_positive", "spearman_rho", "rho_ci_low",
                      "rho_ci_high", "q_holm_within_cohort"]], "{:.3f}"), ""]
        (tmp / "agextend_endpoint_summary.md").write_text("\n".join(lines) + "\n")

        atomic_write_json(tmp / "RUN_MANIFEST.json", {
            "schema_version": f"{SCHEMA}.run_manifest.v1", "run_id": run_id,
            "protocol_sha256": protocol_sha,
            "checkpoint_binding_sha256": checkpoint_binding_sha,
            "scientific_role": "endpoint_aligned_cross_benchmark_evaluation",
            "is_external_validation_of_the_d1_model": False,
            "source_resolution": resolution,
            "curation": {"provider_rows": int(len(ledger)),
                         "excluded_rows": int((~ledger.included).sum()),
                         "conflict_groups": int(len(conflicts)),
                         "curated_identities": int(len(curated)),
                         "d1_overlap_identities": int(curated.overlaps_d1_connectivity.sum())},
            "features": feature_audit,
            "panel": list(PANEL),
            "resolved_model_settings": panel.resolved_settings,
            "folds": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, "
                     f"random_state={SEED}) on curated parent identities",
            "fixed_threshold": 0.5, "threshold_is_tuned": False,
            "challenge_overlap_removed_before_scoring": True,
            "challenge_labels_used_for_tuning_or_selection": False,
            "outcome_used_in_curation": "conflicting-duplicate detection only",
            "runtime": {"python": platform.python_version(),
                        "platform": platform.platform()},
            "existing_runs_modified": False})
        atomic_write_json(tmp / "COMPLETED.json", {
            "schema_version": f"{SCHEMA}.completed.v1",
            "status": "COMPLETE_EXACT_REFIT_BLOCKED_PARTIAL",
            "run_id": run_id,
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
