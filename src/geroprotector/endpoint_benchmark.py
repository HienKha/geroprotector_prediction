"""Shared machinery for the endpoint-aligned cross-benchmark experiments (D and E).

Scientific role, stated once so neither caller can blur it: retraining the fixed
architectures under another dataset's operational endpoint tests whether the
*architecture* transports to that benchmark task.  It is not external validation
of the D1-fitted model, and the resulting blends are the `blend3 architecture`
and the `GB4 architecture`, not the locked D1 model.

What lives here:

  * outcome-blind chemical curation (parent standardisation, identity collapse,
    conflicting-label exclusion, D1 overlap annotation) with a full ledger;
  * the three feature representations the panel needs, built from the curated
    parent structures -- seven DataWarrior descriptors through the pinned
    OpenChemLib CLI (with the exact 405-row D1 parity check first), Morgan bits,
    and the raw RDKit2D matrix;
  * stratified compound-level k-fold cross-validation of the fixed panel with
    every learned transformation fitted inside the fold;
  * a single full-cohort fit used to score a held-aside challenge table once.

No outcome of the benchmark being built ever reaches a model setting, a feature
decision, a threshold or a fold assignment beyond the approximate stratification
that k-fold requires.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from sklearn.model_selection import StratifiedKFold

from geroprotector.study_common import (max_tanimoto, molecular_variables,
                                        morgan_generator, morgan_matrix)
from geroprotector.model_panel import BASE_MODELS, PANEL, Panel, add_blends, run_folds
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.nine_ml_featuresets import _metrics
from geroprotector.screening_blend_altmodels import _rdkit2d_from_smiles
from geroprotector.screening_blend_external import (
    PAPER_COLUMNS,
    _datawarrior_descriptors,
    _datawarrior_parity,
    _parent_identity,
)
from geroprotector.screening_blend_external import load_protocol as load_external_protocol
from geroprotector.traditional_paper405 import _read_sources, paper_split_indices


class EndpointBenchmarkError(RuntimeError):
    """Raised when a curation contract, a parity check or a leakage guard fails."""


PRIMARY_METRICS = ("auprc_average_precision_positive", "auroc", "accuracy", "mcc",
                   "macro_f1", "brier", "cohen_kappa", "recall_sensitivity",
                   "specificity")


# ------------------------------------------------------------------- curation

def curate(raw: pd.DataFrame, *, cohort: str, d1_connectivity: set[str]
           ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Outcome-blind curation of a provider table.

    `raw` must carry `provider_record_id`, `source_smiles`, `label` and
    `compound_name`.  Returns the curated cohort, the full inclusion/exclusion
    ledger (one row per provider row) and the conflicting-identity table.

    The label is carried through untouched and is used for exactly two things:
    detecting conflicting duplicates, and being the outcome afterwards.  It never
    influences an exclusion rule other than the conflict rule itself, which
    section 8.3 mandates.
    """
    records = []
    for row in raw.itertuples(index=False):
        base = {"provider_record_id": str(row.provider_record_id),
                "compound_name": str(row.compound_name),
                "source_smiles": str(row.source_smiles).strip(),
                "label": int(row.label)}
        try:
            identity = _parent_identity(base["source_smiles"])
        except Exception:
            records.append({**base, "included": False,
                            "exclusion_reason": "structure_parse_or_standardize_failed"})
            continue
        # A salt or a hydrate has more than one component but exactly one organic
        # parent, so `FragmentParent` resolves it unambiguously. A true mixture of
        # two distinct organic species does not. Section 8.3 excludes *unresolved*
        # mixtures, so the two cases are separated here rather than lumped
        # together; both counts are kept in the ledger.
        raw_molecule = Chem.MolFromSmiles(base["source_smiles"])
        organic_fragments = 0
        for fragment in Chem.GetMolFrags(raw_molecule, asMols=True, sanitizeFrags=False):
            has_carbon = any(a.GetAtomicNum() == 6 for a in fragment.GetAtoms())
            if has_carbon and fragment.GetNumHeavyAtoms() >= 2:
                organic_fragments += 1
        reason = ""
        if identity["metal_atomic_numbers"]:
            reason = "metal_or_coordination_complex"
        elif organic_fragments > 1:
            reason = "unresolved_multi_organic_mixture"
        else:
            molecule = Chem.MolFromSmiles(identity["standardized_parent_smiles"])
            if molecule is None or molecule.GetNumHeavyAtoms() < 2:
                reason = "degenerate_parent_structure"
        records.append({**base, **{k: v for k, v in identity.items()
                                   if k != "metal_atomic_numbers"},
                        "metal_atomic_numbers": json.dumps(identity["metal_atomic_numbers"]),
                        "organic_fragment_count": int(organic_fragments),
                        "excluded_by_strict_multicomponent_rule":
                            bool(identity["component_count"] > 1),
                        "included": not reason, "exclusion_reason": reason})
    ledger = pd.DataFrame(records)

    kept = ledger[ledger["included"]].copy()
    conflicts, grouped = [], []
    for connectivity, group in kept.groupby("connectivity_inchikey", sort=True):
        labels = set(group["label"].astype(int))
        if len(labels) > 1:
            conflicts.append({"connectivity_inchikey": connectivity,
                              "n_rows": int(len(group)),
                              "labels": json.dumps(sorted(labels)),
                              "compound_names": json.dumps(
                                  sorted(set(group["compound_name"].astype(str))))})
            continue
        representative = group.sort_values(
            ["source_smiles", "compound_name", "provider_record_id"],
            kind="stable").iloc[0]
        grouped.append({
            "compound_id": f"{cohort}::{connectivity}", "cohort": cohort,
            "connectivity_inchikey": connectivity,
            "full_inchikey": representative["full_inchikey"],
            "compound_name": representative["compound_name"],
            "source_smiles": representative["source_smiles"],
            "standardized_parent_smiles": representative["standardized_parent_smiles"],
            "label": int(representative["label"]),
            "collapsed_rows": int(len(group)),
            "provider_record_ids_json": json.dumps(
                sorted(set(group["provider_record_id"].astype(str)))),
            "overlaps_d1_connectivity": bool(connectivity in d1_connectivity)})
    conflict_keys = {c["connectivity_inchikey"] for c in conflicts}
    ledger.loc[ledger["connectivity_inchikey"].isin(conflict_keys)
               if "connectivity_inchikey" in ledger.columns else False,
               "exclusion_reason"] = "conflicting_labels_within_connectivity_group"
    ledger.loc[ledger["connectivity_inchikey"].isin(conflict_keys)
               if "connectivity_inchikey" in ledger.columns else False,
               "included"] = False
    curated = pd.DataFrame(grouped).sort_values("compound_id", kind="stable")
    curated = curated.reset_index(drop=True)
    if curated.empty or curated["compound_id"].duplicated().any():
        raise EndpointBenchmarkError(f"{cohort}: curated cohort is empty or duplicated")
    return curated, ledger, pd.DataFrame(conflicts)


def d1_connectivity_keys(root: Path, positive: Path, negative: Path) -> tuple[set[str], dict]:
    """Connectivity InChIKeys of all 405 D1 compounds, for overlap annotation."""
    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    frame, audit = _read_sources(positive.resolve(), negative.resolve(), traditional)
    smiles, _ = _validated_raw_smiles(frame)
    keys = set()
    for text in smiles:
        try:
            keys.add(_parent_identity(text)["connectivity_inchikey"])
        except Exception:
            continue
    return keys, {"d1_rows": int(len(frame)), "d1_connectivity_keys": int(len(keys)),
                  "d1_data_audit": audit}


# ------------------------------------------------------------------- features

def build_features(root: Path, curated: pd.DataFrame, *, verify_d1_parity: bool = True,
                   nonfinite_descriptor_policy: str = "error"
                   ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build the three fixed representations from curated parent structures.

    DataWarrior occasionally declines a chemically parseable parent and returns
    an entirely missing descriptor row.  Endpoint D/E retain the fail-closed
    default.  A source reconstruction may instead request ``exclude``; that
    outcome-blind rule is recorded with exact input indices and compound IDs so
    the caller can apply the identical mask to its endpoint table and ledger.
    """
    if nonfinite_descriptor_policy not in {"error", "exclude"}:
        raise EndpointBenchmarkError(
            "nonfinite_descriptor_policy must be 'error' or 'exclude'")
    external_protocol, _sha = load_external_protocol(
        root, root / "configs" / "screening_blend_external_protocol.yaml")
    audit: dict[str, Any] = {}
    if verify_d1_parity:
        # The published SVM is only meaningful on genuine DataWarrior descriptors,
        # so the CLI is proved against all 405 D1 rows before it is trusted here.
        audit["datawarrior_d1_parity"] = _datawarrior_parity(root, external_protocol)
    rows = pd.DataFrame({
        "stable_id": curated["compound_id"].astype(str),
        "compound_name": curated["compound_name"].astype(str),
        "source_smiles": curated["standardized_parent_smiles"].astype(str)})
    descriptors, descriptor_audit = _datawarrior_descriptors(
        root, external_protocol, rows, d1_parity=False)
    audit["datawarrior"] = descriptor_audit
    paper = descriptors[list(PAPER_COLUMNS)].to_numpy(dtype=float)
    eligible = np.isfinite(paper).all(axis=1)
    excluded_indices = np.flatnonzero(~eligible).astype(int).tolist()
    audit["descriptor_eligibility"] = {
        "policy": nonfinite_descriptor_policy,
        "input_rows": int(len(curated)),
        "eligible_rows": int(eligible.sum()),
        "excluded_rows": int((~eligible).sum()),
        "eligible_input_indices": np.flatnonzero(eligible).astype(int).tolist(),
        "excluded_input_indices": excluded_indices,
        "excluded_compound_ids": curated.iloc[excluded_indices]["compound_id"]
            .astype(str).tolist(),
        "rule_uses_endpoint_labels": False,
    }
    if excluded_indices and nonfinite_descriptor_policy == "error":
        raise EndpointBenchmarkError(
            "DataWarrior returned non-finite descriptors for input indices "
            f"{excluded_indices}")
    paper = paper[eligible]

    smiles = curated.loc[eligible, "standardized_parent_smiles"].astype(str).tolist()
    from rdkit.Chem import Descriptors
    names = [n for n, _f in Descriptors._descList]
    rdkit2d = _rdkit2d_from_smiles(smiles, names)
    morgan = morgan_matrix(smiles, morgan_generator())
    audit["shapes"] = {"paper": list(paper.shape), "morgan": list(morgan.shape),
                       "rdkit2d": list(rdkit2d.shape)}
    return {"paper": paper, "morgan": morgan, "rdkit2d": rdkit2d}, audit


# ------------------------------------------------------------------ evaluation

def cross_validate(panel: Panel, features: dict[str, np.ndarray], labels: np.ndarray,
                   *, n_splits: int, seed: int, checkpoint: Path | None = None,
                   checkpoint_binding_sha256: str | None = None,
                   fold_assignments: np.ndarray | None = None,
                   tag: str = "cv") -> pd.DataFrame:
    """Stratified compound-level k-fold OOF predictions for the fixed panel."""
    checkpoint_hash_path = None
    if checkpoint is not None:
        binding_path = checkpoint.parent / "CHECKPOINT_BINDING.json"
        if not binding_path.is_file() or checkpoint_binding_sha256 is None:
            raise EndpointBenchmarkError(
                f"{tag}: checkpoint resume requires a verified binding hash")
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if binding.get("binding_sha256") != checkpoint_binding_sha256:
            raise EndpointBenchmarkError(f"{tag}: checkpoint binding hash differs")
        checkpoint_hash_path = checkpoint.with_suffix(checkpoint.suffix + ".sha256.json")
        if checkpoint.is_file():
            if not checkpoint_hash_path.is_file():
                raise EndpointBenchmarkError(f"{tag}: checkpoint hash sidecar is absent")
            recorded = json.loads(checkpoint_hash_path.read_text(encoding="utf-8"))
            if recorded.get("sha256") != sha256_file(checkpoint):
                raise EndpointBenchmarkError(f"{tag}: checkpoint content hash differs")
            print(f"  {tag}: resuming from hash-verified checkpoint", flush=True)
            return pd.read_csv(checkpoint)
    rows = np.arange(len(labels))
    if fold_assignments is None:
        splitter = StratifiedKFold(n_splits=int(n_splits), shuffle=True,
                                   random_state=int(seed))
        fold_id = np.full(len(labels), -1, dtype=int)
        jobs = []
        for fold, (fit, validation) in enumerate(splitter.split(rows, labels)):
            jobs.append((fit, validation))
            fold_id[validation] = fold
    else:
        fold_id = np.asarray(fold_assignments, dtype=int)
        if len(fold_id) != len(labels) or set(np.unique(fold_id)) != set(range(int(n_splits))):
            raise EndpointBenchmarkError(f"{tag}: supplied fold registry is invalid")
        jobs = [(rows[fold_id != fold], rows[fold_id == fold])
                for fold in range(int(n_splits))]
        if any(len(np.unique(labels[validation])) < 2 for _fit, validation in jobs):
            raise EndpointBenchmarkError(f"{tag}: a supplied validation fold lacks one class")
    if (fold_id < 0).any():
        raise EndpointBenchmarkError(f"{tag}: fold assignment is incomplete")
    columns = run_folds(panel, features, labels, jobs,
                        [int(seed) + k for k in range(len(jobs))],
                        n_rows=len(labels), tag=f"{tag} fold")
    frame = pd.DataFrame({"fold": fold_id, "label": labels,
                          **{f"p_{k}": v for k, v in add_blends(columns).items()}})
    active_similarity = np.full(len(labels), np.nan, dtype=float)
    for fit, validation in jobs:
        active_similarity[validation] = max_tanimoto(
            features["morgan"][validation], features["morgan"][fit])
    frame["maximum_tanimoto_to_active_fit_fold"] = active_similarity
    if checkpoint is not None:
        frame.to_csv(checkpoint, index=False, lineterminator="\n")
        atomic_write_json(checkpoint_hash_path, {"sha256": sha256_file(checkpoint)})
    return frame


def score_once(panel: Panel, fit_features: dict[str, np.ndarray], labels: np.ndarray,
               target_features: dict[str, np.ndarray], *, seed: int = 42
               ) -> pd.DataFrame:
    """One fit on the whole curated training cohort, one scoring of the target."""
    panel.reload_tabfm()          # deterministic global RNG state before scoring
    n_fit, n_target = len(labels), len(target_features["paper"])
    combined = {key: np.concatenate([fit_features[key], target_features[key]], axis=0)
                for key in ("paper", "morgan", "rdkit2d")}
    all_labels = np.concatenate([labels, np.full(n_target, -1, dtype=int)])
    probabilities = panel.fold_probabilities(
        combined, all_labels, np.arange(n_fit), np.arange(n_fit, n_fit + n_target),
        component_seed=int(seed), verbose=False)
    # Cross-validation and every downstream metric table use the same explicit
    # probability-column contract.  Returning bare model IDs here previously
    # caused the AgeXtend challenge stage to fail only after its full CV fit.
    frame = pd.DataFrame(
        {f"p_{model}": value
         for model, value in add_blends(probabilities).items()})
    frame["maximum_tanimoto_to_active_fit_fold"] = max_tanimoto(
        target_features["morgan"], fit_features["morgan"])
    return frame


def metric_table(frame: pd.DataFrame, *, group: str | None = None) -> pd.DataFrame:
    rows = []
    blocks = ([(None, frame)] if group is None
              else list(frame.groupby(group)) + [("pooled", frame)])
    for name, block in blocks:
        y = block["label"].to_numpy(int)
        if len(np.unique(y)) < 2:
            continue
        for model in PANEL:
            probability = block[f"p_{model}"].to_numpy(float)
            record = {"model_id": model, **_metrics(y, probability, probability)}
            if group is not None:
                record = {group: name, **record}
            rows.append(record)
    return pd.DataFrame(rows)


def attach_molecular_variables(curated: pd.DataFrame, predictions: pd.DataFrame,
                               *, cohort: str, d1_train_bits: np.ndarray
                               ) -> pd.DataFrame:
    """The Experiment A compound table schema, for an endpoint-aligned cohort."""
    from geroprotector.study_common import max_tanimoto, qed_stratum
    table = predictions.copy().reset_index(drop=True)
    table["cohort"] = cohort
    table["compound_id"] = curated["compound_id"].to_numpy()
    table["smiles"] = curated["standardized_parent_smiles"].to_numpy()
    table["overlaps_d1_connectivity"] = curated["overlaps_d1_connectivity"].to_numpy()
    properties = molecular_variables(table["smiles"].astype(str).tolist())
    for column in properties.columns:
        table[column] = properties[column].to_numpy()
    bits = morgan_matrix(table["smiles"].astype(str).tolist(), morgan_generator())
    table["max_tanimoto_to_d1_train_secondary"] = max_tanimoto(bits, d1_train_bits)
    if "maximum_tanimoto_to_active_fit_fold" not in table.columns:
        raise EndpointBenchmarkError(
            f"{cohort}: predictions lack fold-local active-training similarity")
    table["max_train_tanimoto"] = table[
        "maximum_tanimoto_to_active_fit_fold"].to_numpy(float)
    table["max_train_tanimoto_scope"] = "active_endpoint_fit_fold_or_full_challenge_fit"
    table["qed_stratum"] = qed_stratum(table["qed"].to_numpy(float))
    for model in PANEL:
        table[f"d_{model}"] = (table[f"p_{model}"] >= 0.5).astype(int)
    return table
