"""Audited reconstruction of the 382-compound supervised cohort."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from ..chemistry.standardize import (
    STANDARDIZATION_CONTRACT,
    ChemistryError,
    rdkit_version,
    standardize_smiles,
)
from ..hashing import atomic_write_json, canonical_sha256, sha256_file
from ..logging import utc_now
from .acquire import SOURCE_SPECS, UPSTREAM_COMMIT
from .identities import compound_id, raw_row_id


class CurationError(ValueError):
    """Raised when provenance or the locked cohort cannot be reconstructed."""


PAPER_SEVEN_COLUMNS = (
    "total_molweight",
    "clogp",
    "h_acceptors",
    "h_donors",
    "total_surface_area",
    "relative_psa",
    "rotatable_bonds",
)

PAPER_SEVEN_SOURCE_COLUMNS = {
    "total_molweight": "Total Molweight",
    "clogp": "cLogP",
    "h_acceptors": "H-Acceptors",
    "h_donors": "H-Donors",
    "total_surface_area": "Total Surface Area",
    "relative_psa": "Relative PSA",
    "rotatable_bonds": "Rotatable Bonds",
}


def _decision_key(name: object, smiles: object) -> tuple[str, str]:
    """Match curated decisions after harmless boundary-whitespace normalization.

    The exact source strings remain in the provenance ledger and raw-row hash.  This key
    only prevents source formatting whitespace from bypassing an explicit, structure-bound
    adjudication or repair decision.
    """

    return str(name).strip(), str(smiles).strip()


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite immutable table: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_overrides(path: Path) -> tuple[dict[tuple[str, str], str], str]:
    if path.is_symlink() or not path.is_file():
        raise CurationError(f"Curation overrides must be regular/non-symlink: {path}")
    frame = pd.read_csv(path)
    required = {"name", "original_smiles", "curated_smiles"}
    missing = required - set(frame)
    if missing:
        raise CurationError(f"Curation override columns missing: {sorted(missing)}")
    keys = [
        _decision_key(name, smiles)
        for name, smiles in zip(frame["name"], frame["original_smiles"], strict=True)
    ]
    if len(keys) != len(set(keys)):
        raise CurationError("Curation override keys are duplicated")
    return (
        {
            key: str(value)
            for key, value in zip(keys, frame["curated_smiles"].astype(str), strict=True)
        },
        sha256_file(path),
    )


def _load_metal_adjudication(
    path: Path,
) -> tuple[dict[tuple[str, str], dict[str, str]], str]:
    if path.is_symlink() or not path.is_file():
        raise CurationError(f"Metal adjudication must be regular/non-symlink: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "name",
        "original_smiles",
        "adjudication",
        "retained_representation",
        "note",
    }
    missing = required - set(frame)
    if missing:
        raise CurationError(f"Metal adjudication columns missing: {sorted(missing)}")
    keys = [
        _decision_key(name, smiles)
        for name, smiles in zip(frame["name"], frame["original_smiles"], strict=True)
    ]
    if len(keys) != len(set(keys)):
        raise CurationError("Metal adjudication keys are duplicated")
    allowed = {"simple_counterion_salt_parent", "coordination_formulation_ligand_parent"}
    if not set(frame["adjudication"].astype(str)).issubset(allowed):
        raise CurationError("Metal adjudication contains an unsupported decision")
    records = {
        key: {
            "metal_adjudication": str(row["adjudication"]),
            "metal_retained_representation": str(row["retained_representation"]),
            "metal_adjudication_note": str(row["note"]),
        }
        for key, (_, row) in zip(keys, frame.iterrows(), strict=True)
    }
    return records, sha256_file(path)


def _read_raw_source(raw_dir: Path, spec: Any) -> pd.DataFrame:
    path = raw_dir / spec.cache_name
    if path.is_symlink() or not path.is_file():
        raise CurationError(f"Missing regular pinned source: {path}")
    actual = sha256_file(path)
    if actual != spec.sha256:
        raise CurationError(f"Source hash mismatch: {actual} != {spec.sha256} ({path})")
    frame = pd.read_csv(path, sep=spec.separator, encoding=spec.encoding)
    required = {spec.name_column, spec.smiles_column}
    missing = required - set(frame)
    if missing:
        raise CurationError(f"{spec.source_role} columns missing: {sorted(missing)}")
    if len(frame) != spec.rows:
        raise CurationError(f"{spec.source_role} rows changed: {len(frame)} != {spec.rows}")
    unique_smiles = frame[spec.smiles_column].astype(str).str.strip().nunique()
    if unique_smiles != spec.unique_smiles:
        raise CurationError(
            f"{spec.source_role} unique SMILES changed: {unique_smiles} != {spec.unique_smiles}"
        )
    return frame


def _canonical_cohort_hash(frame: pd.DataFrame) -> str:
    columns = [
        "compound_id",
        "standardized_parent_smiles",
        "full_inchikey",
        "connectivity_inchikey",
        "label",
        "source_role",
    ]
    payload = (
        frame.loc[:, columns]
        .sort_values("compound_id", kind="stable")
        .to_dict(orient="records")
    )
    return canonical_sha256({"schema": "geroprotector.curated.v1", "rows": payload})


def curate_cohort(
    *,
    raw_dir: str | Path,
    curated_table: str | Path,
    provenance_table: str | Path,
    identity_conflicts: str | Path,
    standardization_ledger: str | Path,
    manifest_path: str | Path,
    curation_overrides: str | Path,
    expected_overrides_sha256: str,
    metal_adjudication: str | Path,
    expected_metal_adjudication_sha256: str,
    expected_counts: Mapping[str, int],
    expected_identity_label_sha256: str,
    expected_rdkit_version: str,
    resolved_config_sha256: str,
) -> dict[str, Any]:
    """Create all curation artifacts without ever promoting prior predictions to labels."""

    if rdkit_version() != str(expected_rdkit_version):
        raise CurationError(
            f"RDKit version mismatch: {rdkit_version()} != {expected_rdkit_version}"
        )
    raw_root = Path(raw_dir)
    overrides, overrides_sha256 = _load_overrides(Path(curation_overrides))
    if overrides_sha256 != str(expected_overrides_sha256):
        raise CurationError(
            f"Curation override hash mismatch: {overrides_sha256} != "
            f"{expected_overrides_sha256}"
        )
    metal_decisions, metal_adjudication_sha256 = _load_metal_adjudication(
        Path(metal_adjudication)
    )
    if metal_adjudication_sha256 != str(expected_metal_adjudication_sha256):
        raise CurationError(
            "Metal adjudication hash mismatch: "
            f"{metal_adjudication_sha256} != {expected_metal_adjudication_sha256}"
        )
    records: list[dict[str, Any]] = []
    matched_overrides: dict[tuple[str, str], int] = {key: 0 for key in overrides}
    matched_metal_decisions: dict[tuple[str, str], int] = {key: 0 for key in metal_decisions}
    for spec in SOURCE_SPECS:
        source = _read_raw_source(raw_root, spec)
        for index, row in source.iterrows():
            name = str(row[spec.name_column])
            raw_smiles = str(row[spec.smiles_column])
            key = _decision_key(name, raw_smiles)
            standardization_input = overrides.get(key, raw_smiles)
            if key in overrides:
                matched_overrides[key] += 1
            base = {
                "raw_row_id": raw_row_id(spec.source_role, int(index), name, raw_smiles),
                "source_row_index": int(index),
                "source_role": spec.source_role,
                "raw_name": name,
                "raw_smiles": raw_smiles,
                "standardization_input_smiles": standardization_input,
                "smiles_curated": key in overrides,
                "label": spec.supervised_label,
                "is_prior_ml_candidate": spec.source_role == "prior_ml_unlabeled",
                "input_file_sha256": spec.sha256,
            }
            if spec.supervised_label is not None:
                missing_descriptors = set(PAPER_SEVEN_SOURCE_COLUMNS.values()) - set(source)
                if missing_descriptors:
                    raise CurationError(
                        f"Paper-seven source columns missing: {sorted(missing_descriptors)}"
                    )
                for output_name, source_name in PAPER_SEVEN_SOURCE_COLUMNS.items():
                    value = pd.to_numeric(pd.Series([row[source_name]]), errors="coerce").iloc[
                        0
                    ]
                    if pd.isna(value):
                        raise CurationError(
                            f"Paper-seven descriptor is missing at {spec.source_role}:{index}: "
                            f"{source_name}"
                        )
                    base[f"paper7::{output_name}"] = float(value)
            try:
                molecule = standardize_smiles(standardization_input)
            except ChemistryError as exc:
                if spec.supervised_label is not None:
                    raise CurationError(
                        f"Supervised row cannot be standardized: {base['raw_row_id']}"
                    ) from exc
                records.append(
                    {
                        **base,
                        "curation_status": "excluded",
                        "curation_reason": "unparseable_unlabeled_prior_candidate",
                        "qc_flags": json.dumps(["unparseable"], separators=(",", ":")),
                    }
                )
                continue
            if molecule.metal_atomic_numbers and spec.supervised_label is not None:
                if key not in metal_decisions:
                    raise CurationError(
                        "Supervised metal-containing row lacks adjudication: "
                        f"{base['raw_row_id']}"
                    )
                matched_metal_decisions[key] += 1
                base.update(metal_decisions[key])
                base["metal_sensitive_representation"] = (
                    metal_decisions[key]["metal_adjudication"]
                    == "coordination_formulation_ligand_parent"
                )
            else:
                base.update(
                    {
                        "metal_adjudication": None,
                        "metal_retained_representation": None,
                        "metal_adjudication_note": None,
                        "metal_sensitive_representation": False,
                    }
                )
            records.append(
                {
                    **base,
                    **molecule.to_dict(),
                    "identity_group_id": molecule.connectivity_inchikey,
                    "curation_status": "pending",
                    "curation_reason": "pending_identity_resolution",
                    "qc_flags": json.dumps(list(molecule.qc_flags), separators=(",", ":")),
                }
            )
    invalid_override_counts = {
        key: count for key, count in matched_overrides.items() if count != 1
    }
    if invalid_override_counts:
        raise CurationError(
            "Curation overrides did not match exactly once: "
            f"{sorted(invalid_override_counts.items())}"
        )
    invalid_metal_counts = {
        key: count for key, count in matched_metal_decisions.items() if count != 1
    }
    if invalid_metal_counts:
        raise CurationError(
            "Metal adjudication rows did not match exactly once: "
            f"{sorted(invalid_metal_counts.items())}"
        )
    ledger = pd.DataFrame(records)
    if ledger["raw_row_id"].duplicated().any():
        raise CurationError("Raw-row IDs are not unique")

    supervised_mask = ledger["label"].notna()
    prior_mask = ledger["is_prior_ml_candidate"].astype(bool)
    if int((supervised_mask & prior_mask).sum()) != 0:
        raise CurationError("Prior-predicted candidates entered supervised labels")
    usable = ledger.loc[supervised_mask].copy()
    usable["label"] = usable["label"].astype(int)
    conflicts = (
        usable.groupby("identity_group_id", sort=True)["label"]
        .nunique()
        .loc[lambda values: values > 1]
        .index.astype(str)
    )
    conflict_set = set(conflicts)
    conflict_frame = usable.loc[usable["identity_group_id"].isin(conflict_set)].copy()
    expected_conflicts = {"KLBQZWRITKRQQV", "YEHCICAEULNIGD"}
    if conflict_set != expected_conflicts or len(conflict_frame) != 6:
        raise CurationError(
            "Identity-conflict derivation changed: "
            f"groups={sorted(conflict_set)}, rows={len(conflict_frame)}"
        )
    conflict_ids = set(conflict_frame["raw_row_id"].astype(str))
    ledger.loc[
        ledger["raw_row_id"].isin(conflict_ids), ["curation_status", "curation_reason"]
    ] = [
        "excluded",
        "conflicting_supervised_labels_entire_identity_group",
    ]
    nonconflict = usable.loc[~usable["identity_group_id"].isin(conflict_set)].copy()
    nonconflict = nonconflict.sort_values(
        [
            "identity_group_id",
            "component_count",
            "metal_sensitive_representation",
            "smiles_curated",
            "source_role",
            "raw_row_id",
        ],
        kind="stable",
    )
    nonconflict["representative_selection_rank"] = (
        nonconflict.groupby("identity_group_id", sort=False).cumcount() + 1
    )
    ledger["representative_selection_rank"] = pd.NA
    rank_by_raw_id = nonconflict.set_index("raw_row_id")["representative_selection_rank"]
    ledger.loc[
        ledger["raw_row_id"].isin(rank_by_raw_id.index), "representative_selection_rank"
    ] = ledger.loc[ledger["raw_row_id"].isin(rank_by_raw_id.index), "raw_row_id"].map(
        rank_by_raw_id
    )
    representative_ids = set(
        nonconflict.drop_duplicates("identity_group_id", keep="first")["raw_row_id"].astype(str)
    )
    duplicate_ids = set(nonconflict["raw_row_id"].astype(str)) - representative_ids
    if len(ledger) != 1893 or int(supervised_mask.sum()) != 405 or len(duplicate_ids) != 17:
        raise CurationError(
            "Pinned cohort derivation changed: expected raw/supervised/duplicates "
            f"1893/405/17, observed {len(ledger)}/{int(supervised_mask.sum())}/"
            f"{len(duplicate_ids)}"
        )
    ledger.loc[
        ledger["raw_row_id"].isin(duplicate_ids), ["curation_status", "curation_reason"]
    ] = [
        "excluded",
        "collapsed_consistent_identity_duplicate",
    ]
    ledger.loc[
        ledger["raw_row_id"].isin(representative_ids), ["curation_status", "curation_reason"]
    ] = ["included", "supervised_identity_representative"]
    ledger.loc[
        prior_mask & ledger["curation_status"].eq("pending"),
        ["curation_status", "curation_reason"],
    ] = [
        "excluded",
        "prior_ml_candidate_unlabeled_only",
    ]
    if ledger["curation_status"].eq("pending").any():
        raise CurationError("Some raw rows lack a terminal curation status")

    curated = ledger.loc[ledger["curation_status"].eq("included")].copy()
    curated["label"] = curated["label"].astype(int)
    curated["compound_id"] = curated["connectivity_inchikey"].map(compound_id)
    curated["identity_group_id"] = curated["connectivity_inchikey"]
    curated["label_conflict_flag"] = False
    curated = curated.rename(columns={"raw_name": "compound_name"})
    paper_columns = [f"paper7::{column}" for column in PAPER_SEVEN_COLUMNS]
    if curated[paper_columns].isna().any().any():
        raise CurationError(
            "Original-paper seven-descriptor compatibility mapping is incomplete"
        )
    duplicate_groups = nonconflict.loc[nonconflict.duplicated("identity_group_id", keep=False)]
    discordant_descriptor_groups = sorted(
        identity
        for identity, group in duplicate_groups.groupby("identity_group_id", sort=True)
        if any(group[column].nunique(dropna=False) > 1 for column in paper_columns)
    )
    curated = curated.sort_values("compound_id", kind="stable").reset_index(drop=True)
    if curated["compound_id"].duplicated().any():
        raise CurationError("Curated cohort contains duplicate connectivity identities")
    counts = {
        "total": len(curated),
        "positive": int(curated["label"].eq(1).sum()),
        "weak_reference": int(curated["label"].eq(0).sum()),
    }
    locked_counts = {key: int(value) for key, value in expected_counts.items()}
    if counts != locked_counts:
        raise CurationError(f"Curated counts changed: {counts} != {locked_counts}")
    if int(ledger.loc[prior_mask, "label"].notna().sum()) != 0:
        raise CurationError("The prior-candidate label invariant was violated")
    identity_label_sha256 = canonical_sha256(
        {
            "schema": "geroprotector.curated_identity_label.v1",
            "rows": curated.loc[:, ["identity_group_id", "label"]]
            .sort_values("identity_group_id", kind="stable")
            .to_dict(orient="records"),
        }
    )
    if identity_label_sha256 != str(expected_identity_label_sha256):
        raise CurationError(
            "Curated identity-label membership changed: "
            f"{identity_label_sha256} != {expected_identity_label_sha256}"
        )
    curated_identities = set(curated["identity_group_id"].astype(str))
    prior_overlaps = sorted(
        set(ledger.loc[prior_mask, "identity_group_id"].dropna().astype(str))
        & curated_identities
    )
    expected_prior_overlaps = ["HAPOVYFOVVWLRS", "PVNIIMVLHYAWGP"]
    if prior_overlaps != expected_prior_overlaps:
        raise CurationError(f"Prior/gold identity overlap changed: {prior_overlaps}")
    ledger["overlaps_supervised_identity"] = (
        ledger["identity_group_id"].isin(curated_identities) & prior_mask
    )

    curated_path = Path(curated_table)
    provenance_path = Path(provenance_table)
    conflicts_path = Path(identity_conflicts)
    ledger_path = Path(standardization_ledger)
    manifest_destination = Path(manifest_path)
    final_destinations = (
        curated_path,
        provenance_path,
        conflicts_path,
        ledger_path,
        manifest_destination,
    )
    parent_set = {destination.parent.resolve() for destination in final_destinations}
    if len(parent_set) != 1:
        raise CurationError("All curation artifacts must share one atomic bundle directory")
    final_bundle = next(iter(parent_set))
    if final_bundle.exists():
        raise FileExistsError(f"Refusing to overwrite curation bundle: {final_bundle}")
    final_bundle.parent.mkdir(parents=True, exist_ok=True)
    staging_bundle = Path(
        tempfile.mkdtemp(prefix=f".{final_bundle.name}.work-", dir=final_bundle.parent)
    )
    staged = {
        destination: staging_bundle / destination.name for destination in final_destinations
    }
    _atomic_parquet(curated, staged[curated_path])
    _atomic_parquet(ledger, staged[provenance_path])
    _atomic_parquet(ledger, staged[ledger_path])
    conflict_frame.to_csv(staged[conflicts_path], index=False)
    cohort_hash = _canonical_cohort_hash(curated)
    manifest = {
        "schema_version": "geroprotector.curation.v1",
        "created_utc": utc_now(),
        "upstream_commit": UPSTREAM_COMMIT,
        "resolved_config_sha256": str(resolved_config_sha256),
        "rdkit_version": rdkit_version(),
        "standardization_contract": STANDARDIZATION_CONTRACT,
        "curation_overrides_sha256": overrides_sha256,
        "metal_adjudication_sha256": metal_adjudication_sha256,
        "n_metal_adjudications": len(metal_decisions),
        "n_coordination_ligand_parent_representations": int(
            curated["metal_sensitive_representation"].sum()
        ),
        "paper_seven_descriptor_source": "selected_supervised_raw_row_in_pinned_source",
        "paper_seven_label_column_read": False,
        "paper_seven_descriptor_columns": list(PAPER_SEVEN_COLUMNS),
        "raw_rows_total": len(ledger),
        "supervised_raw_rows": int(supervised_mask.sum()),
        "raw_rows_terminal_status": int(ledger["curation_status"].notna().sum()),
        "raw_row_partition_complete": bool(
            ledger["raw_row_id"].nunique() == len(ledger)
            and ledger["curation_status"].isin(["included", "excluded"]).all()
        ),
        "curated_counts": counts,
        "n_identity_conflict_groups": len(conflict_set),
        "n_identity_conflict_rows": len(conflict_frame),
        "conflict_connectivity_inchikeys": sorted(conflict_set),
        "n_consistent_duplicate_rows_collapsed": len(duplicate_ids),
        "duplicate_representative_policy": (
            "fewest_components_then_noncoordination_then_no_smiles_override_then_"
            "source_role_then_raw_row_id"
        ),
        "paper_seven_duplicate_descriptor_discordance_groups": (discordant_descriptor_groups),
        "n_paper_seven_duplicate_descriptor_discordance_groups": len(
            discordant_descriptor_groups
        ),
        "n_prior_candidates_used_as_labels": 0,
        "source_label_confounding_disclosed": True,
        "scientific_estimand": (
            "source-defined reported geroprotectors versus weak ChEMBL references"
        ),
        "curated_cohort_canonical_sha256": cohort_hash,
        "curated_identity_label_sha256": identity_label_sha256,
        "prior_unlabeled_overlap_connectivity": prior_overlaps,
        "prior_unlabeled_overlap_count": len(prior_overlaps),
        "artifacts": {
            "curated_table": {
                "path": curated_path.as_posix(),
                "sha256": sha256_file(staged[curated_path]),
            },
            "provenance_table": {
                "path": provenance_path.as_posix(),
                "sha256": sha256_file(staged[provenance_path]),
            },
            "identity_conflicts": {
                "path": conflicts_path.as_posix(),
                "sha256": sha256_file(staged[conflicts_path]),
            },
            "standardization_ledger": {
                "path": ledger_path.as_posix(),
                "sha256": sha256_file(staged[ledger_path]),
            },
        },
        "sources": [
            {
                "source_role": spec.source_role,
                "sha256": spec.sha256,
                "rows": spec.rows,
            }
            for spec in SOURCE_SPECS
        ],
    }
    manifest["canonical_sha256"] = canonical_sha256(manifest)
    atomic_write_json(staged[manifest_destination], manifest)
    os.rename(staging_bundle, final_bundle)
    return manifest


def load_curated_cohort(
    *,
    curated_table: str | Path,
    provenance_table: str | Path,
    manifest_path: str | Path,
    resolved_config_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load the curated cohort only after byte, membership and config rebinding."""

    curated_path = Path(curated_table)
    provenance_path = Path(provenance_table)
    manifest_source = Path(manifest_path)
    for role, path in {
        "curated table": curated_path,
        "provenance table": provenance_path,
        "curation manifest": manifest_source,
    }.items():
        if path.is_symlink() or not path.is_file():
            raise CurationError(f"{role} must be a regular non-symlink file: {path}")
    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    payload = dict(manifest)
    claimed = payload.pop("canonical_sha256", None)
    if claimed != canonical_sha256(payload):
        raise CurationError("Curation manifest canonical hash failed")
    if manifest.get("resolved_config_sha256") != str(resolved_config_sha256):
        raise CurationError("Curation manifest was built from a different resolved config")
    if manifest.get("rdkit_version") != rdkit_version():
        raise CurationError(
            "RDKit version differs from the version that produced the curated cohort"
        )
    artifacts = manifest.get("artifacts", {})
    actual_paths = {
        "curated_table": curated_path,
        "provenance_table": provenance_path,
    }
    for role in ("identity_conflicts", "standardization_ledger"):
        record = artifacts.get(role, {})
        if not isinstance(record, dict) or not record.get("path"):
            raise CurationError(f"Curation manifest lacks artifact binding: {role}")
        configured = Path(str(record["path"]))
        candidate = (
            configured if configured.is_absolute() else manifest_source.parent / configured.name
        )
        actual_paths[role] = candidate
    for role, path in actual_paths.items():
        record = artifacts.get(role, {})
        if not isinstance(record, dict) or not record.get("path"):
            raise CurationError(f"Curation manifest lacks artifact binding: {role}")
        configured = Path(str(record["path"]))
        if configured.name != path.name:
            raise CurationError(f"Curation artifact filename binding changed: {role}")
        if path.parent.resolve() != manifest_source.parent.resolve():
            raise CurationError(f"Curation artifact escaped its atomic bundle: {role}")
        if path.is_symlink() or not path.is_file():
            raise CurationError(f"Curation artifact must be regular/non-symlink: {path}")
        if record.get("sha256") != sha256_file(path):
            raise CurationError(f"Curation artifact byte hash changed: {role}")
    curated = pd.read_parquet(curated_path)
    provenance = pd.read_parquet(provenance_path)
    counts = {
        "total": len(curated),
        "positive": int(curated["label"].eq(1).sum()),
        "weak_reference": int(curated["label"].eq(0).sum()),
    }
    if counts != manifest.get("curated_counts"):
        raise CurationError("Curated counts differ from the sealed manifest")
    identity_hash = canonical_sha256(
        {
            "schema": "geroprotector.curated_identity_label.v1",
            "rows": curated.loc[:, ["identity_group_id", "label"]]
            .sort_values("identity_group_id", kind="stable")
            .to_dict(orient="records"),
        }
    )
    if identity_hash != manifest.get("curated_identity_label_sha256"):
        raise CurationError("Curated identity-label membership changed")
    if provenance["raw_row_id"].duplicated().any() or len(provenance) != 1893:
        raise CurationError("Provenance raw-row partition changed")
    if int(provenance.loc[provenance["is_prior_ml_candidate"], "label"].notna().sum()):
        raise CurationError("Prior-predicted candidates acquired supervised labels")
    return curated, provenance, manifest
