"""Immutable identity-safe paper-style 80/20 registry for V5bis and V6bis."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import DataStructs

from ..hashing import atomic_write_json, canonical_sha256, sha256_file
from ..logging import utc_now
from .split_registry import (
    INNER_REGISTRY_COLUMNS,
    _cohort_binding_sha256,
    _validate_cohort,
    grouped_inner_folds,
    validate_selection_folds,
)


class PaperSplitIntegrityError(ValueError):
    """Raised when the contextual paper holdout no longer matches its lock."""


PAPER_REGISTRY_COLUMNS = (
    "compound_id",
    "label",
    "component_id",
    "repeat",
    "outer_fold",
    "role",
)
PAPER_INNER_SEED = 142


def _curated_morgan_bitvectors(smiles: pd.Series):
    """Fingerprint already-standardized parents without a second tautomer pass."""

    from rdkit import Chem
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    generator = GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    vectors = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise PaperSplitIntegrityError("Curated standardized parent failed to parse")
        vectors.append(generator.GetFingerprint(molecule))
    return vectors


def _exact_integer_series(values: pd.Series, *, name: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise PaperSplitIntegrityError(f"{name} must contain exact finite integers") from exc
    array = numeric.to_numpy(dtype=float)
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise PaperSplitIntegrityError(f"{name} must contain exact finite integers")
    return pd.Series(array.astype(np.int64), index=values.index, name=values.name)


def _test_identities(path: Path, *, expected_sha256: str) -> tuple[str, ...]:
    if path.is_symlink() or not path.is_file():
        raise PaperSplitIntegrityError(f"Test-identity lock must be regular: {path}")
    if sha256_file(path) != str(expected_sha256):
        raise PaperSplitIntegrityError("Paper test-identity lock bytes changed")
    values = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    if len(values) != 77 or len(set(values)) != 77 or any(not value for value in values):
        raise PaperSplitIntegrityError("Paper test-identity lock must contain 77 unique IDs")
    return values


def paper_registry_sha256(registry: pd.DataFrame) -> str:
    missing = set(PAPER_REGISTRY_COLUMNS) - set(registry)
    if missing:
        raise PaperSplitIntegrityError(f"Paper registry columns missing: {sorted(missing)}")
    rows = registry.loc[:, list(PAPER_REGISTRY_COLUMNS)].sort_values(
        "compound_id", kind="stable"
    )
    return canonical_sha256(
        {"schema": "geroprotector.paper80_registry.v1", "rows": rows.to_dict("records")}
    )


def _inner_sha256(inner: pd.DataFrame) -> str:
    return canonical_sha256(
        {
            "schema": "geroprotector.paper80_inner_registry.v1",
            "rows": inner.sort_values("compound_id", kind="stable").to_dict("records"),
        }
    )


def _partition_components(curated: pd.DataFrame, *, test_compound_ids: set[str]) -> pd.Series:
    """Fit grouping independently within outer-train and outer-test.

    Only the outer-train components are consumed by model selection.  Computing the
    test partition separately prevents test structures from joining two training
    components through a transitive similarity path.
    """

    parts: list[pd.Series] = []
    for select_test in (False, True):
        mask = curated["compound_id"].astype(str).isin(test_compound_ids)
        subset = curated.loc[mask if select_test else ~mask].copy()
        if subset.empty:
            raise PaperSplitIntegrityError("Paper train/test partition is empty")
        parts.append(_components_from_curated_parents(subset))
    combined = pd.concat(parts)
    if set(combined.index.astype(str)) != set(curated["compound_id"].astype(str)):
        raise PaperSplitIntegrityError("Partition-local component coverage is incomplete")
    return combined


def _components_from_curated_parents(curated: pd.DataFrame) -> pd.Series:
    """Build the locked r2/2048 graph without re-standardizing sealed parents."""

    ids = pd.Index(curated["compound_id"].astype(str), name="compound_id")
    identities = curated["connectivity_inchikey"].astype(str).to_numpy()
    if ids.has_duplicates or len(ids) == 0:
        raise PaperSplitIntegrityError("Component input IDs must be unique and nonempty")
    vectors = _curated_morgan_bitvectors(curated["standardized_parent_smiles"])
    parent = list(range(len(ids)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for right in range(1, len(vectors)):
        for left, similarity in enumerate(
            DataStructs.BulkTanimotoSimilarity(vectors[right], vectors[:right])
        ):
            if float(similarity) >= 0.40:
                union(left, right)
    members: dict[int, list[int]] = {}
    for position in range(len(ids)):
        members.setdefault(find(position), []).append(position)
    values = [""] * len(ids)
    for positions in members.values():
        component = f"SIM0.400::{min(identities[position] for position in positions)}"
        for position in positions:
            values[position] = component
    return pd.Series(values, index=ids, name="component_id", dtype="string")


def validate_paper_registries(
    cohort: pd.DataFrame,
    outer: pd.DataFrame,
    inner: pd.DataFrame,
    *,
    expected_test_identities: tuple[str, ...],
    inner_folds: int,
) -> dict[str, Any]:
    curated = _validate_cohort(cohort)
    missing = set(PAPER_REGISTRY_COLUMNS) - set(outer)
    if missing:
        raise PaperSplitIntegrityError(f"Paper registry columns missing: {sorted(missing)}")
    table = outer.loc[:, list(PAPER_REGISTRY_COLUMNS)].copy()
    if table.isna().any().any() or table["compound_id"].duplicated().any():
        raise PaperSplitIntegrityError("Paper registry contains missing or duplicate rows")
    table["label"] = _exact_integer_series(table["label"], name="paper.label")
    table["repeat"] = _exact_integer_series(table["repeat"], name="paper.repeat")
    table["outer_fold"] = _exact_integer_series(table["outer_fold"], name="paper.outer_fold")
    if not set(table["label"]).issubset({0, 1}):
        raise PaperSplitIntegrityError("Paper registry labels must be binary integers")
    if (table["compound_id"].astype(str).str.strip() == "").any() or (
        table["component_id"].astype(str).str.strip() == ""
    ).any():
        raise PaperSplitIntegrityError("Paper registry IDs/components must be nonblank")
    valid_roles = table["role"].isin({"paper_train", "paper_test"})
    role_fold_match = (table["role"].eq("paper_test") & table["outer_fold"].eq(0)) | (
        table["role"].eq("paper_train") & table["outer_fold"].eq(1)
    )
    if not valid_roles.all() or not role_fold_match.all():
        raise PaperSplitIntegrityError("Paper roles and encoded train/test folds disagree")
    if set(table["compound_id"].astype(str)) != set(curated["compound_id"].astype(str)):
        raise PaperSplitIntegrityError("Paper registry does not cover the curated cohort")
    if set(table["repeat"]) != {0} or set(table["outer_fold"]) != {0, 1}:
        raise PaperSplitIntegrityError("Paper registry must encode one train/test holdout")
    expected_test = {f"cmp::{value}" for value in expected_test_identities}
    observed_test = set(table.loc[table["role"].eq("paper_test"), "compound_id"].astype(str))
    observed_train = set(table.loc[table["role"].eq("paper_train"), "compound_id"].astype(str))
    if observed_test != expected_test or len(observed_test) != 77 or len(observed_train) != 305:
        raise PaperSplitIntegrityError("Paper train/test membership differs from its lock")
    if observed_test & observed_train or observed_test | observed_train != set(
        table["compound_id"].astype(str)
    ):
        raise PaperSplitIntegrityError("Paper train/test identity partition is invalid")
    expected_components = _partition_components(curated, test_compound_ids=expected_test)
    observed_components = table.set_index("compound_id")["component_id"].astype(str)
    if not observed_components.eq(
        expected_components.reindex(observed_components.index).astype(str)
    ).all():
        raise PaperSplitIntegrityError(
            "Paper registry components were not fit independently within partitions"
        )
    indexed = curated.set_index("compound_id")
    metadata = table.set_index("compound_id")
    if not (
        metadata["label"].astype(int).eq(indexed.loc[metadata.index, "label"].astype(int)).all()
    ):
        raise PaperSplitIntegrityError("Paper registry labels differ from curated labels")
    test_counts = metadata.loc[list(observed_test), "label"].astype(int).value_counts()
    if {0: int(test_counts.get(0, 0)), 1: int(test_counts.get(1, 0))} != {0: 29, 1: 48}:
        raise PaperSplitIntegrityError("Paper test class counts differ from V3-1")
    if set(inner.columns) != set(INNER_REGISTRY_COLUMNS):
        raise PaperSplitIntegrityError("Paper inner-registry schema changed")
    inner_table = inner.loc[:, list(INNER_REGISTRY_COLUMNS)].copy()
    if inner_table.isna().any().any() or not inner_table["role"].eq("inner_validation").all():
        raise PaperSplitIntegrityError("Paper inner registry has missing/invalid roles")
    for column in ("label", "repeat", "outer_fold", "inner_fold", "assignment_seed"):
        inner_table[column] = _exact_integer_series(
            inner_table[column], name=f"paper.inner.{column}"
        )
    if not set(inner_table["label"]).issubset({0, 1}):
        raise PaperSplitIntegrityError("Paper inner labels must be binary integers")
    if set(inner_table["repeat"]) != {0} or set(inner_table["outer_fold"]) != {0}:
        raise PaperSplitIntegrityError("Paper inner registry has invalid outer coordinates")
    if set(inner_table["inner_fold"]) != set(range(int(inner_folds))):
        raise PaperSplitIntegrityError("Paper inner registry fold coverage changed")
    if set(inner_table["assignment_seed"]) != {PAPER_INNER_SEED}:
        raise PaperSplitIntegrityError("Paper inner assignment seed changed")
    if (inner_table["compound_id"].astype(str).str.strip() == "").any() or (
        inner_table["component_id"].astype(str).str.strip() == ""
    ).any():
        raise PaperSplitIntegrityError("Paper inner IDs/components must be nonblank")
    if inner_table["compound_id"].duplicated().any():
        raise PaperSplitIntegrityError("Paper inner registry duplicates a compound")
    if set(inner_table["compound_id"].astype(str)) != observed_train:
        raise PaperSplitIntegrityError("Paper inner registry is not exactly outer-train")
    inner_metadata = inner_table.set_index("compound_id")
    expected_train_metadata = metadata.loc[inner_metadata.index]
    if (
        not inner_metadata["label"].eq(expected_train_metadata["label"]).all()
        or not (
            inner_metadata["component_id"]
            .astype(str)
            .eq(expected_train_metadata["component_id"].astype(str))
        ).all()
    ):
        raise PaperSplitIntegrityError("Paper inner metadata differs from outer-train")
    folds = []
    for fold in range(int(inner_folds)):
        valid = tuple(
            sorted(
                inner_table.loc[
                    inner_table["inner_fold"].astype(int).eq(fold), "compound_id"
                ].astype(str)
            )
        )
        train = tuple(sorted(observed_train - set(valid)))
        folds.append((train, valid))
    validate_selection_folds(
        fit_ids=tuple(sorted(observed_train)),
        labels=metadata.loc[list(observed_train), "label"].astype(int),
        groups=metadata.loc[list(observed_train), "component_id"].astype(str),
        folds=folds,
    )
    global_components = _components_from_curated_parents(curated)
    train_components = set(global_components.loc[list(observed_train)].astype(str))
    test_components = set(global_components.loc[list(observed_test)].astype(str))
    return {
        "registry_sha256": paper_registry_sha256(table),
        "inner_registry_sha256": _inner_sha256(inner_table),
        "n_rows": len(table),
        "n_train": len(observed_train),
        "n_test": len(observed_test),
        "test_class_0": 29,
        "test_class_1": 48,
        "identity_overlap_count": 0,
        "component_overlap_count": len(train_components & test_components),
    }


def build_paper_split_registry(
    cohort: pd.DataFrame,
    *,
    output_path: str | Path,
    inner_output_path: str | Path,
    manifest_path: str | Path,
    curated_table_sha256: str,
    data_manifest_path: str | Path,
    resolved_config_sha256: str,
    test_identity_path: str | Path,
    test_identity_file_sha256: str,
    inner_folds: int = 3,
    inner_seed: int = 142,
) -> dict[str, Any]:
    curated = _validate_cohort(cohort)
    identity_path = Path(test_identity_path)
    test_identities = _test_identities(identity_path, expected_sha256=test_identity_file_sha256)
    expected_test = {f"cmp::{value}" for value in test_identities}
    components = _partition_components(curated, test_compound_ids=expected_test)
    rows = []
    for row in curated.itertuples(index=False):
        is_test = str(row.compound_id) in expected_test
        rows.append(
            {
                "compound_id": str(row.compound_id),
                "label": int(row.label),
                "component_id": str(components.loc[str(row.compound_id)]),
                "repeat": 0,
                "outer_fold": 0 if is_test else 1,
                "role": "paper_test" if is_test else "paper_train",
            }
        )
    outer = pd.DataFrame(rows, columns=list(PAPER_REGISTRY_COLUMNS))
    train = outer.loc[outer["role"].eq("paper_train")].set_index("compound_id")
    folds = grouped_inner_folds(
        train["label"].astype(int),
        train["component_id"].astype(str),
        n_splits=int(inner_folds),
        seed=int(inner_seed),
    )
    assignment = {
        compound: fold for fold, (_, validation) in enumerate(folds) for compound in validation
    }
    inner_rows = [
        {
            "compound_id": compound,
            "label": int(train.loc[compound, "label"]),
            "component_id": str(train.loc[compound, "component_id"]),
            "repeat": 0,
            "outer_fold": 0,
            "inner_fold": int(assignment[compound]),
            "role": "inner_validation",
            "assignment_seed": int(inner_seed),
        }
        for compound in sorted(train.index.astype(str))
    ]
    inner = pd.DataFrame(inner_rows, columns=list(INNER_REGISTRY_COLUMNS))
    audit = validate_paper_registries(
        curated,
        outer,
        inner,
        expected_test_identities=test_identities,
        inner_folds=int(inner_folds),
    )
    vectors = _curated_morgan_bitvectors(curated["standardized_parent_smiles"])
    vector_by_id = dict(zip(curated["compound_id"].astype(str), vectors, strict=True))
    train_ids = tuple(sorted(train.index.astype(str)))
    test_ids = tuple(sorted(expected_test))
    train_vectors = [vector_by_id[value] for value in train_ids]
    nearest = [
        max(DataStructs.BulkTanimotoSimilarity(vector_by_id[value], train_vectors))
        for value in test_ids
    ]
    destinations = tuple(map(Path, (output_path, inner_output_path, manifest_path)))
    csv_paths = (destinations[0].with_suffix(".csv"), destinations[1].with_suffix(".csv"))
    parents = {path.parent.resolve() for path in (*destinations, *csv_paths)}
    if len(parents) != 1:
        raise PaperSplitIntegrityError("Paper split artifacts must share one directory")
    final_bundle = next(iter(parents))
    if final_bundle.exists():
        raise FileExistsError(f"Refusing to overwrite paper split bundle: {final_bundle}")
    final_bundle.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{final_bundle.name}.work-", dir=final_bundle.parent)
    )
    staged_outer = staging / destinations[0].name
    staged_inner = staging / destinations[1].name
    staged_outer_csv = staging / csv_paths[0].name
    staged_inner_csv = staging / csv_paths[1].name
    outer.to_parquet(staged_outer, index=False)
    inner.to_parquet(staged_inner, index=False)
    outer.to_csv(staged_outer_csv, index=False)
    inner.to_csv(staged_inner_csv, index=False)
    data_manifest = Path(data_manifest_path)
    manifest = {
        "schema_version": "geroprotector.paper80_split_manifest.v1",
        "created_utc": utc_now(),
        **audit,
        "strategy": "paper_random_80_20",
        "role": "retrospective_contextual_only",
        "identity_resolution_before_split": True,
        "source_paper_rows": 405,
        "curated_rows": 382,
        "random_state": 42,
        "shuffle": True,
        "stratify": False,
        "headline_eligible": False,
        "replaces_grouped_primary_validation": False,
        "outer_test_used_for_selection": False,
        "outer_test_used_for_calibration": False,
        "inner_components_fit_on_outer_train_only": True,
        "outer_test_structures_used_for_inner_grouping": False,
        "component_overlap_is_post_split_audit_only": True,
        "test_identity_file_sha256": str(test_identity_file_sha256),
        "split_input_cohort_sha256": _cohort_binding_sha256(curated),
        "curated_table_sha256": str(curated_table_sha256),
        "data_manifest_sha256": sha256_file(data_manifest),
        "resolved_config_sha256": str(resolved_config_sha256),
        "inner_folds": int(inner_folds),
        "inner_assignment_seed": int(inner_seed),
        "nearest_train_tanimoto_min": float(np.min(nearest)),
        "nearest_train_tanimoto_median": float(np.median(nearest)),
        "nearest_train_tanimoto_max": float(np.max(nearest)),
        "registry_parquet_sha256": sha256_file(staged_outer),
        "registry_csv_sha256": sha256_file(staged_outer_csv),
        "inner_registry_parquet_sha256": sha256_file(staged_inner),
        "inner_registry_csv_sha256": sha256_file(staged_inner_csv),
    }
    manifest["canonical_sha256"] = canonical_sha256(manifest)
    atomic_write_json(staging / destinations[2].name, manifest)
    os.rename(staging, final_bundle)
    return manifest


def load_verified_paper_registries(
    *,
    cohort: pd.DataFrame,
    outer_path: str | Path,
    inner_path: str | Path,
    manifest_path: str | Path,
    curated_table_path: str | Path,
    data_manifest_path: str | Path,
    resolved_config_sha256: str,
    test_identity_path: str | Path,
    test_identity_file_sha256: str,
    inner_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    paths = {
        "outer": Path(outer_path),
        "inner": Path(inner_path),
        "manifest": Path(manifest_path),
        "curated": Path(curated_table_path),
        "data_manifest": Path(data_manifest_path),
    }
    for role, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise PaperSplitIntegrityError(f"Paper {role} must be regular: {path}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    payload = dict(manifest)
    claimed = payload.pop("canonical_sha256", None)
    if claimed != canonical_sha256(payload):
        raise PaperSplitIntegrityError("Paper split manifest canonical hash failed")
    bindings = {
        "registry_parquet_sha256": paths["outer"],
        "inner_registry_parquet_sha256": paths["inner"],
        "registry_csv_sha256": paths["outer"].with_suffix(".csv"),
        "inner_registry_csv_sha256": paths["inner"].with_suffix(".csv"),
        "curated_table_sha256": paths["curated"],
        "data_manifest_sha256": paths["data_manifest"],
    }
    for field, path in bindings.items():
        if path.is_symlink() or not path.is_file() or manifest.get(field) != sha256_file(path):
            raise PaperSplitIntegrityError(f"Paper split artifact binding failed: {field}")
    if manifest.get("resolved_config_sha256") != str(resolved_config_sha256):
        raise PaperSplitIntegrityError("Paper split config binding changed")
    exact_contract = {
        "schema_version": "geroprotector.paper80_split_manifest.v1",
        "strategy": "paper_random_80_20",
        "role": "retrospective_contextual_only",
        "identity_resolution_before_split": True,
        "source_paper_rows": 405,
        "curated_rows": 382,
        "random_state": 42,
        "shuffle": True,
        "stratify": False,
        "headline_eligible": False,
        "replaces_grouped_primary_validation": False,
        "outer_test_used_for_selection": False,
        "outer_test_used_for_calibration": False,
        "inner_components_fit_on_outer_train_only": True,
        "outer_test_structures_used_for_inner_grouping": False,
        "component_overlap_is_post_split_audit_only": True,
        "test_identity_file_sha256": str(test_identity_file_sha256),
        "inner_folds": int(inner_folds),
        "inner_assignment_seed": PAPER_INNER_SEED,
    }
    for field, expected in exact_contract.items():
        if manifest.get(field) != expected:
            raise PaperSplitIntegrityError(f"Paper split contract changed: {field}")
    if manifest.get("split_input_cohort_sha256") != _cohort_binding_sha256(
        _validate_cohort(cohort)
    ):
        raise PaperSplitIntegrityError("Paper split cohort binding changed")
    for field in (
        "nearest_train_tanimoto_min",
        "nearest_train_tanimoto_median",
        "nearest_train_tanimoto_max",
    ):
        value = manifest.get(field)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise PaperSplitIntegrityError(f"Paper split similarity audit invalid: {field}")
    test_ids = _test_identities(
        Path(test_identity_path), expected_sha256=test_identity_file_sha256
    )
    outer = pd.read_parquet(paths["outer"])
    inner = pd.read_parquet(paths["inner"])
    audit = validate_paper_registries(
        cohort,
        outer,
        inner,
        expected_test_identities=test_ids,
        inner_folds=int(inner_folds),
    )
    if audit["registry_sha256"] != manifest.get("registry_sha256") or audit[
        "inner_registry_sha256"
    ] != manifest.get("inner_registry_sha256"):
        raise PaperSplitIntegrityError("Paper registry canonical binding changed")
    for field in (
        "n_rows",
        "n_train",
        "n_test",
        "test_class_0",
        "test_class_1",
        "identity_overlap_count",
        "component_overlap_count",
    ):
        if manifest.get(field) != audit[field]:
            raise PaperSplitIntegrityError(f"Paper split audit changed: {field}")
    return outer, inner, manifest
