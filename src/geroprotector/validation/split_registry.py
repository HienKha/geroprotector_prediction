"""One immutable outer split registry consumed by every pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ..chemistry.components import similarity_components
from ..chemistry.fingerprints import morgan_bitvectors
from ..data.identities import compound_id
from ..hashing import atomic_write_json, canonical_sha256, sha256_file
from ..logging import utc_now


class SplitIntegrityError(ValueError):
    pass


REGISTRY_COLUMNS = (
    "compound_id",
    "label",
    "component_id",
    "repeat",
    "outer_fold",
    "role",
)

INNER_REGISTRY_COLUMNS = (
    "compound_id",
    "label",
    "component_id",
    "repeat",
    "outer_fold",
    "inner_fold",
    "role",
    "assignment_seed",
)


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite split registry: {destination}")
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


def _validate_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "compound_id",
        "standardized_parent_smiles",
        "connectivity_inchikey",
        "label",
        "identity_group_id",
    }
    missing = required - set(frame)
    if missing:
        raise SplitIntegrityError(f"Curated cohort missing split columns: {sorted(missing)}")
    cohort = frame.copy()
    cohort["compound_id"] = cohort["compound_id"].astype(str)
    if (
        cohort["compound_id"].duplicated().any()
        or cohort["identity_group_id"].duplicated().any()
    ):
        raise SplitIntegrityError("Curated identities must be unique before splitting")
    expected_compound = cohort["connectivity_inchikey"].astype(str).map(compound_id)
    if not cohort["compound_id"].eq(expected_compound).all():
        raise SplitIntegrityError("Compound IDs are not bound to connectivity InChIKeys")
    if (
        not cohort["identity_group_id"]
        .astype(str)
        .eq(cohort["connectivity_inchikey"].astype(str))
        .all()
    ):
        raise SplitIntegrityError("Identity group differs from connectivity InChIKey")
    cohort["label"] = _exact_integer_series(cohort["label"], name="cohort.label")
    if set(cohort["label"]) != {0, 1}:
        raise SplitIntegrityError("Both binary classes are required")
    return cohort.sort_values("compound_id", kind="stable").reset_index(drop=True)


def _exact_integer_series(values: pd.Series, *, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    array = numeric.to_numpy(dtype=float)
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise SplitIntegrityError(f"{name} must contain exact finite integers")
    return pd.Series(array.astype(np.int64), index=values.index, name=values.name)


def _cohort_binding_sha256(frame: pd.DataFrame) -> str:
    columns = (
        "compound_id",
        "connectivity_inchikey",
        "standardized_parent_smiles",
        "label",
    )
    rows = frame.loc[:, list(columns)].sort_values("compound_id", kind="stable")
    return canonical_sha256(
        {"schema": "geroprotector.split_input_cohort.v1", "rows": rows.to_dict("records")}
    )


def _sgkf(
    y: pd.Series, groups: pd.Series, *, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Seed-sensitive deterministic SGKF even for symmetric group layouts."""

    permutation = np.random.RandomState(int(seed)).permutation(len(y))
    group_values = groups.astype(str).to_numpy()
    ordered_groups = sorted(
        set(group_values),
        key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).digest(),
    )
    code = {value: index for index, value in enumerate(ordered_groups)}
    encoded_groups = np.asarray([code[value] for value in group_values], dtype=int)
    splitter = StratifiedGroupKFold(
        n_splits=int(n_splits), shuffle=True, random_state=int(seed)
    )
    return [
        (permutation[train], permutation[validation])
        for train, validation in splitter.split(
            np.zeros(len(y)), y.to_numpy()[permutation], encoded_groups[permutation]
        )
    ]


def _derived_inner_seed(seed_portfolio: tuple[int, ...], repeat: int, outer_fold: int) -> int:
    payload = json.dumps(
        {
            "seed_portfolio": list(seed_portfolio),
            "repeat": int(repeat),
            "outer_fold": int(outer_fold),
            "schema": "geroprotector.inner_registry.v1",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def registry_canonical_sha256(registry: pd.DataFrame) -> str:
    required = set(REGISTRY_COLUMNS)
    if required - set(registry):
        raise SplitIntegrityError("Registry schema is incomplete")
    records = (
        registry.loc[:, list(REGISTRY_COLUMNS)]
        .sort_values(["repeat", "compound_id"], kind="stable")
        .to_dict(orient="records")
    )
    return canonical_sha256({"schema": "geroprotector.split_registry.v1", "rows": records})


def validate_registry(
    registry: pd.DataFrame,
    *,
    expected_ids: Iterable[object] | None = None,
    outer_repeats: int = 5,
    outer_folds: int = 5,
) -> dict[str, Any]:
    missing = set(REGISTRY_COLUMNS) - set(registry)
    if missing:
        raise SplitIntegrityError(f"Registry columns missing: {sorted(missing)}")
    table = registry.loc[:, list(REGISTRY_COLUMNS)].copy()
    if table.isna().any().any():
        raise SplitIntegrityError("Registry contains missing values")
    if not table["role"].eq("outer_test").all():
        raise SplitIntegrityError("Registry uses one test-assignment row per compound/repeat")
    if table.duplicated(["compound_id", "repeat"]).any():
        raise SplitIntegrityError("A compound has multiple test assignments in one repeat")
    table["label"] = _exact_integer_series(table["label"], name="registry.label")
    table["repeat"] = _exact_integer_series(table["repeat"], name="registry.repeat")
    table["outer_fold"] = _exact_integer_series(table["outer_fold"], name="registry.outer_fold")
    if not set(table["label"]).issubset({0, 1}):
        raise SplitIntegrityError("Registry labels must be binary integers")
    if (table["compound_id"].astype(str).str.strip() == "").any() or (
        table["component_id"].astype(str).str.strip() == ""
    ).any():
        raise SplitIntegrityError("Registry IDs/components must be nonblank")
    if table.groupby("compound_id")["label"].nunique().ne(1).any():
        raise SplitIntegrityError("Compound labels drift across repeats")
    if table.groupby("compound_id")["component_id"].nunique().ne(1).any():
        raise SplitIntegrityError("Compound components drift across repeats")
    if set(table["repeat"]) != set(range(outer_repeats)):
        raise SplitIntegrityError("Registry repeat indices are incomplete")
    if set(table["outer_fold"]) != set(range(outer_folds)):
        raise SplitIntegrityError("Registry contains an invalid outer fold")
    expected = None if expected_ids is None else {str(value) for value in expected_ids}
    audits: list[dict[str, Any]] = []
    reference_ids: set[str] | None = expected
    for repeat in range(outer_repeats):
        repeated = table.loc[table["repeat"].eq(repeat)]
        observed = set(repeated["compound_id"].astype(str))
        if reference_ids is None:
            reference_ids = observed
        if observed != reference_ids:
            missing_count = len(reference_ids - observed)
            extra_count = len(observed - reference_ids)
            raise SplitIntegrityError(
                f"Repeat {repeat} ID coverage differs "
                f"(missing={missing_count}, extra={extra_count})"
            )
        for fold in range(outer_folds):
            test = repeated.loc[repeated["outer_fold"].eq(fold)]
            train = repeated.loc[~repeated["outer_fold"].eq(fold)]
            id_overlap = set(train["compound_id"]) & set(test["compound_id"])
            component_overlap = set(train["component_id"]) & set(test["component_id"])
            if id_overlap or component_overlap:
                raise SplitIntegrityError(
                    f"Repeat {repeat} fold {fold} leaks IDs/components: "
                    f"{len(id_overlap)}/{len(component_overlap)}"
                )
            counts = test["label"].astype(int).value_counts()
            if any(int(counts.get(label, 0)) == 0 for label in (0, 1)):
                raise SplitIntegrityError(f"Repeat {repeat} fold {fold} has a missing class")
            audits.append(
                {
                    "repeat": repeat,
                    "outer_fold": fold,
                    "n_train": len(train),
                    "n_test": len(test),
                    "test_class_0": int(counts.get(0, 0)),
                    "test_class_1": int(counts.get(1, 0)),
                    "identity_overlap_count": 0,
                    "component_overlap_count": 0,
                }
            )
    return {
        "schema_version": "geroprotector.split_audit.v1",
        "passed": True,
        "representation": (
            "one outer-test assignment per compound and repeat; train is complement"
        ),
        "outcome_stratification_used_for_fold_balance_only": True,
        "model_predictions_seen_during_split_generation": False,
        "registry_sha256": registry_canonical_sha256(table),
        "n_rows": len(table),
        "n_components": int(table["component_id"].nunique()),
        "folds": audits,
    }


def build_split_registry(
    cohort: pd.DataFrame,
    *,
    output_path: str | Path,
    inner_output_path: str | Path,
    manifest_path: str | Path,
    curated_table_sha256: str,
    data_manifest_path: str | Path,
    resolved_config_sha256: str,
    fingerprint_config: Mapping[str, Any],
    edge_threshold: float,
    outer_repeats: int,
    outer_folds: int,
    outer_seeds: Iterable[int],
    inner_folds: int = 3,
    inner_seeds: Iterable[int] = (101, 102, 103),
) -> dict[str, Any]:
    curated = _validate_cohort(cohort)
    data_manifest = Path(data_manifest_path)
    if data_manifest.is_symlink() or not data_manifest.is_file():
        raise SplitIntegrityError(f"Data manifest must be regular/non-symlink: {data_manifest}")
    data_manifest_sha256 = sha256_file(data_manifest)
    if not re.fullmatch(r"[0-9a-f]{64}", str(curated_table_sha256)):
        raise SplitIntegrityError("A literal curated-table SHA-256 is required")
    if not re.fullmatch(r"[0-9a-f]{64}", str(resolved_config_sha256)):
        raise SplitIntegrityError("A literal resolved-config SHA-256 is required")
    seeds = tuple(int(value) for value in outer_seeds)
    if len(seeds) != int(outer_repeats) or len(set(seeds)) != len(seeds):
        raise SplitIntegrityError("Exactly one unique outer seed per repeat is required")
    expected_fp = {
        "family": "morgan_bit",
        "radius": 2,
        "n_bits": 2048,
        "use_chirality": False,
    }
    observed_fp = {
        "family": fingerprint_config.get("family"),
        "radius": int(fingerprint_config.get("radius", -1)),
        "n_bits": int(fingerprint_config.get("n_bits", -1)),
        "use_chirality": bool(fingerprint_config.get("use_chirality")),
    }
    if observed_fp != expected_fp or float(edge_threshold) != 0.40:
        raise SplitIntegrityError(
            f"Primary component contract changed: {observed_fp}, threshold={edge_threshold}"
        )
    components = similarity_components(
        curated["standardized_parent_smiles"],
        compound_ids=curated["compound_id"],
        threshold=float(edge_threshold),
        radius=2,
        n_bits=2048,
        use_chirality=False,
    )
    y = curated.set_index("compound_id")["label"].astype(int)
    groups = components.reindex(y.index)
    rows: list[dict[str, Any]] = []
    assignments: set[str] = set()
    for repeat, seed in enumerate(seeds):
        # Canonical ordering plus explicit seed gives a reproducible approximately
        # stratified allocation. Labels affect fold balance only, never component creation.
        fold_by_id: dict[str, int] = {}
        for fold, (_, test_positions) in enumerate(
            _sgkf(y, groups, n_splits=int(outer_folds), seed=int(seed))
        ):
            for position in test_positions:
                compound = str(y.index[int(position)])
                if compound in fold_by_id:
                    raise SplitIntegrityError("SGKF assigned a compound twice")
                fold_by_id[compound] = int(fold)
        if set(fold_by_id) != set(y.index.astype(str)):
            raise SplitIntegrityError("SGKF did not assign every compound")
        canonical_partitions = sorted(
            tuple(
                sorted(
                    compound for compound, assigned in fold_by_id.items() if assigned == fold
                )
            )
            for fold in range(int(outer_folds))
        )
        assignment_hash = canonical_sha256(canonical_partitions)
        if assignment_hash in assignments:
            raise SplitIntegrityError(
                "Two repeats produced identical assignments; change only the locked seeds"
            )
        assignments.add(assignment_hash)
        for compound in y.index.astype(str):
            rows.append(
                {
                    "compound_id": compound,
                    "label": int(y.loc[compound]),
                    "component_id": str(groups.loc[compound]),
                    "repeat": int(repeat),
                    "outer_fold": int(fold_by_id[compound]),
                    "role": "outer_test",
                }
            )
    registry = pd.DataFrame(rows, columns=list(REGISTRY_COLUMNS))
    audit = validate_registry(
        registry,
        expected_ids=curated["compound_id"],
        outer_repeats=int(outer_repeats),
        outer_folds=int(outer_folds),
    )
    vectors = morgan_bitvectors(
        curated["standardized_parent_smiles"],
        radius=2,
        n_bits=2048,
        use_chirality=False,
    )
    vector_by_id = dict(zip(curated["compound_id"].astype(str), vectors, strict=True))
    from rdkit import DataStructs

    for fold_audit in audit["folds"]:
        train_ids, test_ids = fold_ids(
            registry,
            repeat=int(fold_audit["repeat"]),
            outer_fold=int(fold_audit["outer_fold"]),
        )
        train_vectors = [vector_by_id[value] for value in train_ids]
        nearest = [
            max(DataStructs.BulkTanimotoSimilarity(vector_by_id[value], train_vectors))
            for value in test_ids
        ]
        fold_audit["nearest_train_tanimoto_min"] = float(np.min(nearest))
        fold_audit["nearest_train_tanimoto_median"] = float(np.median(nearest))
        fold_audit["nearest_train_tanimoto_max"] = float(np.max(nearest))
    destination = Path(output_path)
    seed_portfolio = tuple(int(value) for value in inner_seeds)
    if len(seed_portfolio) != int(inner_folds) or len(set(seed_portfolio)) != len(
        seed_portfolio
    ):
        raise SplitIntegrityError(
            "inner_seeds must contain one unique prespecified token per inner fold"
        )
    inner_rows: list[dict[str, Any]] = []
    for repeat in range(int(outer_repeats)):
        repeated = registry.loc[registry["repeat"].eq(repeat)].set_index("compound_id")
        for outer_fold in range(int(outer_folds)):
            outer_train = repeated.loc[~repeated["outer_fold"].eq(outer_fold)].copy()
            inner_y = outer_train["label"].astype(int).sort_index()
            inner_groups = outer_train["component_id"].astype(str).reindex(inner_y.index)
            inner_seed = _derived_inner_seed(seed_portfolio, repeat, outer_fold)
            assignment: dict[str, int] = {}
            for inner_fold, (_, valid_positions) in enumerate(
                _sgkf(inner_y, inner_groups, n_splits=int(inner_folds), seed=inner_seed)
            ):
                for position in valid_positions:
                    compound = str(inner_y.index[int(position)])
                    if compound in assignment:
                        raise SplitIntegrityError("Inner registry assigned a compound twice")
                    assignment[compound] = int(inner_fold)
            if set(assignment) != set(inner_y.index.astype(str)):
                raise SplitIntegrityError("Inner registry failed complete outer-train coverage")
            for compound in inner_y.index.astype(str):
                inner_rows.append(
                    {
                        "compound_id": compound,
                        "label": int(inner_y.loc[compound]),
                        "component_id": str(inner_groups.loc[compound]),
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "inner_fold": assignment[compound],
                        "role": "inner_validation",
                        "assignment_seed": inner_seed,
                    }
                )
    inner_registry = pd.DataFrame(inner_rows, columns=list(INNER_REGISTRY_COLUMNS))
    validate_inner_registry(
        registry,
        inner_registry,
        outer_repeats=int(outer_repeats),
        outer_folds=int(outer_folds),
        inner_folds=int(inner_folds),
    )
    csv_path = destination.with_suffix(".csv")
    inner_path = Path(inner_output_path)
    inner_csv_path = inner_path.with_suffix(".csv")
    manifest_destination = Path(manifest_path)
    final_destinations = (
        destination,
        csv_path,
        inner_path,
        inner_csv_path,
        manifest_destination,
    )
    parent_set = {path.parent.resolve() for path in final_destinations}
    if len(parent_set) != 1:
        raise SplitIntegrityError("All split artifacts must share one atomic bundle directory")
    final_bundle = next(iter(parent_set))
    if final_bundle.exists():
        raise FileExistsError(f"Refusing to overwrite split bundle: {final_bundle}")
    final_bundle.parent.mkdir(parents=True, exist_ok=True)
    staging_bundle = Path(
        tempfile.mkdtemp(prefix=f".{final_bundle.name}.work-", dir=final_bundle.parent)
    )
    staged = {path: staging_bundle / path.name for path in final_destinations}
    _atomic_parquet(registry, staged[destination])
    registry.to_csv(staged[csv_path], index=False)
    _atomic_parquet(inner_registry, staged[inner_path])
    inner_registry.to_csv(staged[inner_csv_path], index=False)
    inner_hash = canonical_sha256(
        {
            "schema": "geroprotector.inner_registry.v1",
            "rows": inner_registry.sort_values(
                ["repeat", "outer_fold", "compound_id"], kind="stable"
            ).to_dict(orient="records"),
        }
    )
    manifest = {
        **audit,
        "created_utc": utc_now(),
        "primary_fingerprint": observed_fp,
        "edge_threshold": float(edge_threshold),
        "outer_seeds": list(seeds),
        "inner_folds": int(inner_folds),
        "inner_seed_portfolio": list(seed_portfolio),
        "inner_registry_sha256": inner_hash,
        "split_input_cohort_sha256": _cohort_binding_sha256(curated),
        "curated_table_sha256": str(curated_table_sha256),
        "data_manifest_sha256": data_manifest_sha256,
        "resolved_config_sha256": str(resolved_config_sha256),
        "software": {
            "rdkit": __import__("rdkit").__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "registry_parquet": destination.as_posix(),
        "registry_parquet_sha256": sha256_file(staged[destination]),
        "registry_csv": csv_path.as_posix(),
        "registry_csv_sha256": sha256_file(staged[csv_path]),
        "inner_registry_parquet": inner_path.as_posix(),
        "inner_registry_parquet_sha256": sha256_file(staged[inner_path]),
        "inner_registry_csv": inner_csv_path.as_posix(),
        "inner_registry_csv_sha256": sha256_file(staged[inner_csv_path]),
    }
    manifest["canonical_sha256"] = canonical_sha256(manifest)
    atomic_write_json(staged[manifest_destination], manifest)
    os.rename(staging_bundle, final_bundle)
    return manifest


def validate_inner_registry(
    outer_registry: pd.DataFrame,
    inner_registry: pd.DataFrame,
    *,
    outer_repeats: int,
    outer_folds: int,
    inner_folds: int,
) -> None:
    missing = set(INNER_REGISTRY_COLUMNS) - set(inner_registry)
    if missing:
        raise SplitIntegrityError(f"Inner registry columns missing: {sorted(missing)}")
    table_all = inner_registry.loc[:, list(INNER_REGISTRY_COLUMNS)].copy()
    if table_all.isna().any().any():
        raise SplitIntegrityError("Inner registry contains missing values")
    if not table_all["role"].eq("inner_validation").all():
        raise SplitIntegrityError("Inner registry role must be inner_validation")
    for column in ("label", "repeat", "outer_fold", "inner_fold", "assignment_seed"):
        table_all[column] = _exact_integer_series(
            table_all[column], name=f"inner_registry.{column}"
        )
    if not set(table_all["label"]).issubset({0, 1}):
        raise SplitIntegrityError("Inner registry labels must be binary")
    if table_all.duplicated(["compound_id", "repeat", "outer_fold"]).any():
        raise SplitIntegrityError("Duplicate compound assignment in inner registry")
    if (table_all["compound_id"].astype(str).str.strip() == "").any() or (
        table_all["component_id"].astype(str).str.strip() == ""
    ).any():
        raise SplitIntegrityError("Inner registry IDs/components must be nonblank")
    outer = outer_registry.copy()
    outer["repeat"] = _exact_integer_series(outer["repeat"], name="outer.repeat")
    outer["outer_fold"] = _exact_integer_series(outer["outer_fold"], name="outer.outer_fold")
    outer["label"] = _exact_integer_series(outer["label"], name="outer.label")
    for repeat in range(outer_repeats):
        outer_repeat = outer.loc[outer["repeat"].eq(repeat)]
        for outer_fold in range(outer_folds):
            expected_train = set(
                outer_repeat.loc[~outer_repeat["outer_fold"].eq(outer_fold), "compound_id"]
            )
            table = table_all.loc[
                table_all["repeat"].eq(repeat) & table_all["outer_fold"].eq(outer_fold)
            ]
            if set(table["compound_id"]) != expected_train:
                raise SplitIntegrityError(
                    "Inner registry does not equal outer-train membership"
                )
            if set(table["inner_fold"].astype(int)) != set(range(inner_folds)):
                raise SplitIntegrityError("Inner fold coverage is incomplete")
            if table["assignment_seed"].nunique() != 1:
                raise SplitIntegrityError("Inner assignment seed drifts within an outer job")
            expected_metadata = outer_repeat.set_index("compound_id").loc[
                list(table["compound_id"]), ["label", "component_id"]
            ]
            observed_metadata = table.set_index("compound_id")[["label", "component_id"]]
            expected_metadata = expected_metadata.sort_index()
            observed_metadata = observed_metadata.sort_index()
            if not observed_metadata["label"].eq(expected_metadata["label"]).all() or not (
                observed_metadata["component_id"]
                .astype(str)
                .eq(expected_metadata["component_id"].astype(str))
                .all()
            ):
                raise SplitIntegrityError("Inner label/component metadata differs from outer")
            for inner_fold in range(inner_folds):
                validation = table.loc[table["inner_fold"].eq(inner_fold)]
                training = table.loc[~table["inner_fold"].eq(inner_fold)]
                if set(validation["component_id"]) & set(training["component_id"]):
                    raise SplitIntegrityError("A primary component crosses an inner fold")
                if set(validation["label"].astype(int)) != {0, 1}:
                    raise SplitIntegrityError("An inner validation fold lacks a class")


def fold_ids(
    registry: pd.DataFrame, *, repeat: int, outer_fold: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    repeated = registry.loc[registry["repeat"].astype(int).eq(int(repeat))]
    if repeated.empty:
        raise SplitIntegrityError(f"Repeat absent from registry: {repeat}")
    is_test = repeated["outer_fold"].astype(int).eq(int(outer_fold))
    test = tuple(sorted(repeated.loc[is_test, "compound_id"].astype(str)))
    train = tuple(sorted(repeated.loc[~is_test, "compound_id"].astype(str)))
    if not test or not train or set(test) & set(train):
        raise SplitIntegrityError("Requested fold is empty or overlapping")
    return train, test


def grouped_inner_folds(
    labels: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    ids = pd.Index(sorted(labels.index.astype(str)), name="compound_id")
    raw_y = labels.reindex(ids)
    raw_g = groups.reindex(ids)
    if raw_y.isna().any() or raw_g.isna().any():
        raise SplitIntegrityError("Inner labels/groups are misaligned")
    y = _exact_integer_series(raw_y, name="inner.labels")
    if not set(y).issubset({0, 1}):
        raise SplitIntegrityError("Inner labels must be exact binary integers")
    g = raw_g.astype(str)
    if g.str.strip().eq("").any() or g.str.lower().isin({"nan", "none", "null"}).any():
        raise SplitIntegrityError("Inner groups must be nonblank and nonmissing")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = []
    for train_position, validation_position in splitter.split(
        np.zeros(len(ids)), y.to_numpy(), g.to_numpy()
    ):
        train_ids = tuple(ids.take(train_position).astype(str))
        validation_ids = tuple(ids.take(validation_position).astype(str))
        if set(g.loc[list(train_ids)]) & set(g.loc[list(validation_ids)]):
            raise SplitIntegrityError("Inner component overlap")
        if set(y.loc[list(validation_ids)]) != {0, 1}:
            raise SplitIntegrityError("Inner validation fold lacks a class")
        result.append((train_ids, validation_ids))
    return result


def validate_selection_folds(
    *,
    fit_ids: Sequence[str],
    labels: pd.Series,
    groups: pd.Series,
    folds: Sequence[tuple[Sequence[str], Sequence[str]]],
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Prove a supplied fold list is a complete grouped OOF partition of fit IDs."""

    universe = {str(value) for value in fit_ids}
    if not universe or len(universe) != len(tuple(fit_ids)):
        raise SplitIntegrityError("Selection fit IDs must be nonempty and unique")
    y = labels.copy()
    y.index = y.index.astype(str)
    g = groups.copy()
    g.index = g.index.astype(str)
    if set(y.index) != universe or set(g.index) != universe:
        raise SplitIntegrityError("Selection labels/groups do not equal fit IDs")
    if y.isna().any() or g.isna().any():
        raise SplitIntegrityError("Selection labels/groups contain missing values")
    y = _exact_integer_series(y, name="selection.labels")
    if not set(y).issubset({0, 1}):
        raise SplitIntegrityError("Selection labels must be exact binary integers")
    g = g.astype(str)
    if g.str.strip().eq("").any() or g.str.lower().isin({"nan", "none", "null"}).any():
        raise SplitIntegrityError("Selection groups must be nonblank and nonmissing")
    validated: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    validation_seen: list[str] = []
    for fold_index, (train_values, validation_values) in enumerate(folds):
        train = tuple(map(str, train_values))
        validation = tuple(map(str, validation_values))
        train_set, validation_set = set(train), set(validation)
        if len(train) != len(train_set) or len(validation) != len(validation_set):
            raise SplitIntegrityError(f"Selection fold {fold_index} has duplicate IDs")
        if train_set & validation_set or train_set | validation_set != universe:
            raise SplitIntegrityError(f"Selection fold {fold_index} is overlapping/incomplete")
        if set(g.loc[list(train_set)]) & set(g.loc[list(validation_set)]):
            raise SplitIntegrityError(f"Selection fold {fold_index} crosses a component")
        if set(y.loc[list(train_set)].astype(int)) != {0, 1} or set(
            y.loc[list(validation_set)].astype(int)
        ) != {0, 1}:
            raise SplitIntegrityError(f"Selection fold {fold_index} lacks a class")
        validation_seen.extend(validation)
        validated.append((tuple(sorted(train)), tuple(sorted(validation))))
    if len(validation_seen) != len(universe) or set(validation_seen) != universe:
        raise SplitIntegrityError("Selection folds do not predict each fit ID exactly once")
    return validated


def locked_inner_fold_ids(
    inner_registry: pd.DataFrame,
    *,
    repeat: int,
    outer_fold: int,
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    table = inner_registry.loc[
        inner_registry["repeat"].astype(int).eq(int(repeat))
        & inner_registry["outer_fold"].astype(int).eq(int(outer_fold))
    ]
    if table.empty:
        raise SplitIntegrityError("Requested outer job is absent from inner registry")
    result = []
    for inner_fold in sorted(table["inner_fold"].astype(int).unique()):
        is_valid = table["inner_fold"].astype(int).eq(inner_fold)
        train = tuple(sorted(table.loc[~is_valid, "compound_id"].astype(str)))
        valid = tuple(sorted(table.loc[is_valid, "compound_id"].astype(str)))
        if not train or not valid:
            raise SplitIntegrityError("Locked inner fold is empty")
        result.append((train, valid))
    return result


def load_verified_registries(
    *,
    outer_path: str | Path,
    inner_path: str | Path,
    manifest_path: str | Path,
    curated_table_path: str | Path,
    data_manifest_path: str | Path,
    resolved_config_sha256: str,
    outer_repeats: int,
    outer_folds: int,
    inner_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load split artifacts only after rebinding them to data/config and full invariants."""

    paths = {
        "registry_parquet": Path(outer_path),
        "inner_registry_parquet": Path(inner_path),
        "manifest": Path(manifest_path),
        "curated": Path(curated_table_path),
        "data_manifest": Path(data_manifest_path),
    }
    for role, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise SplitIntegrityError(f"{role} must be a regular non-symlink file: {path}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    payload = dict(manifest)
    claimed = payload.pop("canonical_sha256", None)
    if claimed != canonical_sha256(payload):
        raise SplitIntegrityError("Split manifest canonical hash failed")
    if manifest.get("registry_parquet_sha256") != sha256_file(paths["registry_parquet"]):
        raise SplitIntegrityError("Outer registry byte hash changed")
    if manifest.get("inner_registry_parquet_sha256") != sha256_file(
        paths["inner_registry_parquet"]
    ):
        raise SplitIntegrityError("Inner registry byte hash changed")
    outer_csv = paths["registry_parquet"].with_suffix(".csv")
    inner_csv = paths["inner_registry_parquet"].with_suffix(".csv")
    for role, csv_path, field in (
        ("outer registry CSV", outer_csv, "registry_csv_sha256"),
        ("inner registry CSV", inner_csv, "inner_registry_csv_sha256"),
    ):
        if csv_path.is_symlink() or not csv_path.is_file():
            raise SplitIntegrityError(f"{role} must be regular/non-symlink: {csv_path}")
        if manifest.get(field) != sha256_file(csv_path):
            raise SplitIntegrityError(f"{role} byte hash changed")
    if manifest.get("curated_table_sha256") != sha256_file(paths["curated"]):
        raise SplitIntegrityError("Curated table differs from split input")
    if manifest.get("data_manifest_sha256") != sha256_file(paths["data_manifest"]):
        raise SplitIntegrityError("Data manifest differs from split input")
    if manifest.get("resolved_config_sha256") != str(resolved_config_sha256):
        raise SplitIntegrityError("Resolved validation configuration changed")
    outer = pd.read_parquet(paths["registry_parquet"])
    inner = pd.read_parquet(paths["inner_registry_parquet"])
    curated = _validate_cohort(pd.read_parquet(paths["curated"]))
    audit = validate_registry(
        outer,
        expected_ids=curated["compound_id"],
        outer_repeats=int(outer_repeats),
        outer_folds=int(outer_folds),
    )
    if audit["registry_sha256"] != manifest.get("registry_sha256"):
        raise SplitIntegrityError("Outer registry canonical hash changed")
    validate_inner_registry(
        outer,
        inner,
        outer_repeats=int(outer_repeats),
        outer_folds=int(outer_folds),
        inner_folds=int(inner_folds),
    )
    inner_hash = canonical_sha256(
        {
            "schema": "geroprotector.inner_registry.v1",
            "rows": inner.sort_values(
                ["repeat", "outer_fold", "compound_id"], kind="stable"
            ).to_dict(orient="records"),
        }
    )
    if inner_hash != manifest.get("inner_registry_sha256"):
        raise SplitIntegrityError("Inner registry canonical hash changed")
    if _cohort_binding_sha256(curated) != manifest.get("split_input_cohort_sha256"):
        raise SplitIntegrityError("Split input cohort binding changed")
    return outer, inner, manifest
