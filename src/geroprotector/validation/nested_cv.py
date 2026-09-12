"""Shared resumable nested-CV runner for references, V5 and V6.

The runner never computes an outer-test metric. It writes one held-out prediction per
compound/repeat/model; reporting is a separate command after the run is sealed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pandas as pd
import yaml

from ..audit import (
    runtime_environment,
    source_tree_files,
    source_tree_sha256,
    validate_core_runtime,
)
from ..config import (
    resolve_config,
    resolved_config_sha256,
    unresolved_placeholders,
    validate_locked_shared_contract,
    validate_protocol_lock,
)
from ..data.curate import load_curated_cohort
from ..hashing import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
    sha256_file,
)
from ..logging import append_event, utc_now
from ..models.references import REFERENCE_MODEL_IDS, ReferenceFeatureStore, ReferencePipeline
from ..models.v5.fingerprint_bank import V5FeatureBank
from ..models.v5.pipeline import V5Pipeline
from ..models.v6.baselines import V6ClassicalPipeline
from ..models.v6.checkpoints import load_checkpoint_ledger
from ..models.v6.feature_panels import V6FeatureStore
from ..models.v6.inference_audit import release_accelerator_memory
from ..models.v6.pipeline import V6Pipeline
from .applicability import outer_train_applicability
from .calibration import BetaCalibrator, PlattCalibrator
from .leakage_checks import assert_internal_config_safe, assert_no_hagr_paths
from .paper_split_registry import load_verified_paper_registries
from .split_registry import (
    fold_ids,
    grouped_inner_folds,
    load_verified_registries,
    locked_inner_fold_ids,
    validate_selection_folds,
)
from .thresholds import select_mcc_threshold


class NestedCVError(RuntimeError):
    """Raised when a run cannot preserve the locked validation contract."""


_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,79}$")


def _pipeline_id(config: Mapping[str, Any]) -> str:
    return {
        "reference": "reference",
        "V5_ELIXIRFP_REBUILT": "V5_ELIXIRFP_REBUILT",
        "V6_TABULAR_FOUNDATION_MODELS": "V6_TABULAR_FOUNDATION_MODELS",
        "V5BIS_PAPER80": "V5BIS_PAPER80",
        "V6BIS_PAPER80": "V6BIS_PAPER80",
    }[str(config["pipeline"])]


def _is_v5(config_or_id: Mapping[str, Any] | str) -> bool:
    value = str(
        config_or_id.get("pipeline") if isinstance(config_or_id, Mapping) else config_or_id
    )
    return value in {"V5_ELIXIRFP_REBUILT", "V5BIS_PAPER80"}


def _is_v6(config_or_id: Mapping[str, Any] | str) -> bool:
    value = str(
        config_or_id.get("pipeline") if isinstance(config_or_id, Mapping) else config_or_id
    )
    return value in {"V6_TABULAR_FOUNDATION_MODELS", "V6BIS_PAPER80"}


def _is_paper80(config_or_id: Mapping[str, Any] | str) -> bool:
    value = str(
        config_or_id.get("pipeline") if isinstance(config_or_id, Mapping) else config_or_id
    )
    return value in {"V5BIS_PAPER80", "V6BIS_PAPER80"}


def _split_strategy(config_or_id: Mapping[str, Any] | str) -> str:
    return "paper_random_80_20" if _is_paper80(config_or_id) else "similarity_components"


def _active_registry_outputs(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return (
        config["paper_registry_outputs"] if _is_paper80(config) else config["registry_outputs"]
    )


def _outer_axes(config: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    if _is_paper80(config):
        return ((0, 0),)
    return tuple(
        (repeat, fold)
        for repeat in range(int(config["primary_split"]["outer_repeats"]))
        for fold in range(int(config["primary_split"]["outer_folds"]))
    )


def _reference_config(root: Path) -> tuple[dict[str, Any], str]:
    config = resolve_config(root / "configs" / "reference.yaml")
    validate_locked_shared_contract(config)
    assert_internal_config_safe(config)
    validate_protocol_lock(root=root, contract_name="reference", config=config)
    return config, resolved_config_sha256(config)


def _project_path(root: Path, value: object, *, role: str) -> Path:
    text = str(value)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise NestedCVError(f"{role} must be project-relative without '..': {text}")
    cursor = root
    for part in path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise NestedCVError(f"{role} path contains a symlink: {cursor}")
    resolved = (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise NestedCVError(f"{role} escapes project root: {text}")
    assert_no_hagr_paths([resolved])
    return resolved


def _seed(*values: object) -> int:
    payload = "\0".join(map(str, values)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _murcko_scaffold_id(value: object) -> str:
    """Return an explicit acyclic sentinel without treating pandas NaN as text."""

    if value is None or pd.isna(value):
        return "ACYCLIC"
    text = str(value).strip()
    return f"MURCKO::{text}" if text else "ACYCLIC"


def _validate_run_manifest_schema(schema_path: Path, manifest: Mapping[str, Any]) -> None:
    if schema_path.is_symlink() or not schema_path.is_file():
        raise NestedCVError(f"Run-manifest schema is unavailable: {schema_path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
    errors = sorted(validator.iter_errors(dict(manifest)), key=lambda item: list(item.path))
    if errors:
        summary = "; ".join(error.message for error in errors[:5])
        raise NestedCVError(f"Run manifest violates its schema: {summary}")


def _id_sha256(values: Sequence[str]) -> str:
    return sha256_bytes(("\n".join(sorted(map(str, values))) + "\n").encode())


def _normalise_trace(frame: pd.DataFrame, **metadata: Any) -> pd.DataFrame:
    output = frame.copy()
    for column in output:
        if output[column].map(lambda value: isinstance(value, (dict, list, tuple))).any():
            output[column] = output[column].map(
                lambda value: (
                    canonical_json_bytes(value).decode()
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
            )
    for key, value in metadata.items():
        output[key] = value
    return output


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable parquet: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_parquet(temporary, index=False)
        os.link(temporary, path)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def _data_and_splits(
    root: Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    data_config = resolve_config(root / "configs" / "data.yaml")
    validation_config = resolve_config(root / "configs" / "validation.yaml")
    validate_protocol_lock(root=root, contract_name="data", config=data_config)
    validate_protocol_lock(root=root, contract_name="validation", config=validation_config)
    # The model config must inherit, not override, every shared scientific field.
    for key in ("project", "upstream", "roles", "curation", "outputs"):
        if config.get(key) != data_config.get(key):
            raise NestedCVError(f"Pipeline overrides locked data config root: {key}")
    for key in (
        "primary_split",
        "inner_cv",
        "secondary_splits",
        "selection",
        "calibration",
        "threshold",
        "metrics",
        "bootstrap",
        "applicability",
        "external_firewall",
        "registry_outputs",
    ):
        if config.get(key) != validation_config.get(key):
            raise NestedCVError(f"Pipeline overrides locked validation root: {key}")
    outputs = data_config["outputs"]
    curated_path = _project_path(root, outputs["curated_table"], role="curated cohort")
    provenance_path = _project_path(
        root, outputs["provenance_table"], role="curation provenance"
    )
    data_manifest_path = _project_path(root, outputs["manifest"], role="curation manifest")
    curated, _, data_manifest = load_curated_cohort(
        curated_table=curated_path,
        provenance_table=provenance_path,
        manifest_path=data_manifest_path,
        resolved_config_sha256=resolved_config_sha256(data_config),
    )
    if _is_paper80(config):
        paper_config = resolve_config(root / "configs" / "paper80.yaml")
        validate_protocol_lock(root=root, contract_name="paper80", config=paper_config)
        for key in ("evaluation_design", "paper_registry_outputs"):
            if config.get(key) != paper_config.get(key):
                raise NestedCVError(f"Pipeline overrides locked paper80 root: {key}")
        registry_paths = config["paper_registry_outputs"]
        design = config["evaluation_design"]
        outer, inner, split_manifest = load_verified_paper_registries(
            cohort=curated,
            outer_path=_project_path(
                root, registry_paths["outer"], role="paper outer registry"
            ),
            inner_path=_project_path(
                root, registry_paths["inner"], role="paper inner registry"
            ),
            manifest_path=_project_path(
                root, registry_paths["manifest"], role="paper split manifest"
            ),
            curated_table_path=curated_path,
            data_manifest_path=data_manifest_path,
            resolved_config_sha256=resolved_config_sha256(paper_config),
            test_identity_path=_project_path(
                root, design["test_identity_file"], role="paper test identity lock"
            ),
            test_identity_file_sha256=design["test_identity_file_sha256"],
            inner_folds=int(design["inner_selection"]["folds"]),
        )
    else:
        registry_paths = validation_config["registry_outputs"]
        outer, inner, split_manifest = load_verified_registries(
            outer_path=_project_path(root, registry_paths["outer"], role="outer registry"),
            inner_path=_project_path(root, registry_paths["inner"], role="inner registry"),
            manifest_path=_project_path(
                root, registry_paths["manifest"], role="split manifest"
            ),
            curated_table_path=curated_path,
            data_manifest_path=data_manifest_path,
            resolved_config_sha256=resolved_config_sha256(validation_config),
            outer_repeats=int(validation_config["primary_split"]["outer_repeats"]),
            outer_folds=int(validation_config["primary_split"]["outer_folds"]),
            inner_folds=int(validation_config["inner_cv"]["folds"]),
        )
    return curated, outer, inner, data_manifest, split_manifest


def _model_specs(config: Mapping[str, Any], *, suite: str) -> tuple[dict[str, Any], ...]:
    if suite not in {"core", "full"}:
        raise NestedCVError("Suite must be 'core' or 'full'")
    pipeline = str(config["pipeline"])
    if pipeline == "reference":
        return tuple(
            {"kind": "reference", "model_id": model_id}
            for model_id in sorted(REFERENCE_MODEL_IDS)
        )
    if pipeline in {"V5_ELIXIRFP_REBUILT", "V5BIS_PAPER80"}:
        variants = (
            ("final_v5",)
            if suite == "core"
            else (
                "raw_best_individual_fingerprint",
                "raw_unweighted_concatenation",
                "selected_lengths_unweighted",
                "weighted_no_reduction",
                "weighted_svd",
                "weighted_nystroem",
                "final_v5",
            )
        )
        references = (
            ("R2_extra_trees_v3_compatible",)
            if suite == "core"
            else (
                "R5_paper_seven_descriptor_linear_svm",
                "R2_extra_trees_v3_compatible",
            )
        )
        return tuple(
            [{"kind": "reference", "model_id": value} for value in references]
            + [{"kind": "v5", "variant": value} for value in variants]
        )
    enabled = tuple(
        model_id
        for model_id, settings in config["models"].items()
        if isinstance(settings, Mapping) and settings.get("enabled") is True
    )
    specs: list[dict[str, Any]] = [
        {"kind": "reference", "model_id": "R2_extra_trees_v3_compatible"}
    ]
    specs.extend(
        {"kind": "v6_baseline", "baseline": value}
        for value in ("elastic_net", "extra_trees", "xgboost")
    )
    specs.append({"kind": "v6", "variant": "selected"})
    if suite == "full":
        anchor = config["inference"]["controlled_ablation_anchor"]
        anchor_model = str(anchor["model_id"])
        anchor_panel = str(anchor["panel_family"])
        anchor_ensemble = int(anchor["ensemble_size"])
        if anchor_model not in enabled:
            raise NestedCVError("V6 controlled-ablation anchor model is not enabled")
        specs.extend(
            {
                "kind": "v6_baseline",
                "baseline": value,
                "fixed_panel_family": anchor_panel,
            }
            for value in ("elastic_net", "extra_trees", "xgboost")
        )
        specs.extend(
            {
                "kind": "v6",
                "variant": f"checkpoint_{model_id}",
                "fixed_model_id": model_id,
                "fixed_panel_family": anchor_panel,
                "fixed_ensemble_size": anchor_ensemble,
            }
            for model_id in enabled
        )
        specs.extend(
            {
                "kind": "v6",
                "variant": f"panel_{panel}",
                "fixed_model_id": anchor_model,
                "fixed_panel_family": panel,
                "fixed_ensemble_size": anchor_ensemble,
            }
            for panel in (
                "rdkit2d_217",
                "chemistry_32",
                "morgan_svd_plus_descriptors",
            )
            # The anchor panel is already represented exactly by
            # checkpoint_<anchor_model>; emitting it again under panel_<anchor_panel>
            # would duplicate predictions and unnecessarily enlarge Holm's family.
            if panel != anchor_panel
        )
        specs.extend(
            {
                "kind": "v6",
                "variant": f"ensemble_{size}",
                "fixed_model_id": anchor_model,
                "fixed_panel_family": anchor_panel,
                "fixed_ensemble_size": size,
            }
            for size in config["inference"]["ablation_ensemble_sizes"]
            if int(size) != 4
        )
    return tuple(specs)


def _model_name(spec: Mapping[str, Any]) -> str:
    if spec["kind"] == "reference":
        return str(spec["model_id"])
    if spec["kind"] == "v5":
        return f"v5_{spec['variant']}"
    if spec["kind"] == "v6_baseline":
        suffix = (
            ""
            if spec.get("fixed_panel_family") is None
            else f"_fixed_{spec['fixed_panel_family']}"
        )
        return f"v6_baseline_{spec['baseline']}{suffix}"
    return f"v6_foundation_{spec['variant']}"


def _scientific_seed_family(spec: Mapping[str, Any]) -> str:
    """Share random draws across prespecified one-factor comparison branches."""

    if spec["kind"] == "reference":
        return f"reference::{spec['model_id']}"
    if spec["kind"] == "v5":
        return "v5_controlled_ablation"
    if spec["kind"] == "v6_baseline":
        return f"v6_baseline::{spec['baseline']}"
    return "v6_foundation_controlled_ablation"


def _build_store(config: Mapping[str, Any], curated: pd.DataFrame):
    pipeline = str(config["pipeline"])
    if pipeline == "reference":
        return ReferenceFeatureStore.build(curated)
    if pipeline in {"V5_ELIXIRFP_REBUILT", "V5BIS_PAPER80"}:
        return {
            "reference": ReferenceFeatureStore.build(curated),
            "v5": V5FeatureBank.build(
                compound_ids=curated["compound_id"],
                smiles=curated["standardized_parent_smiles"],
                config=config,
            ),
        }
    return {
        "reference": ReferenceFeatureStore.build(curated),
        "v6": V6FeatureStore.build(
            compound_ids=curated["compound_id"],
            smiles=curated["standardized_parent_smiles"],
        ),
    }


def _store_for_spec(store: Any, spec: Mapping[str, Any]):
    if isinstance(store, dict):
        if spec["kind"] == "reference":
            return store["reference"]
        if spec["kind"] == "v5":
            return store["v5"]
        return store["v6"]
    return store


def _new_model(
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    seed: int,
    root: Path,
    checkpoint_ledger: Mapping[str, Any] | None,
    v5_preparation_cache_root: Path | None = None,
    v5_preparation_cache_context_sha256: str | None = None,
):
    kind = spec["kind"]
    if kind == "reference":
        reference_config, _ = _reference_config(root)
        return ReferencePipeline(str(spec["model_id"]), reference_config, seed=seed)
    if kind == "v5":
        return V5Pipeline(
            config,
            seed=seed,
            variant=str(spec["variant"]),
            preparation_cache_root=v5_preparation_cache_root,
            preparation_cache_context_sha256=v5_preparation_cache_context_sha256,
        )
    if kind == "v6_baseline":
        return V6ClassicalPipeline(
            str(spec["baseline"]),
            config,
            seed=seed,
            fixed_panel_family=spec.get("fixed_panel_family"),
        )
    if checkpoint_ledger is None:
        raise NestedCVError("V6 foundation model requires a verified checkpoint ledger")
    return V6Pipeline(
        config,
        checkpoint_ledger,
        project_root=root,
        seed=seed,
        device=str(config["execution"]["device"]),
        fixed_model_id=spec.get("fixed_model_id"),
        fixed_panel_family=spec.get("fixed_panel_family"),
        fixed_ensemble_size=(
            None
            if spec.get("fixed_ensemble_size") is None
            else int(spec["fixed_ensemble_size"])
        ),
        variant=str(spec["variant"]),
    )


def _fit(
    model: Any,
    store: Any,
    *,
    ids: Sequence[str],
    labels: pd.Series,
    groups: pd.Series,
    selection_folds: Sequence[tuple[tuple[str, ...], tuple[str, ...]]],
) -> Any:
    active = store.subset(ids)
    model.fit(
        active,
        labels.loc[list(ids)],
        groups=groups.loc[list(ids)],
        selection_folds=selection_folds,
    )
    return model


def _predict(
    model: Any,
    store: Any,
    ids: Sequence[str],
    *,
    full_foundation_audit: bool = True,
) -> tuple[np.ndarray, dict | None]:
    if isinstance(model, V6Pipeline):
        if not full_foundation_audit:
            probability = model.predict_calibration_oof(store, ids)
            return np.asarray(probability[:, 1], dtype=float), None
        probability, audit = model.audit_and_predict(store, ids)
        return np.asarray(probability[:, 1], dtype=float), audit
    return np.asarray(model.predict_proba(store, ids)[:, 1], dtype=float), None


def _selection_trace(model: Any) -> pd.DataFrame:
    if hasattr(model, "get_selection_trace"):
        return model.get_selection_trace()
    if hasattr(model, "selection_trace_"):
        return model.selection_trace_.copy()
    raise NestedCVError("Fitted model does not expose its selection trace")


def _job_paths(directory: Path) -> dict[str, Path]:
    return {
        "outer": directory / "outer_predictions.parquet",
        "inner": directory / "calibration_inner_oof.parquet",
        "trace": directory / "selection_trace.parquet",
        "model": directory / "model.joblib",
        "model_manifest": directory / "model_manifest.json",
        "calibration": directory / "calibration.json",
        "applicability": directory / "applicability.parquet",
        "completed": directory / "JOB_COMPLETED.json",
    }


def _job_artifacts(directory: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(directory.rglob("*")):
        if path.name == "JOB_COMPLETED.json" or path.is_dir():
            continue
        if path.is_symlink() or not path.is_file():
            raise NestedCVError(f"Job artifact is unsafe: {path}")
        records.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _verify_completed_job(directory: Path, expected_binding: str) -> dict[str, Any]:
    completed = directory / "JOB_COMPLETED.json"
    if completed.is_symlink() or not completed.is_file():
        raise NestedCVError(f"Incomplete existing job bundle: {directory}")
    record = json.loads(completed.read_text(encoding="utf-8"))
    payload = dict(record)
    claimed = payload.pop("canonical_sha256", None)
    if (
        claimed != canonical_sha256(payload)
        or record.get("job_binding_sha256") != expected_binding
    ):
        raise NestedCVError(f"Completed job binding/integrity failed: {directory}")
    if record.get("artifacts") != _job_artifacts(directory):
        raise NestedCVError(f"Completed job artifacts changed: {directory}")
    return record


def _run_one_job(
    *,
    root: Path,
    run_id: str,
    pipeline_id: str,
    config: Mapping[str, Any],
    config_hash: str,
    source_hash: str,
    runtime_environment_sha256: str,
    curated: pd.DataFrame,
    outer_registry: pd.DataFrame,
    inner_registry: pd.DataFrame,
    store: Any,
    spec: Mapping[str, Any],
    repeat: int,
    outer_fold: int,
    work_root: Path,
    checkpoint_ledger: Mapping[str, Any] | None,
) -> Path:
    model_name = _model_name(spec)
    seed_family = _scientific_seed_family(spec)
    job_id = f"repeat_{repeat:02d}/fold_{outer_fold:02d}/{model_name}"
    destination = work_root / "jobs" / job_id
    binding = canonical_sha256(
        {
            "schema": "geroprotector.outer_job.v1",
            "run_id": run_id,
            "pipeline_id": pipeline_id,
            "model_spec": dict(spec),
            "repeat": repeat,
            "outer_fold": outer_fold,
            "resolved_config_sha256": config_hash,
            "source_tree_sha256": source_hash,
            "runtime_environment_sha256": runtime_environment_sha256,
        }
    )
    if destination.exists():
        _verify_completed_job(destination, binding)
        return destination
    output_root = work_root.parent
    forensic = output_root / ".forensics"
    forensic.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.{model_name}.", dir=forensic))
    paths = _job_paths(staging)
    indexed = curated.set_index(curated["compound_id"].astype(str), drop=False)
    labels = indexed["label"].astype(int)
    repeated = outer_registry.loc[outer_registry["repeat"].astype(int).eq(repeat)]
    groups = repeated.set_index("compound_id")["component_id"].astype(str)
    outer_train, outer_test = fold_ids(outer_registry, repeat=repeat, outer_fold=outer_fold)
    v5_cache_root = None
    v5_cache_context = None
    if _is_v5(pipeline_id):
        v5_cache_root = (
            work_root
            / "shared"
            / "v5_preparation"
            / f"repeat_{repeat:02d}"
            / f"fold_{outer_fold:02d}"
        )
        v5_cache_context = canonical_sha256(
            {
                "schema": "geroprotector.v5_preparation_run_context.v1",
                "run_id": run_id,
                "pipeline_id": pipeline_id,
                "repeat": int(repeat),
                "outer_fold": int(outer_fold),
                "resolved_config_sha256": config_hash,
                "source_tree_sha256": source_hash,
                "runtime_environment_sha256": runtime_environment_sha256,
                "outer_train_ids_sha256": _id_sha256(outer_train),
                "outer_test_ids_sha256": _id_sha256(outer_test),
                "outer_test_outcomes_in_cache_key": False,
            }
        )
    v5_model_kwargs = (
        {}
        if v5_cache_root is None or spec["kind"] != "v5"
        else {
            "v5_preparation_cache_root": v5_cache_root,
            "v5_preparation_cache_context_sha256": v5_cache_context,
        }
    )
    locked_folds = locked_inner_fold_ids(inner_registry, repeat=repeat, outer_fold=outer_fold)
    validate_selection_folds(
        fit_ids=outer_train,
        labels=labels.loc[list(outer_train)],
        groups=groups.loc[list(outer_train)],
        folds=locked_folds,
    )
    inner_probability = pd.Series(index=list(outer_train), dtype=float)
    inner_fold_by_id: dict[str, str] = {}
    training_ids_by_fold: dict[str, tuple[str, ...]] = {}
    trace_parts: list[pd.DataFrame] = []
    job_started = time.perf_counter()
    crossfit_started = time.perf_counter()
    for calibration_fold, (inner_train, inner_validation) in enumerate(locked_folds):
        fold_token = str(calibration_fold)
        tertiary_seed = _seed(
            config["project"]["random_seed"],
            repeat,
            outer_fold,
            "tertiary",
            calibration_fold,
        )
        tertiary_folds = grouped_inner_folds(
            labels.loc[list(inner_train)],
            groups.loc[list(inner_train)],
            n_splits=int(config["inner_cv"]["folds"]),
            seed=tertiary_seed,
        )
        validate_selection_folds(
            fit_ids=inner_train,
            labels=labels.loc[list(inner_train)],
            groups=groups.loc[list(inner_train)],
            folds=tertiary_folds,
        )
        local_seed = _seed(
            config["project"]["random_seed"],
            seed_family,
            repeat,
            outer_fold,
            "calibration_crossfit",
            calibration_fold,
        )
        fitted = _fit(
            _new_model(
                spec,
                config,
                seed=local_seed,
                root=root,
                checkpoint_ledger=checkpoint_ledger,
                **v5_model_kwargs,
            ),
            store,
            ids=inner_train,
            labels=labels,
            groups=groups,
            selection_folds=tertiary_folds,
        )
        predicted, _ = _predict(
            fitted,
            store,
            inner_validation,
            full_foundation_audit=False,
        )
        inner_probability.loc[list(inner_validation)] = predicted
        training_ids_by_fold[fold_token] = tuple(inner_train)
        inner_fold_by_id.update({value: fold_token for value in inner_validation})
        fitted_trace = _selection_trace(fitted)
        trace_parts.append(
            _normalise_trace(
                fitted_trace,
                phase="full_pipeline_calibration_crossfit",
                model_id=model_name,
                repeat=repeat,
                outer_fold=outer_fold,
                calibration_fold=calibration_fold,
                active_fit_ids_sha256=_id_sha256(inner_train),
                scored_ids_sha256=_id_sha256(inner_validation),
                outer_test_metric_consulted=False,
                hagr_metric_consulted=False,
            )
        )
        if isinstance(fitted, V6Pipeline):
            del fitted
            release_accelerator_memory()
    if inner_probability.isna().any() or set(inner_fold_by_id) != set(outer_train):
        raise NestedCVError("Full-pipeline calibration crossfit is incomplete")
    crossfit_seconds = time.perf_counter() - crossfit_started
    ordered_train = tuple(sorted(outer_train))
    raw_inner = inner_probability.loc[list(ordered_train)].to_numpy(dtype=float)
    y_inner = labels.loc[list(ordered_train)].to_numpy(dtype=int)
    fold_tokens = tuple(inner_fold_by_id[value] for value in ordered_train)
    platt = PlattCalibrator().fit(
        raw_inner,
        y_inner,
        fit_ids=ordered_train,
        fold_ids=fold_tokens,
        training_ids_by_fold=training_ids_by_fold,
    )
    beta = BetaCalibrator().fit(
        raw_inner,
        y_inner,
        fit_ids=ordered_train,
        fold_ids=fold_tokens,
        training_ids_by_fold=training_ids_by_fold,
    )
    calibrated_inner = platt.predict(raw_inner)
    threshold, threshold_mcc = select_mcc_threshold(y_inner, calibrated_inner)
    final_seed = _seed(
        config["project"]["random_seed"],
        seed_family,
        repeat,
        outer_fold,
        "outer_train_final",
    )
    final_fit_started = time.perf_counter()
    final_model = _fit(
        _new_model(
            spec,
            config,
            seed=final_seed,
            root=root,
            checkpoint_ledger=checkpoint_ledger,
            **v5_model_kwargs,
        ),
        store,
        ids=outer_train,
        labels=labels,
        groups=groups,
        selection_folds=locked_folds,
    )
    final_fit_seconds = time.perf_counter() - final_fit_started
    outer_prediction_started = time.perf_counter()
    raw_outer, inference_audit = _predict(final_model, store, outer_test)
    outer_prediction_seconds = time.perf_counter() - outer_prediction_started
    calibrated_outer = platt.predict(raw_outer)
    beta_outer = beta.predict(raw_outer)
    applicability = outer_train_applicability(
        curated,
        train_ids=outer_train,
        query_ids=outer_test,
        min_tanimoto=float(config["applicability"]["min_nearest_train_tanimoto"]),
        descriptor_quantile=float(config["applicability"]["robust_descriptor_train_quantile"]),
    )
    applicability = applicability.set_index("compound_id")
    created = utc_now()
    outer_rows: list[dict[str, Any]] = []
    model_manifest = (
        final_model.get_manifest()
        if hasattr(final_model, "get_manifest")
        else (final_model.get_feature_manifest())
    )
    representation_payload = {
        key: value
        for key, value in model_manifest.items()
        if key
        not in {
            "inference_audit",
            "selection_trace_sha256",
            "outer_test_metric_consulted",
            "hagr_metric_consulted",
        }
    }
    representation = canonical_sha256(
        {
            "schema": "geroprotector.fitted_representation.v1",
            "manifest": representation_payload,
        }
    )
    checkpoint_sha256 = model_manifest.get("model", {}).get("checkpoint_sha256")
    for index, compound in enumerate(outer_test):
        row = indexed.loc[compound]
        probability = float(calibrated_outer[index])
        outer_rows.append(
            {
                "run_id": run_id,
                "pipeline_id": pipeline_id,
                "model_id": model_name,
                "representation_id": representation,
                "checkpoint_sha256": checkpoint_sha256,
                "split_strategy": _split_strategy(pipeline_id),
                "evaluation_role": "outer_test",
                "repeat": repeat,
                "outer_fold": outer_fold,
                "inner_fold": None,
                "seed": final_seed,
                "compound_id": compound,
                "parent_inchikey": str(row["full_inchikey"]),
                "canonical_smiles_sha256": sha256_bytes(
                    str(row["standardized_parent_smiles"]).encode()
                ),
                "identity_group_id": str(row["identity_group_id"]),
                "primary_component_id": str(groups.loc[compound]),
                "murcko_scaffold_id": _murcko_scaffold_id(row.get("murcko_scaffold_smiles")),
                "source_role": str(row["source_role"]),
                "y_true": int(row["label"]),
                "probability_raw": float(raw_outer[index]),
                "probability_calibrated": probability,
                "probability_beta_sensitivity": float(beta_outer[index]),
                "decision_threshold": float(threshold),
                "predicted_class": int(probability >= threshold),
                "uncertainty_score": float(1.0 - abs(2.0 * probability - 1.0)),
                "ensemble_member_count": model_manifest.get("winner", {}).get("ensemble_size"),
                "nearest_train_tanimoto": float(
                    applicability.loc[compound, "nearest_train_tanimoto"]
                ),
                "robust_descriptor_distance": float(
                    applicability.loc[compound, "robust_descriptor_distance"]
                ),
                "inside_applicability_domain": bool(
                    applicability.loc[compound, "inside_applicability_domain"]
                ),
                "abstained": False,
                "adaptation_used": False,
                "adaptation_data_role": None,
                "created_utc": created,
            }
        )
    inner_frame = pd.DataFrame(
        {
            "run_id": run_id,
            "pipeline_id": pipeline_id,
            "model_id": model_name,
            "repeat": repeat,
            "outer_fold": outer_fold,
            "inner_fold": [int(inner_fold_by_id[value]) for value in ordered_train],
            "compound_id": ordered_train,
            "label": y_inner,
            "probability_raw": raw_inner,
            "probability_calibrated": calibrated_inner,
            "probability_beta_sensitivity": beta.predict(raw_inner),
            "fit_scope": "full_pipeline_crossfit",
            "outer_test_ids_sha256": _id_sha256(outer_test),
            "outer_test_metric_consulted": False,
            "hagr_metric_consulted": False,
        }
    )
    trace_parts.append(
        _normalise_trace(
            _selection_trace(final_model),
            phase="outer_train_final_selection",
            model_id=model_name,
            repeat=repeat,
            outer_fold=outer_fold,
            calibration_fold=None,
            active_fit_ids_sha256=_id_sha256(outer_train),
            # Final architecture selection is scored only through the locked inner
            # validation folds whose union is the outer-training set.  Keep the
            # outer-test hash separate as an explicit forbidden universe.
            scored_ids_sha256=_id_sha256(outer_train),
            forbidden_outer_test_ids_sha256=_id_sha256(outer_test),
            outer_test_metric_consulted=False,
            hagr_metric_consulted=False,
        )
    )
    _write_parquet(paths["outer"], pd.DataFrame(outer_rows))
    _write_parquet(paths["inner"], inner_frame)
    _write_parquet(paths["trace"], pd.concat(trace_parts, ignore_index=True))
    _write_parquet(paths["applicability"], applicability.reset_index())
    saved_model_manifest = final_model.save(paths["model"])
    if not isinstance(saved_model_manifest, Mapping):
        raise NestedCVError("Model save did not return an auditable manifest")
    saved_model_sha256 = saved_model_manifest.get(
        "model_artifact_sha256", saved_model_manifest.get("model_sha256")
    )
    if not isinstance(saved_model_sha256, str) or saved_model_sha256 != sha256_file(
        paths["model"]
    ):
        raise NestedCVError("Saved model manifest does not bind its model bytes")
    atomic_write_json(paths["model_manifest"], dict(saved_model_manifest))
    calibration_record = {
        "schema_version": "geroprotector.calibration_bundle.v2",
        "binding": {
            "run_id": run_id,
            "pipeline_id": pipeline_id,
            "model_id": model_name,
            "repeat": int(repeat),
            "outer_fold": int(outer_fold),
            "fit_ids_sha256": _id_sha256(ordered_train),
            "outer_test_ids_sha256": _id_sha256(outer_test),
            "model_artifact_sha256": saved_model_sha256,
        },
        "platt": platt.get_manifest(),
        "beta_sensitivity": beta.get_manifest(),
        "threshold": {
            "objective": "mcc",
            "value": float(threshold),
            "training_oof_mcc": float(threshold_mcc),
            "used_for_primary_model_selection": False,
        },
        "inference_audit": inference_audit,
        "runtime": {
            "calibration_crossfit_seconds": float(crossfit_seconds),
            "outer_train_final_fit_seconds": float(final_fit_seconds),
            "outer_prediction_and_audit_seconds": float(outer_prediction_seconds),
            "job_total_seconds_before_serialization": float(time.perf_counter() - job_started),
        },
    }
    atomic_write_json(paths["calibration"], calibration_record)
    completed = {
        "schema_version": "geroprotector.outer_job_completion.v1",
        "created_utc": utc_now(),
        "job_binding_sha256": binding,
        "run_id": run_id,
        "model_id": model_name,
        "repeat": repeat,
        "outer_fold": outer_fold,
        "one_final_outer_prediction_record_per_compound": True,
        "additional_label_free_inference_audit_calls": bool(inference_audit is not None),
        "outer_test_metrics_computed": False,
        "hagr_labels_loaded": False,
        "artifacts": _job_artifacts(staging),
    }
    completed["canonical_sha256"] = canonical_sha256(completed)
    atomic_write_json(paths["completed"], completed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging, destination)
    if isinstance(final_model, V6Pipeline):
        del final_model
        release_accelerator_memory()
    return destination


def _artifact_inventory(run_directory: Path) -> list[dict[str, Any]]:
    excluded = {"artifact_manifest.json", "run_manifest.json", "COMPLETED.json"}
    records = []
    for path in sorted(run_directory.rglob("*")):
        relative = path.relative_to(run_directory)
        if (
            path.is_dir()
            or path.name in excluded
            or (relative.parts and relative.parts[0] == "report")
        ):
            continue
        if path.is_symlink() or not path.is_file():
            raise NestedCVError(f"Run artifact is unsafe: {path}")
        records.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _archive_partial_aggregation(work: Path) -> Path | None:
    """Archive only incomplete run-level derivatives; preserve all job evidence."""

    if (work / "COMPLETED.json").exists():
        raise NestedCVError("A completed work directory must be published, not recovered")
    candidates = (
        work / "predictions",
        work / "selections",
        work / "audits",
        work / "artifact_manifest.json",
        work / "run_manifest.json",
    )
    present = [path for path in candidates if path.exists() or path.is_symlink()]
    if not present:
        return None
    for path in present:
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise NestedCVError(f"Unsafe partial aggregate artifact: {path}")
    forensic_root = work.parent / ".forensics"
    forensic_root.mkdir(parents=True, exist_ok=True)
    archive = Path(
        tempfile.mkdtemp(prefix=f".{work.name}.aggregate-partial.", dir=forensic_root)
    )
    for path in present:
        os.rename(path, archive / path.name)
    atomic_write_json(
        archive / "RECOVERY.json",
        {
            "schema_version": "geroprotector.aggregate_recovery.v1",
            "run_id": work.name.removeprefix(".").removesuffix(".work"),
            "archived_paths": [path.name for path in present],
            "recovery_reason": "incomplete_aggregate_publication",
            "recoverable": True,
        },
    )
    return archive


def _validate_outer_predictions(
    predictions: pd.DataFrame,
    *,
    model_names: Sequence[str],
    outer_registry: pd.DataFrame,
    repeats: int,
) -> None:
    required = {
        "model_id",
        "compound_id",
        "repeat",
        "outer_fold",
        "y_true",
        "probability_raw",
        "probability_calibrated",
        "primary_component_id",
    }
    if required - set(predictions):
        raise NestedCVError("Aggregated outer predictions have an incomplete schema")
    is_paper = "paper_test" in set(outer_registry["role"].astype(str))
    expected_registry = (
        outer_registry.loc[outer_registry["role"].eq("paper_test")]
        if is_paper
        else outer_registry
    )
    expected_ids = set(expected_registry["compound_id"].astype(str))
    for model_name in model_names:
        table = predictions.loc[predictions["model_id"].eq(model_name)]
        if len(table) != len(expected_ids) * repeats:
            raise NestedCVError(f"Outer OOF coverage incomplete for {model_name}")
        if table.duplicated(["compound_id", "repeat"]).any():
            raise NestedCVError(f"Duplicate outer OOF prediction for {model_name}")
        if set(table["compound_id"].astype(str)) != expected_ids:
            raise NestedCVError(f"Outer OOF identity coverage changed for {model_name}")
        if not np.isfinite(
            table[["probability_raw", "probability_calibrated"]].to_numpy(dtype=float)
        ).all():
            raise NestedCVError(f"Non-finite outer OOF probabilities for {model_name}")
        merged = table.merge(
            expected_registry,
            on=["compound_id", "repeat"],
            how="left",
            validate="one_to_one",
            suffixes=("_prediction", "_registry"),
        )
        if (
            not merged["outer_fold_prediction"]
            .astype(int)
            .eq(merged["outer_fold_registry"].astype(int))
            .all()
        ):
            raise NestedCVError(f"Outer fold assignment changed for {model_name}")
        if not merged["y_true"].astype(int).eq(merged["label"].astype(int)).all():
            raise NestedCVError(f"Outer labels changed for {model_name}")
        if (
            not merged["primary_component_id"]
            .astype(str)
            .eq(merged["component_id"].astype(str))
            .all()
        ):
            raise NestedCVError(f"Outer components changed for {model_name}")


def plan_nested_cv(
    *, root: str | Path, config_path: str | Path, suite: str = "core"
) -> dict[str, Any]:
    project = Path(root).resolve()
    validate_core_runtime(project / "requirements-lock.txt")
    config = resolve_config(config_path)
    validate_locked_shared_contract(config)
    assert_internal_config_safe(config)
    validate_protocol_lock(
        root=project,
        contract_name={
            "reference": "reference",
            "V5_ELIXIRFP_REBUILT": "v5",
            "V6_TABULAR_FOUNDATION_MODELS": "v6",
            "V5BIS_PAPER80": "v5bis",
            "V6BIS_PAPER80": "v6bis",
        }[str(config["pipeline"])],
        config=config,
    )
    curated, outer, inner, data_manifest, split_manifest = _data_and_splits(project, config)
    specs = _model_specs(config, suite=suite)
    if any(spec["kind"] == "reference" for spec in specs):
        _reference_config(project)
    axes = _outer_axes(config)
    scientific_outer_partitions = len(axes)
    plan = {
        "schema_version": "geroprotector.nested_cv_plan.v1",
        "planning_only": True,
        "estimators_fitted": 0,
        "pipeline_id": _pipeline_id(config),
        "suite": suite,
        "n_compounds": len(curated),
        "class_counts": {
            str(key): int(value)
            for key, value in curated["label"].value_counts().sort_index().items()
        },
        "n_components": int(outer["component_id"].nunique()),
        "outer_jobs": scientific_outer_partitions,
        "sealed_model_bundles": scientific_outer_partitions * len(specs),
        "model_ids": [_model_name(spec) for spec in specs],
        "full_pipeline_fit_calls_minimum": scientific_outer_partitions * len(specs) * 4,
        "note": (
            "Each outer job uses three full-pipeline calibration crossfit fits plus one "
            "outer-training refit; internal candidate estimator calls are additional."
        ),
        "curation_manifest_sha256": sha256_file(
            _project_path(project, config["outputs"]["manifest"], role="data manifest")
        ),
        "split_manifest_sha256": sha256_file(
            _project_path(
                project,
                _active_registry_outputs(config)["manifest"],
                role="split manifest",
            )
        ),
        "data_binding": data_manifest["curated_identity_label_sha256"],
        "split_binding": split_manifest["registry_sha256"],
        "inner_registry_rows": len(inner),
        "hagr_labels_loaded": False,
    }
    if _is_paper80(config):
        plan["evaluation_design"] = {
            "strategy": "paper_random_80_20",
            "role": "retrospective_contextual_only",
            "outer_train_rows": int(split_manifest["n_train"]),
            "outer_test_rows": int(split_manifest["n_test"]),
            "outer_test_class_0": int(split_manifest["test_class_0"]),
            "outer_test_class_1": int(split_manifest["test_class_1"]),
            "component_overlap_count": int(split_manifest["component_overlap_count"]),
            "identity_overlap_count": 0,
            "inner_components_fit_on_outer_train_only": bool(
                split_manifest["inner_components_fit_on_outer_train_only"]
            ),
            "outer_test_structures_used_for_inner_grouping": bool(
                split_manifest["outer_test_structures_used_for_inner_grouping"]
            ),
            "nearest_train_tanimoto_median": float(
                split_manifest["nearest_train_tanimoto_median"]
            ),
            "nearest_train_tanimoto_max": float(split_manifest["nearest_train_tanimoto_max"]),
            "headline_eligible": False,
            "replaces_grouped_primary_validation": False,
        }
    if _is_v5(config):
        weighted_variants = sum(
            spec.get("variant")
            in {
                "weighted_no_reduction",
                "weighted_svd",
                "weighted_nystroem",
                "final_v5",
            }
            for spec in specs
            if spec["kind"] == "v5"
        )
        active_scopes = scientific_outer_partitions * 4
        stage_b_states = active_scopes * 4 if weighted_variants else 0
        vector_pairs = (
            stage_b_states
            * len(config["stage_a"]["semantic_families"])
            * int(config["weighting"]["grouped_subcv_folds"])
        )
        plan["v5_dependency_plan"] = {
            "scientific_outer_partitions": scientific_outer_partitions,
            "sealed_model_bundles": scientific_outer_partitions * len(specs),
            "unique_stage_a_active_scopes": active_scopes,
            "unique_stage_b_states_with_cache": stage_b_states,
            "stage_b_states_without_cross_variant_cache": (stage_b_states * weighted_variants),
            "stage_b_subfold_vector_pairs_with_cache": vector_pairs,
            "stage_b_subfold_vector_pairs_without_cross_variant_cache": (
                vector_pairs * weighted_variants
            ),
            "cache_uses_outer_test_outcomes": False,
            "cache_scope": "exact_active_fit_dependency",
        }
    return plan


def verify_completed_run(run_directory: str | Path) -> tuple[dict, dict, dict]:
    """Rehash every sealed training artifact before reporting or downstream use."""

    directory = Path(run_directory)
    if directory.is_symlink() or not directory.is_dir():
        raise NestedCVError(f"Run directory must be regular/non-symlink: {directory}")
    controls = {
        "completion": directory / "COMPLETED.json",
        "manifest": directory / "run_manifest.json",
        "artifacts": directory / "artifact_manifest.json",
    }
    for role, path in controls.items():
        if path.is_symlink() or not path.is_file():
            raise NestedCVError(f"Sealed run lacks {role}: {path}")

    def checked_json(path: Path, *, role: str) -> dict:
        value = json.loads(path.read_text(encoding="utf-8"))
        payload = dict(value)
        claimed = payload.pop("canonical_sha256", None)
        if claimed != canonical_sha256(payload):
            raise NestedCVError(f"{role} canonical integrity failed")
        return value

    completed = checked_json(controls["completion"], role="completion")
    manifest = checked_json(controls["manifest"], role="run manifest")
    artifact_manifest = checked_json(controls["artifacts"], role="artifact manifest")
    if completed.get("run_manifest_sha256") != sha256_file(controls["manifest"]):
        raise NestedCVError("Completion/run-manifest byte binding failed")
    if completed.get("artifact_manifest_sha256") != sha256_file(controls["artifacts"]):
        raise NestedCVError("Completion/artifact-manifest byte binding failed")
    if manifest.get("artifact_manifest_sha256") != sha256_file(controls["artifacts"]):
        raise NestedCVError("Run/artifact-manifest byte binding failed")
    if artifact_manifest.get("entries") != _artifact_inventory(directory):
        raise NestedCVError("Sealed run artifact inventory changed")
    if completed.get("outer_predictions_sha256") != sha256_file(
        directory / "predictions" / "outer_long.parquet"
    ):
        raise NestedCVError("Sealed outer predictions changed")
    if completed.get("schema_version") != "geroprotector.run_completion.v1":
        raise NestedCVError("Completion schema version is unsupported")
    if artifact_manifest.get("schema_version") != "geroprotector.run_artifacts.v1":
        raise NestedCVError("Artifact-manifest schema version is unsupported")
    if completed.get("run_id") != manifest.get("run_id") or completed.get(
        "pipeline_id"
    ) != manifest.get("pipeline_id"):
        raise NestedCVError("Completion and run manifest identify different runs")
    _validate_run_manifest_schema(
        directory / "source_snapshot" / "schemas" / "run_manifest.schema.json",
        manifest,
    )
    return completed, manifest, artifact_manifest


def run_nested_cv(
    *,
    root: str | Path,
    config_path: str | Path,
    run_id: str,
    suite: str = "core",
) -> Path:
    """Run/resume a sealed benchmark. This function never computes outer metrics."""

    if not _RUN_ID.fullmatch(run_id):
        raise NestedCVError("Run ID must use 3-80 lowercase safe characters")
    project = Path(root).resolve()
    validate_core_runtime(project / "requirements-lock.txt")
    locked_environment = runtime_environment()
    locked_environment_sha256 = canonical_sha256(locked_environment)
    requirements_lock = project / "requirements-lock.txt"
    requirements_lock_sha256 = sha256_file(requirements_lock)
    config = resolve_config(config_path)
    validate_locked_shared_contract(config)
    assert_internal_config_safe(config)
    validate_protocol_lock(
        root=project,
        contract_name={
            "reference": "reference",
            "V5_ELIXIRFP_REBUILT": "v5",
            "V6_TABULAR_FOUNDATION_MODELS": "v6",
            "V5BIS_PAPER80": "v5bis",
            "V6BIS_PAPER80": "v6bis",
        }[str(config["pipeline"])],
        config=config,
    )
    placeholders = unresolved_placeholders(config)
    if placeholders:
        raise NestedCVError("Resolve runtime placeholders: " + ", ".join(placeholders))
    pipeline_id = _pipeline_id(config)
    required_prefix = {
        "reference": "reference_",
        "V5_ELIXIRFP_REBUILT": "v5_",
        "V6_TABULAR_FOUNDATION_MODELS": "v6_",
        "V5BIS_PAPER80": "v5bis_",
        "V6BIS_PAPER80": "v6bis_",
    }[pipeline_id]
    if not run_id.startswith(required_prefix):
        raise NestedCVError(f"Run ID for {pipeline_id} must start with {required_prefix}")
    config_hash = resolved_config_sha256(config)
    source_hash = source_tree_sha256(project)
    curated, outer, inner, data_manifest, split_manifest = _data_and_splits(project, config)
    data_manifest_path = _project_path(
        project, config["outputs"]["manifest"], role="data manifest"
    )
    split_manifest_path = _project_path(
        project, _active_registry_outputs(config)["manifest"], role="split manifest"
    )
    data_manifest_sha256 = sha256_file(data_manifest_path)
    split_manifest_sha256 = sha256_file(split_manifest_path)
    checkpoint_ledger = None
    checkpoint_ledger_path: Path | None = None
    checkpoint_ledger_sha256: str | None = None
    if _is_v6(pipeline_id):
        checkpoint_ledger_path = _project_path(
            project, config["checkpoint_ledger"], role="V6 checkpoint ledger"
        )
        checkpoint_ledger = load_checkpoint_ledger(
            checkpoint_ledger_path,
            config=config,
            root=project,
            resolved_config_sha256=config_hash,
        )
        checkpoint_ledger_sha256 = sha256_file(checkpoint_ledger_path)

    def assert_inputs_unchanged() -> None:
        if sha256_file(data_manifest_path) != data_manifest_sha256:
            raise NestedCVError("Data manifest changed during the run")
        if sha256_file(split_manifest_path) != split_manifest_sha256:
            raise NestedCVError("Split manifest changed during the run")
        _, _, _, refreshed_data, refreshed_splits = _data_and_splits(project, config)
        if refreshed_data != data_manifest or refreshed_splits != split_manifest:
            raise NestedCVError("Curated-data or split-registry binding changed during the run")
        if checkpoint_ledger_path is not None:
            if sha256_file(checkpoint_ledger_path) != checkpoint_ledger_sha256:
                raise NestedCVError("V6 checkpoint ledger changed during the run")
            load_checkpoint_ledger(
                checkpoint_ledger_path,
                config=config,
                root=project,
                resolved_config_sha256=config_hash,
            )

    specs = _model_specs(config, suite=suite)
    reference_config_hash = None
    if any(spec["kind"] == "reference" for spec in specs):
        _, reference_config_hash = _reference_config(project)
    names = tuple(_model_name(spec) for spec in specs)
    if len(names) != len(set(names)):
        raise NestedCVError("Model suite contains duplicate model IDs")
    outputs = project / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    final = outputs / run_id
    work = outputs / f".{run_id}.work"
    if final.exists():
        raise FileExistsError(f"Completed run already exists: {final}")
    running_binding = {
        "schema_version": "geroprotector.running_lock.v1",
        "run_id": run_id,
        "pipeline_id": pipeline_id,
        "suite": suite,
        "resolved_config_sha256": config_hash,
        "source_tree_sha256": source_hash,
        "runtime_environment_sha256": locked_environment_sha256,
        "requirements_lock_sha256": requirements_lock_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "checkpoint_ledger_sha256": checkpoint_ledger_sha256,
        "model_ids": list(names),
        "reference_config_sha256": reference_config_hash,
    }
    if work.exists():
        lock_path = work / "RUNNING.json"
        if lock_path.is_symlink() or not lock_path.is_file():
            raise NestedCVError("Existing work directory lacks a safe RUNNING lock")
        observed = json.loads(lock_path.read_text(encoding="utf-8"))
        if observed != running_binding:
            raise NestedCVError("Existing work directory belongs to a different run binding")
        if (work / "COMPLETED.json").exists() or (work / "COMPLETED.json").is_symlink():
            verify_completed_run(work)
            assert_inputs_unchanged()
            if source_tree_sha256(project) != source_hash:
                raise NestedCVError("Source tree changed before completed-run publication")
            os.rename(work, final)
            return final
    else:
        work.mkdir()
        atomic_write_json(work / "RUNNING.json", running_binding)
        atomic_write_bytes(
            work / "resolved_config.yaml",
            yaml.safe_dump(config, sort_keys=True).encode(),
        )
        atomic_write_json(work / "environment.lock.json", locked_environment)
        atomic_write_json(work / "curation_reference.json", data_manifest)
        atomic_write_json(work / "split_reference.json", split_manifest)
        snapshot = work / "source_snapshot"
        for source in source_tree_files(project):
            target = snapshot / source.relative_to(project)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        atomic_write_json(
            work / "source_snapshot_manifest.json",
            {
                "source_tree_sha256": source_hash,
                "files": [
                    {
                        "path": source.relative_to(project).as_posix(),
                        "sha256": sha256_file(source),
                    }
                    for source in source_tree_files(project)
                ],
            },
        )
    bound_control_values = {
        "environment.lock.json": locked_environment,
        "curation_reference.json": data_manifest,
        "split_reference.json": split_manifest,
    }
    for filename, expected_value in bound_control_values.items():
        control_path = work / filename
        if control_path.is_symlink() or not control_path.is_file():
            raise NestedCVError(f"Run work directory lacks safe control: {filename}")
        if json.loads(control_path.read_text(encoding="utf-8")) != expected_value:
            raise NestedCVError(f"Run work control changed: {filename}")
    expected_snapshot_files = [
        {
            "path": source.relative_to(project).as_posix(),
            "sha256": sha256_file(source),
        }
        for source in source_tree_files(project)
    ]
    snapshot_manifest_path = work / "source_snapshot_manifest.json"
    if snapshot_manifest_path.is_symlink() or not snapshot_manifest_path.is_file():
        raise NestedCVError("Run source-snapshot manifest is missing or unsafe")
    if json.loads(snapshot_manifest_path.read_text(encoding="utf-8")) != {
        "source_tree_sha256": source_hash,
        "files": expected_snapshot_files,
    }:
        raise NestedCVError("Run source-snapshot manifest changed")
    for record in expected_snapshot_files:
        snapshot_file = work / "source_snapshot" / record["path"]
        if snapshot_file.is_symlink() or not snapshot_file.is_file():
            raise NestedCVError(f"Run source snapshot is incomplete: {record['path']}")
        if sha256_file(snapshot_file) != record["sha256"]:
            raise NestedCVError(f"Run source snapshot changed: {record['path']}")
    recovered_aggregate = _archive_partial_aggregation(work)
    log_path = work / "logs" / "events.jsonl"
    append_event(log_path, "run_started_or_resumed", run_id=run_id, suite=suite)
    if recovered_aggregate is not None:
        append_event(
            log_path,
            "partial_aggregate_archived",
            archive=str(recovered_aggregate.relative_to(outputs)),
        )
    store = _build_store(config, curated)
    axes = _outer_axes(config)
    repeats = 1 if _is_paper80(config) else int(config["primary_split"]["outer_repeats"])
    folds = 1 if _is_paper80(config) else int(config["primary_split"]["outer_folds"])
    job_directories: list[Path] = []
    for repeat, outer_fold in axes:
        for spec in specs:
            if source_tree_sha256(project) != source_hash:
                raise NestedCVError("Source tree changed during the run; restore it to resume")
            append_event(
                log_path,
                "job_start",
                repeat=repeat,
                outer_fold=outer_fold,
                model_id=_model_name(spec),
            )
            job_directory = _run_one_job(
                root=project,
                run_id=run_id,
                pipeline_id=pipeline_id,
                config=config,
                config_hash=config_hash,
                source_hash=source_hash,
                runtime_environment_sha256=locked_environment_sha256,
                curated=curated,
                outer_registry=outer,
                inner_registry=inner,
                store=_store_for_spec(store, spec),
                spec=spec,
                repeat=repeat,
                outer_fold=outer_fold,
                work_root=work,
                checkpoint_ledger=checkpoint_ledger,
            )
            job_directories.append(job_directory)
            append_event(
                log_path,
                "job_complete",
                repeat=repeat,
                outer_fold=outer_fold,
                model_id=_model_name(spec),
            )
    assert_inputs_unchanged()
    outer_frames = [pd.read_parquet(_job_paths(path)["outer"]) for path in job_directories]
    inner_frames = [pd.read_parquet(_job_paths(path)["inner"]) for path in job_directories]
    trace_frames = [pd.read_parquet(_job_paths(path)["trace"]) for path in job_directories]
    outer_predictions = pd.concat(outer_frames, ignore_index=True)
    _validate_outer_predictions(
        outer_predictions,
        model_names=names,
        outer_registry=outer,
        repeats=repeats,
    )
    _write_parquet(work / "predictions" / "outer_long.parquet", outer_predictions)
    _write_parquet(
        work / "predictions" / "calibration_inner_oof.parquet",
        pd.concat(inner_frames, ignore_index=True),
    )
    _write_parquet(
        work / "selections" / "all_attempts.parquet",
        pd.concat(trace_frames, ignore_index=True),
    )
    runtime_rows: list[dict[str, Any]] = []
    inference_audits: list[dict[str, Any]] = []
    fitted_model_manifests: list[dict[str, Any]] = []
    for job_directory in job_directories:
        job_completion = json.loads(
            _job_paths(job_directory)["completed"].read_text(encoding="utf-8")
        )
        calibration_bundle = json.loads(
            _job_paths(job_directory)["calibration"].read_text(encoding="utf-8")
        )
        fitted_manifest = json.loads(
            _job_paths(job_directory)["model_manifest"].read_text(encoding="utf-8")
        )
        fitted_model_manifests.append(
            {
                "model_id": job_completion["model_id"],
                "repeat": int(job_completion["repeat"]),
                "outer_fold": int(job_completion["outer_fold"]),
                "manifest": fitted_manifest,
            }
        )
        runtime_rows.append(
            {
                "model_id": job_completion["model_id"],
                "repeat": int(job_completion["repeat"]),
                "outer_fold": int(job_completion["outer_fold"]),
                **calibration_bundle["runtime"],
            }
        )
        if calibration_bundle.get("inference_audit") is not None:
            inference_audits.append(
                {
                    "model_id": job_completion["model_id"],
                    "repeat": int(job_completion["repeat"]),
                    "outer_fold": int(job_completion["outer_fold"]),
                    "audit": calibration_bundle["inference_audit"],
                }
            )
    _write_parquet(work / "audits" / "runtime_jobs.parquet", pd.DataFrame(runtime_rows))
    atomic_write_json(
        work / "audits" / "v6_inference_audits.json",
        {
            "schema_version": "geroprotector.v6_inference_audit_collection.v1",
            "pipeline_id": pipeline_id,
            "n_audits": len(inference_audits),
            "audits": inference_audits,
        },
    )
    atomic_write_json(
        work / "audits" / "fitted_model_manifests.json",
        {
            "schema_version": "geroprotector.fitted_model_manifest_collection.v1",
            "pipeline_id": pipeline_id,
            "n_manifests": len(fitted_model_manifests),
            "manifests": fitted_model_manifests,
        },
    )
    leakage_audit = {
        "schema_version": "geroprotector.internal_leakage_audit.v1",
        "passed": True,
        "hagr_labels_loaded_during_development": False,
        "hagr_used_for_model_selection": False,
        "prior_ml_candidates_used_as_labels": 0,
        "outer_test_metrics_computed_during_training": False,
        "outer_test_used_for_architecture_selection": False,
        "calibration_source": "full_pipeline_crossfit_on_locked_outer_train_only",
        "threshold_source": "calibrated_full_pipeline_crossfit_outer_train_only",
        "source_label_confounding_disclosed": True,
    }
    atomic_write_json(work / "audits" / "leakage.json", leakage_audit)
    if source_tree_sha256(project) != source_hash:
        raise NestedCVError("Source tree changed before run publication")
    artifact_manifest = {
        "schema_version": "geroprotector.run_artifacts.v1",
        "entries": _artifact_inventory(work),
    }
    artifact_manifest["canonical_sha256"] = canonical_sha256(artifact_manifest)
    atomic_write_json(work / "artifact_manifest.json", artifact_manifest)
    environment = runtime_environment()
    if canonical_sha256(environment) != locked_environment_sha256:
        raise NestedCVError("Runtime environment changed before run publication")
    if requirements_lock.is_symlink() or not requirements_lock.is_file():
        raise NestedCVError("requirements-lock.txt must be a regular non-symlink file")
    if sha256_file(requirements_lock) != requirements_lock_sha256:
        raise NestedCVError("requirements-lock.txt changed during the run")
    environment["packages_lock_sha256"] = requirements_lock_sha256
    run_manifest = {
        "schema_version": "geroprotector.run_manifest.v2",
        "run_id": run_id,
        "pipeline_id": pipeline_id,
        "suite": suite,
        "created_utc": utc_now(),
        "status": "complete",
        "code": {
            "source_tree_sha256": source_hash,
            "git_commit": None,
            "dirty_worktree": None,
            "entrypoint": "gero run nested-cv",
        },
        "configuration": {
            "resolved_config_sha256": config_hash,
            "resolved_config_path": "resolved_config.yaml",
            "schema_validation_passed": True,
            "reference_config_sha256": reference_config_hash,
            "reference_protocol_lock_validated": reference_config_hash is not None,
        },
        "environment": environment,
        "data": {
            "curation_manifest_sha256": sha256_file(data_manifest_path),
            "curated_cohort_sha256": data_manifest["curated_cohort_canonical_sha256"],
            "curated_identity_label_sha256": data_manifest["curated_identity_label_sha256"],
            "provenance_table_sha256": data_manifest["artifacts"]["provenance_table"]["sha256"],
            "n_total": len(curated),
            "n_positive": int(curated["label"].eq(1).sum()),
            "n_weak_reference": int(curated["label"].eq(0).sum()),
            "n_prior_candidates_used_as_labels": 0,
            "source_label_confounding_disclosed": True,
        },
        "splits": {
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "registry_sha256": split_manifest["registry_sha256"],
            "inner_registry_sha256": split_manifest["inner_registry_sha256"],
            "primary_strategy": _split_strategy(config),
            "outer_test_used_for_selection": False,
            "group_overlap_count": int(split_manifest.get("component_overlap_count", 0)),
            "identity_overlap_count": 0,
        },
        "models": list(names),
        "outer_repeats": repeats,
        "outer_folds": folds,
        "external_firewall": {
            "hagr_labels_loaded_during_development": False,
            "hagr_used_for_model_selection": False,
            "external2_score_once_policy_enforced": True,
            "external2_scored": False,
        },
        "artifact_manifest_sha256": sha256_file(work / "artifact_manifest.json"),
    }
    run_manifest["canonical_sha256"] = canonical_sha256(run_manifest)
    _validate_run_manifest_schema(
        project / "schemas" / "run_manifest.schema.json", run_manifest
    )
    atomic_write_json(work / "run_manifest.json", run_manifest)
    completed = {
        "schema_version": "geroprotector.run_completion.v1",
        "run_id": run_id,
        "pipeline_id": pipeline_id,
        "completed_utc": utc_now(),
        "run_manifest_sha256": sha256_file(work / "run_manifest.json"),
        "artifact_manifest_sha256": sha256_file(work / "artifact_manifest.json"),
        "outer_predictions_sha256": sha256_file(work / "predictions" / "outer_long.parquet"),
        "selection_attempts_sha256": sha256_file(work / "selections" / "all_attempts.parquet"),
        "outer_test_metrics_computed_during_training": False,
        "hagr_labels_loaded_during_development": False,
    }
    completed["canonical_sha256"] = canonical_sha256(completed)
    atomic_write_json(work / "COMPLETED.json", completed)
    assert_inputs_unchanged()
    if source_tree_sha256(project) != source_hash:
        raise NestedCVError("Source tree changed at final atomic publication")
    os.rename(work, final)
    return final
