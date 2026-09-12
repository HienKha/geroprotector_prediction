"""Run-local, content-addressed Stage-A/Stage-B preparation cache for V5.

The cache is deliberately scoped to one sealed run.  Every entry binds the exact
active fit IDs, labels, chemical groups, selection folds, feature bytes, scientific
seed, configuration and run/source context.  No outer-test IDs or outcomes are inputs.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ...hashing import atomic_write_json, canonical_sha256, sha256_file
from .fingerprint_bank import V5FeatureBank
from .resolution_search import select_resolutions
from .stable_importance import StableImportanceState, fit_stable_importance


class V5PreparationCacheError(RuntimeError):
    """Raised when a prepared V5 dependency is incomplete or has changed."""


@dataclass(frozen=True)
class StageAResult:
    selected_specs: tuple[str, ...]
    trace: pd.DataFrame
    binding_sha256: str
    reused: bool


@dataclass(frozen=True)
class StageBResult:
    state: StableImportanceState
    binding_sha256: str
    reused: bool


def _normalise_folds(
    folds: Sequence[tuple[tuple[str, ...], tuple[str, ...]]],
) -> list[dict[str, list[str]]]:
    return [
        {
            "train_ids": sorted(map(str, train_ids)),
            "validation_ids": sorted(map(str, validation_ids)),
        }
        for train_ids, validation_ids in folds
    ]


def _aligned_series_payload(series: pd.Series, ids: Sequence[str], *, integer: bool) -> list:
    aligned = series.copy()
    aligned.index = aligned.index.astype(str)
    ordered = sorted(map(str, ids))
    if set(aligned.index) != set(ordered):
        raise V5PreparationCacheError("Cached V5 series is not aligned to active fit IDs")
    if integer:
        return [[compound, int(aligned.loc[compound])] for compound in ordered]
    return [[compound, str(aligned.loc[compound])] for compound in ordered]


def _entry_paths(root: Path, kind: str, binding: str) -> tuple[Path, Path, Path]:
    directory = root / kind / binding
    return directory, directory / "payload.joblib", directory / "manifest.json"


def _load_entry(root: Path, kind: str, binding: str) -> Any | None:
    directory, artifact, manifest_path = _entry_paths(root, kind, binding)
    if not directory.exists():
        return None
    if directory.is_symlink() or not directory.is_dir():
        raise V5PreparationCacheError(f"Unsafe V5 cache directory: {directory}")
    for path, role in ((artifact, "payload"), (manifest_path, "manifest")):
        if path.is_symlink() or not path.is_file():
            raise V5PreparationCacheError(f"V5 cache {role} is missing or unsafe: {path}")
    manifest = joblib.load(manifest_path) if manifest_path.suffix == ".joblib" else None
    if manifest is None:
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = dict(manifest)
    claimed = payload.pop("canonical_sha256", None)
    if claimed != canonical_sha256(payload):
        raise V5PreparationCacheError(f"V5 cache manifest self-hash failed: {directory}")
    if (
        manifest.get("schema_version") != "geroprotector.v5_preparation_cache.v1"
        or manifest.get("kind") != kind
        or manifest.get("binding_sha256") != binding
        or manifest.get("payload_sha256") != sha256_file(artifact)
    ):
        raise V5PreparationCacheError(f"V5 cache binding failed: {directory}")
    return joblib.load(artifact)


def _publish_entry(root: Path, kind: str, binding: str, value: Any) -> Any:
    parent = root / kind
    parent.mkdir(parents=True, exist_ok=True)
    destination, _, _ = _entry_paths(root, kind, binding)
    staging = Path(tempfile.mkdtemp(prefix=f".{binding}.", dir=parent))
    try:
        artifact = staging / "payload.joblib"
        joblib.dump(value, artifact)
        with artifact.open("rb") as handle:
            os.fsync(handle.fileno())
        manifest = {
            "schema_version": "geroprotector.v5_preparation_cache.v1",
            "kind": kind,
            "binding_sha256": binding,
            "payload_sha256": sha256_file(artifact),
        }
        manifest["canonical_sha256"] = canonical_sha256(manifest)
        atomic_write_json(staging / "manifest.json", manifest)
        try:
            os.rename(staging, destination)
        except FileExistsError:
            existing = _load_entry(root, kind, binding)
            if existing is None:
                raise V5PreparationCacheError(
                    f"Concurrent V5 cache publication is incomplete: {destination}"
                ) from None
            return existing
        return value
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_or_fit_stage_a(
    *,
    cache_root: Path | None,
    cache_context_sha256: str | None,
    bank: V5FeatureBank,
    y: pd.Series,
    groups: pd.Series,
    folds: Sequence[tuple[tuple[str, ...], tuple[str, ...]]],
    config: Mapping[str, Any],
    seed: int,
) -> StageAResult:
    binding_payload = {
        "schema": "geroprotector.v5_stage_a_dependency.v1",
        "cache_context_sha256": cache_context_sha256,
        "feature_contract_sha256": bank.feature_contract_hash,
        "active_feature_content_sha256": bank.bank_hash,
        "active_fit_ids": sorted(bank.ids),
        "labels": _aligned_series_payload(y, bank.ids, integer=True),
        "groups": _aligned_series_payload(groups, bank.ids, integer=False),
        "selection_folds": _normalise_folds(folds),
        "stage_a_config": dict(config["stage_a"]),
        "seed": int(seed),
    }
    binding = canonical_sha256(binding_payload)
    if cache_root is not None:
        cached = _load_entry(cache_root, "stage_a", binding)
        if cached is not None:
            selected = tuple(map(str, cached["selected_specs"]))
            trace = cached["trace"]
            if not isinstance(trace, pd.DataFrame) or not set(selected).issubset(bank.specs):
                raise V5PreparationCacheError("Cached Stage-A payload has invalid contents")
            return StageAResult(selected, trace.copy(), binding, True)
    selected, trace = select_resolutions(
        bank,
        y,
        groups,
        config=config,
        seed=int(seed),
        folds=folds,
    )
    value = {"selected_specs": tuple(selected), "trace": trace.copy()}
    if cache_root is not None:
        value = _publish_entry(cache_root, "stage_a", binding, value)
    return StageAResult(
        tuple(map(str, value["selected_specs"])),
        value["trace"].copy(),
        binding,
        False,
    )


def load_or_fit_stage_b(
    *,
    cache_root: Path | None,
    cache_context_sha256: str | None,
    bank: V5FeatureBank,
    selected_specs: Sequence[str],
    y: pd.Series,
    groups: pd.Series,
    config: Mapping[str, Any],
    seed: int,
) -> StageBResult:
    selected = tuple(map(str, selected_specs))
    binding_payload = {
        "schema": "geroprotector.v5_stage_b_dependency.v1",
        "cache_context_sha256": cache_context_sha256,
        "feature_contract_sha256": bank.feature_contract_hash,
        "active_feature_content_sha256": bank.bank_hash,
        "active_fit_ids": sorted(bank.ids),
        "labels": _aligned_series_payload(y, bank.ids, integer=True),
        "groups": _aligned_series_payload(groups, bank.ids, integer=False),
        "selected_specs": list(selected),
        "weighting_config": dict(config["weighting"]),
        "seed": int(seed),
    }
    binding = canonical_sha256(binding_payload)
    if cache_root is not None:
        cached = _load_entry(cache_root, "stage_b", binding)
        if cached is not None:
            if not isinstance(cached, StableImportanceState):
                raise V5PreparationCacheError("Cached Stage-B payload has the wrong type")
            if cached.selected_specs != selected or set(cached.fit_ids) != set(bank.ids):
                raise V5PreparationCacheError("Cached Stage-B payload differs from fit scope")
            return StageBResult(cached, binding, True)
    vector_root = None if cache_root is None else cache_root / "stage_b_vectors"
    state = fit_stable_importance(
        bank,
        selected,
        y,
        groups,
        config=config,
        seed=int(seed),
        checkpoint_root=vector_root,
        parent_binding_sha256=binding,
    )
    if cache_root is not None:
        state = _publish_entry(cache_root, "stage_b", binding, state)
    return StageBResult(state, binding, False)
