from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import DataStructs

from geroprotector import cli
from geroprotector.audit import (
    RuntimeLockError,
    runtime_environment,
    source_tree_files,
    source_tree_sha256,
    validate_core_runtime,
)
from geroprotector.config import resolve_config, validate_protocol_lock
from geroprotector.data.identities import compound_id
from geroprotector.hashing import sha256_file
from geroprotector.validation import split_registry as split_module
from geroprotector.validation.nested_cv import (
    _model_name,
    _model_specs,
    _scientific_seed_family,
)
from geroprotector.validation.split_registry import (
    SplitIntegrityError,
    build_split_registry,
    load_verified_registries,
)

ROOT = Path(__file__).resolve().parents[1]


def _freeze_manifest_module():
    source = ROOT / "scripts" / "freeze_package_manifest.py"
    spec = importlib.util.spec_from_file_location("freeze_package_manifest_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _balanced_positions(
    y: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    del groups, seed
    labels = y.to_numpy(dtype=int)
    assignment = np.empty(len(labels), dtype=int)
    for label in (0, 1):
        positions = np.flatnonzero(labels == label)
        assignment[positions] = np.arange(len(positions)) % n_splits
    all_positions = np.arange(len(labels))
    return [
        (all_positions[assignment != fold], all_positions[assignment == fold])
        for fold in range(n_splits)
    ]


def _small_curated_cohort() -> pd.DataFrame:
    rows = []
    for index in range(8):
        identity = f"SYNTHETICKEY{index:02d}"
        rows.append(
            {
                "compound_id": compound_id(identity),
                "connectivity_inchikey": identity,
                "identity_group_id": identity,
                "standardized_parent_smiles": "CCO" if index % 2 else "CCC",
                "label": index % 2,
            }
        )
    return pd.DataFrame(rows)


def test_split_registry_bundle_is_atomic_hash_bound_and_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curated = _small_curated_cohort()
    curated_path = tmp_path / "curated.parquet"
    curated.to_parquet(curated_path, index=False)
    data_manifest = tmp_path / "data_manifest.json"
    data_manifest.write_text('{"fixture":true}\n', encoding="utf-8")

    def unique_components(smiles, *, compound_ids, **kwargs):
        del smiles, kwargs
        ids = tuple(map(str, compound_ids))
        return pd.Series(
            [f"component::{value}" for value in ids],
            index=ids,
            name="component_id",
            dtype="string",
        )

    def orthogonal_vectors(smiles, **kwargs):
        del kwargs
        output = []
        for index, _ in enumerate(smiles):
            vector = DataStructs.ExplicitBitVect(64)
            vector.SetBit(index)
            output.append(vector)
        return output

    monkeypatch.setattr(split_module, "similarity_components", unique_components)
    monkeypatch.setattr(split_module, "morgan_bitvectors", orthogonal_vectors)
    monkeypatch.setattr(split_module, "_sgkf", _balanced_positions)

    bundle = tmp_path / "split_bundle"
    outer_path = bundle / "outer.parquet"
    inner_path = bundle / "inner.parquet"
    manifest_path = bundle / "manifest.json"
    config_hash = "a" * 64
    manifest = build_split_registry(
        curated,
        output_path=outer_path,
        inner_output_path=inner_path,
        manifest_path=manifest_path,
        curated_table_sha256=sha256_file(curated_path),
        data_manifest_path=data_manifest,
        resolved_config_sha256=config_hash,
        fingerprint_config={
            "family": "morgan_bit",
            "radius": 2,
            "n_bits": 2048,
            "use_chirality": False,
        },
        edge_threshold=0.40,
        outer_repeats=1,
        outer_folds=2,
        outer_seeds=(41,),
        inner_folds=2,
        inner_seeds=(101, 102),
    )
    assert manifest["registry_parquet_sha256"] == sha256_file(outer_path)
    assert manifest["inner_registry_parquet_sha256"] == sha256_file(inner_path)
    assert not list(tmp_path.glob(".split_bundle.work-*"))

    outer, inner, loaded = load_verified_registries(
        outer_path=outer_path,
        inner_path=inner_path,
        manifest_path=manifest_path,
        curated_table_path=curated_path,
        data_manifest_path=data_manifest,
        resolved_config_sha256=config_hash,
        outer_repeats=1,
        outer_folds=2,
        inner_folds=2,
    )
    assert len(outer) == 8
    assert len(inner) == 8
    assert loaded["canonical_sha256"] == manifest["canonical_sha256"]

    with pytest.raises(FileExistsError, match="overwrite split bundle"):
        build_split_registry(
            curated,
            output_path=outer_path,
            inner_output_path=inner_path,
            manifest_path=manifest_path,
            curated_table_sha256=sha256_file(curated_path),
            data_manifest_path=data_manifest,
            resolved_config_sha256=config_hash,
            fingerprint_config={
                "family": "morgan_bit",
                "radius": 2,
                "n_bits": 2048,
                "use_chirality": False,
            },
            edge_threshold=0.40,
            outer_repeats=1,
            outer_folds=2,
            outer_seeds=(41,),
            inner_folds=2,
            inner_seeds=(101, 102),
        )

    outer_path.with_suffix(".csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(SplitIntegrityError, match="CSV byte hash"):
        load_verified_registries(
            outer_path=outer_path,
            inner_path=inner_path,
            manifest_path=manifest_path,
            curated_table_path=curated_path,
            data_manifest_path=data_manifest,
            resolved_config_sha256=config_hash,
            outer_repeats=1,
            outer_folds=2,
            inner_folds=2,
        )


def test_every_implemented_config_matches_its_protocol_lock() -> None:
    for contract, filename in (
        ("data", "data.yaml"),
        ("validation", "validation.yaml"),
        ("reference", "reference.yaml"),
        ("v5", "v5.yaml"),
        ("v6", "v6.yaml"),
        ("paper80", "paper80.yaml"),
        ("v5bis", "v5bis.yaml"),
        ("v6bis", "v6bis.yaml"),
    ):
        validate_protocol_lock(
            root=ROOT,
            contract_name=contract,
            config=resolve_config(ROOT / "configs" / filename),
        )

    lock = json.loads((ROOT / "configs" / "protocol_lock.json").read_text())
    assert set(lock["contracts"]) == {
        "data",
        "validation",
        "reference",
        "v5",
        "v6",
        "paper80",
        "v5bis",
        "v6bis",
    }


def test_core_runtime_is_exactly_pinned_and_rdkit_is_recorded(tmp_path: Path) -> None:
    observed = validate_core_runtime(ROOT / "requirements-lock.txt")
    assert observed["rdkit"] == runtime_environment()["packages"]["rdkit"]
    changed = (
        (ROOT / "requirements-lock.txt").read_text().replace("numpy==2.0.2", "numpy==0.0.0")
    )
    mismatch = tmp_path / "requirements-lock.txt"
    mismatch.write_text(changed, encoding="utf-8")
    with pytest.raises(RuntimeLockError, match="numpy"):
        validate_core_runtime(mismatch)


def test_source_tree_hash_covers_protocol_and_code_but_excludes_outputs(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "outputs").mkdir()
    source = tmp_path / "src" / "model.py"
    protocol = tmp_path / "configs" / "protocol_lock.json"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    protocol.write_text('{"locked":true}\n', encoding="utf-8")
    initial = source_tree_sha256(tmp_path)

    (tmp_path / "outputs" / "result.json").write_text("{}\n", encoding="utf-8")
    assert source_tree_sha256(tmp_path) == initial
    protocol.write_text('{"locked":false}\n', encoding="utf-8")
    assert source_tree_sha256(tmp_path) != initial

    unsafe = tmp_path / "src" / "linked.py"
    unsafe.symlink_to(source)
    with pytest.raises(ValueError, match="unsafe"):
        source_tree_files(tmp_path)


def test_package_freeze_excludes_generated_trees_and_rejects_source_symlinks(
    tmp_path: Path,
) -> None:
    freeze = _freeze_manifest_module()
    generated_directories = (
        "artifacts",
        "outputs",
        "checkpoints",
        "licenses",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "__pycache__",
    )
    for directory in generated_directories:
        path = tmp_path / directory
        path.mkdir()
        (path / "must-not-freeze.bin").write_bytes(b"generated")
    source = tmp_path / "source.txt"
    source.write_text("locked source\n", encoding="utf-8")
    (tmp_path / "compiled.pyc").write_bytes(b"cache")
    (tmp_path / freeze.CONTENTS_NAME).write_text("old contents\n", encoding="utf-8")
    (tmp_path / freeze.MANIFEST_NAME).write_text("old manifest\n", encoding="utf-8")

    without_contents = {
        path.relative_to(tmp_path).as_posix()
        for path in freeze._source_files(tmp_path, include_contents=False)
    }
    with_contents = {
        path.relative_to(tmp_path).as_posix()
        for path in freeze._source_files(tmp_path, include_contents=True)
    }
    assert without_contents == {"source.txt"}
    assert with_contents == {freeze.CONTENTS_NAME, "source.txt"}
    assert b"must-not-freeze.bin" not in freeze._contents_bytes(tmp_path)
    assert b"must-not-freeze.bin" not in freeze._manifest_bytes(tmp_path)

    (tmp_path / "unsafe-link.txt").symlink_to(source)
    with pytest.raises(RuntimeError, match="cannot contain symlinks"):
        freeze._source_files(tmp_path, include_contents=True)


def test_cli_and_shell_entrypoints_are_static_safe_and_point_to_current_package() -> None:
    parsed = cli._parser().parse_args(
        [
            "--root",
            str(ROOT),
            "run",
            "plan",
            "--config",
            "configs/v5.yaml",
            "--suite",
            "full",
        ]
    )
    assert parsed.command == "run"
    assert parsed.run_command == "plan"
    assert parsed.suite == "full"

    for script in sorted((ROOT / "scripts").glob("*.sh")):
        result = subprocess.run(
            ["bash", "-n", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        source = script.read_text(encoding="utf-8")
        assert "set -euo pipefail" in source
        if script.name in {"status_v5.sh", "status_bis.sh"}:
            assert "JOB_COMPLETED.json" in source
            assert "pgrep -af" in source
        else:
            # The original V5/V6 launchers use the shared CLI. Later locked
            # benchmark, ablation, endpoint-transport, environment-check, and
            # telemetry launchers intentionally use dedicated package modules,
            # inline read-only checks, or versioned Python runner scripts.
            # Requiring every later launcher to mention geroprotector.cli made
            # this test reject valid entrypoints as the repository expanded.
            valid_entrypoint_markers = (
                "geroprotector.cli",
                "-m geroprotector.",
                '"$PYTHON_BIN" -',
                '"$PYTHON_BIN" "scripts/',
                '"$PYTHON_BIN" "$ROOT/scripts/',
                '"$PYTHON_BIN" scripts/',
                'bash "$ROOT/scripts/',
            )
            assert any(marker in source for marker in valid_entrypoint_markers), script.name
        assert "aging-q1" not in source.lower()

    # Only the internal V5/V6/reference launch surface is required to remain
    # independent of retrospective external cohorts. Dedicated DrugAge and
    # AgeXtend scripts are expected elsewhere in scripts/.
    for name in ("run_v5.sh", "run_v6.sh", "run_references.sh"):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8").lower()
        assert "hagr" not in source
        assert "drugage" not in source

    for pipeline in ("v5", "v6", "references"):
        source = (ROOT / "scripts" / f"run_{pipeline}.sh").read_text(encoding="utf-8")
        assert source.index("run plan") < source.index("run nested-cv")


def test_full_suites_cover_declared_v5_and_v6_ablation_families() -> None:
    v5 = resolve_config(ROOT / "configs" / "v5.yaml")
    v5_core = {_model_name(spec) for spec in _model_specs(v5, suite="core")}
    v5_full = {_model_name(spec) for spec in _model_specs(v5, suite="full")}
    assert v5_core <= v5_full
    expected_v5 = {
        "R5_paper_seven_descriptor_linear_svm",
        "R2_extra_trees_v3_compatible",
        "v5_raw_best_individual_fingerprint",
        "v5_raw_unweighted_concatenation",
        "v5_selected_lengths_unweighted",
        "v5_weighted_no_reduction",
        "v5_weighted_svd",
        "v5_weighted_nystroem",
        "v5_final_v5",
    }
    assert expected_v5 == v5_full

    v6 = resolve_config(ROOT / "configs" / "v6.yaml")
    v6_core = {_model_name(spec) for spec in _model_specs(v6, suite="core")}
    v6_full = {_model_name(spec) for spec in _model_specs(v6, suite="full")}
    assert v6_core <= v6_full
    assert {
        "R2_extra_trees_v3_compatible",
        "v6_baseline_elastic_net",
        "v6_baseline_extra_trees",
        "v6_baseline_xgboost",
        "v6_foundation_selected",
    } <= v6_core
    assert {
        "v6_baseline_elastic_net_fixed_chemistry_32",
        "v6_baseline_extra_trees_fixed_chemistry_32",
        "v6_baseline_xgboost_fixed_chemistry_32",
    } <= v6_full
    assert not any("_fixed_" in model_id for model_id in v6_core)
    enabled = {
        name for name, settings in v6["models"].items() if settings.get("enabled") is True
    }
    assert {f"v6_foundation_checkpoint_{name}" for name in enabled} <= v6_full
    assert {
        "v6_foundation_panel_rdkit2d_217",
        "v6_foundation_panel_morgan_svd_plus_descriptors",
        "v6_foundation_ensemble_1",
        "v6_foundation_ensemble_16",
    } <= v6_full
    anchor = v6["inference"]["controlled_ablation_anchor"]
    assert {
        f"v6_baseline_{kind}_fixed_{anchor['panel_family']}"
        for kind in ("elastic_net", "extra_trees", "xgboost")
    } <= v6_full
    full_specs = {_model_name(spec): spec for spec in _model_specs(v6, suite="full")}
    for model_id in enabled:
        spec = full_specs[f"v6_foundation_checkpoint_{model_id}"]
        assert spec["fixed_model_id"] == model_id
        assert spec["fixed_panel_family"] == anchor["panel_family"]
        assert spec["fixed_ensemble_size"] == anchor["ensemble_size"]
    for panel in ("rdkit2d_217", "morgan_svd_plus_descriptors"):
        spec = full_specs[f"v6_foundation_panel_{panel}"]
        assert spec["fixed_model_id"] == anchor["model_id"]
        assert spec["fixed_panel_family"] == panel
        assert spec["fixed_ensemble_size"] == anchor["ensemble_size"]
    assert "v6_foundation_panel_chemistry_32" not in full_specs
    anchor_spec = full_specs[f"v6_foundation_checkpoint_{anchor['model_id']}"]
    assert anchor_spec["fixed_panel_family"] == anchor["panel_family"]
    assert anchor_spec["fixed_ensemble_size"] == anchor["ensemble_size"]
    for size in (1, 16):
        spec = full_specs[f"v6_foundation_ensemble_{size}"]
        assert spec["fixed_model_id"] == anchor["model_id"]
        assert spec["fixed_panel_family"] == anchor["panel_family"]
        assert spec["fixed_ensemble_size"] == size

    v5_specs = _model_specs(v5, suite="full")
    assert {_scientific_seed_family(spec) for spec in v5_specs if spec["kind"] == "v5"} == {
        "v5_controlled_ablation"
    }
    v6_specs = _model_specs(v6, suite="full")
    assert {_scientific_seed_family(spec) for spec in v6_specs if spec["kind"] == "v6"} == {
        "v6_foundation_controlled_ablation"
    }
