from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geroprotector import audit as audit_module
from geroprotector import config as config_module
from geroprotector.chemistry.standardize import rdkit_version
from geroprotector.config import (
    ConfigError,
    resolve_config,
    resolved_config_sha256,
    unresolved_placeholders,
    validate_locked_shared_contract,
    validate_resolved_config,
)
from geroprotector.data import acquire as acquire_module
from geroprotector.data.acquire import AcquisitionError, SourceSpec, acquire_sources
from geroprotector.data.curate import CurationError, _decision_key, load_curated_cohort
from geroprotector.data.identities import compound_id, raw_row_id
from geroprotector.hashing import (
    atomic_write_bytes,
    canonical_sha256,
    sha256_bytes,
    sha256_file,
)
from geroprotector.validation.leakage_checks import (
    LeakageError,
    assert_fit_scope,
    assert_internal_config_safe,
    assert_label_blind,
    assert_no_hagr_paths,
)

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_v5_v6_configs_resolve_and_share_one_locked_protocol() -> None:
    v5 = resolve_config(ROOT / "configs/v5.yaml")
    v6 = resolve_config(ROOT / "configs/v6.yaml")

    validate_locked_shared_contract(v5)
    validate_locked_shared_contract(v6)
    assert v5["primary_split"] == v6["primary_split"]
    assert v5["inner_cv"] == v6["inner_cv"]
    assert v5["curation"] == v6["curation"]
    assert v5["roles"]["prior_ml_candidates"] == "unlabeled_only"
    assert unresolved_placeholders(v5) == []
    assert unresolved_placeholders(v6)
    assert resolved_config_sha256(v5) != resolved_config_sha256(v6)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["roles"].update(prior_ml_candidates="supervised_positive"),
            "unlabeled_only",
        ),
        (lambda value: value["primary_split"].update(edge_threshold=0.41), "Primary split"),
        (lambda value: value["curation"]["expected_counts"].update(total=383), "counts"),
        (
            lambda value: value["external_firewall"].update(hagr_development_access=True),
            "HAGR development access",
        ),
    ],
)
def test_locked_protocol_rejects_semantic_drift(mutation, message: str) -> None:
    value = copy.deepcopy(resolve_config(ROOT / "configs/v5.yaml"))
    mutation(value)
    with pytest.raises(ConfigError, match=message):
        validate_locked_shared_contract(value)


def test_disabled_runtime_placeholder_is_ignored_but_enabled_is_not() -> None:
    value = {
        "disabled": {"enabled": False, "path": "REQUIRED_IF_ENABLED"},
        "enabled": {"enabled": True, "path": "REQUIRED_AT_RUNTIME"},
    }
    assert unresolved_placeholders(value) == ["enabled.path"]
    with pytest.raises(ConfigError, match=r"enabled\.path"):
        validate_resolved_config(value)


def test_config_extends_is_recursive_but_cannot_escape_directory(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("a:\n  b: 1\n  c: 2\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\na:\n  c: 3\n", encoding="utf-8")
    assert resolve_config(tmp_path / "child.yaml")["a"] == {"b": 1, "c": 3}

    outside = tmp_path.parent / "outside.yaml"
    outside.write_text("x: 1\n", encoding="utf-8")
    (tmp_path / "escape.yaml").write_text("extends: ../outside.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must remain"):
        resolve_config(tmp_path / "escape.yaml")


def test_config_extends_cycle_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="cycle"):
        resolve_config(tmp_path / "a.yaml")


def test_internal_firewall_accepts_locked_config_and_rejects_external_variants() -> None:
    config = resolve_config(ROOT / "configs/v5.yaml")
    assert_internal_config_safe(config)

    for path in (
        "/tmp/HAGR2025.csv",
        "/tmp/drug_age.csv",
        "/tmp/DrugAge.tsv",
        "/tmp/external-2.parquet",
    ):
        unsafe = copy.deepcopy(config)
        unsafe["development_input"] = path
        with pytest.raises(LeakageError):
            assert_internal_config_safe(unsafe)
        with pytest.raises(LeakageError):
            assert_no_hagr_paths([path])


def test_internal_firewall_rejects_unknown_external_firewall_keys() -> None:
    config = resolve_config(ROOT / "configs/v5.yaml")
    config["external_firewall"]["hagr_labels_loaded_during_development"] = True
    with pytest.raises((LeakageError, ConfigError)):
        assert_internal_config_safe(config)


def test_fit_scope_and_label_blind_guards() -> None:
    assert_fit_scope(fit_ids=["a", "b"], transform_ids=["held"], forbidden_ids=["held"])
    with pytest.raises(LeakageError, match="forbidden"):
        assert_fit_scope(fit_ids=["a", "held"], transform_ids=["b"], forbidden_ids=["held"])
    with pytest.raises(LeakageError, match="unique"):
        assert_fit_scope(fit_ids=["a", "a"], transform_ids=[])
    assert_label_blind([{"evaluation_role": "external_label_blind", "y_true": None}])
    with pytest.raises(LeakageError, match="y_true"):
        assert_label_blind([{"evaluation_role": "external_label_blind", "y_true": 1}])


def test_identity_helpers_are_stable_and_source_bound() -> None:
    first = raw_row_id("reported_positive", 1, "A", "CC")
    assert first == raw_row_id("reported_positive", 1, "A", "CC")
    assert first != raw_row_id("weak_chembl_reference", 1, "A", "CC")
    assert first != raw_row_id("reported_positive", 2, "A", "CC")
    assert compound_id(" ABC ") == "cmp::ABC"


def test_manual_curation_decision_key_ignores_only_boundary_formatting() -> None:
    raw_name = " Euk-134 "
    raw_smiles = " COC1=CC=CC=C1.[Mn+3]  "
    assert _decision_key(raw_name, raw_smiles) == (
        "Euk-134",
        "COC1=CC=CC=C1.[Mn+3]",
    )
    # Exact raw provenance remains sensitive to the original source bytes.
    assert raw_row_id("reported_positive", 85, raw_name, raw_smiles) != raw_row_id(
        "reported_positive", 85, raw_name.strip(), raw_smiles.strip()
    )


def test_runtime_gate_rejects_imported_or_duplicate_distribution_version_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_module_version = audit_module._module_version
    actual_distribution_versions = audit_module._distribution_versions

    monkeypatch.setattr(
        audit_module,
        "_module_version",
        lambda package: "2.3.3" if package == "pandas" else actual_module_version(package),
    )
    with pytest.raises(audit_module.RuntimeLockError, match=r"Imported.*pandas"):
        audit_module.validate_core_runtime(ROOT / "requirements-lock.txt")

    monkeypatch.setattr(audit_module, "_module_version", actual_module_version)
    for bad_versions in (("2.3.2", "2.3.3"), ("2.3.2", "2.3.2"), ()):
        monkeypatch.setattr(
            audit_module,
            "_distribution_versions",
            lambda package, values=bad_versions: (
                values if package == "pandas" else actual_distribution_versions(package)
            ),
        )
        with pytest.raises(
            audit_module.RuntimeLockError, match=r"metadata is ambiguous.*pandas"
        ):
            audit_module.validate_core_runtime(ROOT / "requirements-lock.txt")


def _write_sealed_curation_bundle(directory: Path, *, config_hash: str = "cfg"):
    directory.mkdir()
    curated = pd.DataFrame(
        {
            "compound_id": ["cmp::A", "cmp::B", "cmp::C", "cmp::D"],
            "identity_group_id": ["A", "B", "C", "D"],
            "label": [0, 1, 0, 1],
        }
    )
    provenance = pd.DataFrame(
        {
            "raw_row_id": [f"raw::{index}" for index in range(1893)],
            "is_prior_ml_candidate": [True] * 1488 + [False] * 405,
            "label": [np.nan] * 1488 + [0.0, 1.0] * 202 + [0.0],
        }
    )
    curated_path = directory / "compounds.parquet"
    provenance_path = directory / "provenance.parquet"
    conflicts_path = directory / "identity_conflicts.csv"
    ledger_path = directory / "standardization_ledger.parquet"
    curated.to_parquet(curated_path, index=False)
    provenance.to_parquet(provenance_path, index=False)
    conflicts_path.write_text("raw_row_id,reason\n", encoding="utf-8")
    provenance.to_parquet(ledger_path, index=False)
    identity_hash = canonical_sha256(
        {
            "schema": "geroprotector.curated_identity_label.v1",
            "rows": curated[["identity_group_id", "label"]]
            .sort_values("identity_group_id", kind="stable")
            .to_dict(orient="records"),
        }
    )
    manifest = {
        "schema_version": "geroprotector.curation.v1",
        "resolved_config_sha256": config_hash,
        "rdkit_version": rdkit_version(),
        "curated_counts": {"total": 4, "positive": 2, "weak_reference": 2},
        "curated_identity_label_sha256": identity_hash,
        "artifacts": {
            "curated_table": {
                "path": str(curated_path),
                "sha256": sha256_file(curated_path),
            },
            "provenance_table": {
                "path": str(provenance_path),
                "sha256": sha256_file(provenance_path),
            },
            "identity_conflicts": {
                "path": str(conflicts_path),
                "sha256": sha256_file(conflicts_path),
            },
            "standardization_ledger": {
                "path": str(ledger_path),
                "sha256": sha256_file(ledger_path),
            },
        },
    }
    manifest["canonical_sha256"] = canonical_sha256(manifest)
    manifest_path = directory / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return curated_path, provenance_path, manifest_path


def test_curated_loader_rebinds_bytes_membership_config_and_prior_labels(
    tmp_path: Path,
) -> None:
    curated, provenance, manifest = _write_sealed_curation_bundle(tmp_path / "bundle")
    loaded, raw, sealed = load_curated_cohort(
        curated_table=curated,
        provenance_table=provenance,
        manifest_path=manifest,
        resolved_config_sha256="cfg",
    )
    assert len(loaded) == 4
    assert int(raw.loc[raw["is_prior_ml_candidate"], "label"].notna().sum()) == 0
    assert sealed["curated_counts"]["total"] == 4

    with pytest.raises(CurationError, match="different resolved config"):
        load_curated_cohort(
            curated_table=curated,
            provenance_table=provenance,
            manifest_path=manifest,
            resolved_config_sha256="other",
        )


def test_curated_loader_rejects_tampered_table(tmp_path: Path) -> None:
    curated, provenance, manifest = _write_sealed_curation_bundle(tmp_path / "bundle")
    changed = pd.read_parquet(curated)
    changed.loc[0, "label"] = 1
    changed.to_parquet(curated, index=False)
    with pytest.raises(CurationError, match="byte hash"):
        load_curated_cohort(
            curated_table=curated,
            provenance_table=provenance,
            manifest_path=manifest,
            resolved_config_sha256="cfg",
        )


@pytest.mark.parametrize(
    "artifact_name",
    ["identity_conflicts.csv", "standardization_ledger.parquet"],
)
def test_curated_loader_rehashes_every_atomic_bundle_artifact(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    curated, provenance, manifest = _write_sealed_curation_bundle(tmp_path / "bundle")
    artifact = manifest.parent / artifact_name
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(CurationError, match="byte hash"):
        load_curated_cohort(
            curated_table=curated,
            provenance_table=provenance,
            manifest_path=manifest,
            resolved_config_sha256="cfg",
        )


def test_acquisition_is_local_hash_pinned_and_existing_manifest_must_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"Name,smiles\nexample,CC\n"
    source = tmp_path / "source.csv"
    source.write_bytes(payload)
    spec = SourceSpec(
        source_role="prior_ml_unlabeled",
        relative_path="fixture.csv",
        cache_name="fixture.csv",
        sha256=sha256_bytes(payload),
        rows=1,
        unique_smiles=1,
        separator=",",
        encoding="utf-8",
        name_column="Name",
        smiles_column="smiles",
        supervised_label=None,
    )
    monkeypatch.setattr(acquire_module, "SOURCE_SPECS", (spec,))
    monkeypatch.setattr(acquire_module, "UPSTREAM_COMMIT", "fixture-commit")
    contract = {
        "prior_ml_unlabeled": {
            "cache_name": "fixture.csv",
            "sha256": spec.sha256,
            "rows": 1,
            "unique_smiles": 1,
            "delimiter": "comma",
            "encoding": "utf-8",
            "name_column": "Name",
            "smiles_column": "smiles",
        }
    }
    cache = tmp_path / "cache"
    manifest = acquire_sources(
        cache,
        allow_network=False,
        source_config=contract,
        expected_upstream_commit="fixture-commit",
        local_sources={"prior_ml_unlabeled": source},
    )
    assert manifest["sources"][0]["verified_sha256"] == spec.sha256

    stale = dict(manifest)
    stale["sources"] = [{"source_role": "fabricated"}]
    stale.pop("canonical_sha256")
    stale["canonical_sha256"] = canonical_sha256(stale)
    (cache / "acquisition_manifest.json").write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(AcquisitionError, match="manifest"):
        acquire_sources(
            cache,
            allow_network=False,
            source_config=contract,
            expected_upstream_commit="fixture-commit",
        )


def test_atomic_artifact_write_never_silently_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "sealed.bin"
    atomic_write_bytes(target, b"first")
    with pytest.raises(FileExistsError):
        atomic_write_bytes(target, b"second")
    assert target.read_bytes() == b"first"


def test_regular_config_loader_rejects_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real.yaml"
    real.write_text("x: 1\n", encoding="utf-8")
    link = tmp_path / "link.yaml"
    link.symlink_to(real)
    with pytest.raises(ConfigError, match="symlink"):
        config_module.resolve_config(link)
