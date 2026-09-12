from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from geroprotector import cli
from geroprotector.data.curate import CurationError
from geroprotector.hashing import sha256_file
from geroprotector.validation.split_registry import SplitIntegrityError


def _data_config(root: Path) -> dict[str, Any]:
    overrides = root / "configs" / "curation_overrides.csv"
    metals = root / "configs" / "metal_adjudication.csv"
    overrides.parent.mkdir(parents=True, exist_ok=True)
    overrides.write_text("identity_group_id,decision\n", encoding="utf-8")
    metals.write_text("identity_group_id,decision\n", encoding="utf-8")
    return {
        "project": {"name": "synthetic"},
        "upstream": {"commit": "fixture", "sources": {}},
        "roles": {"prior_ml_candidates": "unlabeled_only"},
        "curation": {
            "curation_overrides": "configs/curation_overrides.csv",
            "curation_overrides_sha256": sha256_file(overrides),
            "metal_adjudication": "configs/metal_adjudication.csv",
            "metal_adjudication_sha256": sha256_file(metals),
            "expected_counts": {},
            "expected_identity_label_sha256": "fixture",
            "rdkit_version_pin": "fixture",
        },
        "outputs": {
            "raw_dir": "artifacts/raw",
            "curated_table": "artifacts/stage0/curated.parquet",
            "provenance_table": "artifacts/stage0/provenance.parquet",
            "identity_conflicts": "artifacts/stage0/conflicts.csv",
            "standardization_ledger": "artifacts/stage0/ledger.parquet",
            "manifest": "artifacts/stage0/data_manifest.json",
        },
    }


def _validation_config() -> dict[str, Any]:
    return {
        "primary_split": {"outer_repeats": 1, "outer_folds": 2},
        "inner_cv": {"folds": 2},
        "registry_outputs": {
            "outer": "artifacts/stage0/outer.parquet",
            "inner": "artifacts/stage0/inner.parquet",
            "manifest": "artifacts/stage0/split_manifest.json",
        },
    }


def _disable_cli_environment_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "validate_core_runtime", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(cli, "validate_protocol_lock", lambda *_args, **_kwargs: None)


def test_cli_curate_reuses_an_existing_verified_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _data_config(tmp_path)
    manifest = tmp_path / config["outputs"]["manifest"]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('{"sealed":true}\n', encoding="utf-8")
    _disable_cli_environment_guards(monkeypatch)
    monkeypatch.setattr(cli, "resolve_config", lambda _path: config)

    calls: list[dict[str, Any]] = []

    def verified_loader(**kwargs: Any):
        calls.append(kwargs)
        return (
            pd.DataFrame({"compound_id": ["cmp::A"]}),
            pd.DataFrame({"raw_row_id": ["raw::A"]}),
            {"schema_version": "geroprotector.curation.v1", "sealed": True},
        )

    monkeypatch.setattr(cli, "load_curated_cohort", verified_loader)
    monkeypatch.setattr(
        cli,
        "curate_cohort",
        lambda **_kwargs: pytest.fail("a verified bundle must not be rebuilt"),
    )

    assert cli.main(["--root", str(tmp_path), "data", "curate"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage0_resume_status"] == "existing_curated_bundle_verified"
    assert payload["sealed"] is True
    assert len(calls) == 1
    assert calls[0]["manifest_path"] == manifest


def test_cli_curate_rejects_an_existing_tampered_bundle_without_rebuilding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _data_config(tmp_path)
    manifest = tmp_path / config["outputs"]["manifest"]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('{"sealed":true}\n', encoding="utf-8")
    _disable_cli_environment_guards(monkeypatch)
    monkeypatch.setattr(cli, "resolve_config", lambda _path: config)
    monkeypatch.setattr(
        cli,
        "load_curated_cohort",
        lambda **_kwargs: (_ for _ in ()).throw(
            CurationError("Curated artifact byte hash differs")
        ),
    )
    monkeypatch.setattr(
        cli,
        "curate_cohort",
        lambda **_kwargs: pytest.fail("a tampered bundle must not be rebuilt"),
    )

    with pytest.raises(CurationError, match="byte hash"):
        cli.main(["--root", str(tmp_path), "data", "curate"])


def test_cli_split_build_reuses_an_existing_verified_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_config = _data_config(tmp_path)
    validation_config = _validation_config()
    split_manifest = tmp_path / validation_config["registry_outputs"]["manifest"]
    split_manifest.parent.mkdir(parents=True, exist_ok=True)
    split_manifest.write_text('{"sealed":true}\n', encoding="utf-8")
    _disable_cli_environment_guards(monkeypatch)
    monkeypatch.setattr(
        cli,
        "resolve_config",
        lambda path: validation_config if Path(path).name == "validation.yaml" else data_config,
    )
    monkeypatch.setattr(
        cli,
        "load_curated_cohort",
        lambda **_kwargs: (
            pd.DataFrame({"compound_id": ["cmp::A", "cmp::B"]}),
            pd.DataFrame(),
            {"sealed": True},
        ),
    )
    calls: list[dict[str, Any]] = []

    def verified_loader(**kwargs: Any):
        calls.append(kwargs)
        return (
            pd.DataFrame({"compound_id": ["cmp::A", "cmp::B"]}),
            pd.DataFrame({"compound_id": ["cmp::A", "cmp::B"]}),
            {
                "schema_version": "geroprotector.split_registry.v1",
                "registry_sha256": "fixture-registry",
            },
        )

    monkeypatch.setattr(cli, "load_verified_registries", verified_loader)
    monkeypatch.setattr(
        cli,
        "build_split_registry",
        lambda *_args, **_kwargs: pytest.fail("a verified split bundle must not be rebuilt"),
    )

    assert cli.main(["--root", str(tmp_path), "splits", "build"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage0_resume_status"] == "existing_split_bundle_verified"
    assert payload["outer_rows"] == 2
    assert payload["inner_rows"] == 2
    assert len(calls) == 1
    assert calls[0]["manifest_path"] == split_manifest


def test_cli_split_build_rejects_an_existing_tampered_bundle_without_rebuilding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_config = _data_config(tmp_path)
    validation_config = _validation_config()
    split_manifest = tmp_path / validation_config["registry_outputs"]["manifest"]
    split_manifest.parent.mkdir(parents=True, exist_ok=True)
    split_manifest.write_text('{"sealed":true}\n', encoding="utf-8")
    _disable_cli_environment_guards(monkeypatch)
    monkeypatch.setattr(
        cli,
        "resolve_config",
        lambda path: validation_config if Path(path).name == "validation.yaml" else data_config,
    )
    monkeypatch.setattr(
        cli,
        "load_curated_cohort",
        lambda **_kwargs: (pd.DataFrame(), pd.DataFrame(), {"sealed": True}),
    )
    monkeypatch.setattr(
        cli,
        "load_verified_registries",
        lambda **_kwargs: (_ for _ in ()).throw(
            SplitIntegrityError("Split registry parquet byte hash differs")
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_split_registry",
        lambda *_args, **_kwargs: pytest.fail("a tampered split bundle must not be rebuilt"),
    )

    with pytest.raises(SplitIntegrityError, match="byte hash"):
        cli.main(["--root", str(tmp_path), "splits", "build"])
