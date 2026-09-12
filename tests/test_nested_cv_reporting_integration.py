from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import jsonschema
import numpy as np
import pandas as pd
import pytest

from geroprotector import reporting
from geroprotector.hashing import canonical_sha256, sha256_file
from geroprotector.reporting import ReportingError, report_internal
from geroprotector.validation import nested_cv
from geroprotector.validation.nested_cv import (
    NestedCVError,
    run_nested_cv,
    verify_completed_run,
)
from geroprotector.validation.split_registry import (
    INNER_REGISTRY_COLUMNS,
    REGISTRY_COLUMNS,
)

ROOT = Path(__file__).resolve().parents[1]


class _FakeStore:
    def __init__(self, ids: Sequence[str], *, store_hash: str = "fake-store") -> None:
        self.ids = tuple(map(str, ids))
        self.store_hash = store_hash

    def subset(self, ids: Sequence[str]) -> _FakeStore:
        requested = tuple(map(str, ids))
        if not set(requested).issubset(self.ids):
            raise AssertionError("Runner requested IDs outside the molecule-local store")
        return _FakeStore(requested, store_hash=self.store_hash)


class _FakePipeline:
    """Fast spy model: its score is a deterministic function of the compound ID only."""

    initialised_seeds: ClassVar[list[int]] = []
    prediction_scopes: ClassVar[list[tuple[tuple[str, ...], tuple[str, ...]]]] = []

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.initialised_seeds.append(self.seed)

    def fit(
        self,
        store: _FakeStore,
        y: pd.Series,
        *,
        groups: pd.Series,
        selection_folds: Sequence[tuple[tuple[str, ...], tuple[str, ...]]],
    ) -> _FakePipeline:
        self.fit_ids_ = tuple(store.ids)
        assert set(self.fit_ids_) == set(y.index.astype(str))
        assert set(self.fit_ids_) == set(groups.index.astype(str))
        validation_ids = [value for _, valid in selection_folds for value in valid]
        assert len(validation_ids) == len(set(validation_ids)) == len(self.fit_ids_)
        assert set(validation_ids) == set(self.fit_ids_)
        for train, valid in selection_folds:
            assert set(train).isdisjoint(valid)
            assert set(train) | set(valid) == set(self.fit_ids_)
        self.selection_trace_ = pd.DataFrame(
            [
                {
                    "candidate_id": canonical_sha256({"kind": "fake", "seed": self.seed}),
                    "candidate_spec": {"kind": "fake"},
                    "status": "success",
                    "ap_positive": 1.0,
                    "auroc": 1.0,
                    "brier": 0.04,
                    "selected": True,
                    "outer_test_metric_consulted": False,
                    "hagr_metric_consulted": False,
                }
            ]
        )
        return self

    def predict_proba(self, store: _FakeStore, ids: Sequence[str]) -> np.ndarray:
        requested = tuple(map(str, ids))
        assert set(requested).issubset(store.ids)
        assert set(requested).isdisjoint(self.fit_ids_)
        self.prediction_scopes.append((self.fit_ids_, requested))
        positive = np.asarray(
            [0.8 if "::p" in compound else 0.2 for compound in requested], dtype=float
        )
        return np.column_stack([1.0 - positive, positive])

    def get_selection_trace(self) -> pd.DataFrame:
        return self.selection_trace_.copy()

    def get_manifest(self) -> dict[str, Any]:
        return {
            "pipeline": "reference",
            "model_id": "R0_prevalence",
            "winner": {"kind": "fake"},
            "importance": {
                "stability_spearman": {"fold_0": 0.75, "fold_1": 0.85},
            },
            "representation": {
                "stability_fallback_applied": False,
                "embedding": {"n_components": 8},
            },
            "panel": {"panel": "chemistry_32"},
            "model": {
                "model_id": "fake_checkpoint",
                "package": "fake-foundation-package",
                "package_version": "1.0.0",
                "explicit_model_version": "fake-v1",
                "checkpoint_sha256": "a" * 64,
                "license_sha256": "b" * 64,
                "checkpoint_source": "local synthetic fixture",
                "access_date_utc": "2026-08-15",
            },
            "fit_ids": list(self.fit_ids_),
            "outer_test_metric_consulted": False,
            "hagr_metric_consulted": False,
        }

    def save(self, path: str | Path) -> dict[str, Any]:
        destination = Path(path)
        destination.write_bytes(f"fake-model-seed={self.seed}\n".encode())
        manifest = self.get_manifest()
        manifest["model_artifact_sha256"] = sha256_file(destination)
        return manifest


def _synthetic_cohort_and_registries() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    ordinal = 0
    for label, token, count in ((0, "n", 180), (1, "p", 202)):
        for index in range(count):
            compound = f"cmp::{token}{index:03d}"
            first_block = (
                "A" * 12 + chr(ord("A") + (ordinal // 26) % 26) + chr(ord("A") + ordinal % 26)
            )
            rows.append(
                {
                    "compound_id": compound,
                    "label": label,
                    "identity_group_id": f"identity::{token}{index:02d}",
                    "full_inchikey": f"{first_block}-ABCDEFGHIJ-K",
                    "standardized_parent_smiles": "CCO" if label else "CCC",
                    "murcko_scaffold_smiles": "" if index % 2 else "C1CCCCC1",
                    "source_role": ("reported_positive" if label else "weak_chembl_reference"),
                }
            )
            ordinal += 1
    curated = (
        pd.DataFrame(rows).sort_values("compound_id", kind="stable").reset_index(drop=True)
    )
    outer_rows: list[dict[str, Any]] = []
    for repeat in range(5):
        for label in (0, 1):
            label_ids = sorted(curated.loc[curated["label"].eq(label), "compound_id"])
            for position, compound in enumerate(label_ids):
                outer_rows.append(
                    {
                        "compound_id": compound,
                        "label": label,
                        "component_id": f"component::{compound}",
                        "repeat": repeat,
                        "outer_fold": (position + repeat) % 5,
                        "role": "outer_test",
                    }
                )
    outer = pd.DataFrame(outer_rows, columns=REGISTRY_COLUMNS)
    inner_rows: list[dict[str, Any]] = []
    for repeat in range(5):
        repeated = outer.loc[outer["repeat"].eq(repeat)]
        for outer_fold in range(5):
            outer_train = repeated.loc[~repeated["outer_fold"].eq(outer_fold)]
            assignment: dict[str, int] = {}
            for label in (0, 1):
                label_ids = sorted(
                    outer_train.loc[outer_train["label"].eq(label), "compound_id"]
                )
                assignment.update(
                    {
                        compound: (position + repeat + outer_fold) % 3
                        for position, compound in enumerate(label_ids)
                    }
                )
            for row in outer_train.itertuples(index=False):
                inner_rows.append(
                    {
                        "compound_id": row.compound_id,
                        "label": row.label,
                        "component_id": row.component_id,
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "inner_fold": assignment[row.compound_id],
                        "role": "inner_validation",
                        "assignment_seed": 101 + repeat * 5 + outer_fold,
                    }
                )
    return curated, outer, pd.DataFrame(inner_rows, columns=INNER_REGISTRY_COLUMNS)


def _balanced_folds(
    labels: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    del groups, seed
    assert n_splits >= 2
    assignment: dict[str, int] = {}
    for label in (0, 1):
        ids = sorted(labels.index[labels.astype(int).eq(label)].astype(str))
        assignment.update({compound: index % n_splits for index, compound in enumerate(ids)})
    universe = set(labels.index.astype(str))
    return [
        (
            tuple(sorted(value for value in universe if assignment[value] != fold)),
            tuple(sorted(value for value in universe if assignment[value] == fold)),
        )
        for fold in range(n_splits)
    ]


def _patch_fast_run(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    paper80: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curated, outer, inner = _synthetic_cohort_and_registries()
    if paper80:
        test_ids = set(
            sorted(curated.loc[curated["label"].eq(0), "compound_id"])[:29]
            + sorted(curated.loc[curated["label"].eq(1), "compound_id"])[:48]
        )
        outer = pd.DataFrame(
            [
                {
                    "compound_id": row.compound_id,
                    "label": int(row.label),
                    "component_id": f"component::{row.compound_id}",
                    "repeat": 0,
                    "outer_fold": 0 if row.compound_id in test_ids else 1,
                    "role": "paper_test" if row.compound_id in test_ids else "paper_train",
                }
                for row in curated.itertuples(index=False)
            ],
            columns=REGISTRY_COLUMNS,
        )
        train = outer.loc[outer["role"].eq("paper_train")].copy()
        assignment: dict[str, int] = {}
        for label in (0, 1):
            values = sorted(train.loc[train["label"].eq(label), "compound_id"])
            assignment.update({value: index % 3 for index, value in enumerate(values)})
        inner = pd.DataFrame(
            [
                {
                    "compound_id": row.compound_id,
                    "label": int(row.label),
                    "component_id": row.component_id,
                    "repeat": 0,
                    "outer_fold": 0,
                    "inner_fold": assignment[row.compound_id],
                    "role": "inner_validation",
                    "assignment_seed": 142,
                }
                for row in train.itertuples(index=False)
            ],
            columns=INNER_REGISTRY_COLUMNS,
        )
    config = {
        "pipeline": "V5BIS_PAPER80" if paper80 else "reference",
        "project": {"random_seed": 20260408},
        "primary_split": {"outer_repeats": 5, "outer_folds": 5},
        "inner_cv": {"folds": 3},
        "applicability": {
            "min_nearest_train_tanimoto": 0.20,
            "robust_descriptor_train_quantile": 0.975,
        },
        "outputs": {"manifest": "artifacts/data/data_manifest.json"},
        "registry_outputs": {"manifest": "artifacts/splits/split_manifest.json"},
    }
    if paper80:
        config.update(
            {
                "evaluation_design": {
                    "name": "paper_random_80_20",
                    "role": "retrospective_contextual_only",
                },
                "paper_registry_outputs": {
                    "manifest": "artifacts/paper80_splits/paper80_split_manifest.json"
                },
            }
        )
    data_manifest = {
        "curated_cohort_canonical_sha256": "a" * 64,
        "curated_identity_label_sha256": "b" * 64,
        "artifacts": {"provenance_table": {"sha256": "f" * 64}},
    }
    split_manifest = {
        "registry_sha256": "c" * 64,
        "inner_registry_sha256": "d" * 64,
    }
    if paper80:
        split_manifest.update(
            {
                "component_overlap_count": 18,
                "n_train": 305,
                "n_test": 77,
                "test_class_0": 29,
                "test_class_1": 48,
                "inner_components_fit_on_outer_train_only": True,
                "outer_test_structures_used_for_inner_grouping": False,
                "nearest_train_tanimoto_median": 0.29,
                "nearest_train_tanimoto_max": 0.84,
            }
        )
    data_path = root / config["outputs"]["manifest"]
    split_path = root / (
        config["paper_registry_outputs"]["manifest"]
        if paper80
        else config["registry_outputs"]["manifest"]
    )
    data_path.parent.mkdir(parents=True)
    split_path.parent.mkdir(parents=True)
    data_path.write_text("synthetic-data-manifest\n", encoding="utf-8")
    split_path.write_text("synthetic-split-manifest\n", encoding="utf-8")
    shutil.copytree(ROOT / "schemas", root / "schemas")
    shutil.copy2(ROOT / "requirements-lock.txt", root / "requirements-lock.txt")

    monkeypatch.setattr(nested_cv, "resolve_config", lambda path: dict(config))
    monkeypatch.setattr(nested_cv, "validate_locked_shared_contract", lambda value: None)
    monkeypatch.setattr(nested_cv, "assert_internal_config_safe", lambda value: None)
    monkeypatch.setattr(nested_cv, "validate_protocol_lock", lambda **kwargs: None)
    monkeypatch.setattr(nested_cv, "unresolved_placeholders", lambda value: [])
    monkeypatch.setattr(
        nested_cv,
        "_data_and_splits",
        lambda project, value: (curated, outer, inner, data_manifest, split_manifest),
    )
    monkeypatch.setattr(
        nested_cv,
        "_model_specs",
        lambda value, suite: ({"kind": "reference", "model_id": "R0_prevalence"},),
    )
    store = _FakeStore(curated["compound_id"])
    monkeypatch.setattr(nested_cv, "_build_store", lambda value, frame: store)
    monkeypatch.setattr(
        nested_cv,
        "_new_model",
        lambda spec, config, seed, root, checkpoint_ledger: _FakePipeline(seed),
    )
    monkeypatch.setattr(nested_cv, "grouped_inner_folds", _balanced_folds)
    monkeypatch.setattr(
        nested_cv,
        "outer_train_applicability",
        lambda frame, train_ids, query_ids, min_tanimoto, descriptor_quantile: pd.DataFrame(
            {
                "compound_id": list(query_ids),
                "nearest_train_tanimoto": [0.10] * len(query_ids),
                "robust_descriptor_distance": [1.0] * len(query_ids),
                "inside_applicability_domain": [False] * len(query_ids),
            }
        ),
    )
    monkeypatch.setattr(nested_cv, "source_tree_sha256", lambda project: "e" * 64)
    monkeypatch.setattr(
        nested_cv,
        "source_tree_files",
        lambda project: (Path(project) / "schemas" / "run_manifest.schema.json",),
    )
    monkeypatch.setattr(reporting, "source_tree_sha256", lambda project: "e" * 64)
    monkeypatch.setattr(
        reporting,
        "_source_confounding_audit",
        lambda root, run_manifest, aggregated, metric_records: (
            {
                "schema_version": "geroprotector.source_confounding_audit.v1",
                "nearest_neighbor_source_enrichment": {"enrichment_ratio": 1.0},
            },
            pd.DataFrame([{"descriptor": "MolWt"}]),
            pd.DataFrame(
                [
                    {
                        "positive_compound_id": "cmp::p000",
                        "weak_reference_compound_id": "cmp::n000",
                        "tanimoto": 0.25,
                    }
                ]
            ),
        ),
    )
    return curated, outer


def test_nested_run_is_leakage_scoped_seed_stable_sealed_and_reportable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curated, outer = _patch_fast_run(monkeypatch, tmp_path)
    _FakePipeline.initialised_seeds = []
    _FakePipeline.prediction_scopes = []
    first = run_nested_cv(
        root=tmp_path,
        config_path=tmp_path / "ignored.yaml",
        run_id="reference_fixture_a",
        suite="core",
    )
    first_seeds = tuple(_FakePipeline.initialised_seeds)
    # Three calibration cross-fits plus one final refit for each of 25 outer jobs.
    assert len(first_seeds) == 100
    assert all(
        set(fit_ids).isdisjoint(query_ids)
        for fit_ids, query_ids in _FakePipeline.prediction_scopes
    )

    completed, run_manifest, artifact_manifest = verify_completed_run(first)
    assert completed["outer_test_metrics_computed_during_training"] is False
    assert completed["hagr_labels_loaded_during_development"] is False
    assert run_manifest["splits"]["outer_test_used_for_selection"] is False
    assert artifact_manifest["entries"]

    predictions = pd.read_parquet(first / "predictions" / "outer_long.parquet")
    assert len(predictions) == len(curated) * 5
    assert not predictions.duplicated(["model_id", "compound_id", "repeat"]).any()
    assert set(predictions["compound_id"]) == set(outer["compound_id"])
    assert predictions["probability_calibrated"].between(0.0, 1.0).all()

    run_schema = json.loads((tmp_path / "schemas" / "run_manifest.schema.json").read_text())
    jsonschema.Draft202012Validator(run_schema).validate(run_manifest)

    report = report_internal(first)
    assert report == first / "report"
    assert report_internal(first) == report
    summary = json.loads((report / "summary.json").read_text())
    assert summary["external_validation_included"] is False
    assert summary["source_label_confounding_disclosed"] is True
    assert summary["model_metrics"][0]["aggregation_level"] == (
        "one_final_prediction_per_compound"
    )
    reported_metrics = summary["model_metrics"][0]["metrics"]
    assert 0.0 <= reported_metrics["ece_adaptive"] <= 1.0
    assert 0.0 <= reported_metrics["bedroc_alpha_20"] <= 1.0
    assert "recall_at_10" in reported_metrics
    assert summary["selection_audit"]["R0_prevalence"] == {
        "attempted_candidates_across_all_fit_scopes": 100,
        "successful_candidates_across_all_fit_scopes": 100,
        "failed_candidates_across_all_fit_scopes": 0,
        "failure_types": {},
    }
    fitted = summary["fitted_model_diagnostics"]["R0_prevalence"]
    assert fitted["importance_stability_spearman"] == {
        "median": pytest.approx(0.8),
        "minimum": pytest.approx(0.75),
        "n_values": 50,
        "stability_fallback_outer_jobs": 0,
    }
    assert fitted["selected_representation_dimensions"] == {"8": 25}
    assert fitted["selected_panel_outer_jobs"] == {"chemistry_32": 25}
    assert fitted["verified_checkpoint_license_records"]["fake_checkpoint"] == {
        "package": "fake-foundation-package",
        "package_version": "1.0.0",
        "explicit_model_version": "fake-v1",
        "checkpoint_sha256": "a" * 64,
        "license_sha256": "b" * 64,
        "checkpoint_source": "local synthetic fixture",
        "access_date_utc": "2026-08-15",
    }
    reliability = report / "reports" / "reliability.svg"
    assert reliability.is_file()
    assert b"<svg" in reliability.read_bytes()

    _FakePipeline.initialised_seeds = []
    _FakePipeline.prediction_scopes = []
    # A shared reference must retain the same scientific seed when embedded in a
    # different parent pipeline; run/pipeline labels are artifact metadata only.
    monkeypatch.setattr(nested_cv, "_pipeline_id", lambda value: "V5_ELIXIRFP_REBUILT")
    second = run_nested_cv(
        root=tmp_path,
        config_path=tmp_path / "ignored.yaml",
        run_id="v5_fixture_b",
        suite="core",
    )
    assert tuple(_FakePipeline.initialised_seeds) == first_seeds
    second_predictions = pd.read_parquet(second / "predictions" / "outer_long.parquet")
    comparison_columns = [
        "model_id",
        "repeat",
        "outer_fold",
        "compound_id",
        "probability_raw",
        "probability_calibrated",
        "seed",
    ]
    pd.testing.assert_frame_equal(
        predictions[comparison_columns].reset_index(drop=True),
        second_predictions[comparison_columns].reset_index(drop=True),
    )

    (report / "summary.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ReportingError, match="changed"):
        report_internal(first)

    prediction_path = first / "predictions" / "outer_long.parquet"
    prediction_path.write_bytes(prediction_path.read_bytes() + b"tamper")
    with pytest.raises(NestedCVError, match=r"artifact inventory|predictions changed"):
        verify_completed_run(first)


def test_paper80_bis_run_seals_one_contextual_holdout_and_reports_it_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curated, outer = _patch_fast_run(monkeypatch, tmp_path, paper80=True)
    _FakePipeline.initialised_seeds = []
    _FakePipeline.prediction_scopes = []
    run = run_nested_cv(
        root=tmp_path,
        config_path=tmp_path / "ignored.yaml",
        run_id="v5bis_fixture_paper80",
        suite="core",
    )
    # Three calibration cross-fits plus one final fit for the single paper holdout.
    assert len(_FakePipeline.initialised_seeds) == 4
    assert all(
        set(fit_ids).isdisjoint(query_ids)
        for fit_ids, query_ids in _FakePipeline.prediction_scopes
    )
    _, manifest, _ = verify_completed_run(run)
    assert manifest["pipeline_id"] == "V5BIS_PAPER80"
    assert manifest["outer_repeats"] == manifest["outer_folds"] == 1
    assert manifest["splits"]["primary_strategy"] == "paper_random_80_20"
    assert manifest["splits"]["outer_test_used_for_selection"] is False
    schema = json.loads((tmp_path / "schemas" / "run_manifest.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(manifest)

    predictions = pd.read_parquet(run / "predictions" / "outer_long.parquet")
    expected_test = set(outer.loc[outer["role"].eq("paper_test"), "compound_id"])
    assert len(predictions) == 77
    assert set(predictions["compound_id"]) == expected_test
    assert set(predictions["compound_id"]).isdisjoint(
        outer.loc[outer["role"].eq("paper_train"), "compound_id"]
    )
    assert len(curated) == 382

    report = report_internal(run)
    summary = json.loads((report / "summary.json").read_text())
    assert summary["scientific_role"] == "internal_paper_holdout_contextual"
    assert summary["model_metrics"][0]["evaluation_role"] == "paper_holdout_contextual"
    assert summary["model_metrics"][0]["n_compounds"] == 77
    assert summary["external_validation_included"] is False


def test_run_recovers_partial_aggregate_without_refitting_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fast_run(monkeypatch, tmp_path)
    original_write = nested_cv._write_parquet
    interrupted = False

    def interrupt_after_first_aggregate(path: Path, frame: pd.DataFrame) -> None:
        nonlocal interrupted
        if (
            path.name == "calibration_inner_oof.parquet"
            and path.parent.name == "predictions"
            and not interrupted
        ):
            interrupted = True
            raise RuntimeError("synthetic aggregate interruption")
        original_write(path, frame)

    monkeypatch.setattr(nested_cv, "_write_parquet", interrupt_after_first_aggregate)
    with pytest.raises(RuntimeError, match="synthetic aggregate interruption"):
        run_nested_cv(
            root=tmp_path,
            config_path=tmp_path / "ignored.yaml",
            run_id="reference_resume_fixture",
            suite="core",
        )
    work = tmp_path / "outputs" / ".reference_resume_fixture.work"
    assert (work / "predictions" / "outer_long.parquet").is_file()
    assert len(tuple((work / "jobs").glob("repeat_*/fold_*/*/JOB_COMPLETED.json"))) == 25

    monkeypatch.setattr(nested_cv, "_write_parquet", original_write)
    _FakePipeline.initialised_seeds = []
    completed = run_nested_cv(
        root=tmp_path,
        config_path=tmp_path / "ignored.yaml",
        run_id="reference_resume_fixture",
        suite="core",
    )
    assert _FakePipeline.initialised_seeds == []
    verify_completed_run(completed)
    archives = tuple((tmp_path / "outputs" / ".forensics").glob("*aggregate-partial.*"))
    assert len(archives) == 1
    assert (archives[0] / "predictions" / "outer_long.parquet").is_file()
    recovery = json.loads((archives[0] / "RECOVERY.json").read_text())
    assert recovery["recoverable"] is True


def test_post_lock_source_confounding_audit_is_descriptive_and_label_bijective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    curated = pd.DataFrame(
        {
            "compound_id": ["cmp::n0", "cmp::n1", "cmp::p0", "cmp::p1"],
            "label": [0, 0, 1, 1],
            "source_role": [
                "weak_chembl_reference",
                "weak_chembl_reference",
                "reported_positive",
                "reported_positive",
            ],
            "standardized_parent_smiles": ["CC", "CCC", "CCO", "CCCO"],
            "qc_flags": ["[]", '["name_smiles_mismatch"]', "[]", "[]"],
            "smiles_curated": [False, True, False, False],
            "metal_sensitive_representation": [False, False, False, False],
            "component_count": [1, 2, 1, 1],
        }
    )
    provenance = pd.DataFrame({"raw_row_id": ["raw::0", "raw::1"]})
    manifest_path = tmp_path / "curation" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("synthetic curation manifest\n", encoding="utf-8")
    data_config = {
        "outputs": {
            "curated_table": "curation/curated.parquet",
            "provenance_table": "curation/provenance.parquet",
            "manifest": "curation/manifest.json",
        }
    }
    monkeypatch.setattr(reporting, "resolve_config", lambda path: data_config)
    monkeypatch.setattr(reporting, "validate_protocol_lock", lambda **kwargs: None)
    monkeypatch.setattr(
        reporting,
        "load_curated_cohort",
        lambda **kwargs: (curated.copy(), provenance.copy(), {"synthetic": True}),
    )
    aggregated = pd.concat(
        [
            pd.DataFrame(
                {
                    "model_id": model_id,
                    "compound_id": curated["compound_id"],
                    "label": curated["label"],
                    "probability_calibrated": probabilities,
                }
            )
            for model_id, probabilities in (
                ("R2_extra_trees_v3_compatible", [0.1, 0.2, 0.8, 0.9]),
                ("v5_final_v5", [0.2, 0.3, 0.7, 0.8]),
            )
        ],
        ignore_index=True,
    )
    source_control_metrics = {"ap_positive": 1.0, "auroc": 1.0}
    metric_records = [
        {
            "model_id": "R2_extra_trees_v3_compatible",
            "metrics": source_control_metrics,
        }
    ]

    audit, distributions, matches = reporting._source_confounding_audit(
        tmp_path,
        {"data": {"curation_manifest_sha256": reporting.sha256_file(manifest_path)}},
        aggregated,
        metric_records,
    )

    assert audit["post_lock_exploratory_only"] is True
    assert audit["used_for_model_selection"] is False
    assert audit["source_target_is_bijective_with_observed_label"] is True
    assert set(audit["source_label_mapping"].values()) == {0, 1}
    assert audit["source_classifier"]["metrics"] == source_control_metrics
    assert audit["n_raw_provenance_rows_verified"] == 2
    assert set(distributions["descriptor"]) == {
        "MolWt",
        "MolLogP",
        "TPSA",
        "NumHDonors",
        "NumHAcceptors",
        "RingCount",
        "FractionCSP3",
    }
    assert len(matches) == 2
    assert matches["positive_compound_id"].nunique() == 2
    assert matches["weak_reference_compound_id"].nunique() == 2
    weak_qc = audit["source_specific_qc_and_missingness"]["weak_chembl_reference"]
    assert weak_qc["smiles_curated_count"] == 1
    assert weak_qc["multi_component_input_count"] == 1
    assert weak_qc["qc_flag_counts"] == {"name_smiles_mismatch": 1}

    curated.loc[curated["compound_id"].eq("cmp::n0"), "label"] = 1
    with pytest.raises(ReportingError, match="perfect bijection"):
        reporting._source_confounding_audit(
            tmp_path,
            {"data": {"curation_manifest_sha256": reporting.sha256_file(manifest_path)}},
            aggregated,
            metric_records,
        )
