from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from geroprotector.config import resolve_config
from geroprotector.hashing import sha256_bytes, sha256_file
from geroprotector.models import references as reference_module
from geroprotector.models.references import (
    FoldNumericPreprocessor,
    ReferenceFeatureStore,
    ReferencePipeline,
    _reference_feature_content_hash,
    _reference_feature_contract_hash,
    _tanimoto,
)
from geroprotector.validation.bootstrap import paired_component_bootstrap
from geroprotector.validation.calibration import (
    BetaCalibrator,
    CalibrationLeakageError,
    PlattCalibrator,
    load_calibration_bundle,
    validate_oof_lineage,
)
from geroprotector.validation.metrics import aggregate_repeated_oof, metric_bundle
from geroprotector.validation.thresholds import select_mcc_threshold

ROOT = Path(__file__).resolve().parents[1]


def _lineage(ids: tuple[str, ...]) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    midpoint = len(ids) // 2
    folds = ("fold-a",) * midpoint + ("fold-b",) * (len(ids) - midpoint)
    training = {
        "fold-a": ids[midpoint:],
        "fold-b": ids[:midpoint],
    }
    return folds, training


def test_calibration_lineage_rejects_self_fit_duplicate_and_single_fold() -> None:
    ids = ("a", "b", "c", "d")
    folds, training = _lineage(ids)
    validate_oof_lineage(
        fit_ids=ids,
        fold_ids=folds,
        training_ids_by_fold=training,
    )
    leaking = dict(training)
    leaking["fold-a"] = (*leaking["fold-a"], "a")
    with pytest.raises(CalibrationLeakageError, match="predicts fitted IDs"):
        validate_oof_lineage(
            fit_ids=ids,
            fold_ids=folds,
            training_ids_by_fold=leaking,
        )
    with pytest.raises(CalibrationLeakageError, match="duplicated"):
        validate_oof_lineage(
            fit_ids=("a", "a", "c", "d"),
            fold_ids=folds,
            training_ids_by_fold=training,
        )
    with pytest.raises(CalibrationLeakageError, match="at least 2"):
        validate_oof_lineage(
            fit_ids=ids,
            fold_ids=("only",) * 4,
            training_ids_by_fold={"only": ()},
        )

    incomplete = dict(training)
    incomplete["fold-a"] = ("c",)
    with pytest.raises(CalibrationLeakageError, match=r"exact .*complement"):
        validate_oof_lineage(
            fit_ids=ids,
            fold_ids=folds,
            training_ids_by_fold=incomplete,
        )


def test_platt_calibration_is_nondecreasing_and_has_safe_fallbacks() -> None:
    ids = tuple(f"id-{index}" for index in range(8))
    folds, training = _lineage(ids)
    labels = np.asarray([0, 0, 0, 1, 0, 1, 1, 1])
    raw = np.asarray([0.05, 0.10, 0.20, 0.60, 0.30, 0.70, 0.80, 0.95])
    calibrator = PlattCalibrator().fit(
        raw,
        labels,
        fit_ids=ids,
        fold_ids=folds,
        training_ids_by_fold=training,
    )
    grid = np.linspace(0.01, 0.99, 100)
    assert np.all(np.diff(calibrator.predict(grid)) >= 0)
    assert calibrator.get_manifest()["lineage_checked"] is True

    anti = PlattCalibrator().fit(
        1.0 - raw,
        labels,
        fit_ids=ids,
        fold_ids=folds,
        training_ids_by_fold=training,
    )
    assert anti.fit_mode_ == "rank_preserving_intercept_only_fallback"
    assert anti.slope_ == 1.0
    assert np.all(np.diff(anti.predict(grid)) >= 0)

    constant = PlattCalibrator().fit(
        np.full(8, 0.5),
        labels,
        fit_ids=ids,
        fold_ids=folds,
        training_ids_by_fold=training,
    )
    assert constant.fit_mode_ == "constant_input_prevalence"
    np.testing.assert_allclose(constant.predict(np.asarray([0.1, 0.9])), labels.mean())


@pytest.mark.parametrize("bad", [[-0.1] * 8, [1.1] * 8, [np.nan] * 8])
def test_calibrator_rejects_invalid_probabilities(bad) -> None:
    ids = tuple(f"id-{index}" for index in range(8))
    folds, training = _lineage(ids)
    with pytest.raises(CalibrationLeakageError, match=r"\[0, 1\]"):
        PlattCalibrator().fit(
            np.asarray(bad),
            np.asarray([0, 1] * 4),
            fit_ids=ids,
            fold_ids=folds,
            training_ids_by_fold=training,
        )


def _bootstrap_frame() -> pd.DataFrame:
    rows = []
    for component in range(6):
        rows.extend(
            [
                {
                    "component_id": f"c{component}",
                    "label": 0,
                    "candidate": 0.05 + component * 0.01,
                    "reference": 0.45,
                },
                {
                    "component_id": f"c{component}",
                    "label": 1,
                    "candidate": 0.95 - component * 0.01,
                    "reference": 0.55,
                },
            ]
        )
    frame = pd.DataFrame(rows)
    frame.index = np.arange(100, 100 + len(frame))
    return frame


def test_component_bootstrap_is_paired_deterministic_and_index_agnostic() -> None:
    frame = _bootstrap_frame()
    first = paired_component_bootstrap(
        frame,
        candidate_column="candidate",
        reference_column="reference",
        n_resamples=250,
        seed=37,
    )
    second = paired_component_bootstrap(
        frame,
        candidate_column="candidate",
        reference_column="reference",
        n_resamples=250,
        seed=37,
    )
    assert first == second
    assert first["bootstrap_unit"] == "primary_component"
    assert first["bootstrap_units"] == 6
    assert first["n_valid_resamples"] == 250
    assert first["delta_candidate_minus_reference"] >= 0


def test_component_bootstrap_requires_independent_units_and_two_class_draws() -> None:
    frame = _bootstrap_frame()
    with pytest.raises(ValueError, match="At least two"):
        paired_component_bootstrap(
            frame.loc[frame["component_id"].eq("c0")],
            candidate_column="candidate",
            reference_column="reference",
            n_resamples=10,
        )


def _repeated_predictions() -> pd.DataFrame:
    rows = []
    labels = {"a": 0, "b": 1, "c": 0, "d": 1}
    for repeat in (0, 1):
        for index, (compound, label) in enumerate(labels.items()):
            rows.append(
                {
                    "compound_id": compound,
                    "label": label,
                    "component_id": f"component-{index}",
                    "repeat": repeat,
                    "probability_raw": 0.1 + 0.2 * label + 0.01 * repeat,
                    "probability_calibrated": 0.2 + 0.6 * label + 0.02 * repeat,
                }
            )
    return pd.DataFrame(rows)


def test_repeated_oof_is_aggregated_once_per_compound_before_metrics() -> None:
    predictions = _repeated_predictions()
    aggregated = aggregate_repeated_oof(predictions)
    assert len(aggregated) == 4
    assert aggregated["compound_id"].tolist() == ["a", "b", "c", "d"]
    expected = predictions.loc[
        predictions["compound_id"].eq("a"), "probability_calibrated"
    ].mean()
    assert (
        aggregated.loc[aggregated["compound_id"].eq("a"), "probability_calibrated"].iat[0]
        == expected
    )
    metrics = metric_bundle(
        aggregated["label"].to_numpy(),
        aggregated["probability_calibrated"].to_numpy(),
        threshold=0.5,
    )
    assert metrics["ap_positive"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["mcc"] == 1.0


def test_repeated_oof_contract_rejects_duplicates_drift_and_missing_raw_probability() -> None:
    predictions = _repeated_predictions()
    duplicate = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="multiple"):
        aggregate_repeated_oof(duplicate)

    drift = predictions.copy()
    drift.loc[drift["compound_id"].eq("a") & drift["repeat"].eq(1), "label"] = 1
    with pytest.raises(ValueError, match="drift"):
        aggregate_repeated_oof(drift)

    with pytest.raises(ValueError, match="probability_raw"):
        aggregate_repeated_oof(predictions.drop(columns="probability_raw"))


def test_mcc_threshold_selection_is_deterministic_and_training_only_primitive() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.4, 0.6, 0.9])
    first = select_mcc_threshold(labels, scores)
    second = select_mcc_threshold(labels, scores)
    assert first == second
    assert first == (0.5, 1.0)


def test_calibrator_manifests_and_sealed_bundle_reload_with_exact_parity(
    tmp_path: Path,
) -> None:
    ids = tuple(f"cmp::{index}" for index in range(6))
    folds = ("fold-a",) * 3 + ("fold-b",) * 3
    lineage = {"fold-a": ids[3:], "fold-b": ids[:3]}
    raw = np.asarray([0.10, 0.20, 0.40, 0.60, 0.80, 0.90])
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    platt = PlattCalibrator().fit(
        raw,
        labels,
        fit_ids=ids,
        fold_ids=folds,
        training_ids_by_fold=lineage,
    )
    beta = BetaCalibrator().fit(
        raw,
        labels,
        fit_ids=ids,
        fold_ids=folds,
        training_ids_by_fold=lineage,
    )
    query = np.asarray([0.05, 0.35, 0.65, 0.95])
    np.testing.assert_allclose(
        PlattCalibrator.from_manifest(platt.get_manifest()).predict(query),
        platt.predict(query),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        BetaCalibrator.from_manifest(beta.get_manifest()).predict(query),
        beta.predict(query),
        rtol=0.0,
        atol=0.0,
    )
    model_artifact_sha256 = "a" * 64
    job_binding = {
        "run_id": "fixture_run",
        "pipeline_id": "reference",
        "model_id": "R1_elastic_net",
        "repeat": 0,
        "outer_fold": 0,
        "outer_test_ids_sha256": "b" * 64,
    }
    record = {
        "schema_version": "geroprotector.calibration_bundle.v2",
        "binding": {
            **job_binding,
            "fit_ids_sha256": sha256_bytes(("\n".join(sorted(ids)) + "\n").encode()),
            "model_artifact_sha256": model_artifact_sha256,
        },
        "platt": platt.get_manifest(),
        "beta_sensitivity": beta.get_manifest(),
        "threshold": {
            "objective": "mcc",
            "value": 0.55,
            "training_oof_mcc": 1.0,
            "used_for_primary_model_selection": False,
        },
        "inference_audit": None,
        "runtime": {},
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    sealed_hash = sha256_file(path)
    loaded = load_calibration_bundle(
        path,
        expected_sha256=sealed_hash,
        expected_model_artifact_sha256=model_artifact_sha256,
        expected_fit_ids=ids,
        expected_job_binding=job_binding,
    )
    np.testing.assert_allclose(loaded.primary_probability(query), platt.predict(query))
    np.testing.assert_allclose(loaded.beta_probability(query), beta.predict(query))
    np.testing.assert_array_equal(
        loaded.decision(query), (platt.predict(query) >= 0.55).astype(int)
    )

    with pytest.raises(ValueError, match="byte hash"):
        load_calibration_bundle(
            path,
            expected_sha256="0" * 64,
            expected_model_artifact_sha256=model_artifact_sha256,
            expected_fit_ids=ids,
            expected_job_binding=job_binding,
        )
    with pytest.raises(ValueError, match="different model artifact"):
        load_calibration_bundle(
            path,
            expected_sha256=sealed_hash,
            expected_model_artifact_sha256="c" * 64,
            expected_fit_ids=ids,
            expected_job_binding=job_binding,
        )
    with pytest.raises(ValueError, match="fit IDs/folds"):
        load_calibration_bundle(
            path,
            expected_sha256=sealed_hash,
            expected_model_artifact_sha256=model_artifact_sha256,
            expected_fit_ids=ids[::-1],
            expected_job_binding=job_binding,
        )

    for field, wrong_value in (("run_id", "other_run"), ("outer_fold", 1)):
        wrong_job = {**job_binding, field: wrong_value}
        with pytest.raises(ValueError, match="different outer job"):
            load_calibration_bundle(
                path,
                expected_sha256=sealed_hash,
                expected_model_artifact_sha256=model_artifact_sha256,
                expected_fit_ids=ids,
                expected_job_binding=wrong_job,
            )
    malformed_expected_job = dict(job_binding)
    malformed_expected_job.pop("outer_test_ids_sha256")
    with pytest.raises(ValueError, match="Expected calibration job binding schema"):
        load_calibration_bundle(
            path,
            expected_sha256=sealed_hash,
            expected_model_artifact_sha256=model_artifact_sha256,
            expected_fit_ids=ids,
            expected_job_binding=malformed_expected_job,
        )

    malformed_bundle = copy.deepcopy(record)
    malformed_bundle["binding"].pop("pipeline_id")
    path.write_text(json.dumps(malformed_bundle, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="job binding schema"):
        load_calibration_bundle(
            path,
            expected_sha256=sha256_file(path),
            expected_model_artifact_sha256=model_artifact_sha256,
            expected_fit_ids=ids,
            expected_job_binding=job_binding,
        )

    mismatched_folds = copy.deepcopy(record)
    mismatched_folds["beta_sensitivity"]["fold_ids"] = list(reversed(folds))
    path.write_text(json.dumps(mismatched_folds, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="fit IDs/folds"):
        load_calibration_bundle(
            path,
            expected_sha256=sha256_file(path),
            expected_model_artifact_sha256=model_artifact_sha256,
            expected_fit_ids=ids,
            expected_job_binding=job_binding,
        )

    path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="byte hash"):
        load_calibration_bundle(
            path,
            expected_sha256=sealed_hash,
            expected_model_artifact_sha256=model_artifact_sha256,
            expected_fit_ids=ids,
            expected_job_binding=job_binding,
        )


def test_calibrator_manifest_reload_rejects_malformed_and_negative_slopes() -> None:
    ids = ["a", "b", "c", "d"]
    folds = ["left", "left", "right", "right"]
    platt_manifest = {
        "kind": "regularized_monotone_platt_on_full_pipeline_crossfit_oof",
        "epsilon": 1e-6,
        "regularization_c": 1.0,
        "minimum_slope": 1e-6,
        "intercept": 0.0,
        "slope": 1.0,
        "fit_mode": "regularized_monotone_platt",
        "fallback_reason": None,
        "fit_ids": ids,
        "fold_ids": folds,
        "lineage_checked": True,
        "monotone_nondecreasing": True,
    }
    negative_platt = copy.deepcopy(platt_manifest)
    negative_platt["slope"] = -1e-9
    with pytest.raises(ValueError, match="invalid fitted state"):
        PlattCalibrator.from_manifest(negative_platt)
    malformed_platt = copy.deepcopy(platt_manifest)
    malformed_platt["lineage_checked"] = False
    with pytest.raises(ValueError, match="lineage contract"):
        PlattCalibrator.from_manifest(malformed_platt)

    beta_manifest = {
        "kind": "monotone_beta_sensitivity_on_full_pipeline_crossfit_oof",
        "epsilon": 1e-6,
        "l2_penalty": 1e-3,
        "a": 1.0,
        "b": 1.0,
        "intercept": 0.0,
        "fit_ids": ids,
        "fold_ids": folds,
        "lineage_checked": True,
        "used_for_selection": False,
    }
    for field in ("a", "b"):
        negative_beta = copy.deepcopy(beta_manifest)
        negative_beta[field] = -1e-9
        with pytest.raises(ValueError, match="invalid fitted state"):
            BetaCalibrator.from_manifest(negative_beta)
    malformed_beta = copy.deepcopy(beta_manifest)
    malformed_beta["used_for_selection"] = True
    with pytest.raises(ValueError, match="locked sensitivity role"):
        BetaCalibrator.from_manifest(malformed_beta)


def test_fold_numeric_preprocessing_uses_only_fit_rows_and_removes_bad_columns() -> None:
    X = np.asarray(
        [
            [1.0, 1.0, np.nan, 0.0],
            [2.0, 1.0, 2.0, 1.0],
            [3.0, 1.0, 3.0, 0.0],
            [4.0, 1.0, 4.0, 1.0],
        ]
    )
    transformer = FoldNumericPreprocessor(scale=True).fit(X, fit_ids=("a", "b", "c", "d"))
    output = transformer.transform(np.asarray([[10.0, 99.0, np.nan, 1.0]]))
    assert output.shape[0] == 1
    assert output.shape[1] == 3
    assert np.isfinite(output).all()
    assert transformer.fit_ids == ("a", "b", "c", "d")
    with pytest.raises(ValueError, match="fit scope"):
        FoldNumericPreprocessor(scale=False).fit(X, fit_ids=("a", "a", "c", "d"))


def test_exact_tanimoto_kernel_values() -> None:
    bits = np.asarray([[1, 0, 1], [1, 1, 0], [0, 0, 0]], dtype=float)
    kernel = _tanimoto(bits, bits)
    np.testing.assert_allclose(np.diag(kernel)[:2], [1.0, 1.0])
    assert kernel[0, 1] == pytest.approx(1.0 / 3.0)
    assert kernel[2, 2] == 0.0
    np.testing.assert_allclose(kernel, kernel.T)


def _reference_store(ids: tuple[str, ...]) -> ReferenceFeatureStore:
    matrix = np.arange(len(ids) * 4, dtype=float).reshape(len(ids), 4)
    matrices = {"unused": matrix}
    names = {"unused": ("a", "b", "c", "d")}
    contract_hash = _reference_feature_contract_hash(names)
    return ReferenceFeatureStore(
        ids=ids,
        matrices=matrices,
        names=names,
        feature_contract_hash=contract_hash,
        store_hash=_reference_feature_content_hash(
            ids, matrices, feature_contract_hash=contract_hash
        ),
    )


def test_prevalence_reference_runs_nested_selection_and_binds_feature_store(
    balanced_labels: pd.Series,
    unique_groups: pd.Series,
    two_fold_selection,
) -> None:
    ids = tuple(balanced_labels.index)
    config = resolve_config(ROOT / "configs/reference.yaml")
    store = _reference_store(ids)
    fit_ids = ids[:6]
    fit_set = set(fit_ids)
    fit_folds = [
        (
            tuple(value for value in train if value in fit_set),
            tuple(value for value in validation if value in fit_set),
        )
        for train, validation in two_fold_selection
    ]
    fit_store = store.subset(fit_ids)
    pipeline = ReferencePipeline("R0_prevalence", config, seed=3).fit(
        fit_store,
        balanced_labels.loc[list(fit_ids)],
        groups=unique_groups.loc[list(fit_ids)],
        selection_folds=fit_folds,
    )
    query_ids = ids[6:]
    probability = pipeline.predict_proba(store, query_ids)
    np.testing.assert_allclose(probability[:, 1], balanced_labels.loc[list(fit_ids)].mean())
    assert pipeline.get_manifest()["outer_test_metric_consulted"] is False
    assert pipeline.get_manifest()["hagr_metric_consulted"] is False

    query_store = store.subset(query_ids)
    assert len({fit_store.store_hash, store.store_hash, query_store.store_hash}) == 3
    pipeline.predict_proba(query_store, query_ids)
    query_store.matrices["unused"][0, 0] += 1.0
    with pytest.raises(ValueError, match="content hash"):
        pipeline.predict_proba(query_store, query_ids)


class _ReferenceSeedSpy:
    seeds: ClassVar[list[int]] = []

    def __init__(self, model_id: str, candidate: dict[str, Any], *, seed: int) -> None:
        del model_id, candidate
        self.seed = int(seed)
        self.seeds.append(self.seed)

    def fit(
        self,
        store: ReferenceFeatureStore,
        ids: tuple[str, ...],
        y: np.ndarray,
    ) -> _ReferenceSeedSpy:
        del store, y
        self.fit_ids = tuple(ids)
        return self

    def predict_proba(self, store: ReferenceFeatureStore, ids: tuple[str, ...]) -> np.ndarray:
        del store
        probability = np.asarray(
            [0.2 if int(value.rsplit("::", 1)[-1]) % 2 == 0 else 0.8 for value in ids]
        )
        return np.column_stack([1.0 - probability, probability])


def test_reference_candidate_comparison_uses_common_random_numbers(
    balanced_labels: pd.Series,
    unique_groups: pd.Series,
    two_fold_selection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = tuple(balanced_labels.index)
    store = _reference_store(ids)
    config = resolve_config(ROOT / "configs/reference.yaml")
    _ReferenceSeedSpy.seeds = []
    monkeypatch.setattr(
        reference_module,
        "_candidate_specs",
        lambda model_id, value: ({"C": 0.1}, {"C": 10.0}),
    )
    monkeypatch.setattr(reference_module, "_FittedReference", _ReferenceSeedSpy)

    ReferencePipeline("R1_elastic_net", config, seed=17).fit(
        store,
        balanced_labels,
        groups=unique_groups,
        selection_folds=two_fold_selection,
    )

    assert _ReferenceSeedSpy.seeds[:4] == [17, 18, 17, 18]
    assert _ReferenceSeedSpy.seeds[4:] == [17 + 999_983]


def test_reference_resource_exhaustion_aborts_locked_candidate_portfolio(
    balanced_labels: pd.Series,
    unique_groups: pd.Series,
    two_fold_selection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = tuple(balanced_labels.index)
    store = _reference_store(ids)
    config = resolve_config(ROOT / "configs/reference.yaml")

    class ExhaustedReference:
        def __init__(self, model_id: str, candidate: dict[str, Any], *, seed: int) -> None:
            del model_id, candidate, seed

        def fit(self, active_store, fit_ids, labels):
            del active_store, fit_ids, labels
            raise MemoryError("synthetic allocation failure")

    monkeypatch.setattr(
        reference_module,
        "_candidate_specs",
        lambda model_id, value: ({"C": 0.1},),
    )
    monkeypatch.setattr(reference_module, "_FittedReference", ExhaustedReference)
    with pytest.raises(RuntimeError, match="exhausted memory; aborting the locked run"):
        ReferencePipeline("R1_elastic_net", config, seed=17).fit(
            store,
            balanced_labels,
            groups=unique_groups,
            selection_folds=two_fold_selection,
        )
