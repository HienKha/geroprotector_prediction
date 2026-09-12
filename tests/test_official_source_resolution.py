"""Focused contracts for official AgeXtend and Kapsiani source resolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geroprotector import agextend_official_reference as agextend
from geroprotector import drugage_celegans_benchmark as build5
from geroprotector import kapsiani_historical_benchmark as kapsiani
from geroprotector.official_sources import OfficialSourceError, require_pinned_file


ROOT = Path(__file__).resolve().parents[1]


def _require_local_source(relative_path: str) -> None:
    """Skip source-resolution integration checks when licensed data are not staged."""
    if not (ROOT / relative_path).is_file():
        pytest.skip(f"optional external source is not staged: {relative_path}")


def test_agextend_official_sources_resolve_released_inference_not_exact_refit() -> None:
    _require_local_source(
        "external_sources/AgeXtend_official/datasets/Geropredictor/HOLY_AgingDataset.tsv")
    protocol, _ = agextend.load_protocol(
        ROOT / "configs/agextend_official_reference_protocol.yaml")
    resolution, tables, model = agextend.extract_sources(protocol)
    assert (len(tables["training"]), int(tables["training"].status.sum())) == (972, 583)
    assert len(tables["features"]) == 71
    assert len(tables["tenfold"]) == 10
    assert len(tables["loocv"]) == 972
    assert len(tables["challenge"]) == 84
    assert resolution["availability"]["official_model_inference"].startswith("REPRODUCIBLE")
    assert resolution["availability"]["exact_historical_refit"] == "BLOCKED_PARTIAL"
    assert model.C == 1.5 and model.gamma == 2.5


def test_agextend_published_loocv_is_compound_aligned_by_exact_label_order() -> None:
    _require_local_source(
        "external_sources/AgeXtend_official/datasets/Geropredictor/HOLY_AgingDataset.tsv")
    protocol, _ = agextend.load_protocol(
        ROOT / "configs/agextend_official_reference_protocol.yaml")
    _, tables, _ = agextend.extract_sources(protocol)
    np.testing.assert_array_equal(
        tables["training"].status.astype(int),
        tables["loocv"]["Actual Status"].astype(int))
    assert tables["loocv"]["SMILES"].tolist() == tables["training"]["smiles"].tolist()


def test_agextend_parity_compares_probabilities_and_decisions(monkeypatch) -> None:
    class FakeModel:
        classes_ = np.array([0, 1])
        feature_names_in_ = np.array(["A1_0", "A1_1"])

        def predict_proba(self, frame):
            positive = frame["A1_0"].to_numpy(float)
            return np.column_stack([1.0 - positive, positive])

    challenge = pd.DataFrame({
        "Compound Name": ["a", "b"], "Canonical_SMILES": ["CCO", "CCN"],
        "Label": [0, 1], "Anti_Aging_Prob": [0.2, 0.8],
        "Anti_Aging_Status": [0, 1]})
    monkeypatch.setattr(
        agextend, "_signaturizer_features",
        lambda smiles: pd.DataFrame({"A1_0": [0.2, 0.8], "A1_1": [0.0, 0.0]}))
    rows, status = agextend.run_parity(challenge, FakeModel(), probability_atol=1e-12)
    assert status["status"] == "PASS"
    assert status["decision_matches"] == 2
    assert rows["decision_match"].all()


def test_kapsiani_official_source_is_exact_and_build5_is_not_conflated() -> None:
    _require_local_source("external_sources/41598_2021_93070_MOESM2_ESM.xlsx")
    protocol, _ = kapsiani.load_protocol(
        ROOT / "configs/kapsiani_historical_benchmark_protocol.yaml")
    cohort, features, resolution = kapsiani.resolve_sources(protocol)
    assert len(cohort) == 1430
    assert cohort.Target.value_counts().to_dict() == {0: 1126, 1: 304}
    assert len(features) == 69
    assert resolution["not_drugage_build5"] is True
    assert resolution["availability"]["exact_historical_rf_refit"] == "BLOCKED_PARTIAL"
    assert resolution["selection_description"]["nested_inside_each_cv_fold"] is False


def test_build5_background_has_the_noncausal_operational_name() -> None:
    _require_local_source("external_data/drugage_build5/drugage.csv")
    protocol, _ = build5._load_endpoint_protocol(
        ROOT / "configs/drugage_celegans_benchmark_protocol.yaml")
    _, _, resolution = build5.resolve_sources(protocol)
    track = resolution["track_e2_protocol"]
    assert track["negative_class_name"] == "no_recorded_significant_extension"
    assert track["negative_class_is_certified_biological_negative"] is False
    assert track["not_equivalent_to_kapsiani_literature_negative"] is True
    assert resolution["track_e1_historical_reproduction"]["status"] == "BLOCKED_PARTIAL"


def test_build5_runtime_declares_outcome_blind_descriptor_eligibility() -> None:
    source = (ROOT / "src/geroprotector/drugage_celegans_benchmark.py").read_text(
        encoding="utf-8")
    assert 'nonfinite_descriptor_policy="exclude"' in source
    assert "required_datawarrior_descriptors_nonfinite" in source
    assert '"rule_uses_endpoint_labels": False' in source


def test_pinned_source_guard_rejects_hash_drift(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"official")
    with pytest.raises(OfficialSourceError, match="SHA256 differs"):
        require_pinned_file(source, "0" * 64, role="synthetic official source")
