from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import geroprotector.weighted_blend_paper405 as weighted
from geroprotector.hypermoltab_paper405 import select_mcc_threshold
from geroprotector.weighted_blend_paper405 import (
    COMPONENTS,
    _test_sweep,
    load_protocol,
    select_threshold,
    select_weight_candidate,
    weight_grid,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _protocol() -> dict:
    return load_protocol(_root() / "configs" / "weighted_blend_paper405.yaml")[0]


def test_weight_grid_is_the_locked_171_point_positive_simplex() -> None:
    grid = weight_grid(_protocol())
    assert len(grid) == 171
    assert len({row["candidate_id"] for row in grid}) == 171
    for row in grid:
        weights = np.asarray(
            [row["weight_svm"], row["weight_tanimoto"], row["weight_tabpfn"]]
        )
        assert np.isclose(weights.sum(), 1.0)
        assert np.all(weights >= 0.05)
        assert np.allclose(weights / 0.05, np.round(weights / 0.05))


def test_vectorized_mcc_threshold_matches_existing_contract() -> None:
    labels = np.asarray([0, 1, 0, 1, 1, 0, 0, 1])
    probability = np.asarray([0.1, 0.8, 0.6, 0.4, 0.7, 0.2, 0.3, 0.9])
    expected = select_mcc_threshold(labels, probability)
    observed = select_threshold(labels, probability)
    assert observed is not None
    assert np.allclose(observed, expected, rtol=0.0, atol=1e-15)


def test_screening_threshold_honors_both_operating_constraints() -> None:
    labels = np.asarray([0] * 10 + [1] * 10)
    probability = np.asarray(
        [
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.90,
            0.15,
            0.20,
            0.25,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
        ]
    )
    selected = select_threshold(
        labels, probability, recall_floor=0.50, specificity_floor=0.75
    )
    assert selected is not None
    decision = probability >= selected[0]
    assert decision[labels == 1].mean() >= 0.50
    assert 1.0 - decision[labels == 0].mean() >= 0.75


def test_weight_selection_is_deterministic_and_never_claims_test_selection() -> None:
    rng = np.random.default_rng(44)
    labels = np.tile([0, 1], 40)
    latent = labels * 0.45 + rng.normal(0.0, 0.15, len(labels))
    probabilities = {
        "paper_svm": np.clip(0.30 + latent * 0.45, 0.01, 0.99),
        "tanimoto_svc": np.clip(0.25 + latent * 0.80, 0.01, 0.99),
        "tabpfn_v2": np.clip(0.28 + latent * 0.65, 0.01, 0.99),
    }
    first_table, first = select_weight_candidate(labels, probabilities, _protocol())
    second_table, second = select_weight_candidate(labels, probabilities, _protocol())
    pd.testing.assert_frame_equal(first_table, second_table)
    assert first == second
    assert first["selected_using_test_labels"] is False
    assert first["candidate_id"] in set(first_table.candidate_id)


def test_test_sweep_reports_all_candidates_but_cannot_replace_locked_primary() -> None:
    rng = np.random.default_rng(91)
    train_labels = np.tile([0, 1], 30)
    train_probabilities = {
        name: np.clip(
            0.2 + train_labels * (0.45 + offset) + rng.normal(0, 0.12, len(train_labels)),
            0.01,
            0.99,
        )
        for name, offset in zip(COMPONENTS, (-0.10, 0.05, 0.0), strict=True)
    }
    search, winner = select_weight_candidate(
        train_labels, train_probabilities, _protocol()
    )
    test_labels = np.tile([0, 1], 10)
    test_frame = pd.DataFrame(
        {
            "paper_row_index": np.arange(len(test_labels)),
            "probability_paper_svm": np.where(test_labels == 1, 0.9, 0.1),
            "probability_tanimoto_svc": np.where(test_labels == 1, 0.1, 0.9),
            "probability_tabpfn_v2": 0.5,
        }
    )
    metrics, predictions, primary, _equal, audit = _test_sweep(
        labels=test_labels,
        test_frame=test_frame,
        search=search,
        winner=winner,
    )
    assert len(metrics) == 171
    assert len(predictions) == 171 * len(test_labels)
    assert metrics.selected_primary_candidate.sum() == 1
    assert primary["candidate_id"] == winner["candidate_id"]
    assert not metrics.test_used_for_selection.any()
    assert audit["test_sweep_used_to_change_primary"] is False


def test_synthetic_end_to_end_run_seals_selection_before_171_test_rows(
    tmp_path: Path, monkeypatch
) -> None:
    protocol = _protocol()
    frame = pd.DataFrame(
        {
            "paper_row_index": np.arange(40),
            "compound_name": [f"compound-{index}" for index in range(40)],
            "smiles": ["CC"] * 40,
            "label": np.arange(40) % 2,
            "source_role": np.where(
                np.arange(40) % 2, "reported_positive", "weak_reference_negative"
            ),
        }
    )
    train = np.arange(30)
    test = np.arange(30, 40)
    positive = tmp_path / "positive.tsv"
    negative = tmp_path / "negative.csv"
    config = tmp_path / "protocol.yaml"
    positive.write_text("positive\n", encoding="utf-8")
    negative.write_text("negative\n", encoding="utf-8")
    config.write_text("protocol\n", encoding="utf-8")
    (tmp_path / "requirements-lock.txt").write_text("", encoding="utf-8")

    monkeypatch.setattr(weighted, "validate_core_runtime", lambda _path: None)
    monkeypatch.setattr(weighted, "source_tree_sha256", lambda _root: "source-tree")
    monkeypatch.setattr(weighted, "load_protocol", lambda _path: (protocol, "protocol"))
    monkeypatch.setattr(weighted, "_paper_contract", lambda _root, _protocol: {})
    monkeypatch.setattr(
        weighted,
        "_read_sources",
        lambda _positive, _negative, _contract: (frame.copy(), {"source_audit": "PASS"}),
    )
    monkeypatch.setattr(
        weighted,
        "paper_split_indices",
        lambda _contract: (train.copy(), test.copy(), "paper-split"),
    )
    monkeypatch.setattr(
        weighted,
        "_validated_raw_smiles",
        lambda _frame: (["CC"] * len(_frame), {"smiles_audit": "PASS"}),
    )
    monkeypatch.setattr(
        weighted,
        "_features",
        lambda _frame, _smiles, _protocol: {"row_id": np.arange(len(_frame))},
    )
    monkeypatch.setattr(weighted, "_cross_split_audit", lambda *_args: {"status": "PASS"})
    monkeypatch.setattr(weighted, "runtime_environment", lambda: {"synthetic": True})

    def fake_components(
        _features,
        _labels,
        _fit,
        target,
        _settings,
        seed,
        *,
        requested,
    ):
        assert requested == COMPONENTS
        signal = (np.asarray(target) % 2).astype(float)
        wobble = (int(seed) % 7) * 0.001
        probabilities = {
            "paper_svm": np.clip(0.35 + 0.30 * signal + wobble, 0.01, 0.99),
            "tanimoto_svc": np.clip(0.20 + 0.60 * signal + wobble, 0.01, 0.99),
            "tabpfn_v2": np.clip(0.25 + 0.50 * signal + wobble, 0.01, 0.99),
        }
        return probabilities, {"paper_svm": {}, "tanimoto_svc": {}}, {"fake": True}

    monkeypatch.setattr(weighted, "_selected_component_predictions", fake_components)
    destination = weighted.run(
        root=tmp_path,
        config_path=config,
        positive_path=positive,
        negative_path=negative,
        run_id="weightedblend405_synthetic",
    )
    assert destination == tmp_path / "outputs" / "weightedblend405_synthetic"
    assert (destination / "COMPLETED.json").is_file()
    selection = pd.read_csv(destination / "full_train_weight_search.csv")
    metrics = pd.read_csv(destination / "test_weight_sweep_metrics.csv")
    predictions = pd.read_csv(destination / "test_weight_sweep_predictions.csv")
    assert len(selection) == 171
    assert len(metrics) == 171
    assert len(predictions) == 1710
    assert metrics.selected_primary_candidate.sum() == 1
    assert not metrics.test_used_for_selection.any()
    assert not (tmp_path / "outputs" / ".weightedblend405_synthetic.work").exists()
