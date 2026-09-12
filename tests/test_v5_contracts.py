from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from geroprotector.config import resolve_config
from geroprotector.models.v5 import pipeline as pipeline_module
from geroprotector.models.v5 import resolution_search as resolution_module
from geroprotector.models.v5.fingerprint_bank import (
    V5FeatureBank,
    _content_hash,
    _contract_hash,
    candidate_specs,
    semantic_family,
)
from geroprotector.models.v5.nystrom_embedding import WeightedRBFNystrom
from geroprotector.models.v5.pipeline import (
    V5Pipeline,
    _candidate_portfolio,
    _variant_portfolio,
)
from geroprotector.models.v5.preparation_cache import StageAResult
from geroprotector.models.v5.resolution_search import select_resolutions
from geroprotector.models.v5.stable_importance import StableImportanceState
from geroprotector.models.v5.weighted_fusion import (
    DescriptorPreprocessor,
    WeightedLinearSVD,
    weighted_concatenation,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def v5_config():
    return resolve_config(ROOT / "configs/v5.yaml")


def _manual_bank() -> V5FeatureBank:
    from geroprotector.chemistry.fingerprints import FingerprintSpec

    ids = ("cmp::a", "cmp::b", "cmp::c", "cmp::d")
    spec_a = FingerprintSpec(kind="morgan_bit", radius=2, n_bits=4)
    spec_b = FingerprintSpec(kind="maccs", n_bits=167)
    blocks = {
        spec_a.identifier: np.asarray(
            [[1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]],
            dtype=np.float32,
        ),
        spec_b.identifier: np.tile(np.arange(167) % 2, (4, 1)).astype(np.float32),
    }
    descriptors = pd.DataFrame(
        [[1.0, 0.0], [2.0, np.nan], [3.0, 1.0], [4.0, 1.0]],
        index=pd.Index(ids, name="compound_id"),
        columns=["d1", "d2"],
    )
    specs = {spec_a.identifier: spec_a, spec_b.identifier: spec_b}
    contract_hash = _contract_hash(specs, descriptors.columns)
    return V5FeatureBank(
        ids=ids,
        smiles=("CC", "CCC", "CO", "CN"),
        blocks=blocks,
        specs=specs,
        descriptors=descriptors,
        feature_contract_hash=contract_hash,
        bank_hash=_content_hash(
            ids,
            ("CC", "CCC", "CO", "CN"),
            blocks,
            descriptors,
            feature_contract_hash=contract_hash,
        ),
    )


def test_v5_candidate_bank_has_four_semantic_families_and_unique_ids(v5_config) -> None:
    specs = candidate_specs(v5_config)
    assert len(specs) == 40
    assert len({spec.identifier for spec in specs}) == len(specs)
    assert {semantic_family(spec) for spec in specs} == {
        "morgan_bit",
        "morgan_count",
        "rdkit_path",
        "maccs",
    }


def test_feature_bank_is_id_ordered_molecule_local_and_label_free(v5_config) -> None:
    bank = V5FeatureBank.build(
        compound_ids=["cmp::z", "cmp::a", "cmp::m"],
        smiles=["CCC", "CC", "CO"],
        config=v5_config,
    )
    assert bank.ids == ("cmp::a", "cmp::m", "cmp::z")
    assert tuple(bank.descriptors.index) == bank.ids
    assert len(bank.blocks) == 40
    assert all(matrix.shape[0] == 3 for matrix in bank.blocks.values())
    assert bank.bank_hash
    assert not any("label" in name.lower() for name in bank.descriptors.columns)

    subset = bank.subset(["cmp::m", "cmp::a"])
    assert subset.ids == ("cmp::m", "cmp::a")
    assert subset.feature_contract_hash == bank.feature_contract_hash
    assert subset.bank_hash != bank.bank_hash
    subset.assert_integrity()
    with pytest.raises(ValueError, match="duplicated"):
        bank.positions(["cmp::a", "cmp::a"])
    with pytest.raises(KeyError, match="absent"):
        bank.positions(["cmp::missing"])


def test_feature_bank_rejects_duplicate_identity_even_when_structures_differ(v5_config) -> None:
    with pytest.raises(ValueError, match="unique"):
        V5FeatureBank.build(
            compound_ids=["cmp::a", "cmp::a"],
            smiles=["CC", "CCC"],
            config=v5_config,
        )


def test_importance_weights_respect_box_and_exact_mean() -> None:
    state = StableImportanceState(
        selected_specs=("block",),
        base_importance={"block": np.asarray([1.0, 0.0, 0.0, 0.0])},
        stability_spearman={"block": 0.0},
        fit_ids=("a", "b"),
        subfold_validation_ids=(("a",), ("b",)),
        state_hash="state",
    )
    weights = state.weights(1.0, clip=(0.25, 4.0), epsilon=1e-8)["block"]
    assert np.isclose(weights.mean(), 1.0, atol=1e-10)
    assert float(weights.min()) >= 0.25
    assert float(weights.max()) <= 4.0
    assert np.array_equal(
        state.weights(0.0, clip=(0.25, 4.0), epsilon=1e-8)["block"],
        np.ones(4),
    )
    with pytest.raises(ValueError, match="non-negative"):
        state.weights(-0.1, clip=(0.25, 4.0), epsilon=1e-8)


def test_weighted_concatenation_applies_sqrt_weight_and_validates_vectors() -> None:
    bank = _manual_bank()
    spec_id = next(iter(bank.specs))
    weights = {spec_id: np.asarray([4.0, 1.0, 0.25, 1.0])}
    output = weighted_concatenation(bank, bank.ids[:2], [spec_id], weights)
    expected = bank.matrix(spec_id, bank.ids[:2]) * np.asarray([2.0, 1.0, 0.5, 1.0])
    np.testing.assert_allclose(output, expected)
    with pytest.raises(ValueError, match="Invalid V5 weight"):
        weighted_concatenation(
            bank,
            bank.ids[:2],
            [spec_id],
            {spec_id: np.asarray([1.0, 0.0, 1.0, 1.0])},
        )


def test_v5_preprocessors_are_fit_scoped_and_transform_only_after_fit() -> None:
    descriptor = np.asarray([[1.0, np.nan, np.nan], [2.0, 3.0, np.nan], [3.0, 4.0, np.nan]])
    processor = DescriptorPreprocessor()
    with pytest.raises(RuntimeError, match="not fitted"):
        processor.transform(descriptor)
    processor.fit(descriptor, fit_ids=["a", "b", "c"])
    transformed = processor.transform(np.asarray([[4.0, np.nan, 9.0]]))
    assert transformed.shape == (1, 2)
    assert np.isfinite(transformed).all()
    assert processor.get_manifest()["fit_ids"] == ["a", "b", "c"]
    with pytest.raises(ValueError, match="fit IDs"):
        DescriptorPreprocessor().fit(descriptor, fit_ids=["a", "a", "c"])

    svd = WeightedLinearSVD(n_components=10, random_state=9).fit(
        np.asarray([[1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=float),
        fit_ids=["a", "b", "c"],
    )
    assert set(svd.get_manifest()["fit_ids"]) == {"a", "b", "c"}
    assert svd.get_manifest()["effective_components"] <= 2


def test_nystrom_landmarks_are_training_only_and_all_identical_fit_fails() -> None:
    X = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    fit_ids = ("a", "b", "c", "d")
    model = WeightedRBFNystrom(3, 1.0, 17).fit(X, fit_ids=fit_ids)
    transformed = model.transform(np.asarray([[0.5, 0.5], [2.0, 2.0]]))
    assert transformed.shape == (2, 3)
    manifest = model.get_manifest()
    assert manifest["landmarks_subset_of_fit_ids"] is True
    assert set(manifest["landmark_ids"]).issubset(fit_ids)
    with pytest.raises(ValueError, match="all-identical"):
        WeightedRBFNystrom(2, 1.0, 1).fit(np.ones((3, 2)), fit_ids=("a", "b", "c"))


def test_v5_candidate_portfolio_is_deterministic_capped_and_has_both_embeddings(
    v5_config,
) -> None:
    first = _candidate_portfolio(v5_config, seed=11)
    second = _candidate_portfolio(v5_config, seed=11)
    assert first == second
    assert len(first) == v5_config["search"]["max_stage_c_candidates_per_active_fit"]
    assert {item["embedding"] for item in first} == {
        "linear_svd",
        "nystroem_weighted_rbf",
    }
    assert all(item["estimator"]["model"] in {"extra_trees", "xgboost"} for item in first)


def test_v5_identity_ablation_branches_share_the_same_classifier_families(v5_config) -> None:
    variants = (
        "raw_best_individual_fingerprint",
        "raw_unweighted_concatenation",
        "selected_lengths_unweighted",
        "weighted_no_reduction",
    )
    portfolios = {
        variant: _variant_portfolio(v5_config, seed=17, variant=variant) for variant in variants
    }
    for portfolio in portfolios.values():
        assert {item["estimator"]["model"] for item in portfolio} == {
            "extra_trees",
            "xgboost",
        }


def _two_fold_v5_inputs():
    bank = _manual_bank()
    labels = pd.Series([0, 1, 0, 1], index=bank.ids, dtype=int)
    groups = pd.Series([f"group-{index}" for index in range(4)], index=bank.ids)
    first, second = bank.ids[:2], bank.ids[2:]
    folds = [(first, second), (second, first)]
    return bank, labels, groups, folds


class _StageASeedSpy:
    seeds: ClassVar[list[int]] = []
    classes_ = np.asarray([0, 1])

    def __init__(self, *, random_state: int, **kwargs) -> None:
        del kwargs
        self.seeds.append(int(random_state))

    def fit(self, X: np.ndarray, y: np.ndarray):
        del X, y
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive = np.linspace(0.25, 0.75, len(X))
        return np.column_stack([1.0 - positive, positive])


def test_v5_stage_a_candidates_use_common_random_numbers(
    v5_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank, labels, groups, folds = _two_fold_v5_inputs()
    config = dict(v5_config)
    config["stage_a"] = {
        **v5_config["stage_a"],
        "semantic_families": ["morgan_bit", "maccs"],
    }
    _StageASeedSpy.seeds = []
    monkeypatch.setattr(resolution_module, "ExtraTreesClassifier", _StageASeedSpy)

    select_resolutions(bank, labels, groups, config=config, seed=47, folds=folds)

    assert _StageASeedSpy.seeds == [47, 48, 47, 48]


def test_v5_stage_a_resource_exhaustion_aborts_locked_candidate_portfolio(
    v5_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank, labels, groups, folds = _two_fold_v5_inputs()
    config = dict(v5_config)
    config["stage_a"] = {
        **v5_config["stage_a"],
        "semantic_families": ["morgan_bit", "maccs"],
    }

    class ExhaustedEstimator:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def fit(self, X: np.ndarray, y: np.ndarray):
            del X, y
            raise MemoryError("synthetic allocation failure")

    monkeypatch.setattr(resolution_module, "ExtraTreesClassifier", ExhaustedEstimator)
    with pytest.raises(RuntimeError, match="exhausted memory; aborting the locked run"):
        select_resolutions(bank, labels, groups, config=config, seed=47, folds=folds)


class _StageCRepresentationSpy:
    seeds: ClassVar[list[int]] = []

    def __init__(self, seed: int) -> None:
        self.seeds.append(int(seed))

    def transform(self, bank: V5FeatureBank, ids) -> np.ndarray:
        positions = bank.positions(ids).astype(float)
        return positions.reshape(-1, 1)


class _StageCEstimatorSpy:
    seeds: ClassVar[list[int]] = []
    classes_ = np.asarray([0, 1])

    def __init__(self, seed: int) -> None:
        self.seeds.append(int(seed))

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        del X, y, kwargs
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive = np.where(np.asarray(X)[:, 0].astype(int) % 2 == 0, 0.2, 0.8)
        return np.column_stack([1.0 - positive, positive])


def test_v5_stage_c_candidates_use_common_random_numbers(
    v5_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank, labels, groups, folds = _two_fold_v5_inputs()
    selected_spec = next(iter(bank.specs))
    candidates = (
        {
            "alpha": 0.0,
            "embedding": "identity",
            "n_components": 0,
            "gamma_multiplier": None,
            "estimator": {"model": "extra_trees", "n_estimators": 10},
        },
        {
            "alpha": 0.0,
            "embedding": "identity",
            "n_components": 0,
            "gamma_multiplier": None,
            "estimator": {"model": "xgboost", "n_estimators": 10, "max_depth": 1},
        },
    )
    stage_a = pd.DataFrame(
        [
            {
                "stage": "A_resolution",
                "candidate_id": "stage-a",
                "spec_id": selected_spec,
                "status": "success",
                "selected": True,
            }
        ]
    )
    _StageCRepresentationSpy.seeds = []
    _StageCEstimatorSpy.seeds = []
    monkeypatch.setattr(
        pipeline_module,
        "load_or_fit_stage_a",
        lambda *args, **kwargs: StageAResult(
            (selected_spec,), stage_a, "stage-a-binding", False
        ),
    )
    monkeypatch.setattr(
        pipeline_module, "_variant_portfolio", lambda *args, **kwargs: candidates
    )
    monkeypatch.setattr(
        pipeline_module,
        "_fit_representation",
        lambda *args, seed, **kwargs: _StageCRepresentationSpy(seed),
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_estimator",
        lambda spec, *, seed: _StageCEstimatorSpy(seed),
    )

    V5Pipeline(v5_config, seed=53, variant="selected_lengths_unweighted").fit(
        bank, labels, groups=groups, selection_folds=folds
    )

    assert _StageCRepresentationSpy.seeds[:4] == [53, 54, 53, 54]
    assert _StageCEstimatorSpy.seeds[:4] == [53, 54, 53, 54]
    assert _StageCRepresentationSpy.seeds[4:] == [53 + 999_983]
    assert _StageCEstimatorSpy.seeds[4:] == [53 + 999_983]


class _V5RepresentationFixture:
    def transform(self, bank, ids):
        return np.arange(len(ids), dtype=float).reshape(-1, 1)


class _V5EstimatorFixture:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, matrix):
        positive = np.full(len(matrix), 0.6)
        return np.column_stack([1.0 - positive, positive])


class _InvalidV5EstimatorFixture:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, matrix):
        return np.tile(np.asarray([-0.2, 1.2]), (len(matrix), 1))


def test_v5_prediction_is_bound_to_feature_bank_contract(v5_config) -> None:
    full = _manual_bank()
    fitted_subset = full.subset(full.ids[:2])
    query_ids = full.ids[2:]
    pipeline = V5Pipeline(v5_config)
    pipeline.estimator_ = _V5EstimatorFixture()
    pipeline.representation_ = _V5RepresentationFixture()
    pipeline.bank_hash_ = fitted_subset.bank_hash
    pipeline.feature_contract_hash_ = fitted_subset.feature_contract_hash
    pipeline.fit_ids_ = fitted_subset.ids

    # The runner fits on an active subset and predicts from the complete molecule-local
    # store. Content hashes must differ, while the extractor contract remains identical.
    prediction = pipeline.predict_proba(full, query_ids)
    assert prediction.shape == (2, 2)
    np.testing.assert_allclose(prediction.sum(axis=1), 1.0)
    query_only = full.subset(query_ids)
    assert query_only.bank_hash not in {full.bank_hash, fitted_subset.bank_hash}
    pipeline.predict_proba(query_only, query_ids)

    first_block = next(iter(query_only.blocks.values()))
    first_block[0, 0] = 1.0 - first_block[0, 0]
    with pytest.raises(ValueError, match="content hash"):
        pipeline.predict_proba(query_only, query_ids)

    with pytest.raises(ValueError, match="overlap"):
        pipeline.predict_proba(full, (fitted_subset.ids[0],))

    pipeline.estimator_ = _InvalidV5EstimatorFixture()
    with pytest.raises(ValueError, match="invalid probabilities"):
        pipeline.predict_proba(full, query_ids)
