from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from geroprotector.chemistry.fingerprints import FingerprintSpec
from geroprotector.config import resolve_config
from geroprotector.models.v5 import pipeline as pipeline_module
from geroprotector.models.v5 import preparation_cache as cache_module
from geroprotector.models.v5 import stable_importance as importance_module
from geroprotector.models.v5.fingerprint_bank import (
    V5FeatureBank,
    _content_hash,
    _contract_hash,
)
from geroprotector.models.v5.pipeline import V5Pipeline
from geroprotector.models.v5.preparation_cache import (
    V5PreparationCacheError,
    load_or_fit_stage_a,
    load_or_fit_stage_b,
)
from geroprotector.models.v5.stable_importance import (
    StableImportanceState,
    fit_stable_importance,
)


def _inputs():
    ids = tuple(f"cmp::{index}" for index in range(6))
    spec = FingerprintSpec(kind="morgan_bit", radius=2, n_bits=4)
    blocks = {
        spec.identifier: np.asarray(
            [
                [1, 0, 0, 1],
                [0, 1, 0, 1],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [1, 0, 1, 0],
                [0, 1, 1, 0],
            ],
            dtype=np.float32,
        )
    }
    descriptors = pd.DataFrame(
        np.arange(12, dtype=float).reshape(6, 2),
        index=pd.Index(ids, name="compound_id"),
        columns=["d1", "d2"],
    )
    specs = {spec.identifier: spec}
    contract = _contract_hash(specs, descriptors.columns)
    smiles = tuple("C" * (index + 1) for index in range(6))
    bank = V5FeatureBank(
        ids,
        smiles,
        blocks,
        specs,
        descriptors,
        contract,
        _content_hash(
            ids,
            smiles,
            blocks,
            descriptors,
            feature_contract_hash=contract,
        ),
    )
    labels = pd.Series([0, 1, 0, 1, 0, 1], index=ids, dtype=int)
    groups = pd.Series([f"group::{index}" for index in range(6)], index=ids)
    folds = [
        (ids[2:], ids[:2]),
        (ids[:2] + ids[4:], ids[2:4]),
        (ids[:4], ids[4:]),
    ]
    return bank, labels, groups, folds


def test_stage_a_cache_reuses_only_an_exact_fit_scope(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank, labels, groups, folds = _inputs()
    config = {"stage_a": {"contract": "synthetic"}}
    calls = 0

    def fake_select(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        spec_id = next(iter(bank.specs))
        return (spec_id,), pd.DataFrame(
            [{"candidate_id": "a", "spec_id": spec_id, "status": "success"}]
        )

    monkeypatch.setattr(cache_module, "select_resolutions", fake_select)
    first = load_or_fit_stage_a(
        cache_root=tmp_path,
        cache_context_sha256="c" * 64,
        bank=bank,
        y=labels,
        groups=groups,
        folds=folds,
        config=config,
        seed=17,
    )
    second = load_or_fit_stage_a(
        cache_root=tmp_path,
        cache_context_sha256="c" * 64,
        bank=bank,
        y=labels,
        groups=groups,
        folds=folds,
        config=config,
        seed=17,
    )
    changed = labels.copy()
    changed.iloc[0] = 1
    third = load_or_fit_stage_a(
        cache_root=tmp_path,
        cache_context_sha256="c" * 64,
        bank=bank,
        y=changed,
        groups=groups,
        folds=folds,
        config=config,
        seed=17,
    )
    assert calls == 2
    assert first.reused is False and second.reused is True and third.reused is False
    assert first.binding_sha256 == second.binding_sha256
    assert third.binding_sha256 != first.binding_sha256

    artifact = tmp_path / "stage_a" / first.binding_sha256 / "payload.joblib"
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(V5PreparationCacheError, match="binding"):
        load_or_fit_stage_a(
            cache_root=tmp_path,
            cache_context_sha256="c" * 64,
            bank=bank,
            y=labels,
            groups=groups,
            folds=folds,
            config=config,
            seed=17,
        )


def test_stage_b_cache_is_shared_across_variants_but_not_groups(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank, labels, groups, _ = _inputs()
    selected = tuple(bank.specs)
    config = {"weighting": {"contract": "synthetic"}}
    calls = 0

    def fake_importance(active_bank, selected_specs, *args, seed, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        return StableImportanceState(
            selected_specs=tuple(selected_specs),
            base_importance={selected[0]: np.full(4, 0.25)},
            stability_spearman={selected[0]: 0.5},
            fit_ids=tuple(active_bank.ids),
            subfold_validation_ids=(),
            state_hash=f"state-{seed}",
        )

    monkeypatch.setattr(cache_module, "fit_stable_importance", fake_importance)
    first = load_or_fit_stage_b(
        cache_root=tmp_path,
        cache_context_sha256="d" * 64,
        bank=bank,
        selected_specs=selected,
        y=labels,
        groups=groups,
        config=config,
        seed=29,
    )
    second = load_or_fit_stage_b(
        cache_root=tmp_path,
        cache_context_sha256="d" * 64,
        bank=bank,
        selected_specs=selected,
        y=labels,
        groups=groups,
        config=config,
        seed=29,
    )
    changed_groups = groups.copy()
    changed_groups.iloc[0] = "changed"
    third = load_or_fit_stage_b(
        cache_root=tmp_path,
        cache_context_sha256="d" * 64,
        bank=bank,
        selected_specs=selected,
        y=labels,
        groups=changed_groups,
        config=config,
        seed=29,
    )
    assert calls == 2
    assert first.reused is False and second.reused is True and third.reused is False
    assert first.binding_sha256 == second.binding_sha256
    assert third.binding_sha256 != first.binding_sha256


def test_stage_b_subfold_vectors_resume_without_refitting(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank, labels, groups, folds = _inputs()
    config = resolve_config(Path(__file__).resolve().parents[1] / "configs/v5.yaml")
    monkeypatch.setattr(importance_module, "grouped_inner_folds", lambda *a, **k: folds)
    fit_counts = {"forest": 0, "xgboost": 0, "permutation": 0}

    class Forest:
        def __init__(self, **kwargs):
            del kwargs

        def fit(self, X, y):
            del X, y
            fit_counts["forest"] += 1
            return self

    class XGBoost:
        def __init__(self, **kwargs):
            del kwargs
            self.feature_importances_ = np.asarray([0.4, 0.3, 0.2, 0.1])

        def fit(self, X, y):
            del X, y
            fit_counts["xgboost"] += 1
            return self

    def permutation(*args, **kwargs):
        del args, kwargs
        fit_counts["permutation"] += 1
        return SimpleNamespace(importances_mean=np.asarray([0.1, 0.2, 0.3, 0.4]))

    monkeypatch.setattr(importance_module, "ExtraTreesClassifier", Forest)
    monkeypatch.setattr(importance_module, "XGBClassifier", XGBoost)
    monkeypatch.setattr(importance_module, "permutation_importance", permutation)
    first = fit_stable_importance(
        bank,
        tuple(bank.specs),
        labels,
        groups,
        config=config,
        seed=41,
        checkpoint_root=tmp_path,
        parent_binding_sha256="e" * 64,
    )
    assert fit_counts == {"forest": 3, "xgboost": 3, "permutation": 3}
    assert len(list(tmp_path.glob("*/manifest.json"))) == 3

    class MustNotFit:
        def __init__(self, **kwargs):
            del kwargs
            raise AssertionError("completed subfold checkpoint was not reused")

    monkeypatch.setattr(importance_module, "ExtraTreesClassifier", MustNotFit)
    monkeypatch.setattr(importance_module, "XGBClassifier", MustNotFit)
    second = fit_stable_importance(
        bank,
        tuple(bank.specs),
        labels,
        groups,
        config=config,
        seed=41,
        checkpoint_root=tmp_path,
        parent_binding_sha256="e" * 64,
    )
    assert second.state_hash == first.state_hash
    assert fit_counts == {"forest": 3, "xgboost": 3, "permutation": 3}


def test_weighted_pipeline_variants_reuse_identical_stage_a_and_b_dependencies(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank, labels, groups, _ = _inputs()
    config = resolve_config(Path(__file__).resolve().parents[1] / "configs/v5.yaml")
    ids = bank.ids
    folds = [(ids[:3], ids[3:]), (ids[3:], ids[:3])]
    selected = tuple(bank.specs)
    calls = {"stage_a": 0, "stage_b": 0}

    def fake_select(*args, **kwargs):
        del args, kwargs
        calls["stage_a"] += 1
        return selected, pd.DataFrame(
            [
                {
                    "stage": "A_resolution",
                    "candidate_id": "stage-a",
                    "spec_id": selected[0],
                    "status": "success",
                    "selected": True,
                }
            ]
        )

    def fake_importance(active_bank, selected_specs, *args, seed, **kwargs):
        del args, kwargs
        calls["stage_b"] += 1
        return StableImportanceState(
            selected_specs=tuple(selected_specs),
            base_importance={selected[0]: np.full(4, 0.25)},
            stability_spearman={selected[0]: 0.8},
            fit_ids=tuple(active_bank.ids),
            subfold_validation_ids=(),
            state_hash=f"state-{seed}-{active_bank.bank_hash}",
        )

    class Representation:
        def __init__(self, fit_ids):
            self.fit_ids = tuple(fit_ids)

        def transform(self, active_bank, requested_ids):
            return active_bank.positions(requested_ids).astype(float).reshape(-1, 1)

    class Estimator:
        classes_ = np.asarray([0, 1])

        def fit(self, X, y, **kwargs):
            del X, y, kwargs
            return self

        def predict_proba(self, X):
            positive = np.where(np.asarray(X)[:, 0].astype(int) % 2, 0.7, 0.3)
            return np.column_stack([1.0 - positive, positive])

    def portfolio(config, *, seed, variant):
        del config, seed
        embedding = "identity" if variant == "weighted_no_reduction" else "linear_svd"
        components = 0 if embedding == "identity" else 2
        return tuple(
            {
                "alpha": 0.25,
                "embedding": embedding,
                "n_components": components,
                "gamma_multiplier": None,
                "estimator": {"model": model, "n_estimators": 10, "max_depth": 1},
            }
            for model in ("extra_trees", "xgboost")
        )

    monkeypatch.setattr(cache_module, "select_resolutions", fake_select)
    monkeypatch.setattr(cache_module, "fit_stable_importance", fake_importance)
    monkeypatch.setattr(pipeline_module, "_variant_portfolio", portfolio)
    monkeypatch.setattr(
        pipeline_module,
        "_fit_representation",
        lambda active_bank, fit_ids, *args, **kwargs: Representation(fit_ids),
    )
    monkeypatch.setattr(pipeline_module, "build_estimator", lambda *args, **kwargs: Estimator())
    for variant in ("weighted_no_reduction", "weighted_svd"):
        V5Pipeline(
            config,
            seed=101,
            variant=variant,
            preparation_cache_root=tmp_path,
            preparation_cache_context_sha256="f" * 64,
        ).fit(bank, labels, groups=groups, selection_folds=folds)

    assert calls == {"stage_a": 1, "stage_b": 3}
