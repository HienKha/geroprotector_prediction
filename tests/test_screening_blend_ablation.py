from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.svm import SVC

from geroprotector.screening_blend_ablation import (
    ScreeningBlendAblationError,
    _add_blend,
    _svc_probability_at_native_boundary,
    _union_cluster_ids,
    paired_bootstrap,
)


def test_locked_blend_is_exact_and_invalid_component_probability_fails() -> None:
    frame = pd.DataFrame(
        {
            "probability_paper_svm": [0.1, 0.8],
            "probability_tanimoto_svc": [0.2, 0.7],
            "probability_tabpfn_v2": [0.3, 0.6],
        }
    )
    blended = _add_blend(frame)
    assert np.allclose(blended.blend_probability, [0.22, 0.68])
    frame.loc[0, "probability_tabpfn_v2"] = 1.1
    with pytest.raises(ScreeningBlendAblationError, match="invalid"):
        _add_blend(frame)


def test_publication_union_clusters_are_transitive_and_stable() -> None:
    frame = pd.DataFrame(
        {
            "external_id": ["c", "a", "b", "d"],
            "publication_ids_json": [
                json.dumps(["p2"]),
                json.dumps(["p1"]),
                json.dumps(["p1", "p2"]),
                json.dumps(["p3"]),
            ],
        }
    )
    observed = _union_cluster_ids(frame)
    assert observed[0] == observed[1] == observed[2]
    assert observed[3] != observed[0]
    reordered = frame.iloc[::-1].reset_index(drop=True)
    remapped = dict(zip(reordered.external_id, _union_cluster_ids(reordered), strict=True))
    assert remapped["a"] == remapped["b"] == remapped["c"]
    assert remapped["d"] != remapped["a"]


def test_paired_bootstrap_identical_predictions_have_zero_delta() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    probability = np.asarray([0.1, 0.3, 0.4, 0.6, 0.8, 0.9])
    result = paired_bootstrap(
        labels=labels,
        blend_probability=probability,
        component_probability=probability.copy(),
        blend_threshold=0.5,
        component_threshold=0.5,
        resamples=200,
        seed=7,
        confidence_level=0.95,
    )
    assert np.allclose(result.delta_blend_minus_component, 0)
    assert np.allclose(result.ci_lower, 0)
    assert np.allclose(result.ci_upper, 0)
    assert not result.ci_excludes_zero.any()


def test_cluster_bootstrap_rejects_misaligned_clusters() -> None:
    with pytest.raises(ScreeningBlendAblationError, match="cluster IDs"):
        paired_bootstrap(
            labels=np.asarray([0, 0, 1, 1]),
            blend_probability=np.asarray([0.1, 0.2, 0.8, 0.9]),
            component_probability=np.asarray([0.2, 0.3, 0.7, 0.8]),
            blend_threshold=0.5,
            component_threshold=0.5,
            resamples=100,
            seed=9,
            confidence_level=0.95,
            cluster_ids=np.asarray(["a", "b"]),
        )


def test_published_svc_probability_boundary_reproduces_native_predict() -> None:
    X = np.asarray([[-3.0], [-2.0], [-1.0], [0.5], [1.5], [3.0]])
    y = np.asarray([0, 0, 0, 1, 1, 1])
    model = SVC(
        kernel="linear", C=1.0, gamma=1.0, probability=True, random_state=42
    ).fit(X, y)
    boundary = _svc_probability_at_native_boundary(model)
    assert np.array_equal(
        model.predict(X), (model.predict_proba(X)[:, 1] >= boundary).astype(int)
    )
