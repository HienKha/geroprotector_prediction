from __future__ import annotations

from pathlib import Path

import numpy as np

from geroprotector.fixed_blend_paper405 import (
    BLENDS,
    _blends,
    _tanimoto,
    load_protocol,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_fixed_blend_protocol_has_three_equal_weight_recipes() -> None:
    protocol, _ = load_protocol(_root() / "configs" / "fixed_blend_paper405.yaml")
    assert tuple(protocol["blends"]) == BLENDS
    assert protocol["blends"]["sota_fixed_blend_original"]["components"] == [
        "extra_trees",
        "tanimoto_svc",
        "tabpfn_v2",
    ]
    assert protocol["blends"]["sota_fixed_blend_plus_svm"]["components"][-1] == ("paper_svm")
    assert (
        protocol["blends"]["sota_fixed_blend_svm_replaces_extratrees"]["components"][0]
        == "paper_svm"
    )
    assert protocol["evaluation"]["outer_test_used_for_components_or_weights"] is False


def test_fixed_blends_are_exact_arithmetic_means() -> None:
    protocol, _ = load_protocol(_root() / "configs" / "fixed_blend_paper405.yaml")
    probabilities = {
        "extra_trees": np.array([0.1, 0.9]),
        "tanimoto_svc": np.array([0.2, 0.8]),
        "tabpfn_v2": np.array([0.3, 0.7]),
        "paper_svm": np.array([0.4, 0.6]),
    }
    observed = _blends(probabilities, protocol)
    assert np.allclose(observed["sota_fixed_blend_original"], [0.2, 0.8])
    assert np.allclose(observed["sota_fixed_blend_plus_svm"], [0.25, 0.75])
    assert np.allclose(observed["sota_fixed_blend_svm_replaces_extratrees"], [0.3, 0.7])


def test_exact_tanimoto_kernel() -> None:
    left = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    right = np.array([[1, 1, 0], [1, 0, 1]], dtype=np.uint8)
    observed = _tanimoto(left, right)
    assert np.allclose(observed, [[1 / 3, 1.0], [0.5, 0.0]])
