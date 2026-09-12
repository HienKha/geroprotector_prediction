from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geroprotector.hashing import sha256_file
from geroprotector.screening_blend_paper405 import (
    LOCKED_OOF_THRESHOLD,
    LOCKED_WEIGHTS,
    ScreeningBlendPaper405Error,
    _array_sha256,
    _write_bundle,
    load_locked_bundle,
    load_protocol,
    paired_stratified_bootstrap,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_protocol_locks_train_selected_weights_and_two_thresholds() -> None:
    protocol, _sha256 = load_protocol(
        _root() / "configs" / "screening_blend_paper405.yaml"
    )
    assert np.allclose(
        protocol["locked_model"]["weights"], LOCKED_WEIGHTS, rtol=0.0, atol=1e-15
    )
    assert protocol["locked_model"]["outer_test_used_for_weight_selection"] is False
    assert protocol["decision_bundles"]["oof_mcc"]["decision_threshold"] == (
        LOCKED_OOF_THRESHOLD
    )
    assert protocol["decision_bundles"]["fixed_0p5"]["decision_threshold"] == 0.5
    assert protocol["post_lock_analysis"][
        "test_labels_may_change_model_or_threshold"
    ] is False


def test_paired_bootstrap_is_deterministic_and_keeps_pairing() -> None:
    labels = np.asarray([0] * 20 + [1] * 20)
    better = np.r_[np.linspace(0.02, 0.35, 20), np.linspace(0.65, 0.98, 20)]
    weaker = np.r_[np.linspace(0.10, 0.70, 20), np.linspace(0.30, 0.90, 20)]
    first = paired_stratified_bootstrap(
        labels=labels,
        first_probability=better,
        second_probability=weaker,
        first_threshold=0.5,
        second_threshold=0.5,
        resamples=200,
        seed=17,
        confidence_level=0.95,
    )
    second = paired_stratified_bootstrap(
        labels=labels,
        first_probability=better,
        second_probability=weaker,
        first_threshold=0.5,
        second_threshold=0.5,
        resamples=200,
        seed=17,
        confidence_level=0.95,
    )
    assert first.equals(second)
    assert set(first.metric) == {
        "auprc_average_precision_positive",
        "auroc",
        "brier",
        "mcc",
        "macro_f1",
    }
    assert not first.test_used_for_model_or_threshold_selection.any()
    assert (
        first[first.metric.isin(["auprc_average_precision_positive", "auroc", "mcc"])]
        .delta_first_minus_second
        .ge(0)
        .all()
    )


def test_two_portable_bundles_share_component_state_and_fail_on_tamper(
    tmp_path: Path,
) -> None:
    fit_indices = np.asarray([1, 2, 3, 4], dtype=np.int64)
    fit_labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    bits = np.asarray([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.uint8)
    context_features = np.arange(12, dtype=np.float32).reshape(4, 3)
    shared = {
        "weights": LOCKED_WEIGHTS.tolist(),
        "components": ["paper_svm", "tanimoto_svc", "tabpfn_v2"],
        "component_state_sha256": "component-state",
        "source_binding": {"selection_lock_sha256": "selection"},
        "fit_paper_indices": fit_indices,
        "fit_labels": fit_labels,
        "fit_paper_indices_sha256": _array_sha256(fit_indices),
        "fit_labels_sha256": _array_sha256(fit_labels),
        "paper_svm": {"synthetic": True},
        "paper_svm_feature_names": ["one"],
        "tanimoto_svc": {"synthetic": True},
        "tanimoto_train_bits": bits,
        "tanimoto_train_bits_sha256": _array_sha256(bits),
        "tabpfn_context": {"context_features": context_features},
        "tabpfn_context_sha256": _array_sha256(context_features),
        "tabpfn_checkpoint": {"checkpoint_sha256": "checkpoint"},
        "descriptor_names": ["descriptor"],
        "portable_contract": {
            "paper_descriptor_substitution_allowed": False,
            "tabpfn_checkpoint_embedded": False,
        },
    }
    oof = _write_bundle(
        directory=tmp_path,
        bundle_id="blend_010_060_030_oof_mcc",
        threshold=LOCKED_OOF_THRESHOLD,
        threshold_source="train_oof",
        shared=shared,
    )
    fixed = _write_bundle(
        directory=tmp_path,
        bundle_id="blend_010_060_030_fixed_0p5",
        threshold=0.5,
        threshold_source="prespecified",
        shared=shared,
    )
    first = load_locked_bundle(
        tmp_path / oof["path"], expected_artifact_sha256=oof["sha256"]
    )
    second = load_locked_bundle(
        tmp_path / fixed["path"], expected_artifact_sha256=fixed["sha256"]
    )
    assert first["component_state_sha256"] == second["component_state_sha256"]
    assert first["decision_threshold"] == LOCKED_OOF_THRESHOLD
    assert second["decision_threshold"] == 0.5
    tampered = tmp_path / fixed["path"]
    original_hash = sha256_file(tampered)
    with tampered.open("ab") as handle:
        handle.write(b"tamper")
    assert sha256_file(tampered) != original_hash
    with pytest.raises(ScreeningBlendPaper405Error, match="Bundle bytes differ"):
        load_locked_bundle(tampered, expected_artifact_sha256=fixed["sha256"])
