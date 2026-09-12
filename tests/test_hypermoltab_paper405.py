from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from geroprotector.hypermoltab_core import focal_loss, variant_config
from geroprotector.hypermoltab_paper405 import VARIANTS, load_protocol


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_locked_hypermoltab_loss_and_learning_rate() -> None:
    protocol, _ = load_protocol(_root() / "configs" / "hypermoltab_paper405.yaml")
    assert tuple(protocol["variants"]) == VARIANTS
    assert protocol["training"]["loss"] == "focal"
    assert protocol["training"]["focal_alpha"] == 1.0
    assert protocol["training"]["focal_gamma"] == 1.0
    assert protocol["training"]["learning_rate"] == 4e-4
    for name in VARIANTS:
        config = variant_config(name, protocol["training"])
        assert config.focal_alpha == 1.0
        assert config.focal_gamma == 1.0
        assert config.learning_rate == 4e-4


def test_focal_loss_alpha_one_gamma_one_matches_definition() -> None:
    logits = torch.tensor([[1.2, -0.2], [-0.3, 0.8], [0.1, 0.4]], dtype=torch.float64)
    labels = torch.tensor([0, 1, 1])
    observed = focal_loss(logits, labels, alpha=1.0, gamma=1.0)
    probabilities = torch.softmax(logits, dim=1)
    pt = probabilities[torch.arange(3), labels]
    expected = (-(1.0 - pt) * torch.log(pt)).mean()
    assert torch.allclose(observed, expected)


def test_focal_alpha_changes_positive_rows_only() -> None:
    logits = torch.zeros((2, 2), dtype=torch.float64)
    labels = torch.tensor([0, 1])
    base = focal_loss(logits, labels, alpha=1.0, gamma=1.0)
    weighted = focal_loss(logits, labels, alpha=2.0, gamma=1.0)
    per_row = -0.5 * np.log(0.5)
    assert base.item() == pytest.approx(per_row)
    assert weighted.item() == pytest.approx(1.5 * per_row)


def test_tuned_protocol_locks_search_scheduler_and_budget() -> None:
    protocol, _ = load_protocol(_root() / "configs" / "hypermoltab_paper405_tuned.yaml")
    assert protocol["focal_search"]["outer_test_consulted"] is False
    assert protocol["focal_search"]["alpha_positive_grid"] == [0.75, 1.0, 1.25]
    assert protocol["focal_search"]["gamma_grid"] == [0.5, 1.0, 2.0]
    assert protocol["training"]["learning_rate"] == 1e-3
    assert protocol["training"]["epochs"] == 100
    assert protocol["training"]["patience"] == 20
    assert protocol["training"]["lr_step_size"] == 10
    assert protocol["training"]["lr_scheduler_gamma"] == 0.5


def test_variant_ablation_semantics_are_explicit() -> None:
    settings = yaml.safe_load(
        (_root() / "configs" / "hypermoltab_paper405.yaml").read_text(encoding="utf-8")
    )["training"]
    assert not variant_config("hyper_moltab_no_graph", settings).use_graph
    assert not variant_config("hyper_moltab_no_tabm", settings).use_tabm
    assert not variant_config("hyper_moltab_no_kan", settings).use_kan
    assert not variant_config("hyper_moltab_no_tree", settings).use_tree
    assert variant_config("hyper_moltab_no_cl", settings).pretrain_epochs == 0
    combined = variant_config("hyper_moltab_no_graph_no_cl", settings)
    assert not combined.use_graph and combined.pretrain_epochs == 0
    sensitivity = variant_config("hyper_moltab_distill_rank", settings)
    assert sensitivity.distill_weight > 0 and sensitivity.rank_weight > 0


def test_no_nan_in_simple_focal_batch() -> None:
    generator = np.random.default_rng(42)
    logits = torch.tensor(generator.normal(size=(16, 2)), dtype=torch.float32)
    labels = torch.tensor([0, 1] * 8)
    assert torch.isfinite(focal_loss(logits, labels, alpha=1.0, gamma=1.0))
