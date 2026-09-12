from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from geroprotector.traditional_paper405 import evaluation_metrics, paper_split_indices


def _protocol(root):
    return yaml.safe_load((root / "configs" / "traditional_paper405.yaml").read_text())


def test_exact_publication_split_indices():
    project_root = Path(__file__).resolve().parents[1]
    train, test, binding = paper_split_indices(_protocol(project_root))
    assert len(train) == 324
    assert len(test) == 81
    assert not set(train) & set(test)
    assert set(train) | set(test) == set(range(405))
    assert binding == "0eb31569408c3db463a097b016a1972ef83ce552edcfc7a41297bed4248ccdf8"


def test_metrics_include_auprc_mcc_and_macro_f1():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = evaluation_metrics(labels, scores, scores, threshold=0.5)
    assert metrics["auprc_average_precision_positive"] == pytest.approx(1.0)
    assert metrics["mcc"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["tn"] == 2 and metrics["tp"] == 2


def test_ols_ranking_can_be_unbounded_but_probability_must_not_be():
    labels = np.array([0, 0, 1, 1])
    ranking = np.array([-0.2, 0.2, 0.8, 1.2])
    probability = np.clip(ranking, 0.0, 1.0)
    metrics = evaluation_metrics(labels, ranking, probability, threshold=0.5)
    assert metrics["auroc"] == pytest.approx(1.0)
    with pytest.raises(Exception, match="scores are invalid"):
        evaluation_metrics(labels, ranking, ranking, threshold=0.5)
