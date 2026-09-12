"""Training-OOF-only decision threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import matthews_corrcoef


def select_mcc_threshold(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(y, dtype=int)
    scores = np.asarray(probability, dtype=float)
    if len(labels) != len(scores) or set(labels) != {0, 1}:
        raise ValueError("MCC threshold selection requires aligned two-class data")
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    best = None
    for threshold in candidates:
        decision = (scores >= threshold).astype(int)
        mcc = float(matthews_corrcoef(labels, decision))
        key = (mcc, -abs(float(threshold) - 0.5), -float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), mcc)
    assert best is not None
    return best[1], best[2]
