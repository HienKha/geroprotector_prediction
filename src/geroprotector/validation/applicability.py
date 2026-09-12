"""Outcome-blind, outer-training-scoped applicability diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from rdkit import DataStructs

from ..chemistry.descriptors import descriptor_frame
from ..chemistry.fingerprints import morgan_bitvectors


def _robust_descriptor_distances(
    train_smiles: Sequence[str], query_smiles: Sequence[str], *, quantile: float
) -> tuple[np.ndarray, float]:
    train = descriptor_frame(train_smiles, panel="chemistry_32").to_numpy(dtype=float)
    query = descriptor_frame(query_smiles, panel="chemistry_32").to_numpy(dtype=float)
    finite = np.isfinite(train).any(axis=0)
    if not finite.any():
        raise ValueError("Applicability descriptor panel has no finite training columns")
    train = train[:, finite]
    query = query[:, finite]
    median = np.nanmedian(np.where(np.isfinite(train), train, np.nan), axis=0)
    train = np.where(np.isfinite(train), train, median)
    query = np.where(np.isfinite(query), query, median)
    q25, q75 = np.quantile(train, [0.25, 0.75], axis=0)
    scale = q75 - q25
    nonconstant = scale > 1e-12
    if not nonconstant.any():
        raise ValueError("Applicability descriptor panel has no varying columns")
    train = (train[:, nonconstant] - median[nonconstant]) / scale[nonconstant]
    query = (query[:, nonconstant] - median[nonconstant]) / scale[nonconstant]
    normalizer = np.sqrt(train.shape[1])
    query_distance = np.asarray(
        [np.min(np.linalg.norm(train - row, axis=1) / normalizer) for row in query],
        dtype=float,
    )
    if len(train) < 2:
        raise ValueError("Applicability distance requires at least two training compounds")
    within = np.linalg.norm(train[:, None, :] - train[None, :, :], axis=2) / normalizer
    np.fill_diagonal(within, np.inf)
    nearest_train = np.min(within, axis=1)
    threshold = float(np.quantile(nearest_train, float(quantile)))
    return query_distance, threshold


def outer_train_applicability(
    curated: pd.DataFrame,
    *,
    train_ids: Sequence[str],
    query_ids: Sequence[str],
    min_tanimoto: float,
    descriptor_quantile: float,
) -> pd.DataFrame:
    """Compute diagnostics without labels or query-query relationships."""

    indexed = curated.set_index(curated["compound_id"].astype(str), drop=False)
    train = tuple(map(str, train_ids))
    query = tuple(map(str, query_ids))
    if set(train) & set(query) or (set(train) | set(query)) - set(indexed.index):
        raise ValueError("Applicability train/query IDs are overlapping or unknown")
    train_smiles = tuple(indexed.loc[list(train), "standardized_parent_smiles"].astype(str))
    query_smiles = tuple(indexed.loc[list(query), "standardized_parent_smiles"].astype(str))
    train_vectors = morgan_bitvectors(train_smiles, radius=2, n_bits=2048, use_chirality=False)
    query_vectors = morgan_bitvectors(query_smiles, radius=2, n_bits=2048, use_chirality=False)
    nearest_tanimoto = np.asarray(
        [
            max(DataStructs.BulkTanimotoSimilarity(value, train_vectors))
            for value in query_vectors
        ],
        dtype=float,
    )
    distance, threshold = _robust_descriptor_distances(
        train_smiles, query_smiles, quantile=descriptor_quantile
    )
    inside = (nearest_tanimoto >= float(min_tanimoto)) & (distance <= threshold)
    return pd.DataFrame(
        {
            "compound_id": query,
            "nearest_train_tanimoto": nearest_tanimoto,
            "robust_descriptor_distance": distance,
            "descriptor_distance_threshold": threshold,
            "inside_applicability_domain": inside,
            "applicability_thresholds_selected_with_labels": False,
        }
    )
