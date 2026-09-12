"""Order-invariant label-free molecular similarity components."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .fingerprints import morgan_bitvectors
from .standardize import standardize_smiles


def similarity_components(
    smiles: Iterable[object],
    *,
    compound_ids: Iterable[object],
    threshold: float = 0.40,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = False,
) -> pd.Series:
    from rdkit import DataStructs

    ids = pd.Index([str(value).strip() for value in compound_ids], name="compound_id")
    structures = list(smiles)
    if len(ids) != len(structures) or ids.has_duplicates or (ids == "").any():
        raise ValueError("Similarity components require aligned unique nonblank IDs")
    if not 0.0 < float(threshold) <= 1.0:
        raise ValueError("Similarity threshold must lie in (0, 1]")
    standardized = [standardize_smiles(value) for value in structures]
    vectors = morgan_bitvectors(
        [item.standardized_parent_smiles for item in standardized],
        radius=radius,
        n_bits=n_bits,
        use_chirality=use_chirality,
    )
    parent = list(range(len(ids)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for right in range(1, len(vectors)):
        similarities = DataStructs.BulkTanimotoSimilarity(vectors[right], vectors[:right])
        for left, similarity in enumerate(similarities):
            if float(similarity) >= float(threshold):
                union(left, right)
    members: dict[int, list[int]] = {}
    for position in range(len(ids)):
        members.setdefault(find(position), []).append(position)
    values = [""] * len(ids)
    for positions in members.values():
        key = min(standardized[position].connectivity_inchikey for position in positions)
        component = f"SIM{threshold:.3f}::{key}"
        for position in positions:
            values[position] = component
    return pd.Series(values, index=ids, name="component_id", dtype="string")
