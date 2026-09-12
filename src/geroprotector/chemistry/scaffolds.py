"""Murcko groups with label-free clustering for acyclic molecules."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .components import similarity_components
from .standardize import standardize_smiles


def hybrid_scaffold_groups(
    smiles: Iterable[object], *, compound_ids: Iterable[object], acyclic_threshold: float = 0.55
) -> pd.Series:
    structures = list(smiles)
    ids = pd.Index([str(value) for value in compound_ids], name="compound_id")
    standardized = [standardize_smiles(value) for value in structures]
    if len(ids) != len(standardized):
        raise ValueError("Scaffold inputs are misaligned")
    acyclic_positions = [
        position
        for position, item in enumerate(standardized)
        if not item.murcko_scaffold_smiles
    ]
    acyclic_groups: dict[int, str] = {}
    if acyclic_positions:
        subset = similarity_components(
            [
                standardized[position].standardized_parent_smiles
                for position in acyclic_positions
            ],
            compound_ids=[ids[position] for position in acyclic_positions],
            threshold=acyclic_threshold,
        )
        acyclic_groups = {
            position: f"ACYCLIC::{subset.loc[ids[position]]}" for position in acyclic_positions
        }
    values = [
        (
            f"MURCKO::{item.murcko_scaffold_smiles}"
            if item.murcko_scaffold_smiles
            else acyclic_groups[position]
        )
        for position, item in enumerate(standardized)
    ]
    return pd.Series(values, index=ids, name="scaffold_group", dtype="string")
