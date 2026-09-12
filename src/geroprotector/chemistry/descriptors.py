"""Pinned molecule-local RDKit descriptor panels."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .standardize import standardize_smiles

CHEMISTRY_32: tuple[str, ...] = (
    "MolWt",
    "HeavyAtomMolWt",
    "ExactMolWt",
    "MolLogP",
    "MolMR",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "FormalCharge",
    "HeavyAtomCount",
    "NitrogenCount",
    "OxygenCount",
    "SulfurCount",
    "HalogenCount",
    "BertzCT",
    "BalabanJ",
    "HallKierAlpha",
    "LabuteASA",
    "Chi0",
    "Chi1",
    "Chi2n",
    "Chi3n",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumAromaticHeterocycles",
)


def _descriptor_registry() -> dict[str, object]:
    from rdkit.Chem import Descriptors

    return {name: function for name, function in Descriptors._descList}


def _chemistry_registry() -> dict[str, object]:
    registry = _descriptor_registry()
    registry.update(
        {
            "FormalCharge": lambda mol: sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
            "NitrogenCount": lambda mol: sum(
                atom.GetAtomicNum() == 7 for atom in mol.GetAtoms()
            ),
            "OxygenCount": lambda mol: sum(atom.GetAtomicNum() == 8 for atom in mol.GetAtoms()),
            "SulfurCount": lambda mol: sum(
                atom.GetAtomicNum() == 16 for atom in mol.GetAtoms()
            ),
            "HalogenCount": lambda mol: sum(
                atom.GetAtomicNum() in {9, 17, 35, 53} for atom in mol.GetAtoms()
            ),
        }
    )
    return registry


def descriptor_names(panel: str) -> tuple[str, ...]:
    if panel == "chemistry_32":
        registry = _chemistry_registry()
        missing = set(CHEMISTRY_32) - set(registry)
        if missing:
            raise RuntimeError(
                f"Pinned chemistry_32 descriptors unavailable: {sorted(missing)}"
            )
        return CHEMISTRY_32
    if panel == "rdkit2d_217":
        registry = _descriptor_registry()
        names = tuple(registry)
        # The name is historical; the exact registry and count are always manifest-bound.
        if len(names) < 200:
            raise RuntimeError(f"Unexpectedly small RDKit descriptor registry: {len(names)}")
        return names
    raise ValueError(f"Unknown descriptor panel: {panel}")


def descriptor_frame(
    smiles: Iterable[object], *, panel: str, index: Iterable[object] | None = None
) -> pd.DataFrame:
    from rdkit import Chem

    names = descriptor_names(panel)
    registry = _chemistry_registry() if panel == "chemistry_32" else _descriptor_registry()
    rows: list[list[float]] = []
    for raw in smiles:
        molecule = standardize_smiles(raw)
        mol = Chem.MolFromSmiles(molecule.standardized_parent_smiles)
        if mol is None:
            raise ValueError("Standardized molecule unexpectedly failed to parse")
        values: list[float] = []
        for name in names:
            try:
                value = float(registry[name](mol))
            except Exception:
                value = float("nan")
            values.append(value if np.isfinite(value) else float("nan"))
        rows.append(values)
    frame = pd.DataFrame(rows, columns=list(names), dtype=float)
    if index is not None:
        frame.index = pd.Index([str(value) for value in index], name="compound_id")
        if frame.index.has_duplicates:
            raise ValueError("Descriptor frame requires unique compound IDs")
    return frame
