from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geroprotector.chemistry.components import similarity_components
from geroprotector.chemistry.descriptors import CHEMISTRY_32, descriptor_frame
from geroprotector.chemistry.fingerprints import (
    FingerprintSpec,
    fingerprint,
    fingerprint_matrix,
)
from geroprotector.chemistry.standardize import ChemistryError, standardize_smiles


def test_standardization_selects_parent_but_preserves_raw_component_and_metal_audit() -> None:
    molecule = standardize_smiles("CC(=O)[O-].[Na+]")
    assert molecule.component_count == 2
    assert molecule.metal_atomic_numbers == (11,)
    assert len(molecule.raw_component_smiles) == 2
    assert "multi_component_parent_selected" in molecule.qc_flags
    assert "metal_present_in_input" in molecule.qc_flags
    assert "metal_removed_by_parent_selection" in molecule.qc_flags
    assert "Na" not in molecule.standardized_parent_smiles


def test_standardization_preserves_stereo_but_connectivity_identity_is_stereo_agnostic() -> (
    None
):
    left = standardize_smiles("C[C@H](O)C(=O)O")
    right = standardize_smiles("C[C@@H](O)C(=O)O")
    assert left.standardized_parent_smiles != right.standardized_parent_smiles
    assert left.full_inchikey != right.full_inchikey
    assert left.connectivity_inchikey == right.connectivity_inchikey
    with pytest.raises(ChemistryError):
        standardize_smiles("not-a-smiles")


def test_chemistry32_is_exact_named_molecule_local_panel() -> None:
    frame = descriptor_frame(["CCO", "c1ccccc1"], panel="chemistry_32", index=["a", "b"])
    assert tuple(frame.columns) == CHEMISTRY_32
    assert frame.shape == (2, 32)
    assert frame.index.tolist() == ["a", "b"]
    assert np.isfinite(frame.to_numpy()).all()
    assert frame.loc["a", "OxygenCount"] == 1.0
    assert frame.loc["b", "NumAromaticRings"] == 1.0


@pytest.mark.parametrize(
    "spec",
    [
        FingerprintSpec(kind="morgan_bit", radius=2, n_bits=64),
        FingerprintSpec(kind="morgan_count", radius=2, n_bits=64),
        FingerprintSpec(kind="rdkit_path", n_bits=64),
        FingerprintSpec(kind="maccs", n_bits=167),
    ],
)
def test_fingerprint_shapes_are_explicit_and_finite(spec: FingerprintSpec) -> None:
    single = fingerprint("CCO", spec)
    matrix = fingerprint_matrix(["CCO", "CCC"], spec)
    assert single.shape == (spec.n_bits,)
    assert matrix.shape == (2, spec.n_bits)
    assert np.isfinite(matrix).all()


def test_similarity_components_are_order_invariant_and_id_aligned() -> None:
    ids = ["ethane", "propane", "benzene", "phenol"]
    smiles = ["CC", "CCC", "c1ccccc1", "Oc1ccccc1"]
    first = similarity_components(smiles, compound_ids=ids, threshold=0.40)
    permutation = [2, 0, 3, 1]
    second = similarity_components(
        [smiles[index] for index in permutation],
        compound_ids=[ids[index] for index in permutation],
        threshold=0.40,
    )
    pd.testing.assert_series_equal(first.sort_index(), second.sort_index())
    assert first.index.tolist() == ids
    with pytest.raises(ValueError, match="unique"):
        similarity_components(["CC", "CCC"], compound_ids=["x", "x"])
