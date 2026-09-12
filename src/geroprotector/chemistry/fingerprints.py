"""Molecule-local fingerprint extraction with explicit contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .standardize import rdkit_version, standardize_smiles

FingerprintKind = Literal["morgan_bit", "morgan_count", "rdkit_path", "maccs"]


@dataclass(frozen=True)
class FingerprintSpec:
    kind: FingerprintKind
    n_bits: int
    radius: int | None = None
    use_chirality: bool = False
    min_path: int = 1
    max_path: int = 7

    @property
    def identifier(self) -> str:
        fields = (
            self.kind,
            self.n_bits,
            self.radius,
            self.use_chirality,
            self.min_path,
            self.max_path,
            rdkit_version(),
        )
        digest = hashlib.sha256(repr(fields).encode("utf-8")).hexdigest()[:12]
        return f"{self.kind}_{self.n_bits}_{digest}"


def fingerprint(smiles: object, spec: FingerprintSpec) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator

    molecule = standardize_smiles(smiles)
    mol = Chem.MolFromSmiles(molecule.standardized_parent_smiles)
    if mol is None:
        raise ValueError("Standardized molecule unexpectedly failed to parse")
    if spec.kind == "morgan_bit":
        if spec.radius is None:
            raise ValueError("Morgan fingerprint requires radius")
        generator = GetMorganGenerator(
            radius=int(spec.radius),
            fpSize=int(spec.n_bits),
            includeChirality=bool(spec.use_chirality),
        )
        return generator.GetFingerprintAsNumPy(mol).astype(np.float32, copy=False)
    if spec.kind == "morgan_count":
        if spec.radius is None:
            raise ValueError("Morgan count fingerprint requires radius")
        generator = GetMorganGenerator(
            radius=int(spec.radius),
            fpSize=int(spec.n_bits),
            includeChirality=bool(spec.use_chirality),
        )
        return generator.GetCountFingerprintAsNumPy(mol).astype(np.float32, copy=False)
    if spec.kind == "rdkit_path":
        generator = GetRDKitFPGenerator(
            minPath=int(spec.min_path),
            maxPath=int(spec.max_path),
            fpSize=int(spec.n_bits),
        )
        return generator.GetFingerprintAsNumPy(mol).astype(np.float32, copy=False)
    if spec.kind == "maccs":
        if spec.n_bits != 167:
            raise ValueError("RDKit MACCS has exactly 167 bits")
        return np.fromiter(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32, count=167)
    raise ValueError(f"Unsupported fingerprint kind: {spec.kind}")


def fingerprint_matrix(smiles: Iterable[object], spec: FingerprintSpec) -> np.ndarray:
    rows = [fingerprint(value, spec) for value in smiles]
    if not rows:
        return np.empty((0, spec.n_bits), dtype=np.float32)
    matrix = np.vstack(rows).astype(np.float32, copy=False)
    if matrix.shape[1] != spec.n_bits or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid fingerprint matrix for {spec.identifier}: {matrix.shape}")
    return matrix


def morgan_bitvectors(
    smiles: Iterable[object],
    *,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = False,
):
    from rdkit import Chem
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    generator = GetMorganGenerator(
        radius=int(radius), fpSize=int(n_bits), includeChirality=bool(use_chirality)
    )
    vectors = []
    for value in smiles:
        molecule = standardize_smiles(value)
        mol = Chem.MolFromSmiles(molecule.standardized_parent_smiles)
        if mol is None:
            raise ValueError("Standardized molecule unexpectedly failed to parse")
        vectors.append(generator.GetFingerprint(mol))
    return vectors
