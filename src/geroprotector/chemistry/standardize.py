"""Deterministic, version-recorded RDKit parent standardisation."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any


class ChemistryError(ValueError):
    """Raised when a structure cannot be parsed or standardised safely."""


@dataclass(frozen=True)
class StandardizedMolecule:
    standardization_input_smiles: str
    standardized_parent_smiles: str
    full_inchikey: str
    connectivity_inchikey: str
    tautomer_hash: str
    murcko_scaffold_smiles: str
    molecular_formula: str
    heavy_atom_count: int
    formal_charge: int
    component_count: int
    raw_component_smiles: tuple[str, ...]
    metal_atomic_numbers: tuple[int, ...]
    qc_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["qc_flags"] = list(self.qc_flags)
        return value


STANDARDIZATION_CONTRACT = {
    "schema_version": "geroprotector.standardize.v1",
    "cleanup": True,
    "fragment_parent": "rdMolStandardize.FragmentParent",
    "uncharger": "rdMolStandardize.Uncharger",
    "tautomer_identity_role": "diagnostic_only",
    "canonical_isomeric_smiles": True,
    "identity_key": "connectivity_inchikey",
}


def rdkit_version() -> str:
    import rdkit

    return str(rdkit.__version__)


def _fallback_key(smiles: str) -> tuple[str, str]:
    digest = hashlib.sha256(smiles.encode("utf-8")).hexdigest().upper()
    return f"CANON-{digest[:20]}", f"CANON-{digest[:14]}-{digest[14:24]}-X"


def standardize_smiles(smiles: object) -> StandardizedMolecule:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold

    text = str(smiles).strip()
    if not text:
        raise ChemistryError("SMILES is blank")
    raw_mol = Chem.MolFromSmiles(text)
    if raw_mol is None:
        raise ChemistryError(f"RDKit cannot parse SMILES: {text!r}")
    raw_fragments = Chem.GetMolFrags(raw_mol, asMols=True, sanitizeFrags=True)
    component_count = len(raw_fragments)
    raw_component_smiles = tuple(
        sorted(
            Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
            for fragment in raw_fragments
        )
    )
    metal_atomic_numbers = tuple(
        sorted(
            {
                atom.GetAtomicNum()
                for atom in raw_mol.GetAtoms()
                if atom.GetAtomicNum() not in {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}
            }
        )
    )
    flags: list[str] = []
    if component_count > 1:
        flags.append("multi_component_parent_selected")
    if metal_atomic_numbers:
        flags.append("metal_present_in_input")
        metal_indices = {
            atom.GetIdx()
            for atom in raw_mol.GetAtoms()
            if atom.GetAtomicNum() in metal_atomic_numbers
        }
        if any(
            bond.GetBeginAtomIdx() in metal_indices or bond.GetEndAtomIdx() in metal_indices
            for bond in raw_mol.GetBonds()
        ):
            flags.append("metal_covalent_or_coordinate_bond_present")
        elif component_count > 1:
            flags.append("disconnected_metal_component_present")
    try:
        cleaned = rdMolStandardize.Cleanup(raw_mol)
        parent = rdMolStandardize.FragmentParent(cleaned)
        parent = rdMolStandardize.Uncharger().uncharge(parent)
        Chem.SanitizeMol(parent)
    except Exception as exc:
        raise ChemistryError(f"RDKit cannot standardise SMILES: {text!r}") from exc
    if not any(atom.GetAtomicNum() == 6 for atom in parent.GetAtoms()):
        flags.append("no_carbon_parent")
    canonical = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    full_key = Chem.MolToInchiKey(parent)
    if not full_key:
        raise ChemistryError("RDKit could not derive a primary InChIKey")
    connectivity = full_key.split("-", maxsplit=1)[0]
    parent_atomic_numbers = {atom.GetAtomicNum() for atom in parent.GetAtoms()}
    if metal_atomic_numbers and not set(metal_atomic_numbers).issubset(parent_atomic_numbers):
        flags.append("metal_removed_by_parent_selection")
    tautomer = rdMolStandardize.TautomerEnumerator().Canonicalize(parent)
    tautomer_smiles = Chem.MolToSmiles(tautomer, canonical=True, isomericSmiles=False)
    tautomer_hash = hashlib.sha256(tautomer_smiles.encode("utf-8")).hexdigest()
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=parent, includeChirality=False)
    return StandardizedMolecule(
        standardization_input_smiles=text,
        standardized_parent_smiles=canonical,
        full_inchikey=full_key,
        connectivity_inchikey=connectivity,
        tautomer_hash=tautomer_hash,
        murcko_scaffold_smiles=scaffold,
        molecular_formula=rdMolDescriptors.CalcMolFormula(parent),
        heavy_atom_count=int(parent.GetNumHeavyAtoms()),
        formal_charge=int(Chem.GetFormalCharge(parent)),
        component_count=int(component_count),
        raw_component_smiles=raw_component_smiles,
        metal_atomic_numbers=metal_atomic_numbers,
        qc_flags=tuple(sorted(flags)),
    )
