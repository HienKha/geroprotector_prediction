"""Global cache of strictly molecule-local V5 features."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...chemistry.descriptors import descriptor_frame
from ...chemistry.fingerprints import FingerprintSpec, fingerprint_matrix
from ...chemistry.standardize import rdkit_version
from ...hashing import atomic_write_json, canonical_sha256, sha256_bytes, sha256_file


def _contract_hash(
    specs: Mapping[str, FingerprintSpec], descriptor_names: Iterable[object]
) -> str:
    return canonical_sha256(
        {
            "schema": "geroprotector.v5_feature_contract.v1",
            "rdkit_version": rdkit_version(),
            "specs": [asdict(specs[key]) for key in sorted(specs)],
            "descriptor_panel": "chemistry_32",
            "descriptor_names": list(map(str, descriptor_names)),
        }
    )


def _content_hash(
    ids: tuple[str, ...],
    smiles: tuple[str, ...],
    blocks: Mapping[str, np.ndarray],
    descriptors: pd.DataFrame,
    *,
    feature_contract_hash: str,
) -> str:
    return canonical_sha256(
        {
            "schema": "geroprotector.v5_feature_content.v1",
            "feature_contract_sha256": feature_contract_hash,
            "ids": list(ids),
            "smiles": list(smiles),
            "block_sha256": {
                name: sha256_bytes(np.ascontiguousarray(matrix).tobytes())
                for name, matrix in sorted(blocks.items())
            },
            "descriptor_matrix_sha256": sha256_bytes(
                np.ascontiguousarray(descriptors.to_numpy(dtype=np.float64)).tobytes()
            ),
        }
    )


def candidate_specs(config: Mapping[str, Any]) -> tuple[FingerprintSpec, ...]:
    candidates = config["representation"]["fingerprint_candidates"]
    specs: list[FingerprintSpec] = []
    for kind in ("morgan_bit", "morgan_count"):
        block = candidates[kind]
        for radius in block["radii"]:
            for chirality in block["use_chirality"]:
                for n_bits in block["n_bits"]:
                    specs.append(
                        FingerprintSpec(
                            kind=kind,
                            radius=int(radius),
                            n_bits=int(n_bits),
                            use_chirality=bool(chirality),
                        )
                    )
    path = candidates["rdkit_path"]
    for n_bits in path["n_bits"]:
        specs.append(
            FingerprintSpec(
                kind="rdkit_path",
                n_bits=int(n_bits),
                min_path=int(path["min_path"]),
                max_path=int(path["max_path"]),
            )
        )
    specs.append(FingerprintSpec(kind="maccs", n_bits=167))
    if len({spec.identifier for spec in specs}) != len(specs):
        raise ValueError("V5 fingerprint candidate IDs are not unique")
    return tuple(specs)


def semantic_family(spec: FingerprintSpec) -> str:
    return spec.kind


@dataclass(frozen=True)
class V5FeatureBank:
    ids: tuple[str, ...]
    smiles: tuple[str, ...]
    blocks: Mapping[str, np.ndarray]
    specs: Mapping[str, FingerprintSpec]
    descriptors: pd.DataFrame
    feature_contract_hash: str
    bank_hash: str

    def __post_init__(self) -> None:
        if len(self.ids) != len(set(self.ids)) or len(self.ids) != len(self.smiles):
            raise ValueError("Feature bank IDs/SMILES are invalid")
        for name, matrix in self.blocks.items():
            if matrix.shape[0] != len(self.ids) or not np.isfinite(matrix).all():
                raise ValueError(f"Feature block is invalid: {name} {matrix.shape}")
        if tuple(self.descriptors.index.astype(str)) != self.ids:
            raise ValueError("Descriptor index differs from feature-bank IDs")

    def assert_integrity(self) -> None:
        expected_contract = _contract_hash(self.specs, self.descriptors.columns)
        if self.feature_contract_hash != expected_contract:
            raise ValueError("V5 feature-contract hash is invalid")
        expected_content = _content_hash(
            self.ids,
            self.smiles,
            self.blocks,
            self.descriptors,
            feature_contract_hash=self.feature_contract_hash,
        )
        if self.bank_hash != expected_content:
            raise ValueError("V5 feature-bank content hash is invalid")

    @classmethod
    def build(
        cls,
        *,
        compound_ids: Iterable[object],
        smiles: Iterable[object],
        config: Mapping[str, Any],
    ) -> V5FeatureBank:
        pairs = sorted(
            zip(map(str, compound_ids), map(str, smiles), strict=True), key=lambda item: item[0]
        )
        ids = tuple(item[0] for item in pairs)
        structures = tuple(item[1] for item in pairs)
        if len(ids) != len(set(ids)):
            raise ValueError("Feature bank requires unique compound IDs")
        specs = candidate_specs(config)
        blocks = {spec.identifier: fingerprint_matrix(structures, spec) for spec in specs}
        spec_map = {spec.identifier: spec for spec in specs}
        descriptors = descriptor_frame(structures, panel="chemistry_32", index=ids)
        feature_contract_hash = _contract_hash(spec_map, descriptors.columns)
        bank_hash = _content_hash(
            ids,
            structures,
            blocks,
            descriptors,
            feature_contract_hash=feature_contract_hash,
        )
        bank = cls(
            ids,
            structures,
            blocks,
            spec_map,
            descriptors,
            feature_contract_hash,
            bank_hash,
        )
        bank.assert_integrity()
        return bank

    def positions(self, requested_ids: Iterable[object]) -> np.ndarray:
        lookup = {compound: index for index, compound in enumerate(self.ids)}
        requested = tuple(map(str, requested_ids))
        if len(requested) != len(set(requested)):
            raise ValueError("Requested feature IDs are duplicated")
        missing = set(requested) - set(lookup)
        if missing:
            raise KeyError(f"Feature IDs absent from bank: {sorted(missing)[:5]}")
        return np.asarray([lookup[value] for value in requested], dtype=int)

    def matrix(self, spec_id: str, requested_ids: Iterable[object]) -> np.ndarray:
        return self.blocks[spec_id][self.positions(requested_ids)]

    def descriptor_matrix(self, requested_ids: Iterable[object]) -> np.ndarray:
        requested = tuple(map(str, requested_ids))
        return self.descriptors.loc[list(requested)].to_numpy(dtype=float)

    def subset(self, requested_ids: Iterable[object]) -> V5FeatureBank:
        requested = tuple(map(str, requested_ids))
        positions = self.positions(requested)
        structures = tuple(self.smiles[position] for position in positions)
        blocks = {key: value[positions] for key, value in self.blocks.items()}
        descriptors = self.descriptors.loc[list(requested)].copy()
        feature_contract_hash = self.feature_contract_hash
        subset = V5FeatureBank(
            requested,
            structures,
            blocks,
            dict(self.specs),
            descriptors,
            feature_contract_hash,
            _content_hash(
                requested,
                structures,
                blocks,
                descriptors,
                feature_contract_hash=feature_contract_hash,
            ),
        )
        subset.assert_integrity()
        return subset

    def save(self, directory: str | Path) -> dict[str, Any]:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        manifest_path = destination / "feature_manifest.json"
        if manifest_path.exists():
            raise FileExistsError(f"Refusing to overwrite feature bank: {destination}")
        artifacts: list[dict[str, Any]] = []
        for spec_id in sorted(self.blocks):
            path = destination / f"{spec_id}.npz"
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", dir=destination
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle, ids=np.asarray(self.ids), X=self.blocks[spec_id]
                    )
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            artifacts.append(
                {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "spec": asdict(self.specs[spec_id]),
                }
            )
        descriptor_path = destination / "chemistry_32.parquet"
        descriptor_frame_to_save = self.descriptors.reset_index()
        descriptor_frame_to_save.to_parquet(descriptor_path, index=False)
        artifacts.append({"path": descriptor_path.name, "sha256": sha256_file(descriptor_path)})
        manifest = {
            "schema_version": "geroprotector.v5_feature_bank.v1",
            "feature_contract_sha256": self.feature_contract_hash,
            "bank_hash": self.bank_hash,
            "n_compounds": len(self.ids),
            "label_columns_present": False,
            "cohort_statistics_used": False,
            "artifacts": artifacts,
        }
        atomic_write_json(manifest_path, manifest)
        return manifest
