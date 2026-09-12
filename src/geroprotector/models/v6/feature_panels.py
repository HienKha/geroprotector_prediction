"""Compact molecule-local inputs with fold-local panel transformations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from ...chemistry.descriptors import descriptor_frame
from ...chemistry.fingerprints import FingerprintSpec, fingerprint_matrix
from ...chemistry.standardize import rdkit_version
from ...hashing import canonical_sha256, sha256_bytes

_V6_MORGAN_SPEC = FingerprintSpec(kind="morgan_count", radius=2, n_bits=2048)


def _feature_contract_hash(
    rdkit_names: Sequence[object], chemistry_names: Sequence[object]
) -> str:
    return canonical_sha256(
        {
            "schema": "geroprotector.v6_feature_contract.v1",
            "rdkit_version": rdkit_version(),
            "rdkit2d_panel": "rdkit2d_217",
            "rdkit_names": list(map(str, rdkit_names)),
            "chemistry_panel": "chemistry_32",
            "chemistry_names": list(map(str, chemistry_names)),
            "morgan": asdict(_V6_MORGAN_SPEC),
            "molecule_local": True,
            "label_columns_present": False,
        }
    )


def _feature_content_hash(
    ids: tuple[str, ...],
    smiles: tuple[str, ...],
    rdkit2d: pd.DataFrame,
    chemistry32: pd.DataFrame,
    morgan_count: np.ndarray,
    *,
    feature_contract_hash: str,
) -> str:
    return canonical_sha256(
        {
            "schema": "geroprotector.v6_feature_content.v1",
            "feature_contract_sha256": feature_contract_hash,
            "ids": list(ids),
            "smiles": list(smiles),
            "rdkit_matrix_sha256": sha256_bytes(
                np.ascontiguousarray(rdkit2d.to_numpy(dtype=np.float64)).tobytes()
            ),
            "chemistry_matrix_sha256": sha256_bytes(
                np.ascontiguousarray(chemistry32.to_numpy(dtype=np.float64)).tobytes()
            ),
            "morgan_matrix_sha256": sha256_bytes(np.ascontiguousarray(morgan_count).tobytes()),
        }
    )


@dataclass(frozen=True)
class V6FeatureStore:
    ids: tuple[str, ...]
    smiles: tuple[str, ...]
    rdkit2d: pd.DataFrame
    chemistry32: pd.DataFrame
    morgan_count: np.ndarray
    feature_contract_hash: str
    store_hash: str

    def __post_init__(self) -> None:
        if len(self.ids) != len(set(self.ids)) or len(self.ids) != len(self.smiles):
            raise ValueError("V6 feature-store IDs/SMILES are invalid")
        expected_index = self.ids
        if tuple(map(str, self.rdkit2d.index)) != expected_index:
            raise ValueError("V6 RDKit2D rows differ from feature-store IDs")
        if tuple(map(str, self.chemistry32.index)) != expected_index:
            raise ValueError("V6 chemistry32 rows differ from feature-store IDs")
        if self.morgan_count.shape != (len(self.ids), int(_V6_MORGAN_SPEC.n_bits)):
            raise ValueError("V6 Morgan matrix has an invalid shape")

    def assert_integrity(self) -> None:
        expected_contract = _feature_contract_hash(
            self.rdkit2d.columns, self.chemistry32.columns
        )
        if self.feature_contract_hash != expected_contract:
            raise ValueError("V6 feature-contract hash is invalid")
        expected_content = _feature_content_hash(
            self.ids,
            self.smiles,
            self.rdkit2d,
            self.chemistry32,
            self.morgan_count,
            feature_contract_hash=self.feature_contract_hash,
        )
        if self.store_hash != expected_content:
            raise ValueError("V6 feature-store content hash is invalid")

    @classmethod
    def build(
        cls, *, compound_ids: Iterable[object], smiles: Iterable[object]
    ) -> V6FeatureStore:
        pairs = sorted(
            zip(map(str, compound_ids), map(str, smiles), strict=True), key=lambda item: item[0]
        )
        ids = tuple(item[0] for item in pairs)
        structures = tuple(item[1] for item in pairs)
        if len(ids) != len(set(ids)):
            raise ValueError("V6 store IDs are duplicated")
        rdkit = descriptor_frame(structures, panel="rdkit2d_217", index=ids)
        chemistry = descriptor_frame(structures, panel="chemistry_32", index=ids)
        morgan = fingerprint_matrix(structures, _V6_MORGAN_SPEC)
        feature_contract_hash = _feature_contract_hash(rdkit.columns, chemistry.columns)
        store = cls(
            ids,
            structures,
            rdkit,
            chemistry,
            morgan,
            feature_contract_hash,
            _feature_content_hash(
                ids,
                structures,
                rdkit,
                chemistry,
                morgan,
                feature_contract_hash=feature_contract_hash,
            ),
        )
        store.assert_integrity()
        return store

    def positions(self, ids: Sequence[str]) -> np.ndarray:
        requested = tuple(map(str, ids))
        if len(requested) != len(set(requested)):
            raise ValueError("Requested V6 IDs are duplicated")
        lookup = {value: index for index, value in enumerate(self.ids)}
        missing = set(requested) - set(lookup)
        if missing:
            raise KeyError(f"V6 IDs absent from store: {sorted(missing)[:5]}")
        return np.asarray([lookup[value] for value in requested], dtype=int)

    def subset(self, ids: Sequence[str]) -> V6FeatureStore:
        requested = tuple(map(str, ids))
        positions = self.positions(requested)
        structures = tuple(self.smiles[position] for position in positions)
        rdkit = self.rdkit2d.loc[list(requested)].copy()
        chemistry = self.chemistry32.loc[list(requested)].copy()
        morgan = self.morgan_count[positions]
        subset = V6FeatureStore(
            requested,
            structures,
            rdkit,
            chemistry,
            morgan,
            self.feature_contract_hash,
            _feature_content_hash(
                requested,
                structures,
                rdkit,
                chemistry,
                morgan,
                feature_contract_hash=self.feature_contract_hash,
            ),
        )
        subset.assert_integrity()
        return subset


@dataclass
class DescriptorPanelTransformer:
    panel: str
    imputer: SimpleImputer | None = None
    keep_: np.ndarray | None = None
    fit_ids_: tuple[str, ...] = ()
    feature_names_: tuple[str, ...] = ()

    def fit(
        self, store: V6FeatureStore, *, fit_ids: Sequence[str]
    ) -> DescriptorPanelTransformer:
        ids = tuple(map(str, fit_ids))
        frame = store.rdkit2d if self.panel == "rdkit2d_217" else store.chemistry32
        X = frame.loc[list(ids)].to_numpy(dtype=float)
        finite = np.isfinite(X).any(axis=0)
        if not finite.any():
            raise ValueError("Descriptor panel has no finite training columns")
        provisional = SimpleImputer(strategy="median").fit_transform(X[:, finite])
        nonconstant = np.ptp(provisional, axis=0) > 0
        kept_indices = np.flatnonzero(finite)[nonconstant]
        if not len(kept_indices):
            raise ValueError("Descriptor panel has no nonconstant training columns")
        self.keep_ = kept_indices
        self.imputer = SimpleImputer(strategy="median").fit(X[:, kept_indices])
        self.fit_ids_ = ids
        self.feature_names_ = tuple(str(frame.columns[index]) for index in kept_indices)
        return self

    def transform(self, store: V6FeatureStore, ids: Sequence[str]) -> pd.DataFrame:
        if self.imputer is None or self.keep_ is None:
            raise RuntimeError("Descriptor panel is not fitted")
        frame = store.rdkit2d if self.panel == "rdkit2d_217" else store.chemistry32
        requested = tuple(map(str, ids))
        output = self.imputer.transform(
            frame.loc[list(requested)].to_numpy(dtype=float)[:, self.keep_]
        )
        if not np.isfinite(output).all():
            raise ValueError("Descriptor panel transform is non-finite")
        return pd.DataFrame(output, index=requested, columns=self.feature_names_)

    def get_manifest(self) -> dict[str, Any]:
        return {
            "kind": "descriptor_panel_train_median_drop_constant",
            "panel": self.panel,
            "fit_ids": list(self.fit_ids_),
            "feature_names": list(self.feature_names_),
        }


@dataclass
class MorganSVDPanelTransformer:
    n_components: int
    random_state: int
    support_: np.ndarray | None = None
    svd_: TruncatedSVD | None = None
    descriptor_imputer_: SimpleImputer | None = None
    descriptor_scaler_: RobustScaler | None = None
    descriptor_keep_: np.ndarray | None = None
    fit_ids_: tuple[str, ...] = ()
    feature_names_: tuple[str, ...] = ()

    def fit(
        self, store: V6FeatureStore, *, fit_ids: Sequence[str]
    ) -> MorganSVDPanelTransformer:
        ids = tuple(map(str, fit_ids))
        positions = store.positions(ids)
        counts = store.morgan_count[positions]
        support = np.count_nonzero(counts, axis=0) >= 2
        if not support.any():
            raise ValueError("Morgan panel has no supported columns")
        maximum = max(1, min(counts[:, support].shape) - 1)
        if int(self.n_components) > maximum:
            raise ValueError(
                f"Requested Morgan SVD dimension {self.n_components} exceeds "
                f"active-fit maximum {maximum}"
            )
        effective = int(self.n_components)
        self.svd_ = TruncatedSVD(n_components=effective, random_state=self.random_state)
        self.svd_.fit(counts[:, support])
        descriptor = store.chemistry32.loc[list(ids)].to_numpy(dtype=float)
        finite = np.isfinite(descriptor).any(axis=0)
        self.descriptor_keep_ = np.flatnonzero(finite)
        self.descriptor_imputer_ = SimpleImputer(strategy="median").fit(
            descriptor[:, self.descriptor_keep_]
        )
        imputed = self.descriptor_imputer_.transform(descriptor[:, self.descriptor_keep_])
        self.descriptor_scaler_ = RobustScaler().fit(imputed)
        self.support_ = support
        self.fit_ids_ = ids
        self.feature_names_ = tuple(
            [f"morgan_svd::{index}" for index in range(effective)]
            + [
                f"chemistry32::{store.chemistry32.columns[index]}"
                for index in self.descriptor_keep_
            ]
        )
        return self

    def transform(self, store: V6FeatureStore, ids: Sequence[str]) -> pd.DataFrame:
        if (
            self.support_ is None
            or self.svd_ is None
            or self.descriptor_keep_ is None
            or self.descriptor_imputer_ is None
            or self.descriptor_scaler_ is None
        ):
            raise RuntimeError("Morgan SVD panel is not fitted")
        requested = tuple(map(str, ids))
        positions = store.positions(requested)
        latent = self.svd_.transform(store.morgan_count[positions][:, self.support_])
        descriptor = store.chemistry32.loc[list(requested)].to_numpy(dtype=float)
        descriptor = self.descriptor_scaler_.transform(
            self.descriptor_imputer_.transform(descriptor[:, self.descriptor_keep_])
        )
        output = np.concatenate([latent, descriptor], axis=1)
        if not np.isfinite(output).all():
            raise ValueError("Morgan SVD panel transform is non-finite")
        return pd.DataFrame(output, index=requested, columns=self.feature_names_)

    def get_manifest(self) -> dict[str, Any]:
        if self.support_ is None or self.svd_ is None:
            raise RuntimeError("Morgan SVD panel is not fitted")
        return {
            "kind": "morgan_count_train_support_svd_plus_chemistry32",
            "requested_components": int(self.n_components),
            "effective_components": int(self.svd_.n_components),
            "supported_morgan_columns": np.flatnonzero(self.support_).tolist(),
            "fit_ids": list(self.fit_ids_),
            "feature_names": list(self.feature_names_),
        }


def panel_candidates(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    panels = config["input_panels"]
    candidates: list[dict[str, Any]] = []
    for name in ("rdkit2d_217", "chemistry_32"):
        if panels[name]["enabled"]:
            candidates.append({"panel": name, "n_components": None})
    block = panels["morgan_svd_plus_descriptors"]
    if block["enabled"]:
        if (
            block.get("append_maccs") is not False
            or block.get("append_chemistry_32") is not True
        ):
            raise ValueError("Primary V6 Panel C contract changed")
        candidates.extend(
            {"panel": "morgan_svd_plus_descriptors", "n_components": int(value)}
            for value in block["svd_components"]
        )
    for unsupported in ("v5_locked", "frozen_embedding_plus_descriptors"):
        if panels[unsupported]["enabled"]:
            raise NotImplementedError(
                f"{unsupported} requires a separately frozen upstream artifact "
                "and is not primary V6"
            )
    return tuple(candidates)


def panel_declared_dimension(spec: Mapping[str, Any]) -> int:
    """Prespecified nominal dimension used only for deterministic simplicity ties."""

    name = str(spec["panel"])
    if name == "rdkit2d_217":
        return 217
    if name == "chemistry_32":
        return 32
    if name == "morgan_svd_plus_descriptors":
        return int(spec["n_components"]) + 32
    raise ValueError(f"Unsupported V6 panel: {name}")


def build_panel_transformer(spec: Mapping[str, Any], *, seed: int):
    name = spec["panel"]
    if name in {"rdkit2d_217", "chemistry_32"}:
        return DescriptorPanelTransformer(name)
    if name == "morgan_svd_plus_descriptors":
        return MorganSVDPanelTransformer(int(spec["n_components"]), int(seed))
    raise ValueError(f"Unsupported V6 panel: {name}")
