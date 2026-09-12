"""Mandatory reference models evaluated on the shared nested split registry."""

from __future__ import annotations

import hashlib
import itertools
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ..chemistry.descriptors import descriptor_frame
from ..chemistry.fingerprints import FingerprintSpec, fingerprint_matrix
from ..chemistry.standardize import rdkit_version
from ..hashing import atomic_write_json, canonical_sha256, sha256_bytes, sha256_file
from ..validation.split_registry import validate_selection_folds
from .resources import abort_on_resource_exhaustion

REFERENCE_MODEL_IDS = {
    "R0_prevalence",
    "R1_elastic_net",
    "R2_extra_trees_v3_compatible",
    "R3_xgboost",
    "R4_exact_tanimoto_svc",
    "R5_paper_seven_descriptor_linear_svm",
}

PAPER_SEVEN_FEATURES = (
    "paper7::total_molweight",
    "paper7::clogp",
    "paper7::h_acceptors",
    "paper7::h_donors",
    "paper7::total_surface_area",
    "paper7::relative_psa",
    "paper7::rotatable_bonds",
)


def _reference_feature_contract_hash(
    names: Mapping[str, tuple[str, ...]],
) -> str:
    """Hash extractor semantics without binding them to one cohort."""

    return canonical_sha256(
        {
            "schema": "geroprotector.reference_feature_contract.v1",
            "rdkit_version": rdkit_version(),
            "feature_names": {key: list(value) for key, value in sorted(names.items())},
            "fingerprint_contracts": {
                "morgan_bit_r2_2048": {
                    "kind": "morgan_bit",
                    "radius": 2,
                    "n_bits": 2048,
                    "use_chirality": False,
                },
                "rdkit2d_morgan_count_maccs": {
                    "morgan_kind": "morgan_count",
                    "morgan_radius": 2,
                    "morgan_n_bits": 2048,
                    "morgan_use_chirality": False,
                    "maccs_n_bits": 167,
                },
                "v3_rdkit2d_multifp": "explicit_v3_generator_arguments_in_code",
            },
            "molecule_local": True,
            "label_columns_present": False,
        }
    )


def _reference_feature_content_hash(
    ids: tuple[str, ...],
    matrices: Mapping[str, np.ndarray],
    *,
    feature_contract_hash: str,
) -> str:
    return canonical_sha256(
        {
            "schema": "geroprotector.reference_feature_content.v1",
            "feature_contract_sha256": feature_contract_hash,
            "ids": list(ids),
            "feature_matrix_sha256": {
                key: sha256_bytes(
                    np.ascontiguousarray(np.asarray(matrix, dtype=np.float64)).tobytes()
                )
                for key, matrix in sorted(matrices.items())
            },
        }
    )


def _v3_multifingerprint(smiles: Sequence[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Reproduce the molecule-local V3 RDKit2D + multi-fingerprint contract."""

    from rdkit import Chem
    from rdkit.Chem import Descriptors, MACCSkeys
    from rdkit.Chem.rdFingerprintGenerator import (
        GetAtomPairGenerator,
        GetMorganGenerator,
        GetRDKitFPGenerator,
        GetTopologicalTorsionGenerator,
    )

    descriptors = list(Descriptors._descList)
    descriptor_values = descriptor_frame(smiles, panel="rdkit2d_217").to_numpy(dtype=float)
    n_bits = 2048
    morgan_r2 = GetMorganGenerator(
        radius=2,
        countSimulation=False,
        includeChirality=False,
        useBondTypes=True,
        onlyNonzeroInvariants=False,
        includeRingMembership=True,
        countBounds=None,
        fpSize=n_bits,
        atomInvariantsGenerator=None,
        bondInvariantsGenerator=None,
        includeRedundantEnvironments=False,
    )
    morgan_r3 = GetMorganGenerator(
        radius=3,
        countSimulation=False,
        includeChirality=False,
        useBondTypes=True,
        onlyNonzeroInvariants=False,
        includeRingMembership=True,
        countBounds=None,
        fpSize=n_bits,
        atomInvariantsGenerator=None,
        bondInvariantsGenerator=None,
        includeRedundantEnvironments=False,
    )
    generators = (
        (
            "morgan_bit_r2",
            morgan_r2,
            "bit",
        ),
        (
            "morgan_bit_r3",
            morgan_r3,
            "bit",
        ),
        (
            "morgan_count_r2",
            morgan_r2,
            "count",
        ),
        (
            "rdkit_path_bit",
            GetRDKitFPGenerator(
                minPath=1,
                maxPath=7,
                useHs=True,
                branchedPaths=True,
                useBondOrder=True,
                countSimulation=False,
                countBounds=None,
                fpSize=n_bits,
                numBitsPerFeature=2,
                atomInvariantsGenerator=None,
            ),
            "bit",
        ),
        (
            "atom_pair_bit",
            GetAtomPairGenerator(
                minDistance=1,
                maxDistance=30,
                includeChirality=False,
                use2D=True,
                countSimulation=True,
                countBounds=None,
                fpSize=n_bits,
                atomInvariantsGenerator=None,
            ),
            "bit",
        ),
        (
            "topological_torsion_bit",
            GetTopologicalTorsionGenerator(
                includeChirality=False,
                torsionAtomCount=4,
                countSimulation=True,
                countBounds=None,
                fpSize=n_bits,
                atomInvariantsGenerator=None,
            ),
            "bit",
        ),
    )
    rows: list[np.ndarray] = []
    for text in smiles:
        mol = Chem.MolFromSmiles(str(text))
        if mol is None:
            raise ValueError("Curated standardized SMILES unexpectedly failed to parse")
        blocks: list[np.ndarray] = []
        for _, generator, kind in generators:
            value = (
                generator.GetCountFingerprintAsNumPy(mol)
                if kind == "count"
                else generator.GetFingerprintAsNumPy(mol)
            )
            blocks.append(np.asarray(value, dtype=np.float64))
        blocks.append(np.fromiter(MACCSkeys.GenMACCSKeys(mol), dtype=np.float64, count=167))
        rows.append(np.concatenate(blocks))
    names = tuple(f"rdkit2d::{name}" for name, _ in descriptors)
    for block_name, _, _ in generators:
        names += tuple(f"{block_name}::{index}" for index in range(n_bits))
    names += tuple(f"maccs::{index}" for index in range(167))
    output = np.concatenate([descriptor_values, np.vstack(rows)], axis=1)
    if output.shape[1] != len(names):
        raise RuntimeError("V3-compatible feature-name contract is misaligned")
    return output, names


@dataclass(frozen=True)
class ReferenceFeatureStore:
    ids: tuple[str, ...]
    matrices: Mapping[str, np.ndarray]
    names: Mapping[str, tuple[str, ...]]
    feature_contract_hash: str
    store_hash: str

    def __post_init__(self) -> None:
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Reference feature-store IDs are duplicated")
        if set(self.matrices) != set(self.names):
            raise ValueError("Reference feature blocks and names differ")
        for key, matrix in self.matrices.items():
            if np.asarray(matrix).shape != (len(self.ids), len(self.names[key])):
                raise ValueError(f"Reference feature block is misaligned: {key}")

    def assert_integrity(self) -> None:
        expected_contract = _reference_feature_contract_hash(self.names)
        if self.feature_contract_hash != expected_contract:
            raise ValueError("Reference feature-contract hash is invalid")
        expected_content = _reference_feature_content_hash(
            self.ids,
            self.matrices,
            feature_contract_hash=self.feature_contract_hash,
        )
        if self.store_hash != expected_content:
            raise ValueError("Reference feature-store content hash is invalid")

    @classmethod
    def build(cls, curated: pd.DataFrame) -> ReferenceFeatureStore:
        required = {"compound_id", "standardized_parent_smiles", *PAPER_SEVEN_FEATURES}
        missing = required - set(curated)
        if missing:
            raise ValueError(f"Reference feature inputs missing: {sorted(missing)}")
        frame = curated.sort_values("compound_id", kind="stable").reset_index(drop=True)
        ids = tuple(frame["compound_id"].astype(str))
        if len(ids) != len(set(ids)):
            raise ValueError("Reference feature store requires unique compound IDs")
        smiles = tuple(frame["standardized_parent_smiles"].astype(str))
        chemistry = descriptor_frame(smiles, panel="chemistry_32", index=ids)
        rdkit2d = descriptor_frame(smiles, panel="rdkit2d_217", index=ids)
        morgan_bit_spec = FingerprintSpec(kind="morgan_bit", radius=2, n_bits=2048)
        morgan_count_spec = FingerprintSpec(kind="morgan_count", radius=2, n_bits=2048)
        maccs_spec = FingerprintSpec(kind="maccs", n_bits=167)
        morgan_bit = fingerprint_matrix(smiles, morgan_bit_spec).astype(np.float64)
        morgan_count = fingerprint_matrix(smiles, morgan_count_spec).astype(np.float64)
        maccs = fingerprint_matrix(smiles, maccs_spec).astype(np.float64)
        v3_matrix, v3_names = _v3_multifingerprint(smiles)
        paper_seven = frame.loc[:, list(PAPER_SEVEN_FEATURES)].to_numpy(dtype=float)
        matrices = {
            "chemistry_32": chemistry.to_numpy(dtype=float),
            "rdkit2d_morgan_count_maccs": np.concatenate(
                [rdkit2d.to_numpy(dtype=float), morgan_count, maccs], axis=1
            ),
            "v3_rdkit2d_multifp": v3_matrix,
            "morgan_bit_r2_2048": morgan_bit,
            "paper_seven": paper_seven,
        }
        names = {
            "chemistry_32": tuple(map(str, chemistry.columns)),
            "rdkit2d_morgan_count_maccs": (
                tuple(f"rdkit2d::{value}" for value in rdkit2d.columns)
                + tuple(f"morgan_count_r2::{index}" for index in range(2048))
                + tuple(f"maccs::{index}" for index in range(167))
            ),
            "v3_rdkit2d_multifp": v3_names,
            "morgan_bit_r2_2048": tuple(f"morgan_bit_r2::{index}" for index in range(2048)),
            "paper_seven": PAPER_SEVEN_FEATURES,
        }
        for key, matrix in matrices.items():
            if matrix.shape != (len(ids), len(names[key])):
                raise ValueError(f"Reference feature block is misaligned: {key}")
        feature_contract_hash = _reference_feature_contract_hash(names)
        store = cls(
            ids,
            matrices,
            names,
            feature_contract_hash,
            _reference_feature_content_hash(
                ids,
                matrices,
                feature_contract_hash=feature_contract_hash,
            ),
        )
        store.assert_integrity()
        return store

    def positions(self, ids: Sequence[str]) -> np.ndarray:
        requested = tuple(map(str, ids))
        if len(requested) != len(set(requested)):
            raise ValueError("Requested reference IDs are duplicated")
        lookup = {compound: index for index, compound in enumerate(self.ids)}
        missing = set(requested) - set(lookup)
        if missing:
            raise KeyError(f"Reference IDs missing: {sorted(missing)[:5]}")
        return np.asarray([lookup[value] for value in requested], dtype=int)

    def matrix(self, panel: str, ids: Sequence[str]) -> np.ndarray:
        return self.matrices[panel][self.positions(ids)]

    def subset(self, ids: Sequence[str]) -> ReferenceFeatureStore:
        requested = tuple(map(str, ids))
        positions = self.positions(requested)
        matrices = {key: value[positions] for key, value in self.matrices.items()}
        subset = ReferenceFeatureStore(
            requested,
            matrices,
            dict(self.names),
            self.feature_contract_hash,
            _reference_feature_content_hash(
                requested,
                matrices,
                feature_contract_hash=self.feature_contract_hash,
            ),
        )
        subset.assert_integrity()
        return subset


@dataclass
class FoldNumericPreprocessor:
    scale: bool
    min_nonzero_support: int = 1
    fit_ids: tuple[str, ...] = ()
    keep_: np.ndarray | None = None
    imputer_: SimpleImputer | None = None
    scaler_: StandardScaler | None = None

    def fit(self, X: np.ndarray, *, fit_ids: Sequence[str]) -> FoldNumericPreprocessor:
        matrix = np.asarray(X, dtype=float)
        ids = tuple(map(str, fit_ids))
        if matrix.shape[0] != len(ids) or len(ids) != len(set(ids)):
            raise ValueError("Reference preprocessor fit scope is invalid")
        finite = np.isfinite(matrix)
        support = np.sum(finite & (matrix != 0), axis=0)
        maximum = np.max(np.where(finite, matrix, -np.inf), axis=0)
        minimum = np.min(np.where(finite, matrix, np.inf), axis=0)
        keep = (
            finite.any(axis=0)
            & (maximum > minimum)
            & (support >= int(self.min_nonzero_support))
        )
        if not keep.any():
            raise ValueError("Reference preprocessing removed every feature")
        provisional_indices = np.flatnonzero(keep)
        provisional = SimpleImputer(strategy="median").fit_transform(
            matrix[:, provisional_indices]
        )
        duplicated = pd.DataFrame(provisional.T).duplicated(keep="first").to_numpy()
        final_indices = provisional_indices[~duplicated]
        self.keep_ = np.zeros(matrix.shape[1], dtype=bool)
        self.keep_[final_indices] = True
        self.imputer_ = SimpleImputer(strategy="median").fit(matrix[:, self.keep_])
        transformed = self.imputer_.transform(matrix[:, self.keep_])
        if self.scale:
            self.scaler_ = StandardScaler().fit(transformed)
        self.fit_ids = ids
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.keep_ is None or self.imputer_ is None:
            raise RuntimeError("Reference preprocessor is not fitted")
        transformed = self.imputer_.transform(np.asarray(X, dtype=float)[:, self.keep_])
        if self.scaler_ is not None:
            transformed = self.scaler_.transform(transformed)
        if not np.isfinite(transformed).all():
            raise ValueError("Reference preprocessing produced non-finite values")
        return np.asarray(transformed, dtype=float)


def _tanimoto(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left > 0, dtype=np.float64)
    b = np.asarray(right > 0, dtype=np.float64)
    intersection = a @ b.T
    union = a.sum(axis=1)[:, None] + b.sum(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


class _FittedReference:
    def __init__(self, model_id: str, candidate: Mapping[str, Any], *, seed: int):
        self.model_id = model_id
        self.candidate = dict(candidate)
        self.seed = int(seed)

    def fit(
        self,
        store: ReferenceFeatureStore,
        ids: Sequence[str],
        y: np.ndarray,
    ) -> _FittedReference:
        store.assert_integrity()
        fit_ids = tuple(map(str, ids))
        labels = np.asarray(y, dtype=int)
        if len(labels) != len(fit_ids) or set(labels) != {0, 1}:
            raise ValueError("Reference fit requires aligned two-class data")
        self.fit_ids_ = fit_ids
        if self.model_id == "R0_prevalence":
            self.prevalence_ = float(labels.mean())
            return self
        panel, scale, support = _reference_panel(self.model_id)
        raw = store.matrix(panel, fit_ids)
        if self.model_id == "R4_exact_tanimoto_svc":
            self.train_bits_ = raw
            self.estimator_ = SVC(
                C=float(self.candidate["C"]),
                kernel="precomputed",
                class_weight="balanced",
                probability=False,
                random_state=self.seed,
            ).fit(_tanimoto(raw, raw), labels)
            if tuple(map(int, self.estimator_.classes_)) != (0, 1):
                raise ValueError("Reference SVC class order differs from [0, 1]")
            return self
        self.preprocessor_ = FoldNumericPreprocessor(scale, support).fit(raw, fit_ids=fit_ids)
        X = self.preprocessor_.transform(raw)
        if self.model_id == "R1_elastic_net":
            estimator = LogisticRegression(
                C=float(self.candidate["C"]),
                l1_ratio=float(self.candidate["l1_ratio"]),
                penalty="elasticnet",
                solver="saga",
                class_weight="balanced",
                max_iter=10000,
                random_state=self.seed,
            )
        elif self.model_id == "R2_extra_trees_v3_compatible":
            estimator = ExtraTreesClassifier(
                n_estimators=int(self.candidate["n_estimators"]),
                min_samples_leaf=int(self.candidate["min_samples_leaf"]),
                max_features=self.candidate["max_features"],
                class_weight=self.candidate["class_weight"],
                n_jobs=-1,
                random_state=self.seed,
            )
        elif self.model_id == "R3_xgboost":
            estimator = XGBClassifier(
                **self.candidate,
                eval_metric="logloss",
                n_jobs=1,
                random_state=self.seed,
            )
        elif self.model_id == "R5_paper_seven_descriptor_linear_svm":
            estimator = LinearSVC(
                C=float(self.candidate["C"]),
                class_weight="balanced",
                random_state=self.seed,
                max_iter=10000,
            )
        else:
            raise ValueError(f"Unsupported reference model: {self.model_id}")
        fit_kwargs: dict[str, Any] = {}
        if self.model_id == "R3_xgboost":
            fit_kwargs["sample_weight"] = compute_sample_weight("balanced", labels)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            estimator.fit(X, labels, **fit_kwargs)
        if tuple(map(int, estimator.classes_)) != (0, 1):
            raise ValueError("Reference estimator class order differs from [0, 1]")
        self.estimator_ = estimator
        return self

    def predict_proba(self, store: ReferenceFeatureStore, ids: Sequence[str]) -> np.ndarray:
        store.assert_integrity()
        requested = tuple(map(str, ids))
        if set(requested) & set(self.fit_ids_):
            raise ValueError("Reference held-out prediction IDs overlap fitted IDs")
        if self.model_id == "R0_prevalence":
            positive = np.full(len(requested), self.prevalence_, dtype=float)
        else:
            panel, _, _ = _reference_panel(self.model_id)
            raw = store.matrix(panel, requested)
            if self.model_id == "R4_exact_tanimoto_svc":
                positive = expit(
                    self.estimator_.decision_function(_tanimoto(raw, self.train_bits_))
                )
            else:
                X = self.preprocessor_.transform(raw)
                if hasattr(self.estimator_, "predict_proba"):
                    positive = self.estimator_.predict_proba(X)[:, 1]
                else:
                    positive = expit(self.estimator_.decision_function(X))
        output = np.column_stack([1.0 - positive, positive])
        if output.shape != (len(requested), 2) or not np.isfinite(output).all():
            raise ValueError("Reference model returned invalid probabilities")
        return output


def _reference_panel(model_id: str) -> tuple[str, bool, int]:
    mapping = {
        "R1_elastic_net": ("chemistry_32", True, 1),
        "R2_extra_trees_v3_compatible": ("v3_rdkit2d_multifp", False, 3),
        "R3_xgboost": ("rdkit2d_morgan_count_maccs", False, 3),
        "R4_exact_tanimoto_svc": ("morgan_bit_r2_2048", False, 1),
        "R5_paper_seven_descriptor_linear_svm": ("paper_seven", True, 1),
    }
    if model_id not in mapping:
        raise ValueError(f"No feature panel for reference model: {model_id}")
    return mapping[model_id]


def _candidate_specs(model_id: str, config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    models = config["models"]
    if model_id == "R0_prevalence":
        return ({},)
    if model_id == "R1_elastic_net":
        return tuple(
            {"C": float(c), "l1_ratio": float(ratio)}
            for c, ratio in itertools.product(
                models["elastic_net"]["C"], models["elastic_net"]["l1_ratio"]
            )
        )
    if model_id == "R2_extra_trees_v3_compatible":
        settings = models["extra_trees"]
        return tuple(
            {
                "n_estimators": int(settings["n_estimators"]),
                "min_samples_leaf": int(leaf),
                "max_features": value,
                "class_weight": weight,
            }
            for leaf, value, weight in itertools.product(
                settings["min_samples_leaf"],
                settings["max_features"],
                settings["class_weight"],
            )
        )
    if model_id == "R3_xgboost":
        settings = models["xgboost"]
        keys = (
            "max_depth",
            "learning_rate",
            "n_estimators",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
        )
        values = itertools.product(*(settings[key] for key in keys))
        raw = [dict(zip(keys, item, strict=True)) for item in values]
        raw.sort(key=lambda item: hashlib.sha256(repr(item).encode()).digest())
        return tuple(raw[: int(settings["n_trials"])])
    if model_id == "R4_exact_tanimoto_svc":
        return tuple({"C": float(value)} for value in models["exact_tanimoto_svc"]["C"])
    if model_id == "R5_paper_seven_descriptor_linear_svm":
        values = models["paper_seven_descriptor_linear_svm"]["C"]
        return tuple({"C": float(value)} for value in values)
    raise ValueError(f"Unknown reference model: {model_id}")


class ReferencePipeline:
    """One prespecified reference, with selection confined to active training IDs."""

    def __init__(self, model_id: str, config: Mapping[str, Any], *, seed: int):
        if model_id not in REFERENCE_MODEL_IDS:
            raise ValueError(f"Unknown reference model ID: {model_id}")
        self.model_id = model_id
        self.config = dict(config)
        self.seed = int(seed)

    def fit(
        self,
        store: ReferenceFeatureStore,
        y: pd.Series,
        *,
        groups: pd.Series,
        selection_folds: Sequence[tuple[tuple[str, ...], tuple[str, ...]]],
    ) -> ReferencePipeline:
        store.assert_integrity()
        labels = y.copy()
        labels.index = labels.index.astype(str)
        if set(store.ids) != set(labels.index):
            raise ValueError("Reference store and labels are not ID-aligned")
        group_values = groups.copy()
        group_values.index = group_values.index.astype(str)
        folds = validate_selection_folds(
            fit_ids=store.ids,
            labels=labels,
            groups=group_values,
            folds=selection_folds,
        )
        rows: list[dict[str, Any]] = []
        for candidate in _candidate_specs(self.model_id, self.config):
            try:
                oof = pd.Series(index=labels.index, dtype=float)
                for fold_index, (train_ids, validation_ids) in enumerate(folds):
                    fitted = _FittedReference(
                        self.model_id,
                        candidate,
                        seed=self.seed + fold_index,
                    ).fit(
                        store,
                        train_ids,
                        labels.loc[list(train_ids)].to_numpy(dtype=int),
                    )
                    oof.loc[list(validation_ids)] = fitted.predict_proba(store, validation_ids)[
                        :, 1
                    ]
                if oof.isna().any():
                    raise RuntimeError(
                        "Reference candidate lacks complete inner OOF predictions"
                    )
                row = {
                    "status": "success",
                    "failure_type": None,
                    "failure_message": None,
                    "ap_positive": float(average_precision_score(labels, oof)),
                    "auroc": float(roc_auc_score(labels, oof)),
                    "brier": float(brier_score_loss(labels, oof)),
                }
            except Exception as exc:
                abort_on_resource_exhaustion(exc, stage=f"Reference {self.model_id}")
                row = {
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc)[:1000],
                    "ap_positive": np.nan,
                    "auroc": np.nan,
                    "brier": np.nan,
                }
            rows.append(
                {
                    "candidate_id": canonical_sha256(candidate),
                    "candidate_spec": dict(candidate),
                    **row,
                    "complexity": len(candidate),
                    "outer_test_metric_consulted": False,
                    "hagr_metric_consulted": False,
                }
            )
        trace = pd.DataFrame(rows)
        successful = trace.loc[trace["status"].eq("success")]
        if successful.empty:
            raise RuntimeError(f"Every candidate failed for reference {self.model_id}")
        best_ap = float(successful["ap_positive"].max())
        eligible = successful.loc[successful["ap_positive"].ge(best_ap - 0.005)]
        best_auc = float(eligible["auroc"].max())
        eligible = eligible.loc[eligible["auroc"].ge(best_auc - 0.005)]
        winner_row = eligible.sort_values(
            ["brier", "complexity", "candidate_id"], kind="stable"
        ).iloc[0]
        winner = dict(winner_row["candidate_spec"])
        self.estimator_ = _FittedReference(self.model_id, winner, seed=self.seed + 999983).fit(
            store, store.ids, labels.loc[list(store.ids)].to_numpy(dtype=int)
        )
        self.fit_ids_ = tuple(store.ids)
        self.feature_store_hash_ = store.store_hash
        self.feature_contract_hash_ = store.feature_contract_hash
        self.winner_ = winner
        self.selection_trace_ = trace
        self.selection_trace_["selected"] = self.selection_trace_["candidate_id"].eq(
            canonical_sha256(winner)
        )
        return self

    def predict_proba(self, store: ReferenceFeatureStore, ids: Sequence[str]) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("Reference pipeline is not fitted")
        store.assert_integrity()
        if store.feature_contract_hash != self.feature_contract_hash_:
            raise ValueError("Reference feature-extractor contract differs from fit")
        requested = tuple(map(str, ids))
        if set(requested) & set(self.fit_ids_):
            raise ValueError("Reference held-out prediction IDs overlap fitted IDs")
        return self.estimator_.predict_proba(store, requested)

    def get_manifest(self) -> dict[str, Any]:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("Reference pipeline is not fitted")
        return {
            "pipeline": "reference",
            "model_id": self.model_id,
            "winner": self.winner_,
            "fit_ids": list(self.fit_ids_),
            "fit_feature_content_sha256": self.feature_store_hash_,
            "feature_contract_sha256": self.feature_contract_hash_,
            "selection_trace_sha256": canonical_sha256(
                self.selection_trace_.fillna("<NA>").to_dict(orient="records")
            ),
            "outer_test_metric_consulted": False,
            "hagr_metric_consulted": False,
        }

    def save(self, path: str | Path) -> dict[str, Any]:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite reference model: {destination}")
        joblib.dump(self, destination)
        manifest = self.get_manifest()
        manifest["model_sha256"] = sha256_file(destination)
        atomic_write_json(
            destination.with_suffix(destination.suffix + ".manifest.json"), manifest
        )
        return manifest
