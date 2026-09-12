"""Hierarchical V5 fit; all selection is confined to the supplied training rows."""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from ...hashing import atomic_write_json, canonical_sha256, sha256_file
from ...validation.split_registry import grouped_inner_folds, validate_selection_folds
from ..resources import abort_on_resource_exhaustion
from .estimators import build_estimator, downstream_candidates
from .fingerprint_bank import V5FeatureBank
from .nystrom_embedding import WeightedRBFNystrom
from .preparation_cache import load_or_fit_stage_a, load_or_fit_stage_b
from .stable_importance import StableImportanceState
from .weighted_fusion import (
    DescriptorPreprocessor,
    WeightedIdentityTransform,
    WeightedLinearSVD,
    weighted_concatenation,
)

V5_VARIANTS = {
    "raw_best_individual_fingerprint",
    "raw_unweighted_concatenation",
    "selected_lengths_unweighted",
    "weighted_no_reduction",
    "weighted_svd",
    "weighted_nystroem",
    "final_v5",
}


def _assert_implemented_v5_contract(config: Mapping[str, Any]) -> None:
    """Fail closed when declarative scientific switches differ from implementation."""

    stage_a = config.get("stage_a", {})
    if (
        list(stage_a.get("semantic_families", []))
        != ["morgan_bit", "morgan_count", "rdkit_path", "maccs"]
        or stage_a.get("select_exactly_one_per_semantic_family") is not True
        or stage_a.get("estimator") != "extra_trees"
        or stage_a.get("tie_break") != "smaller_fingerprint"
    ):
        raise ValueError("V5 Stage-A config differs from implemented semantics")
    weighting = config.get("weighting", {})
    if (
        list(weighting.get("methods", [])) != ["extra_trees_permutation", "xgboost_gain"]
        or int(weighting.get("grouped_subcv_folds", -1)) != 3
        or int(weighting.get("importance_estimator_n_jobs", -1)) != 1
        or int(weighting.get("permutation_n_jobs", -1)) != 16
        or weighting.get("aggregate") != "median"
        or weighting.get("negative_importance_policy") != "clip_zero"
        or weighting.get("normalize_mean_per_block") is not True
        or weighting.get("fit_scope") != "active_training_partition_only"
    ):
        raise ValueError("V5 Stage-B config differs from implemented semantics")
    if list(config.get("embedding", {}).get("methods", [])) != [
        "linear_svd",
        "nystroem_weighted_rbf",
    ]:
        raise ValueError("V5 embedding config differs from implemented semantics")
    if list(config.get("models", {}).get("downstream", [])) != [
        "extra_trees",
        "xgboost",
    ]:
        raise ValueError("V5 downstream config differs from implemented semantics")
    search = config.get("search", {})
    if (
        search.get("strategy") != "deterministic_hierarchical"
        or search.get("outer_test_metric_consulted") is not False
        or search.get("hagr_metric_consulted") is not False
    ):
        raise ValueError("V5 search config differs from implemented semantics")


def _validated_probability(estimator: Any, matrix: np.ndarray) -> np.ndarray:
    if tuple(map(int, np.asarray(estimator.classes_).tolist())) != (0, 1):
        raise ValueError("V5 estimator class order differs from [0, 1]")
    output = np.asarray(estimator.predict_proba(matrix), dtype=float)
    if (
        output.ndim != 2
        or output.shape[1] != 2
        or not np.isfinite(output).all()
        or np.any((output < 0) | (output > 1))
        or not np.allclose(output.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("V5 estimator returned invalid probabilities")
    return output


@dataclass
class FittedV5Representation:
    selected_specs: tuple[str, ...]
    weights: Mapping[str, np.ndarray]
    embedding: WeightedIdentityTransform | WeightedLinearSVD | WeightedRBFNystrom
    descriptor_preprocessor: DescriptorPreprocessor
    fit_ids: tuple[str, ...]
    requested_alpha: float
    effective_alpha: float
    importance_stability_median: float
    stability_fallback_applied: bool

    def transform(self, bank: V5FeatureBank, ids: Sequence[str]) -> np.ndarray:
        fingerprint = weighted_concatenation(bank, ids, self.selected_specs, self.weights)
        latent = self.embedding.transform(fingerprint)
        descriptor = self.descriptor_preprocessor.transform(bank.descriptor_matrix(ids))
        output = np.concatenate([latent, descriptor], axis=1)
        if not np.isfinite(output).all():
            raise ValueError("V5 representation produced non-finite values")
        return output

    def get_manifest(self) -> dict[str, Any]:
        return {
            "selected_specs": list(self.selected_specs),
            "fit_ids": list(self.fit_ids),
            "weight_sha256": {
                key: hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()
                for key, value in self.weights.items()
            },
            "embedding": self.embedding.get_manifest(),
            "descriptors": self.descriptor_preprocessor.get_manifest(),
            "requested_alpha": self.requested_alpha,
            "effective_alpha": self.effective_alpha,
            "importance_stability_median": self.importance_stability_median,
            "stability_fallback_applied": self.stability_fallback_applied,
        }


def _fit_representation(
    bank: V5FeatureBank,
    ids: tuple[str, ...],
    selected_specs: tuple[str, ...],
    importance: StableImportanceState,
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    seed: int,
) -> FittedV5Representation:
    weighting = config["weighting"]
    stability_values = np.asarray(list(importance.stability_spearman.values()), dtype=float)
    stability_median = (
        float(np.median(stability_values[np.isfinite(stability_values)]))
        if np.isfinite(stability_values).any()
        else 0.0
    )
    requested_alpha = float(candidate["alpha"])
    fallback_applied = stability_median < float(weighting["stability_min_median_spearman"])
    effective_alpha = (
        float(weighting["stability_fallback_alpha"]) if fallback_applied else requested_alpha
    )
    weights = importance.weights(
        effective_alpha,
        clip=tuple(map(float, weighting["weight_clip"])),
        epsilon=float(weighting["epsilon"]),
    )
    fingerprint = weighted_concatenation(bank, ids, selected_specs, weights)
    if candidate["embedding"] == "identity":
        embedding: WeightedIdentityTransform | WeightedLinearSVD | WeightedRBFNystrom
        embedding = WeightedIdentityTransform().fit(fingerprint, fit_ids=ids)
    elif candidate["embedding"] == "linear_svd":
        embedding = WeightedLinearSVD(int(candidate["n_components"]), int(seed)).fit(
            fingerprint, fit_ids=ids
        )
    elif candidate["embedding"] == "nystroem_weighted_rbf":
        embedding = WeightedRBFNystrom(
            int(candidate["n_components"]),
            float(candidate["gamma_multiplier"]),
            int(seed),
        ).fit(fingerprint, fit_ids=ids)
    else:
        raise ValueError(f"Unknown V5 embedding: {candidate['embedding']}")
    descriptor = DescriptorPreprocessor().fit(bank.descriptor_matrix(ids), fit_ids=ids)
    return FittedV5Representation(
        selected_specs,
        weights,
        embedding,
        descriptor,
        ids,
        requested_alpha,
        effective_alpha,
        stability_median,
        fallback_applied,
    )


def _candidate_complexity(
    candidate: Mapping[str, Any], bank: V5FeatureBank, selected_specs: Sequence[str]
) -> int:
    if candidate["embedding"] == "identity":
        representation_dimension = sum(bank.blocks[value].shape[1] for value in selected_specs)
    else:
        representation_dimension = int(candidate["n_components"])
    representation_dimension += int(bank.descriptors.shape[1])
    estimator = candidate["estimator"]
    if estimator["model"] == "extra_trees":
        estimator_complexity = int(estimator["n_estimators"])
    else:
        estimator_complexity = int(estimator["n_estimators"]) * (
            2 ** int(estimator["max_depth"])
        )
    return int(representation_dimension * 1_000_000 + estimator_complexity)


def _candidate_portfolio(config: Mapping[str, Any], *, seed: int) -> tuple[dict[str, Any], ...]:
    embeddings = config["embedding"]
    models = downstream_candidates(config, seed=seed)
    representative_models: list[dict[str, Any]] = []
    for name in ("extra_trees", "xgboost"):
        representative_models.append(next(item for item in models if item["model"] == name))
    raw: list[dict[str, Any]] = []
    for alpha, model in itertools.product(config["weighting"]["alpha"], representative_models):
        raw.append(
            {
                "alpha": float(alpha),
                "embedding": "linear_svd",
                "n_components": 64,
                "gamma_multiplier": None,
                "estimator": model,
            }
        )
        raw.append(
            {
                "alpha": float(alpha),
                "embedding": "nystroem_weighted_rbf",
                "n_components": 64,
                "gamma_multiplier": 1.0,
                "estimator": model,
            }
        )
    full = []
    for alpha, model, n_components in itertools.product(
        config["weighting"]["alpha"], models, embeddings["linear_svd_components"]
    ):
        full.append(
            {
                "alpha": float(alpha),
                "embedding": "linear_svd",
                "n_components": int(n_components),
                "gamma_multiplier": None,
                "estimator": model,
            }
        )
    for alpha, model, n_components, gamma in itertools.product(
        config["weighting"]["alpha"],
        models,
        embeddings["nystroem_components"],
        embeddings["gamma_multiplier"],
    ):
        full.append(
            {
                "alpha": float(alpha),
                "embedding": "nystroem_weighted_rbf",
                "n_components": int(n_components),
                "gamma_multiplier": float(gamma),
                "estimator": model,
            }
        )
    full.sort(key=lambda item: hashlib.sha256(f"{seed}\0{item}".encode()).digest())
    unique: dict[str, dict[str, Any]] = {}
    for item in (*raw, *full):
        unique.setdefault(canonical_sha256(item), item)
    cap = int(config["search"]["max_stage_c_candidates_per_active_fit"])
    return tuple(list(unique.values())[:cap])


def _variant_portfolio(
    config: Mapping[str, Any], *, seed: int, variant: str
) -> tuple[dict[str, Any], ...]:
    full = _candidate_portfolio(config, seed=seed)
    models = downstream_candidates(config, seed=seed)
    anchors = [
        next(value for value in models if value["model"] == name)
        for name in (
            "extra_trees",
            "xgboost",
        )
    ]
    if variant in {
        "raw_best_individual_fingerprint",
        "raw_unweighted_concatenation",
        "selected_lengths_unweighted",
    }:
        return tuple(
            {
                "alpha": 0.0,
                "embedding": "identity",
                "n_components": 0,
                "gamma_multiplier": None,
                "estimator": estimator,
            }
            for estimator in anchors
        )
    if variant == "weighted_no_reduction":
        return tuple(
            {
                "alpha": float(alpha),
                "embedding": "identity",
                "n_components": 0,
                "gamma_multiplier": None,
                "estimator": estimator,
            }
            for alpha, estimator in itertools.product(config["weighting"]["alpha"], anchors)
        )
    if variant == "weighted_svd":
        return tuple(item for item in full if item["embedding"] == "linear_svd")
    if variant == "weighted_nystroem":
        return tuple(item for item in full if item["embedding"] == "nystroem_weighted_rbf")
    return full


def _uniform_importance(
    bank: V5FeatureBank, selected_specs: tuple[str, ...]
) -> StableImportanceState:
    base = {
        spec_id: np.full(
            bank.blocks[spec_id].shape[1], 1.0 / bank.blocks[spec_id].shape[1], dtype=float
        )
        for spec_id in selected_specs
    }
    payload = {
        "kind": "uniform_unweighted_ablation",
        "selected_specs": list(selected_specs),
        "fit_ids": list(bank.ids),
    }
    return StableImportanceState(
        selected_specs=selected_specs,
        base_importance=base,
        stability_spearman={spec_id: 1.0 for spec_id in selected_specs},
        fit_ids=tuple(bank.ids),
        subfold_validation_ids=(),
        state_hash=canonical_sha256(payload),
    )


def _choose_candidate(trace: pd.DataFrame) -> dict[str, Any]:
    best_ap = float(trace["ap_positive"].max())
    eligible = trace.loc[trace["ap_positive"].ge(best_ap - 0.005)]
    best_auc = float(eligible["auroc"].max())
    eligible = eligible.loc[eligible["auroc"].ge(best_auc - 0.005)]
    eligible = eligible.sort_values(
        ["brier", "complexity", "candidate_id"], ascending=[True, True, True], kind="stable"
    )
    return dict(eligible.iloc[0]["candidate_spec"])


class V5Pipeline:
    """A fitted V5 estimator; callers supply only one active training partition."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        seed: int = 20260408,
        variant: str = "final_v5",
        preparation_cache_root: str | Path | None = None,
        preparation_cache_context_sha256: str | None = None,
    ):
        if variant not in V5_VARIANTS:
            raise ValueError(f"Unknown V5 ablation variant: {variant}")
        _assert_implemented_v5_contract(config)
        self.config = dict(config)
        self.seed = int(seed)
        self.variant = str(variant)
        self.model_id = f"v5_{self.variant}"
        self.preparation_cache_root = (
            None if preparation_cache_root is None else Path(preparation_cache_root)
        )
        self.preparation_cache_context_sha256 = preparation_cache_context_sha256
        if (self.preparation_cache_root is None) != (
            self.preparation_cache_context_sha256 is None
        ):
            raise ValueError("V5 preparation cache root/context must be supplied together")

    def fit(
        self,
        bank: V5FeatureBank,
        y: pd.Series,
        *,
        groups: pd.Series,
        selection_folds: Sequence[tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
    ) -> V5Pipeline:
        ids = tuple(bank.ids)
        y = y.copy()
        y.index = y.index.astype(str)
        groups = groups.copy()
        groups.index = groups.index.astype(str)
        if set(ids) != set(y.index) or set(ids) != set(groups.index):
            raise ValueError("V5 fit inputs are not ID-aligned")
        bank.assert_integrity()
        if set(y.astype(int)) != {0, 1}:
            raise ValueError("V5 requires two classes")
        folds = (
            list(selection_folds)
            if selection_folds is not None
            else grouped_inner_folds(y, groups, n_splits=3, seed=self.seed)
        )
        folds = validate_selection_folds(
            fit_ids=ids,
            labels=y,
            groups=groups,
            folds=folds,
        )
        stage_a_result = load_or_fit_stage_a(
            cache_root=self.preparation_cache_root,
            cache_context_sha256=self.preparation_cache_context_sha256,
            bank=bank,
            y=y,
            groups=groups,
            folds=folds,
            config=self.config,
            seed=self.seed,
        )
        selected_specs = stage_a_result.selected_specs
        stage_a = stage_a_result.trace
        stage_a["preparation_cache_binding_sha256"] = stage_a_result.binding_sha256
        stage_a["preparation_cache_reused"] = stage_a_result.reused
        if self.variant == "raw_best_individual_fingerprint":
            winner = (
                stage_a.loc[stage_a["status"].eq("success")]
                .sort_values(
                    ["selection_objective", "n_bits", "spec_id"],
                    ascending=[False, True, True],
                    kind="stable",
                )
                .iloc[0]
            )
            selected_specs = (str(winner["spec_id"]),)
        elif self.variant == "raw_unweighted_concatenation":
            selected_specs = tuple(sorted(bank.specs))
        unweighted = self.variant in {
            "raw_best_individual_fingerprint",
            "raw_unweighted_concatenation",
            "selected_lengths_unweighted",
        }
        fold_importance: dict[int, StableImportanceState] = {}
        importance_cache_lineage: dict[int, dict[str, Any]] = {}
        for fold_index, (train_ids, _) in enumerate(folds):
            active_bank = bank.subset(train_ids)
            if unweighted:
                fold_importance[fold_index] = _uniform_importance(active_bank, selected_specs)
                importance_cache_lineage[fold_index] = {
                    "binding_sha256": None,
                    "reused": False,
                    "role": "uniform_unweighted_control",
                }
            else:
                result = load_or_fit_stage_b(
                    cache_root=self.preparation_cache_root,
                    cache_context_sha256=self.preparation_cache_context_sha256,
                    bank=active_bank,
                    selected_specs=selected_specs,
                    y=y.loc[list(train_ids)],
                    groups=groups.loc[list(train_ids)],
                    config=self.config,
                    seed=self.seed + fold_index * 1009,
                )
                fold_importance[fold_index] = result.state
                importance_cache_lineage[fold_index] = {
                    "binding_sha256": result.binding_sha256,
                    "reused": result.reused,
                    "role": "fold_local_stable_importance",
                }
        importance_lineage = [
            {
                "selection_fold": fold_index,
                "state_hash": state.state_hash,
                "fit_ids_sha256": hashlib.sha256(
                    ("\n".join(sorted(state.fit_ids)) + "\n").encode()
                ).hexdigest(),
                "subfold_validation_ids_sha256": [
                    hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()
                    for values in state.subfold_validation_ids
                ],
                "stability_spearman": dict(state.stability_spearman),
                "preparation_cache": importance_cache_lineage[fold_index],
            }
            for fold_index, state in sorted(fold_importance.items())
        ]
        rows: list[dict[str, Any]] = []
        for candidate in _variant_portfolio(self.config, seed=self.seed, variant=self.variant):
            try:
                oof = pd.Series(index=y.index, dtype=float)
                for fold_index, (train_ids, validation_ids) in enumerate(folds):
                    common_fold_seed = self.seed + fold_index
                    representation = _fit_representation(
                        bank,
                        tuple(train_ids),
                        selected_specs,
                        fold_importance[fold_index],
                        candidate,
                        self.config,
                        seed=common_fold_seed,
                    )
                    estimator = build_estimator(
                        candidate["estimator"],
                        seed=common_fold_seed,
                    )
                    X_train = representation.transform(bank, train_ids)
                    y_train = y.loc[list(train_ids)].to_numpy(dtype=int)
                    fit_kwargs = {}
                    if candidate["estimator"]["model"] == "xgboost":
                        fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train)
                    estimator.fit(X_train, y_train, **fit_kwargs)
                    oof.loc[list(validation_ids)] = _validated_probability(
                        estimator, representation.transform(bank, validation_ids)
                    )[:, 1]
                if oof.isna().any():
                    raise RuntimeError("V5 candidate did not produce complete inner OOF")
                outcome = {
                    "status": "success",
                    "failure_type": None,
                    "failure_message": None,
                    "ap_positive": float(average_precision_score(y, oof)),
                    "auroc": float(roc_auc_score(y, oof)),
                    "brier": float(brier_score_loss(y, oof)),
                }
            except Exception as exc:
                abort_on_resource_exhaustion(exc, stage=f"V5 {self.variant} Stage C")
                outcome = {
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc)[:1000],
                    "ap_positive": np.nan,
                    "auroc": np.nan,
                    "brier": np.nan,
                }
            rows.append(
                {
                    "stage": "C_embedding_classifier",
                    "candidate_id": canonical_sha256(candidate),
                    "candidate_spec": candidate,
                    **outcome,
                    "complexity": _candidate_complexity(candidate, bank, selected_specs),
                    "importance_fit_lineage": importance_lineage,
                    "outer_test_metric_consulted": False,
                    "hagr_metric_consulted": False,
                }
            )
        stage_c = pd.DataFrame(rows)
        successful = stage_c.loc[stage_c["status"].eq("success")]
        if successful.empty:
            raise RuntimeError(f"Every V5 Stage-C candidate failed: {self.variant}")
        required_embeddings = {
            "final_v5": {"linear_svd", "nystroem_weighted_rbf"},
            "weighted_svd": {"linear_svd"},
            "weighted_nystroem": {"nystroem_weighted_rbf"},
            "weighted_no_reduction": {"identity"},
        }.get(self.variant, {"identity"})
        observed_embeddings = {value["embedding"] for value in successful["candidate_spec"]}
        if not required_embeddings.issubset(observed_embeddings):
            raise RuntimeError("A required V5 embedding branch had no successful candidate")
        required_estimators = {"extra_trees", "xgboost"}
        observed_estimators = {
            value["estimator"]["model"] for value in successful["candidate_spec"]
        }
        if not required_estimators.issubset(observed_estimators):
            raise RuntimeError("A required V5 downstream family had no successful candidate")
        winner = _choose_candidate(successful)
        if unweighted:
            final_importance = _uniform_importance(bank, selected_specs)
            final_importance_cache = {
                "binding_sha256": None,
                "reused": False,
                "role": "uniform_unweighted_control",
            }
        else:
            final_result = load_or_fit_stage_b(
                cache_root=self.preparation_cache_root,
                cache_context_sha256=self.preparation_cache_context_sha256,
                bank=bank,
                selected_specs=selected_specs,
                y=y,
                groups=groups,
                config=self.config,
                seed=self.seed + 99991,
            )
            final_importance = final_result.state
            final_importance_cache = {
                "binding_sha256": final_result.binding_sha256,
                "reused": final_result.reused,
                "role": "outer_active_fit_stable_importance",
            }
        representation = _fit_representation(
            bank,
            ids,
            selected_specs,
            final_importance,
            winner,
            self.config,
            seed=self.seed + 999983,
        )
        estimator = build_estimator(winner["estimator"], seed=self.seed + 999983)
        X = representation.transform(bank, ids)
        labels = y.loc[list(ids)].to_numpy(dtype=int)
        fit_kwargs = {}
        if winner["estimator"]["model"] == "xgboost":
            fit_kwargs["sample_weight"] = compute_sample_weight("balanced", labels)
        estimator.fit(X, labels, **fit_kwargs)
        self.selected_specs_ = selected_specs
        self.importance_ = final_importance
        self.representation_ = representation
        self.estimator_ = estimator
        self.winner_ = winner
        self.fit_ids_ = ids
        self.bank_hash_ = bank.bank_hash
        self.feature_contract_hash_ = bank.feature_contract_hash
        self.preparation_cache_lineage_ = {
            "stage_a": {
                "binding_sha256": stage_a_result.binding_sha256,
                "reused": stage_a_result.reused,
            },
            "selection_fold_importance": importance_cache_lineage,
            "final_importance": final_importance_cache,
            "variant_changes_stage_a_or_b_dependency": False,
        }
        self.selection_trace_ = pd.concat([stage_a, stage_c], ignore_index=True)
        self.selection_trace_.loc[
            self.selection_trace_["candidate_id"].eq(canonical_sha256(winner)), "selected"
        ] = True
        return self

    def predict_proba(self, bank: V5FeatureBank, ids: Sequence[str]) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("V5 pipeline is not fitted")
        bank.assert_integrity()
        if bank.feature_contract_hash != self.feature_contract_hash_:
            raise ValueError("V5 feature-extractor contract differs from fitted bank")
        requested = tuple(map(str, ids))
        if set(requested) & set(self.fit_ids_):
            raise ValueError("V5 held-out prediction IDs overlap fitted IDs")
        output = _validated_probability(
            self.estimator_, self.representation_.transform(bank, requested)
        )
        if output.shape != (len(requested), 2):
            raise ValueError("V5 estimator returned the wrong number of rows")
        return output

    def get_feature_manifest(self) -> dict[str, Any]:
        if not hasattr(self, "representation_"):
            raise RuntimeError("V5 pipeline is not fitted")
        return {
            "pipeline": "V5_ELIXIRFP_REBUILT",
            "model_id": self.model_id,
            "ablation_variant": self.variant,
            "fit_feature_content_sha256": getattr(self, "bank_hash_", None),
            "feature_contract_sha256": getattr(self, "feature_contract_hash_", None),
            "fit_ids": list(self.fit_ids_),
            "importance": self.importance_.get_manifest(),
            "preparation_cache": self.preparation_cache_lineage_,
            "representation": self.representation_.get_manifest(),
            "winner": self.winner_,
            "selection_trace_sha256": canonical_sha256(
                self.selection_trace_.fillna("<NA>").to_dict(orient="records")
            ),
        }

    def get_selection_trace(self) -> pd.DataFrame:
        if not hasattr(self, "selection_trace_"):
            raise RuntimeError("V5 pipeline is not fitted")
        return self.selection_trace_.copy()

    def save(self, path: str | Path) -> dict[str, Any]:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite V5 model: {destination}")
        joblib.dump(self, destination)
        manifest = self.get_feature_manifest()
        manifest["model_sha256"] = sha256_file(destination)
        atomic_write_json(
            destination.with_suffix(destination.suffix + ".manifest.json"), manifest
        )
        return manifest
