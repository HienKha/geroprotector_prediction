"""Nested, canonical-inference V6 foundation-model selector."""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ...config import (
    resolve_config,
    resolved_config_sha256,
    validate_locked_shared_contract,
    validate_protocol_lock,
)
from ...hashing import atomic_write_json, canonical_sha256, sha256_file
from ...validation.split_registry import grouped_inner_folds, validate_selection_folds
from ..resources import abort_on_resource_exhaustion, is_resource_exhaustion
from .checkpoints import _project_file, checkpoint_record, load_checkpoint_ledger
from .feature_panels import (
    V6FeatureStore,
    build_panel_transformer,
    panel_candidates,
    panel_declared_dimension,
)
from .finetune import assert_primary_zero_shot
from .inference_audit import audit_inference_semantics, release_accelerator_memory
from .tabicl_adapter import TabICLAdapter
from .tabpfn_adapter import TabPFNAdapter


def _context_features_sha256(frame: pd.DataFrame) -> str:
    matrix = np.ascontiguousarray(frame.to_numpy(dtype="<f8"))
    return canonical_sha256(
        {
            "schema": "geroprotector.v6_context_features.v1",
            "index": list(map(str, frame.index)),
            "columns": list(map(str, frame.columns)),
            "shape": list(matrix.shape),
            "matrix_sha256": hashlib.sha256(matrix.tobytes(order="C")).hexdigest(),
        }
    )


def _fit_labels_sha256(fit_ids: Sequence[str], labels: np.ndarray) -> str:
    values = np.asarray(labels, dtype="<i8")
    return canonical_sha256(
        {
            "schema": "geroprotector.v6_fit_labels.v1",
            "fit_ids": list(map(str, fit_ids)),
            "labels_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        }
    )


def _query_binding_sha256(ids: Sequence[str], frame: pd.DataFrame) -> str:
    matrix = np.ascontiguousarray(frame.to_numpy(dtype="<f8"))
    return canonical_sha256(
        {
            "query_ids": list(map(str, ids)),
            "feature_names": list(map(str, frame.columns)),
            "feature_sha256": hashlib.sha256(matrix.tobytes(order="C")).hexdigest(),
        }
    )


def _adapter(
    model_id: str,
    record: Mapping[str, Any],
    *,
    root: Path,
    ensemble_size: int,
    seed: int,
    device: str,
):
    kwargs = {
        "project_root": root,
        "n_estimators": ensemble_size,
        "random_state": seed,
        "device": device,
    }
    if model_id.startswith("tabpfn"):
        return TabPFNAdapter(record, **kwargs)
    if model_id.startswith("tabicl"):
        return TabICLAdapter(record, **kwargs)
    raise ValueError(f"Unsupported V6 model: {model_id}")


def _portfolio(
    config: Mapping[str, Any],
    ledger: Mapping[str, Any],
    *,
    seed: int,
    maximum_svd_components: int,
    fixed_model_id: str | None = None,
    fixed_panel_family: str | None = None,
    fixed_ensemble_size: int | None = None,
) -> tuple[dict[str, Any], ...]:
    staged = {record["model_id"] for record in ledger["models"]}
    models = [
        model_id
        for model_id, settings in config["models"].items()
        if isinstance(settings, Mapping)
        and settings.get("enabled") is True
        and model_id in staged
    ]
    panels = tuple(
        panel
        for panel in panel_candidates(config)
        if panel["panel"] != "morgan_svd_plus_descriptors"
        or int(panel["n_components"]) <= int(maximum_svd_components)
    )
    ensembles = tuple(int(value) for value in config["inference"]["ensemble_sizes"])
    # Stage 1 covers every checkpoint x panel at the smallest ensemble. Stage 2 uses
    # six locked ensemble contrasts on the primary current checkpoint; no outcome can
    # remove a checkpoint/panel ablation from the cap-30 portfolio.
    base = [
        {"model_id": model, "panel": panel, "ensemble_size": ensembles[0]}
        for model, panel in itertools.product(models, panels)
    ]
    primary_model = "tabpfn3" if "tabpfn3" in models else models[0]
    contrast_panels = list(panels[:3])
    contrasts = [
        {"model_id": primary_model, "panel": panel, "ensemble_size": ensemble}
        for panel, ensemble in itertools.product(contrast_panels, ensembles[1:])
    ]
    unique: dict[str, dict[str, Any]] = {}
    for item in (*base, *contrasts):
        unique.setdefault(canonical_sha256(item), item)
    cap = int(config["search"]["max_candidates_per_active_fit"])
    if len(unique) > cap:
        raise ValueError(
            f"Locked V6 ablation portfolio has {len(unique)} candidates but cap is {cap}"
        )
    selected = tuple(unique.values())
    if fixed_model_id is not None:
        selected = tuple(item for item in selected if item["model_id"] == fixed_model_id)
    if fixed_panel_family is not None:
        selected = tuple(
            item for item in selected if item["panel"]["panel"] == fixed_panel_family
        )
    if fixed_ensemble_size is not None:
        # Ensemble ablations are prespecified and may include size 1 even when the main
        # portfolio begins at 4; panel/checkpoint choices remain inner-selected.
        selected = tuple(
            {**item, "ensemble_size": int(fixed_ensemble_size)} for item in selected
        )
        selected = tuple({canonical_sha256(item): item for item in selected}.values())
    if not selected:
        raise ValueError("V6 fixed ablation constraint leaves no feasible candidate")
    return selected


def _select(trace: pd.DataFrame) -> dict[str, Any]:
    best_ap = float(trace["ap_positive"].max())
    eligible = trace.loc[trace["ap_positive"].ge(best_ap - 0.005)]
    best_auc = float(eligible["auroc"].max())
    eligible = eligible.loc[eligible["auroc"].ge(best_auc - 0.005)]
    eligible = eligible.sort_values(["brier", "complexity", "candidate_id"], kind="stable")
    return dict(eligible.iloc[0]["candidate_spec"])


class V6Pipeline:
    model_id = "v6_tabular_foundation_zero_shot"

    def __init__(
        self,
        config: Mapping[str, Any],
        checkpoint_ledger: Mapping[str, Any],
        *,
        project_root: str | Path,
        seed: int = 20260408,
        device: str = "auto",
        fixed_model_id: str | None = None,
        fixed_panel_family: str | None = None,
        fixed_ensemble_size: int | None = None,
        variant: str = "selected",
    ) -> None:
        self.config = dict(config)
        self.checkpoint_ledger = dict(checkpoint_ledger)
        self.project_root = Path(project_root).resolve()
        self.seed = int(seed)
        self.device = str(device)
        self.fixed_model_id = fixed_model_id
        self.fixed_panel_family = fixed_panel_family
        self.fixed_ensemble_size = fixed_ensemble_size
        self.variant = str(variant)
        self.model_id = f"v6_foundation_{self.variant}"
        assert_primary_zero_shot(self.config)

    def fit(
        self,
        store: V6FeatureStore,
        y: pd.Series,
        *,
        groups: pd.Series,
        selection_folds: Sequence[tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
    ) -> V6Pipeline:
        store.assert_integrity()
        ids = tuple(store.ids)
        y = y.copy()
        y.index = y.index.astype(str)
        groups = groups.copy()
        groups.index = groups.index.astype(str)
        if set(ids) != set(y.index) or set(ids) != set(groups.index):
            raise ValueError("V6 fit inputs are not ID-aligned")
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
        maximum_svd_components = min(len(train_ids) - 1 for train_ids, _ in folds)
        candidates = _portfolio(
            self.config,
            self.checkpoint_ledger,
            seed=self.seed,
            maximum_svd_components=maximum_svd_components,
            fixed_model_id=self.fixed_model_id,
            fixed_panel_family=self.fixed_panel_family,
            fixed_ensemble_size=self.fixed_ensemble_size,
        )
        if not candidates:
            raise RuntimeError("No staged enabled V6 candidates")
        trace_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            try:
                oof = pd.Series(index=y.index, dtype=float)
                for fold_index, (train_ids, validation_ids) in enumerate(folds):
                    local_seed = self.seed + fold_index
                    panel = build_panel_transformer(candidate["panel"], seed=local_seed)
                    panel.fit(store, fit_ids=train_ids)
                    X_train = panel.transform(store, train_ids)
                    X_valid = panel.transform(store, validation_ids)
                    record = checkpoint_record(self.checkpoint_ledger, candidate["model_id"])
                    model = None
                    try:
                        model = _adapter(
                            candidate["model_id"],
                            record,
                            root=self.project_root,
                            ensemble_size=int(candidate["ensemble_size"]),
                            seed=local_seed,
                            device=self.device,
                        )
                        model.fit_context(X_train, y.loc[list(train_ids)].to_numpy(dtype=int))
                        prediction = model.predict_proba(X_valid.copy(), canonical_mode=True)[
                            :, 1
                        ]
                        repeated = model.predict_proba(X_valid.copy(), canonical_mode=True)[
                            :, 1
                        ]
                        reordered = model.predict_proba(
                            X_valid.iloc[::-1].copy(), canonical_mode=True
                        )[::-1, 1]
                        atol = float(self.config["inference"]["audit_atol"])
                        if not np.allclose(
                            prediction, repeated, rtol=0.0, atol=atol
                        ) or not np.allclose(prediction, reordered, rtol=0.0, atol=atol):
                            raise RuntimeError(
                                "V6 candidate singleton inference is not "
                                "repeat/query-order stable"
                            )
                        oof.loc[list(validation_ids)] = prediction
                    finally:
                        if model is not None:
                            del model
                        release_accelerator_memory()
                if oof.isna().any():
                    raise RuntimeError("V6 candidate did not produce complete inner OOF")
                outcome = {
                    "status": "success",
                    "failure_type": None,
                    "failure_message": None,
                    "ap_positive": float(average_precision_score(y, oof)),
                    "auroc": float(roc_auc_score(y, oof)),
                    "brier": float(brier_score_loss(y, oof)),
                }
            except Exception as exc:
                if is_resource_exhaustion(exc):
                    release_accelerator_memory()
                    abort_on_resource_exhaustion(exc, stage="V6 foundation selection")
                outcome = {
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc)[:1000],
                    "ap_positive": np.nan,
                    "auroc": np.nan,
                    "brier": np.nan,
                }
            trace_rows.append(
                {
                    "candidate_id": canonical_sha256(candidate),
                    "candidate_spec": candidate,
                    **outcome,
                    "complexity": int(
                        panel_declared_dimension(candidate["panel"]) * 1_000
                        + int(candidate["ensemble_size"])
                    ),
                    "outer_test_metric_consulted": False,
                    "hagr_metric_consulted": False,
                }
            )
        trace = pd.DataFrame(trace_rows)
        successful = trace.loc[trace["status"].eq("success")]
        if successful.empty:
            raise RuntimeError(f"Every V6 candidate failed: {self.variant}")
        expected_models = {item["model_id"] for item in candidates}
        successful_models = {item["model_id"] for item in successful["candidate_spec"]}
        if expected_models != successful_models:
            missing = sorted(expected_models - successful_models)
            raise RuntimeError(f"Required V6 checkpoint families failed completely: {missing}")
        expected_panels = {canonical_sha256(item["panel"]) for item in candidates}
        successful_panels = {
            canonical_sha256(item["panel"]) for item in successful["candidate_spec"]
        }
        if expected_panels != successful_panels:
            raise RuntimeError(
                "A required V6 foundation input-panel candidate failed completely"
            )
        expected_ensembles = {int(item["ensemble_size"]) for item in candidates}
        successful_ensembles = {
            int(item["ensemble_size"]) for item in successful["candidate_spec"]
        }
        if expected_ensembles != successful_ensembles:
            raise RuntimeError("A required V6 foundation ensemble setting failed completely")
        anchor_ensemble = min(expected_ensembles)
        expected_anchor_pairs = {
            (item["model_id"], canonical_sha256(item["panel"]))
            for item in candidates
            if int(item["ensemble_size"]) == anchor_ensemble
        }
        successful_anchor_pairs = {
            (item["model_id"], canonical_sha256(item["panel"]))
            for item in successful["candidate_spec"]
            if int(item["ensemble_size"]) == anchor_ensemble
        }
        if expected_anchor_pairs != successful_anchor_pairs:
            raise RuntimeError("A required V6 checkpoint-by-panel anchor comparison failed")
        winner = _select(successful)
        panel = build_panel_transformer(winner["panel"], seed=self.seed + 999983)
        panel.fit(store, fit_ids=ids)
        X = panel.transform(store, ids)
        record = checkpoint_record(self.checkpoint_ledger, winner["model_id"])
        model = _adapter(
            winner["model_id"],
            record,
            root=self.project_root,
            ensemble_size=int(winner["ensemble_size"]),
            seed=self.seed + 999983,
            device=self.device,
        )
        model.fit_context(X, y.loc[list(ids)].to_numpy(dtype=int))
        self.panel_ = panel
        self.adapter_ = model
        self.winner_ = winner
        self.fit_ids_ = ids
        self.fit_labels_ = y.loc[list(ids)].to_numpy(dtype=int)
        self.context_features_ = X.copy()
        self.store_hash_ = store.store_hash
        self.feature_contract_hash_ = store.feature_contract_hash
        self.selection_trace_ = trace
        self.selection_trace_["selected"] = self.selection_trace_["candidate_id"].eq(
            canonical_sha256(winner)
        )
        return self

    def audit_inference(
        self,
        store: V6FeatureStore,
        *,
        query_ids: Sequence[str],
    ) -> dict[str, Any]:
        if not hasattr(self, "adapter_"):
            raise RuntimeError("V6 pipeline is not fitted")
        store.assert_integrity()
        if store.feature_contract_hash != self.feature_contract_hash_:
            raise ValueError("V6 inference-audit feature contract differs from fit")
        requested = tuple(map(str, query_ids))
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("V6 inference audit query IDs must be nonempty and unique")
        if set(requested) & set(self.fit_ids_):
            raise ValueError("V6 inference audit query overlaps the exact fitted context")
        X_context = self.context_features_.copy()
        if tuple(map(str, X_context.index)) != self.fit_ids_:
            raise RuntimeError("Stored V6 context does not match fitted IDs")
        X_query = self.panel_.transform(store, requested)
        record = checkpoint_record(self.checkpoint_ledger, self.winner_["model_id"])

        def factory():
            return _adapter(
                self.winner_["model_id"],
                record,
                root=self.project_root,
                ensemble_size=int(self.winner_["ensemble_size"]),
                seed=self.seed + 999983,
                device=self.device,
            )

        settings = self.config["inference"]
        gpu_peak_memory_bytes: int | None = None
        try:
            import torch

            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except (ImportError, RuntimeError):
            torch = None  # type: ignore[assignment]
        audit_started = time.perf_counter()
        pre_reconstruction = self.adapter_.predict_proba(X_query.copy(), canonical_mode=True)
        # The selected live foundation estimator can occupy most of GPU memory. Rebuild it
        # deterministically after the independent semantics audit so only one checkpoint
        # instance is resident at a time.
        del self.adapter_
        release_accelerator_memory()
        audit = audit_inference_semantics(
            factory,
            X_context,
            self.fit_labels_,
            X_query,
            atol=float(settings["audit_atol"]),
            seed=self.seed + 700001,
            random_compositions_per_query=int(settings["random_batch_compositions_per_query"]),
            hard_fail_on_context_row_order=bool(settings["hard_fail_on_context_row_order"]),
            hard_fail_on_feature_order=bool(settings["hard_fail_on_feature_order"]),
        )
        if not audit["passed"]:
            raise RuntimeError("V6 inference semantics audit failed")
        self.adapter_ = factory().fit_context(X_context.copy(), self.fit_labels_.copy())
        actual = self.adapter_.predict_proba(X_query.copy(), canonical_mode=True)
        repeated = self.adapter_.predict_proba(X_query.copy(), canonical_mode=True)
        reversed_prediction = self.adapter_.predict_proba(
            X_query.iloc[::-1].copy(), canonical_mode=True
        )[::-1]
        atol = float(settings["audit_atol"])
        reconstruction_difference = float(np.max(np.abs(pre_reconstruction - actual)))
        if (
            reconstruction_difference > atol
            or not np.allclose(actual, repeated, rtol=0.0, atol=atol)
            or not np.allclose(actual, reversed_prediction, rtol=0.0, atol=atol)
        ):
            raise RuntimeError("Actual fitted V6 adapter failed canonical stability audit")
        audit_seconds = time.perf_counter() - audit_started
        if torch is not None:
            try:
                if str(self.device).startswith("cuda") and torch.cuda.is_available():
                    gpu_peak_memory_bytes = int(torch.cuda.max_memory_allocated())
            except RuntimeError:
                gpu_peak_memory_bytes = None
        query_binding = _query_binding_sha256(requested, X_query)
        audit["actual_fitted_adapter_checked"] = True
        audit["reconstruction_max_abs_difference"] = reconstruction_difference
        audit["reconstruction_within_tolerance"] = reconstruction_difference <= atol
        # Canonical calls issue one estimator forward pass per query. This deterministic
        # count covers the semantic-audit factory calls and the three actual-adapter calls.
        semantic_passes = (5 + int(settings["random_batch_compositions_per_query"])) * len(
            requested
        ) + 1
        actual_passes = 4 * len(requested)
        audit["runtime"] = {
            "wall_clock_seconds": float(audit_seconds),
            "gpu_peak_memory_bytes": gpu_peak_memory_bytes,
            "semantic_audit_estimated_forward_passes": int(semantic_passes),
            "actual_adapter_forward_passes": int(actual_passes),
            "total_estimated_forward_passes": int(semantic_passes + actual_passes),
            "device": self.device,
        }
        audit["query_binding_sha256"] = query_binding
        audit["query_ids"] = list(requested)
        self.inference_audit_ = audit
        self.audited_query_ids_ = requested
        self.audited_query_binding_sha256_ = query_binding
        self.audited_probabilities_ = np.asarray(actual, dtype=float)
        return audit

    def predict_calibration_oof(self, store: V6FeatureStore, ids: Sequence[str]) -> np.ndarray:
        """Canonical prediction for inner OOF; full semantics audit is outer-only."""

        if not hasattr(self, "adapter_"):
            raise RuntimeError("V6 pipeline is not fitted")
        store.assert_integrity()
        if store.feature_contract_hash != self.feature_contract_hash_:
            raise ValueError("V6 calibration feature contract differs from fit")
        requested = tuple(map(str, ids))
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("V6 calibration query IDs must be nonempty and unique")
        if set(requested) & set(self.fit_ids_):
            raise ValueError("V6 calibration query IDs overlap fitted context")
        X_query = self.panel_.transform(store, requested)
        probability = self.adapter_.predict_proba(X_query.copy(), canonical_mode=True)
        repeated = self.adapter_.predict_proba(X_query.copy(), canonical_mode=True)
        reversed_probability = self.adapter_.predict_proba(
            X_query.iloc[::-1].copy(), canonical_mode=True
        )[::-1]
        atol = float(self.config["inference"]["audit_atol"])
        if not np.allclose(probability, repeated, rtol=0.0, atol=atol) or not np.allclose(
            probability, reversed_probability, rtol=0.0, atol=atol
        ):
            raise RuntimeError(
                "V6 calibration canonical inference is not repeat/query-order stable"
            )
        return probability

    def predict_proba(self, store: V6FeatureStore, ids: Sequence[str]) -> np.ndarray:
        if not hasattr(self, "adapter_"):
            raise RuntimeError("V6 pipeline is not fitted")
        if not hasattr(self, "inference_audit_"):
            raise RuntimeError("V6 inference audit must pass before held-out prediction")
        store.assert_integrity()
        if store.feature_contract_hash != self.feature_contract_hash_:
            raise ValueError("V6 feature-extractor contract differs from fit")
        requested = tuple(map(str, ids))
        if requested != self.audited_query_ids_:
            raise RuntimeError("V6 prediction IDs/order differ from the audited held-out query")
        query_binding = _query_binding_sha256(
            requested, self.panel_.transform(store, requested)
        )
        if query_binding != self.audited_query_binding_sha256_:
            raise RuntimeError("V6 query features differ from the audited held-out query")
        return self.audited_probabilities_.copy()

    def audit_and_predict(
        self, store: V6FeatureStore, ids: Sequence[str]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        audit = self.audit_inference(store, query_ids=ids)
        return self.predict_proba(store, ids), audit

    def get_manifest(self) -> dict[str, Any]:
        if not hasattr(self, "adapter_"):
            raise RuntimeError("V6 pipeline is not fitted")
        if not hasattr(self, "inference_audit_"):
            raise RuntimeError("V6 inference audit must pass before artifact export")
        return {
            "pipeline": "V6_TABULAR_FOUNDATION_MODELS",
            "model_id": self.model_id,
            "ablation_variant": self.variant,
            "phase": "zero_shot_only",
            "winner": self.winner_,
            "fit_ids": list(self.fit_ids_),
            "fit_feature_content_sha256": self.store_hash_,
            "feature_contract_sha256": self.feature_contract_hash_,
            "context_features_sha256": _context_features_sha256(self.context_features_),
            "fit_labels_sha256": _fit_labels_sha256(self.fit_ids_, self.fit_labels_),
            "panel": self.panel_.get_manifest(),
            "model": self.adapter_.save_manifest(),
            "checkpoint_ledger_sha256": self.checkpoint_ledger["canonical_sha256"],
            "selection_trace_sha256": canonical_sha256(
                self.selection_trace_.fillna("<NA>").to_dict(orient="records")
            ),
            "outer_test_metric_consulted": False,
            "hagr_metric_consulted": False,
            "inference_audit": getattr(self, "inference_audit_", None),
        }

    def save(self, path: str | Path) -> dict[str, Any]:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite V6 artifact: {destination}")
        # Do not serialize a live foundation estimator/GPU/checkpoint object. The portable
        # artifact stores the fitted panel, exact context and verified reconstruction recipe.
        portable = {
            "schema_version": "geroprotector.v6_portable_context_bundle.v2",
            "config": self.config,
            "checkpoint_ledger": self.checkpoint_ledger,
            "seed": self.seed,
            "device": self.device,
            "variant": self.variant,
            "model_id": self.model_id,
            "fixed_model_id": self.fixed_model_id,
            "fixed_panel_family": self.fixed_panel_family,
            "fixed_ensemble_size": self.fixed_ensemble_size,
            "winner": self.winner_,
            "panel": self.panel_,
            "fit_ids": self.fit_ids_,
            "fit_labels": self.fit_labels_,
            "context_features": self.context_features_,
            "fit_feature_content_sha256": self.store_hash_,
            "feature_contract_sha256": self.feature_contract_hash_,
            "context_features_sha256": _context_features_sha256(self.context_features_),
            "fit_labels_sha256": _fit_labels_sha256(self.fit_ids_, self.fit_labels_),
            "selection_trace": self.selection_trace_.to_dict(orient="records"),
            "inference_audit": getattr(self, "inference_audit_", None),
        }
        joblib.dump(portable, destination)
        manifest = self.get_manifest()
        manifest["model_artifact_sha256"] = sha256_file(destination)
        manifest["canonical_sha256"] = canonical_sha256(manifest)
        atomic_write_json(
            destination.with_suffix(destination.suffix + ".manifest.json"), manifest
        )
        return manifest

    @classmethod
    def load_portable(
        cls,
        path: str | Path,
        *,
        project_root: str | Path,
        expected_artifact_sha256: str,
        expected_manifest_sha256: str,
        config_path: str | Path = "configs/v6.yaml",
        checkpoint_ledger_path: str | Path | None = None,
    ) -> V6Pipeline:
        """Reconstruct a sealed V6 context from current verified offline assets.

        The prior outer-query audit is retained only as history. A new query must pass
        :meth:`audit_and_predict`, so a saved audit can never authorize different IDs.
        """

        project = Path(project_root).resolve()
        artifact = _project_file(project, path, role="V6 portable artifact")
        sidecar = artifact.with_suffix(artifact.suffix + ".manifest.json")
        if sidecar.is_symlink() or not sidecar.is_file():
            raise ValueError("V6 portable artifact manifest is missing or unsafe")
        if sha256_file(artifact) != str(expected_artifact_sha256):
            raise ValueError("V6 portable artifact differs from sealed job inventory")
        if sha256_file(sidecar) != str(expected_manifest_sha256):
            raise ValueError("V6 portable manifest differs from sealed job inventory")
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        manifest_payload = dict(manifest)
        claimed_manifest_hash = manifest_payload.pop("canonical_sha256", None)
        if claimed_manifest_hash != canonical_sha256(manifest_payload):
            raise ValueError("V6 portable artifact manifest integrity failed")
        if manifest.get("model_artifact_sha256") != sha256_file(artifact):
            raise ValueError("V6 portable artifact hash differs from its manifest")

        config_source = _project_file(project, config_path, role="V6 config")
        config = resolve_config(config_source)
        validate_locked_shared_contract(config)
        validate_protocol_lock(root=project, contract_name="v6", config=config)
        assert_primary_zero_shot(config)
        config_hash = resolved_config_sha256(config)
        ledger_source_value = (
            checkpoint_ledger_path
            if checkpoint_ledger_path is not None
            else config["checkpoint_ledger"]
        )
        ledger_source = _project_file(project, ledger_source_value, role="V6 checkpoint ledger")
        ledger = load_checkpoint_ledger(
            ledger_source,
            config=config,
            root=project,
            resolved_config_sha256=config_hash,
        )
        portable = joblib.load(artifact)
        if not isinstance(portable, Mapping) or portable.get("schema_version") != (
            "geroprotector.v6_portable_context_bundle.v2"
        ):
            raise ValueError("Unsupported V6 portable artifact schema")
        if canonical_sha256(portable.get("config")) != canonical_sha256(config):
            raise ValueError("V6 portable artifact config differs from current lock")
        embedded_ledger = portable.get("checkpoint_ledger")
        if not isinstance(embedded_ledger, Mapping):
            raise ValueError("V6 portable artifact checkpoint ledger is invalid")
        embedded_ledger_payload = dict(embedded_ledger)
        embedded_ledger_claim = embedded_ledger_payload.pop("canonical_sha256", None)
        if embedded_ledger_claim != canonical_sha256(
            embedded_ledger_payload
        ) or embedded_ledger_claim != ledger.get("canonical_sha256"):
            raise ValueError("V6 portable artifact checkpoint ledger differs from current")
        if manifest.get("model_id") != portable.get("model_id"):
            raise ValueError("V6 portable artifact model ID differs from manifest")
        if manifest.get("winner") != portable.get("winner"):
            raise ValueError("V6 portable artifact winner differs from manifest")
        if (
            manifest.get("pipeline") != "V6_TABULAR_FOUNDATION_MODELS"
            or manifest.get("phase") != "zero_shot_only"
            or manifest.get("outer_test_metric_consulted") is not False
            or manifest.get("hagr_metric_consulted") is not False
        ):
            raise ValueError("V6 portable artifact scientific-role manifest is invalid")

        fitted = cls(
            config,
            ledger,
            project_root=project,
            seed=int(portable["seed"]),
            device=str(portable["device"]),
            fixed_model_id=portable.get("fixed_model_id"),
            fixed_panel_family=portable.get("fixed_panel_family"),
            fixed_ensemble_size=portable.get("fixed_ensemble_size"),
            variant=str(portable["variant"]),
        )
        if fitted.model_id != portable["model_id"]:
            raise ValueError("V6 reconstructed model ID differs from portable artifact")
        fitted.winner_ = dict(portable["winner"])
        fitted.panel_ = portable["panel"]
        fitted.fit_ids_ = tuple(map(str, portable["fit_ids"]))
        fitted.fit_labels_ = np.asarray(portable["fit_labels"], dtype=int)
        fitted.context_features_ = portable["context_features"].copy()
        if (
            tuple(map(str, fitted.context_features_.index)) != fitted.fit_ids_
            or len(fitted.fit_labels_) != len(fitted.fit_ids_)
            or set(fitted.fit_labels_) != {0, 1}
            or not np.isfinite(fitted.context_features_.to_numpy(dtype=float)).all()
        ):
            raise ValueError("V6 portable context/labels/IDs are invalid")
        panel_manifest = fitted.panel_.get_manifest()
        if list(fitted.context_features_.columns) != panel_manifest.get("feature_names"):
            raise ValueError("V6 portable context differs from fitted panel schema")
        fitted.store_hash_ = str(portable["fit_feature_content_sha256"])
        fitted.feature_contract_hash_ = str(portable["feature_contract_sha256"])
        context_hash = _context_features_sha256(fitted.context_features_)
        labels_hash = _fit_labels_sha256(fitted.fit_ids_, fitted.fit_labels_)
        if (
            portable.get("context_features_sha256") != context_hash
            or portable.get("fit_labels_sha256") != labels_hash
        ):
            raise ValueError("V6 portable context/label payload hash is invalid")
        fitted.selection_trace_ = pd.DataFrame(portable["selection_trace"])
        fitted.historical_inference_audit_ = portable.get("inference_audit")
        expected_manifest_fields = {
            "ablation_variant": fitted.variant,
            "fit_ids": list(fitted.fit_ids_),
            "fit_feature_content_sha256": fitted.store_hash_,
            "feature_contract_sha256": fitted.feature_contract_hash_,
            "context_features_sha256": context_hash,
            "fit_labels_sha256": labels_hash,
            "panel": fitted.panel_.get_manifest(),
            "checkpoint_ledger_sha256": ledger["canonical_sha256"],
            "selection_trace_sha256": canonical_sha256(
                fitted.selection_trace_.fillna("<NA>").to_dict(orient="records")
            ),
            "inference_audit": fitted.historical_inference_audit_,
        }
        if any(manifest.get(key) != value for key, value in expected_manifest_fields.items()):
            raise ValueError("V6 portable payload differs from its manifest")
        record = checkpoint_record(ledger, fitted.winner_["model_id"])
        fitted.adapter_ = _adapter(
            fitted.winner_["model_id"],
            record,
            root=project,
            ensemble_size=int(fitted.winner_["ensemble_size"]),
            seed=fitted.seed + 999983,
            device=fitted.device,
        ).fit_context(fitted.context_features_.copy(), fitted.fit_labels_.copy())
        if fitted.adapter_.save_manifest() != manifest.get("model"):
            raise ValueError("V6 reconstructed adapter differs from saved model manifest")
        return fitted
