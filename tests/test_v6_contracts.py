from __future__ import annotations

import copy
import importlib.metadata
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import joblib
import numpy as np
import pandas as pd
import pytest

from geroprotector.config import resolve_config
from geroprotector.hashing import canonical_sha256, sha256_file
from geroprotector.models.v6 import baselines as baseline_module
from geroprotector.models.v6 import inference_audit as inference_audit_module
from geroprotector.models.v6 import pipeline as pipeline_module
from geroprotector.models.v6.baselines import V6ClassicalPipeline
from geroprotector.models.v6.checkpoints import (
    CheckpointError,
    _project_file,
    _torch_environment,
    checkpoint_record,
    load_checkpoint_ledger,
    stage_checkpoints,
)
from geroprotector.models.v6.feature_panels import (
    DescriptorPanelTransformer,
    MorganSVDPanelTransformer,
    V6FeatureStore,
    _feature_content_hash,
    _feature_contract_hash,
    panel_candidates,
)
from geroprotector.models.v6.finetune import assert_primary_zero_shot
from geroprotector.models.v6.inference_audit import audit_inference_semantics
from geroprotector.models.v6.pipeline import V6Pipeline, _portfolio, _query_binding_sha256
from geroprotector.models.v6.tabicl_adapter import TabICLAdapter
from geroprotector.models.v6.tabpfn_adapter import TabPFNAdapter
from geroprotector.validation import nested_cv as nested_cv_module

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def v6_config():
    return resolve_config(ROOT / "configs/v6.yaml")


def _manual_store(*, name_suffix: str = "") -> V6FeatureStore:
    ids = tuple(f"cmp::{index}" for index in range(8))
    rdkit = pd.DataFrame(
        {
            f"varying{name_suffix}": np.arange(8, dtype=float),
            f"missing{name_suffix}": [np.nan, 1, 2, 3, 4, 5, 6, 7],
            f"constant{name_suffix}": np.ones(8),
        },
        index=ids,
    )
    chemistry = pd.DataFrame(
        {
            f"c1{name_suffix}": np.arange(8, dtype=float),
            f"c2{name_suffix}": np.arange(8, dtype=float) % 3,
            f"c3{name_suffix}": [np.nan, 1, 0, 1, 0, 1, 0, 1],
        },
        index=ids,
    )
    morgan_small = np.asarray(
        [
            [1, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [1, 0, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ],
        dtype=np.float32,
    )
    morgan = np.zeros((len(ids), 2048), dtype=np.float32)
    morgan[:, : morgan_small.shape[1]] = morgan_small
    smiles = tuple("CC" for _ in ids)
    contract_hash = _feature_contract_hash(rdkit.columns, chemistry.columns)
    return V6FeatureStore(
        ids=ids,
        smiles=smiles,
        rdkit2d=rdkit,
        chemistry32=chemistry,
        morgan_count=morgan,
        feature_contract_hash=contract_hash,
        store_hash=_feature_content_hash(
            ids,
            smiles,
            rdkit,
            chemistry,
            morgan,
            feature_contract_hash=contract_hash,
        ),
    )


def test_v6_panel_portfolio_is_exact_checkpoint_by_panel_ablation(v6_config) -> None:
    panels = panel_candidates(v6_config)
    assert len(panels) == 6
    staged = [
        {"model_id": name}
        for name, settings in v6_config["models"].items()
        if settings.get("enabled") is True
    ]
    portfolio = _portfolio(
        v6_config,
        {"models": staged},
        seed=19,
        maximum_svd_components=300,
    )
    assert len(portfolio) == 30
    base_ensemble = v6_config["inference"]["ensemble_sizes"][0]
    base_pairs = {
        (item["model_id"], canonical_sha256(item["panel"]))
        for item in portfolio
        if item["ensemble_size"] == base_ensemble
    }
    assert len(base_pairs) == 4 * 6
    assert len(portfolio) <= v6_config["search"]["max_candidates_per_active_fit"]


def test_v6_infeasible_svd_candidates_are_filtered_without_silent_clamping(v6_config) -> None:
    staged = [
        {"model_id": name}
        for name, settings in v6_config["models"].items()
        if settings.get("enabled") is True
    ]
    portfolio = _portfolio(
        v6_config,
        {"models": staged},
        seed=19,
        maximum_svd_components=40,
    )
    dimensions = {
        item["panel"]["n_components"]
        for item in portfolio
        if item["panel"]["panel"] == "morgan_svd_plus_descriptors"
    }
    assert dimensions == {32}


def test_fold_local_descriptor_and_morgan_panels_record_fit_scope() -> None:
    store = _manual_store()
    fit_ids = store.ids[:6]
    query_ids = store.ids[6:]

    descriptor = DescriptorPanelTransformer("rdkit2d_217").fit(store, fit_ids=fit_ids)
    transformed = descriptor.transform(store, query_ids)
    assert transformed.index.tolist() == list(query_ids)
    assert "constant" not in transformed.columns
    assert np.isfinite(transformed.to_numpy()).all()
    assert descriptor.get_manifest()["fit_ids"] == list(fit_ids)

    with pytest.raises(ValueError, match="exceeds"):
        MorganSVDPanelTransformer(n_components=6, random_state=3).fit(store, fit_ids=fit_ids)
    morgan = MorganSVDPanelTransformer(n_components=2, random_state=3).fit(
        store, fit_ids=fit_ids
    )
    output = morgan.transform(store, query_ids)
    assert output.shape[0] == 2
    assert output.shape[1] == 2 + store.chemistry32.shape[1]
    assert morgan.get_manifest()["effective_components"] == 2


class _StableAuditAdapter:
    def fit_context(self, X: pd.DataFrame, y: np.ndarray):
        self.columns = tuple(X.columns)
        self.context_ids = set(map(str, X.index))
        return self

    def predict_proba(self, X: pd.DataFrame, *, canonical_mode: bool):
        assert tuple(X.columns) == self.columns
        assert not (set(map(str, X.index)) & self.context_ids)
        positive = 1.0 / (1.0 + np.exp(-X.to_numpy(dtype=float).sum(axis=1)))
        return np.column_stack([1.0 - positive, positive])


class _ContextOrderSensitiveAdapter(_StableAuditAdapter):
    def fit_context(self, X: pd.DataFrame, y: np.ndarray):
        super().fit_context(X, y)
        self.offset = float(X.iloc[0, 0])
        return self

    def predict_proba(self, X: pd.DataFrame, *, canonical_mode: bool):
        values = X.to_numpy(dtype=float).sum(axis=1) + self.offset
        positive = 1.0 / (1.0 + np.exp(-values))
        return np.column_stack([1.0 - positive, positive])


class _BatchPositionSpyAdapter(_StableAuditAdapter):
    noncanonical_batches: ClassVar[list[tuple[str, ...]]] = []

    def predict_proba(self, X: pd.DataFrame, *, canonical_mode: bool):
        if not canonical_mode:
            self.noncanonical_batches.append(tuple(map(str, X.index)))
        return super().predict_proba(X, canonical_mode=canonical_mode)


def _audit_frames() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    context = pd.DataFrame(
        [[-2.0, 0.0], [-1.0, 1.0], [1.0, -1.0], [3.0, 1.0]],
        index=["train-a", "train-b", "train-c", "train-d"],
        columns=["x", "z"],
    )
    labels = np.asarray([0, 0, 1, 1])
    query = pd.DataFrame(
        [[-0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0, 1.0]],
        index=["test-a", "test-b", "test-c", "test-d"],
        columns=context.columns,
    )
    return context, labels, query


def test_v6_inference_audit_checks_repeat_query_and_context_order() -> None:
    context, labels, query = _audit_frames()
    audit = audit_inference_semantics(
        _StableAuditAdapter,
        context,
        labels,
        query,
        atol=1e-12,
        seed=7,
        random_compositions_per_query=3,
    )
    assert audit["passed"] is True
    assert all(audit["hard_checks"].values())
    assert audit["random_compositions_per_query"] == 3

    sensitive = audit_inference_semantics(
        _ContextOrderSensitiveAdapter,
        context,
        labels,
        query,
        atol=1e-12,
        seed=7,
        hard_fail_on_context_row_order=True,
    )
    assert sensitive["passed"] is False
    assert sensitive["hard_checks"]["context_row_order"] is False


def test_v6_inference_audit_rejects_context_query_overlap() -> None:
    context, labels, query = _audit_frames()
    query.index = ["train-a", "test-b", "test-c", "test-d"]
    with pytest.raises(ValueError, match="overlap"):
        audit_inference_semantics(
            _StableAuditAdapter,
            context,
            labels,
            query,
            atol=1e-6,
            seed=7,
        )


class _FailingAuditAdapter:
    def fit_context(self, X: pd.DataFrame, y: np.ndarray):
        del X, y
        return self

    def predict_proba(self, X: pd.DataFrame, *, canonical_mode: bool):
        del X, canonical_mode
        raise RuntimeError("synthetic inference failure")


def test_v6_inference_audit_releases_adapter_memory_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, labels, query = _audit_frames()
    release_calls: list[bool] = []
    monkeypatch.setattr(
        inference_audit_module,
        "release_accelerator_memory",
        lambda: release_calls.append(True),
    )
    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        audit_inference_semantics(
            _FailingAuditAdapter,
            context,
            labels,
            query,
            atol=1e-6,
            seed=7,
        )
    assert release_calls == [True]


def test_v6_random_composition_audit_varies_focal_query_position() -> None:
    context, labels, _ = _audit_frames()
    query = pd.DataFrame(
        np.arange(24, dtype=float).reshape(12, 2) / 10.0,
        index=[f"query-{index}" for index in range(12)],
        columns=context.columns,
    )
    _BatchPositionSpyAdapter.noncanonical_batches = []
    repetitions = 10
    audit_inference_semantics(
        _BatchPositionSpyAdapter,
        context,
        labels,
        query,
        atol=1e-12,
        seed=71,
        random_compositions_per_query=repetitions,
    )
    # The first noncanonical call is the whole query batch; subsequent calls are grouped
    # by focal query in audit order.
    batches = _BatchPositionSpyAdapter.noncanonical_batches[1:]
    observed_positions = []
    for query_index, query_id in enumerate(query.index):
        block = batches[query_index * repetitions : (query_index + 1) * repetitions]
        observed_positions.extend(batch.index(query_id) for batch in block)
    assert len(set(observed_positions)) > 1


class _ProbabilityEstimator:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive = 1.0 / (1.0 + np.exp(-np.asarray(X)[:, 0]))
        return np.column_stack([1.0 - positive, positive])


@pytest.mark.parametrize("adapter_class", [TabPFNAdapter, TabICLAdapter])
def test_foundation_adapters_enforce_query_schema_identity_and_probability_contract(
    adapter_class,
    tmp_path: Path,
) -> None:
    adapter = adapter_class(
        {}, project_root=tmp_path, n_estimators=1, random_state=1, device="cpu"
    )
    adapter.estimator_ = _ProbabilityEstimator()
    adapter.feature_names_ = ("x", "z")
    adapter.context_ids_ = ("train",)
    query = pd.DataFrame([[1.0, 0.0], [-1.0, 0.0]], index=["q2", "q1"], columns=["x", "z"])
    prediction = adapter.predict_proba(query, canonical_mode=True)
    reversed_prediction = adapter.predict_proba(query.iloc[::-1], canonical_mode=True)[::-1]
    np.testing.assert_allclose(prediction, reversed_prediction)
    assert np.allclose(prediction.sum(axis=1), 1.0)

    with pytest.raises(ValueError, match="schema/order"):
        adapter.predict_proba(query[["z", "x"]], canonical_mode=True)
    overlapping = query.copy()
    overlapping.index = ["train", "q1"]
    with pytest.raises(ValueError, match="overlap"):
        adapter.predict_proba(overlapping, canonical_mode=True)


def test_tabpfn_requires_telemetry_disabled_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TABPFN_DISABLE_TELEMETRY", raising=False)
    adapter = TabPFNAdapter(
        {}, project_root=tmp_path, n_estimators=1, random_state=1, device="cpu"
    )
    context = pd.DataFrame([[0.0], [1.0]], index=["a", "b"], columns=["x"])
    with pytest.raises(RuntimeError, match="TABPFN_DISABLE_TELEMETRY"):
        adapter.fit_context(context, np.asarray([0, 1]))


def _install_fake_tabpfn(
    monkeypatch: pytest.MonkeyPatch,
    *,
    effective_overrides: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    class FakeModelVersion:
        def __class_getitem__(cls, name: str) -> str:
            return f"resolved::{name}"

    class FakeEstimator:
        def __init__(self, parameters: dict[str, object]) -> None:
            self.parameters = dict(parameters)

        def get_params(self, *, deep: bool) -> dict[str, object]:
            assert deep is False
            return {**self.parameters, **(effective_overrides or {})}

        def fit(self, matrix: np.ndarray, labels: np.ndarray):
            assert matrix.shape == (4, 2)
            assert labels.tolist() == [0, 1, 0, 1]
            self.n_estimators_ = int(self.parameters["n_estimators"])
            self.classes_ = np.asarray([0, 1])
            return self

    class FakeClassifier:
        @staticmethod
        def create_default_for_version(version: object, **parameters: object):
            calls.append({"version": version, **parameters})
            return FakeEstimator(parameters)

    package = ModuleType("tabpfn")
    package.__path__ = []
    package.TabPFNClassifier = FakeClassifier
    constants = ModuleType("tabpfn.constants")
    constants.ModelVersion = FakeModelVersion
    monkeypatch.setitem(sys.modules, "tabpfn", package)
    monkeypatch.setitem(sys.modules, "tabpfn.constants", constants)
    return calls


def _tabpfn_fit_fixture(tmp_path: Path) -> tuple[dict[str, str], pd.DataFrame, np.ndarray]:
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"offline-tabpfn-checkpoint")
    record = {
        "checkpoint_path": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "explicit_model_version": "fixture-v1",
    }
    context = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]],
        index=["a", "b", "c", "d"],
        columns=["x", "z"],
    )
    return record, context, np.asarray([0, 1, 0, 1])


def test_tabpfn_disables_auto_scaling_and_records_effective_constructor_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fake_tabpfn(monkeypatch)
    monkeypatch.setenv("TABPFN_DISABLE_TELEMETRY", "1")
    record, context, labels = _tabpfn_fit_fixture(tmp_path)
    adapter = TabPFNAdapter(
        record,
        project_root=tmp_path,
        n_estimators=7,
        random_state=19,
        device="cpu",
    ).fit_context(context, labels)

    checkpoint = str((tmp_path / record["checkpoint_path"]).resolve())
    assert calls == [
        {
            "version": "resolved::fixture-v1",
            "model_path": checkpoint,
            "n_estimators": 7,
            "auto_scale_n_estimators": False,
            "random_state": 19,
            "device": "cpu",
            "show_progress_bar": False,
        }
    ]
    manifest = adapter.save_manifest()
    assert manifest["auto_scale_n_estimators"] is False
    assert manifest["effective_constructor_parameters"] == {
        "n_estimators": "7",
        "auto_scale_n_estimators": "False",
        "random_state": "19",
        "device": "cpu",
        "model_path": checkpoint,
        "show_progress_bar": "False",
    }


def test_tabpfn_rejects_effective_auto_scaling_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_tabpfn(
        monkeypatch,
        effective_overrides={"auto_scale_n_estimators": True},
    )
    monkeypatch.setenv("TABPFN_DISABLE_TELEMETRY", "1")
    record, context, labels = _tabpfn_fit_fixture(tmp_path)
    adapter = TabPFNAdapter(
        record,
        project_root=tmp_path,
        n_estimators=7,
        random_state=19,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="auto_scale_n_estimators"):
        adapter.fit_context(context, labels)


def _checkpoint_fixture(tmp_path: Path):
    checkpoint = tmp_path / "checkpoints/model.bin"
    license_path = tmp_path / "licenses/model.txt"
    checkpoint.parent.mkdir()
    license_path.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint-one")
    license_path.write_text("license-one", encoding="utf-8")
    settings = {
        "enabled": True,
        "package": "fixture-package",
        "package_version": "1.0",
        "explicit_model_version": "fixture-v1",
        "checkpoint_path": "checkpoints/model.bin",
        "checkpoint_sha256": sha256_file(checkpoint),
        "license_path": "licenses/model.txt",
        "license_sha256": sha256_file(license_path),
        "checkpoint_source": "fixture://local",
        "access_date_utc": "2026-08-15",
    }
    config = {"models": {"fixture": settings}}
    record = {"model_id": "fixture", **{k: v for k, v in settings.items() if k != "enabled"}}
    ledger = {
        "schema_version": "geroprotector.v6_checkpoint_ledger.v1",
        "implicit_downloads_allowed": False,
        "telemetry_disabled": True,
        "offline_environment": {
            "TABPFN_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "resolved_config_sha256": "config-hash",
        "environment": _torch_environment(),
        "models": [record],
    }
    ledger["canonical_sha256"] = canonical_sha256(ledger)
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    return config, ledger, ledger_path


def _duplicate_checkpoint_fixture(
    tmp_path: Path, *, duplicate_kind: str
) -> tuple[dict[str, object], dict[str, object], Path]:
    config, ledger, ledger_path = _checkpoint_fixture(tmp_path)
    config["inference"] = {"telemetry_disabled": True}
    first = config["models"]["fixture"]

    second_checkpoint = tmp_path / "checkpoints/model-two.bin"
    second_checkpoint.write_bytes(b"checkpoint-one")
    second_license = tmp_path / "licenses/model-two.txt"
    second_license.write_text("license-two", encoding="utf-8")
    second = {
        **first,
        "explicit_model_version": "fixture-v2",
        "checkpoint_path": (
            first["checkpoint_path"]
            if duplicate_kind == "path"
            else "checkpoints/model-two.bin"
        ),
        "checkpoint_sha256": sha256_file(second_checkpoint),
        "license_path": "licenses/model-two.txt",
        "license_sha256": sha256_file(second_license),
        "checkpoint_source": "fixture://local-two",
    }
    config["models"]["fixture_two"] = second
    second_record = {
        "model_id": "fixture_two",
        **{key: value for key, value in second.items() if key != "enabled"},
    }
    ledger["models"].append(second_record)
    ledger.pop("canonical_sha256")
    ledger["canonical_sha256"] = canonical_sha256(ledger)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    return config, ledger, ledger_path


@pytest.mark.parametrize(
    ("duplicate_kind", "stage_message"),
    [("path", "share a checkpoint path"), ("sha", "share checkpoint bytes")],
)
def test_duplicate_enabled_checkpoint_identity_is_rejected_at_stage_and_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    duplicate_kind: str,
    stage_message: str,
) -> None:
    config, _, ledger_path = _duplicate_checkpoint_fixture(
        tmp_path, duplicate_kind=duplicate_kind
    )
    real_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda package: "1.0" if package == "fixture-package" else real_version(package),
    )
    monkeypatch.setenv("TABPFN_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    with pytest.raises(CheckpointError, match=stage_message):
        stage_checkpoints(
            config,
            root=tmp_path,
            output_path=tmp_path / "staged-ledger.json",
            resolved_config_sha256="config-hash",
        )
    with pytest.raises(CheckpointError, match="share checkpoint bytes/path"):
        load_checkpoint_ledger(
            ledger_path,
            config=config,
            root=tmp_path,
            resolved_config_sha256="config-hash",
        )


def test_checkpoint_loader_rebinds_ledger_to_config_and_current_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, ledger, ledger_path = _checkpoint_fixture(tmp_path)
    real_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda package: "1.0" if package == "fixture-package" else real_version(package),
    )
    monkeypatch.setenv("TABPFN_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    loaded = load_checkpoint_ledger(
        ledger_path,
        config=config,
        root=tmp_path,
        resolved_config_sha256="config-hash",
    )
    assert (
        checkpoint_record(loaded, "fixture")["checkpoint_sha256"]
        == config["models"]["fixture"]["checkpoint_sha256"]
    )

    changed_config = copy.deepcopy(config)
    changed_config["models"]["fixture"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(CheckpointError):
        load_checkpoint_ledger(
            ledger_path,
            config=changed_config,
            root=tmp_path,
            resolved_config_sha256="config-hash",
        )

    Path(tmp_path / ledger["models"][0]["checkpoint_path"]).write_bytes(b"tampered")
    with pytest.raises(CheckpointError, match="changed"):
        load_checkpoint_ledger(
            ledger_path,
            config=config,
            root=tmp_path,
            resolved_config_sha256="config-hash",
        )


def test_checkpoint_paths_cannot_escape_or_traverse_symlinks(tmp_path: Path) -> None:
    real = tmp_path / "real.bin"
    real.write_bytes(b"x")
    link = tmp_path / "link.bin"
    link.symlink_to(real)
    assert _project_file(tmp_path.resolve(), "real.bin", role="fixture") == real
    with pytest.raises(CheckpointError):
        _project_file(tmp_path.resolve(), "../outside.bin", role="fixture")
    with pytest.raises(CheckpointError, match="symlink"):
        _project_file(tmp_path.resolve(), "link.bin", role="fixture")


def test_primary_v6_must_remain_zero_shot(v6_config) -> None:
    assert_primary_zero_shot(v6_config)
    adapted = copy.deepcopy(v6_config)
    adapted["adaptation"]["context_only_ttt"]["enabled"] = True
    with pytest.raises(ValueError, match="separate config/run"):
        assert_primary_zero_shot(adapted)


def test_v6_prediction_is_bound_to_audited_query_order_and_feature_store(v6_config) -> None:
    full = _manual_store()
    fitted_subset = full.subset(full.ids[:6])
    query_ids = full.ids[6:]
    pipeline = V6Pipeline(
        v6_config,
        {"models": [], "canonical_sha256": "ledger"},
        project_root=ROOT,
    )
    pipeline.adapter_ = SimpleNamespace()
    pipeline.panel_ = DescriptorPanelTransformer("chemistry_32").fit(
        full, fit_ids=fitted_subset.ids
    )
    pipeline.inference_audit_ = {"passed": True}
    pipeline.store_hash_ = fitted_subset.store_hash
    pipeline.feature_contract_hash_ = fitted_subset.feature_contract_hash
    pipeline.fit_ids_ = fitted_subset.ids
    pipeline.audited_query_ids_ = query_ids
    pipeline.audited_probabilities_ = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    pipeline.audited_query_binding_sha256_ = _query_binding_sha256(
        query_ids, pipeline.panel_.transform(full, query_ids)
    )
    np.testing.assert_array_equal(
        pipeline.predict_proba(full, query_ids),
        pipeline.audited_probabilities_,
    )
    query_store = full.subset(query_ids)
    assert len({fitted_subset.store_hash, full.store_hash, query_store.store_hash}) == 3
    pipeline.predict_proba(query_store, query_ids)
    with pytest.raises(RuntimeError, match="IDs/order"):
        pipeline.predict_proba(full, query_ids[::-1])
    with pytest.raises(ValueError, match="feature-extractor contract"):
        pipeline.predict_proba(_manual_store(name_suffix="_other"), query_ids)
    with pytest.raises(ValueError, match="overlap"):
        pipeline.audit_inference(full, query_ids=(fitted_subset.ids[0],))

    changed_chemistry = full.chemistry32.copy()
    changed_chemistry.loc[query_ids[0], changed_chemistry.columns[0]] += 100.0
    changed_store = V6FeatureStore(
        ids=full.ids,
        smiles=full.smiles,
        rdkit2d=full.rdkit2d.copy(),
        chemistry32=changed_chemistry,
        morgan_count=full.morgan_count.copy(),
        feature_contract_hash=full.feature_contract_hash,
        store_hash=_feature_content_hash(
            full.ids,
            full.smiles,
            full.rdkit2d,
            changed_chemistry,
            full.morgan_count,
            feature_contract_hash=full.feature_contract_hash,
        ),
    )
    changed_store.assert_integrity()
    with pytest.raises(RuntimeError, match="features differ from the audited"):
        pipeline.predict_proba(changed_store, query_ids)

    query_store.morgan_count[0, 0] = 1.0 - query_store.morgan_count[0, 0]
    with pytest.raises(ValueError, match="content hash"):
        pipeline.predict_proba(query_store, query_ids)


class _SeedSpyPanel:
    seeds: ClassVar[list[int]] = []

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.seeds.append(self.seed)

    def fit(self, store: V6FeatureStore, *, fit_ids):
        del store
        self.fit_ids = tuple(fit_ids)
        return self

    def transform(self, store: V6FeatureStore, ids) -> pd.DataFrame:
        del store
        requested = tuple(ids)
        values = np.asarray([[float(str(value).rsplit("::", 1)[-1])] for value in requested])
        return pd.DataFrame(values, index=requested, columns=["x"])

    def get_manifest(self):
        return {
            "panel": "chemistry_32",
            "fit_ids": list(self.fit_ids),
            "feature_names": ["x"],
        }


class _SeedSpyAdapter:
    seeds: ClassVar[list[int]] = []

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.seeds.append(self.seed)

    def fit_context(self, X: pd.DataFrame, y: np.ndarray):
        del X, y
        return self

    def predict_proba(self, X: pd.DataFrame, *, canonical_mode: bool) -> np.ndarray:
        del canonical_mode
        positive = np.where(X.to_numpy(dtype=float)[:, 0] % 2 == 0, 0.2, 0.8)
        return np.column_stack([1.0 - positive, positive])


def _two_fold_v6_inputs():
    store = _manual_store()
    labels = pd.Series(
        [index % 2 for index in range(len(store.ids))], index=store.ids, dtype=int
    )
    groups = pd.Series([f"group-{index}" for index in range(len(store.ids))], index=store.ids)
    first, second = store.ids[:4], store.ids[4:]
    folds = [(first, second), (second, first)]
    return store, labels, groups, folds


def test_v6_foundation_candidates_use_common_random_numbers(
    v6_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, labels, groups, folds = _two_fold_v6_inputs()
    candidates = (
        {
            "model_id": "model-a",
            "panel": {"panel": "chemistry_32", "n_components": None},
            "ensemble_size": 4,
        },
        {
            "model_id": "model-b",
            "panel": {"panel": "chemistry_32", "n_components": None},
            "ensemble_size": 4,
        },
    )
    ledger = {
        "canonical_sha256": "fixture-ledger",
        "models": [{"model_id": "model-a"}, {"model_id": "model-b"}],
    }
    _SeedSpyPanel.seeds = []
    _SeedSpyAdapter.seeds = []
    monkeypatch.setattr(pipeline_module, "_portfolio", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(
        pipeline_module,
        "build_panel_transformer",
        lambda spec, seed: _SeedSpyPanel(seed),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_adapter",
        lambda model_id, record, *, root, ensemble_size, seed, device: _SeedSpyAdapter(seed),
    )

    V6Pipeline(v6_config, ledger, project_root=ROOT, seed=23).fit(
        store, labels, groups=groups, selection_folds=folds
    )

    assert _SeedSpyPanel.seeds[:4] == [23, 24, 23, 24]
    assert _SeedSpyAdapter.seeds[:4] == [23, 24, 23, 24]
    assert _SeedSpyPanel.seeds[4:] == [23 + 999_983]
    assert _SeedSpyAdapter.seeds[4:] == [23 + 999_983]


def test_v6_classical_candidates_use_common_random_numbers(
    v6_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, labels, groups, folds = _two_fold_v6_inputs()
    candidates = (
        {
            "panel": {"panel": "chemistry_32", "n_components": None},
            "estimator": {"n_estimators": 10, "min_samples_leaf": 1, "max_features": "sqrt"},
        },
        {
            "panel": {"panel": "chemistry_32", "n_components": None},
            "estimator": {"n_estimators": 20, "min_samples_leaf": 2, "max_features": "sqrt"},
        },
    )
    observed: list[int] = []

    def fake_fit_candidate(self, active_store, ids, active_labels, candidate, *, seed):
        del self, active_store, active_labels
        observed.append(int(seed))
        return SimpleNamespace(candidate=candidate, fit_ids=tuple(ids)), object(), None

    def fake_predict(panel, estimator, scaler, active_store, ids):
        del panel, estimator, scaler, active_store
        positive = np.asarray(
            [0.2 if int(str(value).rsplit("::", 1)[-1]) % 2 == 0 else 0.8 for value in ids]
        )
        return np.column_stack([1.0 - positive, positive])

    monkeypatch.setattr(baseline_module, "_portfolio", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(V6ClassicalPipeline, "_fit_candidate", fake_fit_candidate)
    monkeypatch.setattr(V6ClassicalPipeline, "_predict", staticmethod(fake_predict))

    V6ClassicalPipeline("extra_trees", v6_config, seed=31).fit(
        store, labels, groups=groups, selection_folds=folds
    )

    assert observed[:4] == [31, 32, 31, 32]
    assert observed[4:] == [31 + 999_983]


class _ResourceExhaustedAdapter:
    def fit_context(self, X: pd.DataFrame, y: np.ndarray):
        del X, y
        raise RuntimeError("CUDA out of memory in synthetic fixture")


def test_v6_resource_exhaustion_aborts_locked_candidate_portfolio(
    v6_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, labels, groups, folds = _two_fold_v6_inputs()
    candidate = {
        "model_id": "model-a",
        "panel": {"panel": "chemistry_32", "n_components": None},
        "ensemble_size": 4,
    }
    ledger = {
        "canonical_sha256": "fixture-ledger",
        "models": [{"model_id": "model-a"}],
    }
    release_calls: list[bool] = []
    monkeypatch.setattr(pipeline_module, "_portfolio", lambda *args, **kwargs: (candidate,))
    monkeypatch.setattr(
        pipeline_module,
        "build_panel_transformer",
        lambda spec, seed: _SeedSpyPanel(seed),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_adapter",
        lambda *args, **kwargs: _ResourceExhaustedAdapter(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "release_accelerator_memory",
        lambda: release_calls.append(True),
    )

    with pytest.raises(RuntimeError, match="exhausted memory; aborting the locked run"):
        V6Pipeline(v6_config, ledger, project_root=ROOT, seed=23).fit(
            store, labels, groups=groups, selection_folds=folds
        )
    assert release_calls


def test_v6_classical_resource_exhaustion_aborts_locked_candidate_portfolio(
    v6_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, labels, groups, folds = _two_fold_v6_inputs()
    candidate = {
        "panel": {"panel": "chemistry_32", "n_components": None},
        "estimator": {
            "n_estimators": 10,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
    }

    def exhaust_memory(*args, **kwargs):
        del args, kwargs
        raise MemoryError("synthetic allocation failure")

    monkeypatch.setattr(baseline_module, "_portfolio", lambda *args, **kwargs: (candidate,))
    monkeypatch.setattr(V6ClassicalPipeline, "_fit_candidate", exhaust_memory)
    with pytest.raises(RuntimeError, match="exhausted memory; aborting the locked run"):
        V6ClassicalPipeline("extra_trees", v6_config, seed=31).fit(
            store, labels, groups=groups, selection_folds=folds
        )


def test_v6_reconstructed_adapter_must_match_pre_audit_predictions(
    v6_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _manual_store()
    fit_ids = store.ids[:6]
    query_ids = store.ids[6:]
    panel = _SeedSpyPanel(seed=13).fit(store, fit_ids=fit_ids)
    context = panel.transform(store, fit_ids)
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=int)
    ledger = {
        "canonical_sha256": "fixture-ledger",
        "models": [{"model_id": "model-a"}],
    }
    pipeline = V6Pipeline(
        v6_config,
        ledger,
        project_root=ROOT,
        seed=29,
        device="cpu",
    )
    pipeline.panel_ = panel
    pipeline.adapter_ = _SeedSpyAdapter(seed=29 + 999_983).fit_context(context, labels)
    pipeline.winner_ = {"model_id": "model-a", "ensemble_size": 4}
    pipeline.fit_ids_ = fit_ids
    pipeline.fit_labels_ = labels
    pipeline.context_features_ = context
    fitted_store = store.subset(fit_ids)
    pipeline.store_hash_ = fitted_store.store_hash
    pipeline.feature_contract_hash_ = fitted_store.feature_contract_hash
    monkeypatch.setattr(
        pipeline_module,
        "_adapter",
        lambda *args, seed, **kwargs: _SeedSpyAdapter(seed),
    )

    audit = pipeline.audit_inference(store, query_ids=query_ids)

    assert audit["reconstruction_within_tolerance"] is True
    assert audit["reconstruction_max_abs_difference"] == pytest.approx(0.0)
    assert audit["runtime"]["actual_adapter_forward_passes"] == 4 * len(query_ids)
    np.testing.assert_array_equal(
        pipeline.predict_proba(store, query_ids), pipeline.audited_probabilities_
    )


def test_nested_runner_uses_lightweight_v6_path_only_for_calibration(
    v6_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = V6Pipeline(
        v6_config,
        {"models": [], "canonical_sha256": "fixture-ledger"},
        project_root=ROOT,
        device="cpu",
    )
    calls: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "predict_calibration_oof",
        lambda store, ids: calls.append("calibration") or np.asarray([[0.8, 0.2], [0.3, 0.7]]),
    )
    monkeypatch.setattr(
        pipeline,
        "audit_and_predict",
        lambda store, ids: (
            calls.append("outer") or (np.asarray([[0.6, 0.4], [0.1, 0.9]]), {"passed": True})
        ),
    )

    calibration, calibration_audit = nested_cv_module._predict(
        pipeline, object(), ("a", "b"), full_foundation_audit=False
    )
    outer, outer_audit = nested_cv_module._predict(
        pipeline, object(), ("a", "b"), full_foundation_audit=True
    )

    np.testing.assert_allclose(calibration, [0.2, 0.7])
    assert calibration_audit is None
    np.testing.assert_allclose(outer, [0.4, 0.9])
    assert outer_audit == {"passed": True}
    assert calls == ["calibration", "outer"]


class _PortableAdapter(_SeedSpyAdapter):
    def __init__(self, seed: int, model_id: str = "model-a") -> None:
        super().__init__(seed)
        self.model_id = model_id

    def fit_context(self, X: pd.DataFrame, y: np.ndarray):
        self.context_ids = tuple(map(str, X.index))
        self.feature_names = tuple(map(str, X.columns))
        return super().fit_context(X, y)

    def save_manifest(self):
        return {
            "model_id": self.model_id,
            "random_state": self.seed,
            "context_ids": list(self.context_ids),
            "feature_names": list(self.feature_names),
            "canonical_inference": "one_query_per_call",
        }


def test_v6_portable_reload_is_sealed_reaudited_and_tamper_evident(
    tmp_path: Path,
    v6_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _manual_store()
    fit_ids = store.ids[:5]
    historical_query_ids = store.ids[5:7]
    new_query_ids = store.ids[6:]
    panel = _SeedSpyPanel(seed=13).fit(store, fit_ids=fit_ids)
    context = panel.transform(store, fit_ids)
    labels = np.asarray([0, 1, 0, 1, 0], dtype=int)
    ledger_payload = {
        "schema_version": "geroprotector.v6_checkpoint_ledger.v1",
        "models": [{"model_id": "model-a"}],
    }
    ledger = {**ledger_payload, "canonical_sha256": canonical_sha256(ledger_payload)}
    pipeline = V6Pipeline(
        v6_config,
        ledger,
        project_root=tmp_path,
        seed=37,
        device="cpu",
    )
    pipeline.panel_ = panel
    pipeline.adapter_ = _PortableAdapter(seed=37 + 999_983).fit_context(context, labels)
    pipeline.winner_ = {"model_id": "model-a", "ensemble_size": 4}
    pipeline.fit_ids_ = fit_ids
    pipeline.fit_labels_ = labels
    pipeline.context_features_ = context
    fitted_store = store.subset(fit_ids)
    pipeline.store_hash_ = fitted_store.store_hash
    pipeline.feature_contract_hash_ = fitted_store.feature_contract_hash
    pipeline.selection_trace_ = pd.DataFrame(
        [
            {
                "candidate_id": "candidate-a",
                "status": "success",
                "selected": True,
            }
        ]
    )
    pipeline.inference_audit_ = {
        "passed": True,
        "query_ids": list(historical_query_ids),
    }
    pipeline.audited_query_ids_ = historical_query_ids
    pipeline.audited_probabilities_ = pipeline.adapter_.predict_proba(
        panel.transform(store, historical_query_ids), canonical_mode=True
    )
    expected_new = pipeline.adapter_.predict_proba(
        panel.transform(store, new_query_ids), canonical_mode=True
    )
    artifact = tmp_path / "portable.joblib"
    pipeline.save(artifact)
    sidecar = artifact.with_suffix(".joblib.manifest.json")
    original_artifact = artifact.read_bytes()
    original_sidecar = sidecar.read_bytes()
    sealed_artifact_sha256 = sha256_file(artifact)
    sealed_manifest_sha256 = sha256_file(sidecar)

    monkeypatch.setattr(
        pipeline_module, "resolve_config", lambda path: copy.deepcopy(v6_config)
    )
    monkeypatch.setattr(pipeline_module, "validate_locked_shared_contract", lambda config: None)
    monkeypatch.setattr(pipeline_module, "validate_protocol_lock", lambda **kwargs: None)
    monkeypatch.setattr(
        pipeline_module,
        "load_checkpoint_ledger",
        lambda *args, **kwargs: copy.deepcopy(ledger),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_adapter",
        lambda model_id, record, *, seed, **kwargs: _PortableAdapter(seed, model_id),
    )
    config_source = tmp_path / "v6.yaml"
    ledger_source = tmp_path / "checkpoint-ledger.json"
    config_source.write_text("fixture: true\n", encoding="utf-8")
    ledger_source.write_text("{}\n", encoding="utf-8")

    def load_portable(
        *,
        artifact_sha256: str = sealed_artifact_sha256,
        manifest_sha256: str = sealed_manifest_sha256,
    ) -> V6Pipeline:
        return V6Pipeline.load_portable(
            artifact.name,
            project_root=tmp_path,
            expected_artifact_sha256=artifact_sha256,
            expected_manifest_sha256=manifest_sha256,
            config_path=config_source.name,
            checkpoint_ledger_path=ledger_source.name,
        )

    def rewrite_sidecar_for_current_artifact() -> None:
        rewritten = json.loads(original_sidecar)
        rewritten["model_artifact_sha256"] = sha256_file(artifact)
        rewritten.pop("canonical_sha256")
        rewritten["canonical_sha256"] = canonical_sha256(rewritten)
        sidecar.write_text(json.dumps(rewritten), encoding="utf-8")

    loaded = load_portable()
    assert loaded.historical_inference_audit_ == pipeline.inference_audit_
    assert not hasattr(loaded, "inference_audit_")
    with pytest.raises(RuntimeError, match="audit must pass"):
        loaded.predict_proba(store, new_query_ids)
    reloaded_probability, audit = loaded.audit_and_predict(store, new_query_ids)
    assert audit["passed"] is True
    np.testing.assert_allclose(reloaded_probability, expected_new)

    artifact.write_bytes(original_artifact + b"tamper")
    with pytest.raises(ValueError, match="sealed job inventory"):
        load_portable()
    artifact.write_bytes(original_artifact)

    manifest = json.loads(original_sidecar)
    manifest["phase"] = "tampered"
    sidecar.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest integrity"):
        load_portable(manifest_sha256=sha256_file(sidecar))
    sidecar.write_bytes(original_sidecar)

    changed_config = copy.deepcopy(v6_config)
    changed_config["search"]["max_candidates_per_active_fit"] += 1
    monkeypatch.setattr(
        pipeline_module, "resolve_config", lambda path: copy.deepcopy(changed_config)
    )
    with pytest.raises(ValueError, match="config differs"):
        load_portable()
    monkeypatch.setattr(
        pipeline_module, "resolve_config", lambda path: copy.deepcopy(v6_config)
    )

    changed_ledger_payload = {**ledger_payload, "environment": {"tampered": True}}
    changed_ledger = {
        **changed_ledger_payload,
        "canonical_sha256": canonical_sha256(changed_ledger_payload),
    }
    monkeypatch.setattr(
        pipeline_module,
        "load_checkpoint_ledger",
        lambda *args, **kwargs: copy.deepcopy(changed_ledger),
    )
    with pytest.raises(ValueError, match="checkpoint ledger differs"):
        load_portable()
    monkeypatch.setattr(
        pipeline_module,
        "load_checkpoint_ledger",
        lambda *args, **kwargs: copy.deepcopy(ledger),
    )

    for field in ("context_features", "fit_labels"):
        artifact.write_bytes(original_artifact)
        sidecar.write_bytes(original_sidecar)
        portable = joblib.load(artifact)
        if field == "context_features":
            portable[field].iloc[0, 0] += 1.0
        else:
            portable[field] = np.roll(np.asarray(portable[field]), 1)
        joblib.dump(portable, artifact)
        rewrite_sidecar_for_current_artifact()
        with pytest.raises(ValueError, match="sealed job inventory"):
            load_portable()
        with pytest.raises(ValueError, match="context/label payload hash"):
            load_portable(
                artifact_sha256=sha256_file(artifact),
                manifest_sha256=sha256_file(sidecar),
            )

    artifact.write_bytes(original_artifact)
    sidecar.write_bytes(original_sidecar)
    portable = joblib.load(artifact)
    portable["checkpoint_ledger"]["models"][0]["tampered"] = True
    joblib.dump(portable, artifact)
    rewrite_sidecar_for_current_artifact()
    with pytest.raises(ValueError, match="checkpoint ledger differs"):
        load_portable(
            artifact_sha256=sha256_file(artifact),
            manifest_sha256=sha256_file(sidecar),
        )
