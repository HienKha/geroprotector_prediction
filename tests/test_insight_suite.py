"""Fast contract tests for the experiment suite in ``prespecified insight-analysis protocol``.

These tests intentionally exercise protocol and orchestration boundaries without
loading a foundation model or touching a sealed output.  A failing test here is
an audit finding: it should be fixed in a new source/config revision before the
affected long-running experiment is treated as manuscript evidence.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from geroprotector import chemical_space_cv as chemical
from geroprotector import model_panel
from geroprotector import drugage_celegans_benchmark as drugage
from geroprotector import drugage_celegans_finalize as drugage_finalize
from geroprotector import endpoint_benchmark as endpoint
from geroprotector import repeated_rank_stability as repeated
from geroprotector.nine_ml_featuresets import _metrics


ROOT = Path(__file__).resolve().parents[1]

FIXED_BASE_MODELS = (
    "paper_svm",
    "tanimoto_svc",
    "tabpfn_v2",
    "tabfm",
    "catboost_full",
    "xgboost_full",
    "lightgbm_full",
    "tabm_full",
)
FIXED_BLENDS = {
    "blend3_equal": ("paper_svm", "tabpfn_v2", "tabfm"),
    "gb4_equal": ("paper_svm", "tanimoto_svc", "tabpfn_v2", "tabfm"),
}
PRIMARY_METRICS = {
    "auprc_average_precision_positive",
    "auroc",
    "accuracy",
    "mcc",
    "macro_f1",
    "brier",
    "cohen_kappa",
}

PROTOCOLS = {
    "modelwide_druglikeness_bias_protocol.yaml":
        "geroprotector.modelwide_druglikeness_bias.protocol.v1",
    "repeated_rank_stability_protocol.yaml":
        "geroprotector.repeated_rank_stability.protocol.v1",
    "chemical_space_cv_protocol.yaml":
        "geroprotector.chemical_space_cv.protocol.v1",
    "agextend_endpoint_benchmark_protocol.yaml":
        "geroprotector.agextend_endpoint_benchmark.protocol.v1",
    "drugage_celegans_benchmark_protocol.yaml":
        "geroprotector.drugage_celegans_benchmark.protocol.v1",
    "agextend_official_reference_protocol.yaml":
        "geroprotector.agextend_official_reference.protocol.v1",
    "kapsiani_historical_benchmark_protocol.yaml":
        "geroprotector.kapsiani_historical_benchmark.protocol.v1",
}


def _protocol(name: str) -> dict[str, Any]:
    path = ROOT / "configs" / name
    assert path.is_file(), f"required versioned protocol is missing: {path}"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"protocol is not a mapping: {path}"
    return payload


@pytest.mark.parametrize("name, schema", PROTOCOLS.items())
def test_every_experiment_has_a_versioned_protocol(name: str, schema: str) -> None:
    payload = _protocol(name)
    assert payload.get("schema_version") == schema


def test_fixed_panel_and_threshold_are_identical_in_b_and_c_protocols() -> None:
    assert tuple(model_panel.BASE_MODELS) == FIXED_BASE_MODELS
    assert model_panel.BLENDS == FIXED_BLENDS
    assert tuple(model_panel.PANEL) == FIXED_BASE_MODELS + tuple(FIXED_BLENDS)
    for name in ("repeated_rank_stability_protocol.yaml",
                 "chemical_space_cv_protocol.yaml"):
        payload = _protocol(name)
        assert tuple(payload["panel"]["base_models"]) == FIXED_BASE_MODELS
        assert {key: tuple(value) for key, value in payload["panel"]["blends"].items()} \
            == FIXED_BLENDS
        assert payload["threshold"] == {"fixed": 0.5, "tuned": False}
        assert set(payload["metrics"]["primary"]) == PRIMARY_METRICS
        assert payload["metrics"]["lower_is_better"] == ["brier"]

    # A, D and E use a flat panel because they do not need to restate the blend
    # definitions, but the expanded ten-model order and threshold are identical.
    for name, threshold_path in (
        ("modelwide_druglikeness_bias_protocol.yaml", ("threshold", "fixed")),
        ("agextend_endpoint_benchmark_protocol.yaml", ("evaluation", "fixed_threshold")),
        ("drugage_celegans_benchmark_protocol.yaml", ("evaluation", "fixed_threshold")),
        ("kapsiani_historical_benchmark_protocol.yaml", ("evaluation", "fixed_threshold")),
    ):
        payload = _protocol(name)
        assert tuple(payload["panel"]) == FIXED_BASE_MODELS + tuple(FIXED_BLENDS)
        threshold = payload[threshold_path[0]][threshold_path[1]]
        assert threshold == 0.5


def test_b_and_c_protocols_lock_the_d1_train_only_boundary() -> None:
    for name in ("repeated_rank_stability_protocol.yaml",
                 "chemical_space_cv_protocol.yaml"):
        boundary = _protocol(name)["data_boundary"]
        assert boundary["rows"] == "d1_train_only_324"
        assert boundary["d1_test_labels_loaded"] is False
        assert boundary["drugage_outcomes_loaded"] is False
        assert boundary["agextend_outcomes_loaded"] is False


def test_tabfm_receives_the_same_explicit_component_seed_as_each_fold_pass(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The second TabFM pass must not fall back to ambient/global seed state."""

    class SpyPanel:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[str, ...], int, tuple[int, ...], tuple[int, ...]]] = []
            self.reloads = 0

        def reload_tabfm(self) -> None:
            self.reloads += 1

        def fold_probabilities(self, features, labels, fit, target, *, models,
                               component_seed, verbose):
            del features, labels, verbose
            self.calls.append((tuple(models), int(component_seed), tuple(fit), tuple(target)))
            return {model: np.full(len(target), component_seed / 1000.0)
                    for model in models}

    monkeypatch.setattr(model_panel, "BASE_MODELS", ("paper_svm", "tabfm"))
    panel = SpyPanel()
    folds = [
        (np.array([2, 3]), np.array([0, 1])),
        (np.array([0, 1]), np.array([2, 3])),
    ]
    result = model_panel.run_folds(
        panel, {}, np.array([0, 1, 0, 1]), folds, [71, 72], n_rows=4)

    assert panel.reloads == 1
    assert [call[1] for call in panel.calls] == [71, 72, 71, 72]
    assert [call[0] for call in panel.calls] == [
        ("paper_svm",), ("paper_svm",), ("tabfm",), ("tabfm",)]
    np.testing.assert_allclose(result["tabfm"], [0.071, 0.071, 0.072, 0.072])


def test_parent_standardization_precedes_identity_grouping_and_conflicts_are_removed() -> None:
    raw = pd.DataFrame([
        {"provider_record_id": "a", "compound_name": "ethanol-a",
         "source_smiles": "CCO", "label": 1},
        {"provider_record_id": "b", "compound_name": "ethanol-b",
         "source_smiles": "CCO.O", "label": 1},
        {"provider_record_id": "c", "compound_name": "ethylamine-a",
         "source_smiles": "CCN", "label": 0},
        {"provider_record_id": "d", "compound_name": "ethylamine-b",
         "source_smiles": "NCC", "label": 1},
    ])
    curated, ledger, conflicts = endpoint.curate(
        raw, cohort="synthetic", d1_connectivity=set())

    ethanol = curated[curated["standardized_parent_smiles"] == "CCO"].iloc[0]
    assert ethanol["collapsed_rows"] == 2
    assert json.loads(ethanol["provider_record_ids_json"]) == ["a", "b"]
    assert len(conflicts) == 1
    conflict_key = conflicts.iloc[0]["connectivity_inchikey"]
    assert conflict_key not in set(curated["connectivity_inchikey"])
    conflict_ledger = ledger[ledger["connectivity_inchikey"] == conflict_key]
    assert not conflict_ledger["included"].any()
    assert set(conflict_ledger["exclusion_reason"]) == {
        "conflicting_labels_within_connectivity_group"}


def test_group_assignment_keeps_each_group_in_one_fold() -> None:
    registry = pd.DataFrame({
        "row": np.arange(20),
        "group": np.repeat([f"g{i}" for i in range(10)], 2),
        "label": np.tile([0, 1], 10),
    })
    feasibility: dict[str, Any] = {}
    assigned = chemical._assign_folds(
        registry,
        {"n_splits": 2, "shuffle": True, "random_state": 7,
         "feasibility": {"min_positive_per_fold": 1,
                         "min_negative_per_fold": 1}},
        feasibility,
        "synthetic",
    )
    assert assigned.groupby("group")["fold"].nunique().max() == 1
    assert assigned.groupby("fold")["label"].nunique().min() == 2


def test_drugage_publication_infeasibility_is_reported_without_split_search() -> None:
    labels = np.array([1, 1, 1, 1, 0, 0])
    groups = np.array([0, 0, 1, 1, 2, 2])
    assignment, audit, status = drugage_finalize.publication_fold_feasibility(
        labels, groups, folds=3, seed=20260825)

    assert len(assignment) == len(labels)
    assert not status["feasible"]
    assert status["status"] == drugage_finalize.PUBLICATION_STATUS
    assert status["post_hoc_seed_or_fold_search_performed"] is False
    assert status["model_fitting_performed_for_grouped_analysis"] is False
    assert (~audit["estimable"]).any()
    assert pd.DataFrame({"group": groups, "fold": assignment}).groupby(
        "group")["fold"].nunique().max() == 1


def test_group_assignment_enforces_the_prespecified_minimum_class_counts(
        monkeypatch: pytest.MonkeyPatch) -> None:
    class FixedSplitter:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def split(self, rows, labels, groups):
            del rows, labels, groups
            yield np.array([3, 4, 5]), np.array([0, 1, 2])
            yield np.array([0, 1, 2]), np.array([3, 4, 5])

    monkeypatch.setattr(chemical, "StratifiedGroupKFold", FixedSplitter)
    registry = pd.DataFrame({
        "row": np.arange(6),
        "group": [f"g{i}" for i in range(6)],
        "label": [1, 0, 0, 1, 0, 0],
    })
    settings = {
        "n_splits": 2, "shuffle": True, "random_state": 7,
        "feasibility": {"min_positive_per_fold": 2, "min_negative_per_fold": 2},
    }
    with pytest.raises(chemical.ChemicalSpaceCVError, match="minimum|class"):
        chemical._assign_folds(registry, settings, {}, "synthetic")


def test_checkpoint_binding_rejects_contract_or_hash_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bound-work"
    contract = {"protocol_sha256": "a" * 64, "source_sha256": {"module": "b" * 64}}
    first = repeated.bind_checkpoint_directory(
        checkpoint, contract, error_cls=repeated.RankStabilityError)
    assert repeated.bind_checkpoint_directory(
        checkpoint, contract, error_cls=repeated.RankStabilityError) == first
    with pytest.raises(repeated.RankStabilityError, match="does not match"):
        repeated.bind_checkpoint_directory(
            checkpoint,
            {"protocol_sha256": "a" * 64,
             "source_sha256": {"module": "c" * 64}},
            error_cls=repeated.RankStabilityError,
        )

    unbound = tmp_path / "unbound-work"
    unbound.mkdir()
    with pytest.raises(repeated.RankStabilityError, match="Unbound"):
        repeated.bind_checkpoint_directory(
            unbound, contract, error_cls=repeated.RankStabilityError)


def test_b_and_c_bind_the_checkpoint_directory_before_resuming() -> None:
    for module_name in ("repeated_rank_stability.py", "chemical_space_cv.py"):
        source = (ROOT / "src" / "geroprotector" / module_name).read_text(
            encoding="utf-8")
        assert "bind_checkpoint_directory(" in source
        assert "protocol_sha" in source and "source_sha256" in source


def test_endpoint_cv_rejects_an_unbound_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "primary_oof.csv"
    pd.DataFrame({"fold": [0], "label": [0]}).to_csv(checkpoint, index=False)
    with pytest.raises(endpoint.EndpointBenchmarkError, match="binding|hash"):
        endpoint.cross_validate(
            object(), {}, np.array([0, 1]), n_splits=2, seed=42,
            checkpoint=checkpoint)


def test_endpoint_full_fit_scoring_uses_the_same_probability_column_contract() -> None:
    class FakePanel:
        def reload_tabfm(self) -> None:
            pass

        def fold_probabilities(self, features, labels, fit, target, *,
                               component_seed, verbose):
            del features, labels, fit, component_seed, verbose
            return {model: np.full(len(target), 0.25, dtype=float)
                    for model in endpoint.BASE_MODELS}

    fit = {"paper": np.zeros((2, 7)), "morgan": np.zeros((2, 8)),
           "rdkit2d": np.zeros((2, 4))}
    target = {"paper": np.zeros((1, 7)), "morgan": np.zeros((1, 8)),
              "rdkit2d": np.zeros((1, 4))}
    scored = endpoint.score_once(
        FakePanel(), fit, np.array([0, 1]), target, seed=42)
    assert {f"p_{model}" for model in endpoint.PANEL}.issubset(scored.columns)
    assert not (set(endpoint.PANEL) & set(scored.columns))


def test_descriptor_ineligibility_can_be_excluded_outcome_blindly(monkeypatch) -> None:
    curated = pd.DataFrame({
        "compound_id": ["eligible", "missing"],
        "compound_name": ["eligible", "missing"],
        "standardized_parent_smiles": ["CCO", "CCN"],
        "label": [1, 0],
    })
    descriptor_values = pd.DataFrame({
        "stable_id": curated["compound_id"],
        **{column: [1.0, np.nan] for column in endpoint.PAPER_COLUMNS},
    })
    monkeypatch.setattr(endpoint, "load_external_protocol", lambda root, path: ({}, "x"))
    monkeypatch.setattr(
        endpoint, "_datawarrior_descriptors",
        lambda root, protocol, rows, d1_parity: (descriptor_values.copy(), {"ok": True}))
    monkeypatch.setattr(
        endpoint, "_rdkit2d_from_smiles",
        lambda smiles, names: np.zeros((len(smiles), 3), dtype=float))
    monkeypatch.setattr(
        endpoint, "morgan_matrix",
        lambda smiles, generator: np.zeros((len(smiles), 8), dtype=np.uint8))

    features, audit = endpoint.build_features(
        ROOT, curated, verify_d1_parity=False,
        nonfinite_descriptor_policy="exclude")
    assert features["paper"].shape == (1, len(endpoint.PAPER_COLUMNS))
    assert features["morgan"].shape[0] == features["rdkit2d"].shape[0] == 1
    eligibility = audit["descriptor_eligibility"]
    assert eligibility["eligible_input_indices"] == [0]
    assert eligibility["excluded_input_indices"] == [1]
    assert eligibility["excluded_compound_ids"] == ["missing"]
    assert eligibility["rule_uses_endpoint_labels"] is False


def test_drugage_builds_connectivity_identity_before_label_aggregation() -> None:
    """Different names for one parent must become one endpoint unit before labels."""
    ledger = pd.DataFrame([
        {"name_key": "alias-a", "compound_name": "alias A",
         "canonical_smiles": "CCO", "structure_mapped": True,
         "significant_increase": True, "significant_decrease": False,
         "average_change": 10.0, "pubmed_id": "1"},
        {"name_key": "alias-b", "compound_name": "alias B",
         "canonical_smiles": "OCC", "structure_mapped": True,
         "significant_increase": False, "significant_decrease": True,
         "average_change": -5.0, "pubmed_id": "2"},
    ])
    endpoints = drugage.compound_endpoints(ledger)
    assert "connectivity_inchikey" in endpoints.columns
    assert endpoints["connectivity_inchikey"].is_unique
    assert len(endpoints) == 1
    assert endpoints.iloc[0]["conflict_status"] == "increase_and_decrease"


def _contains_call_inside_loop(tree: ast.AST, loop_token: str,
                               call_name: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.AsyncFor)):
            continue
        iterator_text = ast.unparse(node.iter)
        if loop_token not in iterator_text:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = child.func.id if isinstance(child.func, ast.Name) else \
                    child.func.attr if isinstance(child.func, ast.Attribute) else ""
                if name == call_name:
                    return True
    return False


def test_drugage_executes_repeated_variant_and_publication_grouped_protocols() -> None:
    """Audits must be actual refits, not labels rescored on the primary OOF stream."""
    source = (ROOT / "src/geroprotector/drugage_celegans_benchmark.py").read_text()
    tree = ast.parse(source)
    assert _contains_call_inside_loop(tree, "REPEATED", "cross_validate"), \
        "five repeats of five-fold CV are declared but not executed"
    assert _contains_call_inside_loop(tree, "LABEL_VARIANTS", "cross_validate"), \
        "endpoint variants must refit models, not only rescore primary predictions"
    calls = {
        (node.func.id if isinstance(node.func, ast.Name) else
         node.func.attr if isinstance(node.func, ast.Attribute) else "")
        for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    assert {"StratifiedGroupKFold", "GroupKFold"} & calls, \
        "publication-grouped CV is required; a publication-overlap audit is not enough"


REQUIRED_OUTPUTS = {
    "modelwide_druglikeness_bias.py": {
        "model_cohort_predictions.csv", "positive_only_qed_associations.csv",
        "qed_stratum_recall.csv", "qed_logistic_models.csv",
        "qed_adjusted_models.csv", "qed_trend_tests.csv",
        "basic_property_class_shifts.csv", "simple_baseline_metrics.csv",
        "label_permutation_controls.csv", "modelwide_bias_summary.csv",
        "modelwide_bias_summary.md", "RUN_MANIFEST.json", "COMPLETED.json",
    },
    "repeated_rank_stability.py": {
        "split_registry.csv", "per_row_repeated_oof_predictions.csv",
        "per_fold_metrics.csv", "per_repeat_pooled_metrics.csv",
        "rank_distribution.csv", "pairwise_win_matrix.csv",
        "compound_cluster_bootstrap_differences.csv", "seed42_parity_checks.csv",
        "rank_stability_summary.md", "RUN_MANIFEST.json", "COMPLETED.json",
    },
    "chemical_space_cv.py": {
        "scaffold_group_registry.csv", "similarity_component_registry.csv",
        "group_feasibility_audit.json", "per_row_grouped_oof_predictions.csv",
        "per_fold_grouped_metrics.csv", "pooled_grouped_metrics.csv",
        "random_vs_grouped_deltas.csv", "applicability_by_registry.csv",
        "chemical_space_cv_summary.md", "RUN_MANIFEST.json", "COMPLETED.json",
    },
    "agextend_endpoint_benchmark.py": {
        "SOURCE_RESOLUTION.json", "raw_to_curated_ledger.csv",
        "identity_conflicts.csv", "d1_overlap_ledger.csv",
        "agextend_split_registry.csv", "reference_reproduction_metrics.csv",
        "fixed_panel_oof_predictions.csv", "fixed_panel_fold_metrics.csv",
        "fixed_panel_pooled_metrics.csv", "challenge_predictions.csv",
        "challenge_metrics.csv", "agextend_qed_bias.csv",
        "agextend_endpoint_summary.md", "RUN_MANIFEST.json", "COMPLETED.json",
    },
    "drugage_celegans_benchmark.py": {
        "SOURCE_RESOLUTION.json", "historical_reproduction_status.md",
        "observation_level_ledger.csv", "compound_level_endpoint_ledger.csv",
        "identity_mapping_ledger.csv", "identity_conflicts.csv",
        "publication_group_audit.csv", "drugage_split_registry.csv",
        "reference_reproduction_metrics.csv", "fixed_panel_oof_predictions.csv",
        "fixed_panel_fold_metrics.csv", "fixed_panel_pooled_metrics.csv",
        "endpoint_sensitivity_metrics.csv", "drugage_qed_bias.csv",
        "drugage_celegans_summary.md", "RUN_MANIFEST.json", "COMPLETED.json",
    },
    "agextend_official_reference.py": {
        "SOURCE_RESOLUTION.json", "official_training_cohort.csv",
        "official_selected_features.csv", "published_tenfold_metrics.csv",
        "published_loocv_predictions.csv", "official_challenge_reference.csv",
        "official_challenge_inference_parity.csv",
        "official_challenge_parity_status.json",
        "agextend_official_reference_summary.md", "RUN_MANIFEST.json",
        "COMPLETED.json",
    },
    "kapsiani_historical_benchmark.py": {
        "SOURCE_RESOLUTION.json", "official_1430_source_cohort.csv",
        "official_69_selected_moe_features.csv", "raw_to_curated_ledger.csv",
        "identity_conflicts.csv", "d1_overlap_ledger.csv",
        "kapsiani_split_registry.csv", "fixed_panel_oof_predictions.csv",
        "fixed_panel_fold_metrics.csv", "fixed_panel_fold_mean_sd.csv",
        "fixed_panel_pooled_metrics.csv", "d1_nonoverlap_sensitivity_metrics.csv",
        "historical_rf_reproduction_status.csv",
        "kapsiani_historical_summary.md", "RUN_MANIFEST.json", "COMPLETED.json",
    },
    "insight_suite_report.py": {
        "all_experiment_metrics_long.csv", "all_experiment_model_ranks.csv",
        "all_qed_effects.csv", "claim_evidence_matrix.csv", "claims_allowed.json",
        "recommended_main_text_results.md", "recommended_supplementary_results.md",
        "recommended_title_and_claims.md", "logical_risk_register.md",
        "RUN_MANIFEST.json", "COMPLETED.json",
    },
}


@pytest.mark.parametrize("module_name, required", REQUIRED_OUTPUTS.items())
def test_stage_source_declares_the_required_output_contract(
        module_name: str, required: set[str]) -> None:
    source = (ROOT / "src" / "geroprotector" / module_name).read_text(encoding="utf-8")
    missing = sorted(name for name in required if name not in source)
    assert not missing, f"{module_name} omits required artifacts: {missing}"


def test_metric_math_and_fixed_threshold_on_known_arrays() -> None:
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.6, 0.4, 0.9])
    decision = (probability >= 0.5).astype(int)
    result = _metrics(y, probability, probability)

    assert PRIMARY_METRICS <= set(result)
    assert result["threshold"] == 0.5
    assert result["auprc_average_precision_positive"] == pytest.approx(
        average_precision_score(y, probability))
    assert result["auroc"] == pytest.approx(roc_auc_score(y, probability))
    assert result["accuracy"] == pytest.approx(accuracy_score(y, decision))
    assert result["mcc"] == pytest.approx(matthews_corrcoef(y, decision))
    assert result["macro_f1"] == pytest.approx(
        f1_score(y, decision, average="macro"))
    assert result["brier"] == pytest.approx(brier_score_loss(y, probability))
    assert result["cohen_kappa"] == pytest.approx(cohen_kappa_score(y, decision))
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (1, 1, 1, 1)


def test_brier_ranking_is_lower_is_better() -> None:
    frame = pd.DataFrame({
        "seed": [42, 42, 43, 43],
        "model_id": ["good", "bad", "good", "bad"],
        "brier": [0.10, 0.30, 0.20, 0.40],
    })
    assert repeated._ranks(frame, "brier").tolist() == [1.0, 2.0, 1.0, 2.0]


def test_cluster_bootstrap_draws_compounds_not_prediction_rows(
        monkeypatch: pytest.MonkeyPatch) -> None:
    class FixedRng:
        def integers(self, low, high, size):
            assert low == 0
            assert high == 4, "bootstrap population must be four compound identities"
            assert size == (101, 4), "each draw must contain four compound identities"
            return np.tile(np.arange(4), (101, 1))

    monkeypatch.setattr(repeated.np.random, "default_rng", lambda seed: FixedRng())
    monkeypatch.setattr(repeated, "PANEL", ("a", "b"))
    monkeypatch.setattr(repeated, "PRIMARY_METRICS", ("brier",))
    rows = []
    for seed in (42, 43):
        for compound, label in enumerate((0, 0, 1, 1)):
            rows.append({
                "seed": seed, "paper_row_index": compound, "label": label,
                "p_a": (0.1 if label == 0 else 0.9),
                "p_b": (0.4 if label == 0 else 0.6),
            })
    result = repeated._cluster_bootstrap(
        pd.DataFrame(rows), {"n_resamples": 101, "seed": 20260825})
    assert len(result) == 1
    assert result.iloc[0]["n_resamples"] == 101
    assert result.iloc[0]["unit"] == "compound_cluster_all_repeats_together"


def test_master_runner_verifies_hashes_before_skipping_completed_stages() -> None:
    source = (ROOT / "scripts" / "run_insight_suite.sh").read_text(encoding="utf-8")
    verification_tokens = (
        "verify_completed_run", "verify_artifact", "artifact_hashes", "sha256sum")
    assert any(token in source for token in verification_tokens), (
        "--resume currently trusts COMPLETED.json presence alone; it must validate "
        "the manifest and every recorded artifact hash before skipping")
