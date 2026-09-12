from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geroprotector.hashing import sha256_file
from geroprotector.insight_suite_report import (
    EVIDENCE_COLUMNS,
    EXPERIMENTS,
    InsightReportError,
    _load_protocol,
    _verify_run,
    adjudicate,
    resolve_and_verify_runs,
    run,
)


ROOT = Path(__file__).resolve().parents[1]


def _sealed_run(path: Path, experiment: str, stamp: str, schema: str,
                status: str = "COMPLETE") -> dict:
    path.mkdir(parents=True)
    run_id = f"{experiment}_{stamp}"
    manifest = path / "RUN_MANIFEST.json"
    manifest.write_text(json.dumps({"schema_version": schema, "run_id": run_id}))
    artifact = path / "table.csv"
    artifact.write_text("x\n1\n")
    completed = {
        "status": status, "run_id": run_id,
        "run_manifest_sha256": sha256_file(manifest),
        "artifact_hashes": {"RUN_MANIFEST.json": sha256_file(manifest),
                            "table.csv": sha256_file(artifact)},
    }
    (path / "COMPLETED.json").write_text(json.dumps(completed))
    return {"manifest_schema": schema, "allowed_completion_statuses": [status],
            "required_files": ["RUN_MANIFEST.json", "table.csv"]}


def test_report_protocol_keeps_all_fourteen_section_10_columns():
    protocol, digest = _load_protocol(ROOT / "configs/insight_suite_report_protocol.yaml")
    assert len(EVIDENCE_COLUMNS) == 14
    assert tuple(protocol["evidence_columns"]) == EVIDENCE_COLUMNS
    assert protocol["title_decision"]["endpoint_replication_required"] is True
    assert protocol["endpoint_transport_decision"]["arbitrary_absolute_auprc_cutoff"] is None
    assert len(digest) == 64


def test_completion_verification_rejects_changed_artifact(tmp_path):
    experiment, stamp = "modelwide_druglikeness_bias", "unit"
    run = tmp_path / f"{experiment}_{stamp}"
    spec = _sealed_run(run, experiment, stamp, "example.manifest.v1")
    assert _verify_run(run, experiment, stamp, spec)["artifact_count"] == 2
    (run / "table.csv").write_text("x\n2\n")
    with pytest.raises(InsightReportError, match="missing/changed artifact"):
        _verify_run(run, experiment, stamp, spec)


def test_exact_stamp_resolution_never_falls_back_to_latest(tmp_path):
    protocol, _ = _load_protocol(ROOT / "configs/insight_suite_report_protocol.yaml")
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    # A complete differently stamped run must not satisfy a requested stamp.
    experiment = "modelwide_druglikeness_bias"
    spec = protocol["experiments"][experiment]
    _sealed_run(outputs / f"{experiment}_other", experiment, "other",
                spec["manifest_schema"], spec["allowed_completion_statuses"][0])
    with pytest.raises(InsightReportError, match="Expected regular directory"):
        resolve_and_verify_runs(tmp_path, "wanted", protocol)


def _rank_rows() -> pd.DataFrame:
    metrics = ["auprc_average_precision_positive", "auroc", "accuracy", "mcc",
               "macro_f1", "brier", "cohen_kappa"]
    rows = []
    for seed in range(42, 52):
        for metric in metrics:
            winner = "paper_svm" if seed % 2 else "tabfm"
            for model in ("paper_svm", "tabfm"):
                rows.append({"experiment": "B", "analysis": "repeat_pooled_oof",
                             "seed": seed, "metric": metric, "model_id": model,
                             "rank": 1 if model == winner else 2})
    for registry, winner in (("random_seed42", "paper_svm"),
                             ("C1_scaffold", "tabfm"),
                             ("C2_similarity_component", "tabfm")):
        for metric in metrics:
            for model in ("paper_svm", "tabfm"):
                rows.append({"experiment": "C", "analysis": "registry_pooled_oof",
                             "registry": registry, "metric": metric, "model_id": model,
                             "rank": 1 if model == winner else 2})
    return pd.DataFrame(rows)


def _qed_rows(protocol: dict) -> pd.DataFrame:
    rows = []
    models = list(protocol["bias_decision"]["family_representatives"].values())
    for experiment, cohort in (("A", "d1_train_oof"), ("D", "agextend_oof"),
                               ("E", "drugage_celegans_oof")):
        for model in models:
            rows.append({"experiment": experiment, "cohort": cohort,
                         "model_id": model, "effect_kind": "spearman_rho",
                         "effect": -0.4, "ci_low": -0.55, "ci_high": -0.2,
                         "q_holm_within_cohort": 0.01})
    for kind in ("qed_logistic_unadjusted", "qed_logistic_similarity_adjusted"):
        for model in models:
            rows.append({"experiment": "A", "cohort": "d1_train_oof",
                         "model_id": model, "effect_kind": kind, "effect": -0.5,
                         "ci_low": 0.3, "ci_high": 0.9,
                         "q_holm_within_cohort": 0.01})
    return pd.DataFrame(rows)


def test_adjudication_requires_endpoint_replication_for_broad_title(tmp_path):
    protocol, _ = _load_protocol(ROOT / "configs/insight_suite_report_protocol.yaml")
    a, c, d, e = (tmp_path / name for name in ("A", "C", "D", "E"))
    a.mkdir(); c.mkdir(); d.mkdir(); e.mkdir()
    pd.DataFrame([
        {"baseline": "basic_properties", "auroc": 0.75},
        {"baseline": "prevalence_only", "auroc": 0.5},
    ]).to_csv(a / "simple_baseline_metrics.csv", index=False)
    pd.DataFrame({"auroc": np.linspace(0.4, 0.6, 100)}).to_csv(
        a / "label_permutation_controls.csv", index=False)
    deltas = []
    for registry in ("C1_scaffold", "C2_similarity_component"):
        for metric, cutoff in protocol["grouped_decision"]["practical_delta"].items():
            deltas.append({"registry": registry, "model_id": "paper_svm", "metric": metric,
                           "delta_grouped_minus_random": cutoff + 0.01})
    pd.DataFrame(deltas).to_csv(c / "random_vs_grouped_deltas.csv", index=False)
    pd.DataFrame({"label": [0, 1, 1, 1]}).to_csv(
        d / "fixed_panel_oof_predictions.csv", index=False)
    pd.DataFrame({"label": [0, 0, 1, 1, 1]}).to_csv(
        e / "fixed_panel_oof_predictions.csv", index=False)
    metric_rows = []
    for model in ("paper_svm", "tabfm"):
        metric_rows.append({"experiment": "B", "analysis": "repeat_pooled_oof",
                            "model_id": model, "metric": "auroc", "value": 0.7})
    for experiment, replication in (("D", "d1_nonoverlap_sensitivity"),
                                    ("E", "publication_grouped")):
        metric_rows += [
            {"experiment": experiment, "analysis": "endpoint_pooled_oof",
             "model_id": "gb4_equal", "metric": "auroc", "value": 0.7},
            {"experiment": experiment, "analysis": "endpoint_pooled_oof",
             "model_id": "gb4_equal", "metric": "auprc_average_precision_positive", "value": 0.65},
            {"experiment": experiment, "analysis": "endpoint_baselines",
             "model_id": "prevalence_only", "metric": "auprc_average_precision_positive", "value": 0.5},
            {"experiment": experiment, "analysis": replication,
             "model_id": "gb4_equal", "metric": "auroc", "value": 0.65},
        ]
    evidence, claims = adjudicate(pd.DataFrame(metric_rows), _rank_rows(),
                                  _qed_rows(protocol), {EXPERIMENTS[0]: a,
                                                       EXPERIMENTS[2]: c,
                                                       EXPERIMENTS[3]: d,
                                                       EXPERIMENTS[4]: e}, protocol)
    assert evidence.shape == (8, 14)
    assert tuple(evidence.columns) == EVIDENCE_COLUMNS
    assert claims["claim_8"]["verdict"] == "BROAD_TITLE_SUPPORTED"
    # Remove endpoint Holm support: D1 evidence alone must no longer support it.
    no_endpoint = _qed_rows(protocol)
    no_endpoint.loc[no_endpoint.experiment.isin(["D", "E"]), "q_holm_within_cohort"] = 0.5
    _, claims = adjudicate(pd.DataFrame(metric_rows), _rank_rows(), no_endpoint,
                           {EXPERIMENTS[0]: a, EXPERIMENTS[2]: c,
                            EXPERIMENTS[3]: d, EXPERIMENTS[4]: e}, protocol)
    assert claims["claim_8"]["verdict"] == "NARROW_TITLE_TO_D1_STYLE_BENCHMARK"


def test_endpoint_prevalence_uses_labels_not_prevalence_baseline_auprc(tmp_path):
    protocol, _ = _load_protocol(ROOT / "configs/insight_suite_report_protocol.yaml")
    a, c, d, e = (tmp_path / name for name in ("A", "C", "D", "E"))
    a.mkdir(); c.mkdir(); d.mkdir(); e.mkdir()
    pd.DataFrame([
        {"baseline": "basic_properties", "auroc": 0.75},
    ]).to_csv(a / "simple_baseline_metrics.csv", index=False)
    pd.DataFrame({"auroc": np.linspace(0.4, 0.6, 100)}).to_csv(
        a / "label_permutation_controls.csv", index=False)
    pd.DataFrame([
        {"registry": registry, "model_id": "paper_svm", "metric": metric,
         "delta_grouped_minus_random": 0.0}
        for registry in ("C1_scaffold", "C2_similarity_component")
        for metric in protocol["primary_metrics"]
    ]).to_csv(c / "random_vs_grouped_deltas.csv", index=False)
    pd.DataFrame({"label": [0, 1, 1, 1]}).to_csv(
        d / "fixed_panel_oof_predictions.csv", index=False)
    pd.DataFrame({"label": [0, 0, 1, 1, 1]}).to_csv(
        e / "fixed_panel_oof_predictions.csv", index=False)
    rows = []
    for experiment in ("D", "E"):
        rows.extend([
            {"experiment": experiment, "analysis": "endpoint_pooled_oof",
             "model_id": "gb4_equal", "metric": "auroc", "value": 0.7},
            {"experiment": experiment, "analysis": "endpoint_pooled_oof",
             "model_id": "gb4_equal", "metric": "auprc_average_precision_positive",
             "value": 0.8},
            # Deliberately wrong value: adjudication must not call this prevalence.
            {"experiment": experiment, "analysis": "endpoint_baselines",
             "model_id": "prevalence_only", "metric":
             "auprc_average_precision_positive", "value": 0.123},
        ])
    _, claims = adjudicate(pd.DataFrame(rows), _rank_rows(), _qed_rows(protocol),
                           {EXPERIMENTS[0]: a, EXPERIMENTS[2]: c,
                            EXPERIMENTS[3]: d, EXPERIMENTS[4]: e}, protocol)
    assert claims["claim_6"]["prevalence"] == pytest.approx(0.75)
    assert claims["claim_7"]["prevalence"] == pytest.approx(0.60)
    assert claims["claim_7"]["sensitivity_replication_available"] is False
    _, claims = adjudicate(pd.DataFrame(rows), _rank_rows(), _qed_rows(protocol),
                           {EXPERIMENTS[0]: a, EXPERIMENTS[2]: c,
                            EXPERIMENTS[3]: d, EXPERIMENTS[4]: e}, protocol)
    # No publication-grouped rows means infeasible/unavailable, not below chance.
    evidence, _ = adjudicate(pd.DataFrame(rows), _rank_rows(), _qed_rows(protocol),
                             {EXPERIMENTS[0]: a, EXPERIMENTS[2]: c,
                              EXPERIMENTS[3]: d, EXPERIMENTS[4]: e}, protocol)
    replication = evidence.loc[evidence.claim_id == "claim_7", "replication_evidence"].iloc[0]
    assert "not estimable" in replication
    assert "above chance=False" not in replication


def test_run_writes_real_long_tables_and_refuses_overwrite(tmp_path, monkeypatch):
    protocol_path = ROOT / "configs/insight_suite_report_protocol.yaml"
    protocol, _ = _load_protocol(protocol_path)
    fake_runs = {name: tmp_path / name for name in protocol["experiments"]}
    verification = {name: {"run_id": f"{name}_unit", "status": "COMPLETE",
                           "completed_sha256": "a" * 64,
                           "run_manifest_sha256": "b" * 64, "artifact_count": 2}
                    for name in fake_runs}
    metric = pd.DataFrame([{"experiment": "B", "analysis": "repeat_pooled_oof",
                            "model_id": "paper_svm", "metric": "auroc", "value": 0.6}])
    rank = pd.DataFrame([{"experiment": "B", "analysis": "repeat_pooled_oof",
                          "model_id": "paper_svm", "metric": "auroc", "rank": 1}])
    qed = pd.DataFrame([{"experiment": "A", "cohort": "d1_train_oof",
                         "model_id": "paper_svm", "effect_kind": "spearman_rho",
                         "effect": -0.2}])
    evidence = pd.DataFrame([
        dict(zip(EVIDENCE_COLUMNS, [f"claim_{i}", "text", "primary", "replication",
                                    "none", "cohort", "models", "effect", "CI", "Holm",
                                    "limits", "allowed", "prohibited", "SUPPORTED"]))
        for i in range(1, 9)
    ])
    claims = {f"claim_{i}": {"verdict": "SUPPORTED"} for i in range(1, 9)}
    claims["claim_8"]["recommended_title"] = "Neutral synthetic title"
    monkeypatch.setattr("geroprotector.insight_suite_report.resolve_and_verify_runs",
                        lambda root, stamp, p: (fake_runs, verification))
    monkeypatch.setattr("geroprotector.insight_suite_report.collect_metrics",
                        lambda runs, p: metric)
    monkeypatch.setattr("geroprotector.insight_suite_report.collect_ranks",
                        lambda runs, p: rank)
    monkeypatch.setattr("geroprotector.insight_suite_report.collect_qed_effects",
                        lambda runs: qed)
    monkeypatch.setattr("geroprotector.insight_suite_report.adjudicate",
                        lambda metrics, ranks, effects, runs, p: (evidence, claims))
    destination = run(root=tmp_path, config_path=protocol_path,
                      run_id="insight_suite_report_unit", stamp="unit")
    assert len(pd.read_csv(destination / "all_experiment_metrics_long.csv")) == 1
    assert len(pd.read_csv(destination / "all_experiment_model_ranks.csv")) == 1
    assert len(pd.read_csv(destination / "all_qed_effects.csv")) == 1
    assert pd.read_csv(destination / "claim_evidence_matrix.csv").shape == (8, 14)
    with pytest.raises(InsightReportError, match="Run exists"):
        run(root=tmp_path, config_path=protocol_path,
            run_id="insight_suite_report_unit", stamp="unit")
