"""Sealed, prespecified adjudication of Experiments A through E.

One explicit stamp is required.  The exact five stamped runs must be complete and
every recorded artifact hash must verify.  No "latest" fallback, partial report,
model refit, or manuscript edit is permitted.  Decisions follow the versioned YAML
protocol and ``prespecified insight-analysis protocol`` section 10.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.nine_ml_featuresets import _metrics

SCHEMA = "geroprotector.insight_suite_report"
PROTOCOL_SCHEMA = f"{SCHEMA}.protocol.v1"
EXPERIMENTS = (
    "modelwide_druglikeness_bias", "repeated_rank_stability", "chemical_space_cv",
    "agextend_endpoint_benchmark", "drugage_celegans_benchmark",
)
EVIDENCE_COLUMNS = (
    "claim_id", "claim_text", "primary_evidence", "replication_evidence",
    "contradictory_evidence", "cohorts", "models", "effect_size",
    "confidence_interval", "multiplicity_status", "limitations", "allowed_wording",
    "prohibited_wording", "verdict",
)
LOWER_IS_BETTER = {"brier"}


class InsightReportError(RuntimeError):
    """An integrated-evidence contract failed."""


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise InsightReportError(f"Protocol is not a regular file: {path}")
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise InsightReportError("Unknown report protocol schema")
    if tuple(protocol.get("experiments", {})) != EXPERIMENTS:
        raise InsightReportError("Protocol must declare exact A-E experiment order")
    if tuple(protocol.get("evidence_columns", ())) != EVIDENCE_COLUMNS:
        raise InsightReportError("Evidence columns differ from section 10.1")
    required = {"auprc_average_precision_positive", "auroc", "accuracy", "mcc",
                "macro_f1", "brier", "cohen_kappa"}
    if not required.issubset(protocol.get("metrics", ())):
        raise InsightReportError("Protocol omits required metrics")
    if protocol.get("title_decision", {}).get("endpoint_replication_required") is not True:
        raise InsightReportError("Broad-title rule must require endpoint replication")
    return protocol, sha256_file(path)


def _verify_run(run: Path, experiment: str, stamp: str,
                specification: dict[str, Any]) -> dict[str, Any]:
    run_id = f"{experiment}_{stamp}"
    if run.name != run_id or run.is_symlink() or not run.is_dir():
        raise InsightReportError(f"Expected regular directory {run_id}")
    completed_path, manifest_path = run / "COMPLETED.json", run / "RUN_MANIFEST.json"
    if any(p.is_symlink() or not p.is_file() for p in (completed_path, manifest_path)):
        raise InsightReportError(f"{run_id}: completion lock or manifest absent/unsafe")
    try:
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise InsightReportError(f"{run_id}: invalid completion JSON") from exc
    allowed = tuple(map(str, specification["allowed_completion_statuses"]))
    if completed.get("status") not in allowed:
        raise InsightReportError(f"{run_id}: status {completed.get('status')!r} not in {allowed}")
    if completed.get("run_id") != run_id or manifest.get("run_id") != run_id:
        raise InsightReportError(f"{run_id}: mixed run IDs")
    if manifest.get("schema_version") != specification["manifest_schema"]:
        raise InsightReportError(f"{run_id}: unexpected manifest schema")
    manifest_hash = sha256_file(manifest_path)
    if completed.get("run_manifest_sha256") != manifest_hash:
        raise InsightReportError(f"{run_id}: manifest hash mismatch")
    hashes = completed.get("artifact_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise InsightReportError(f"{run_id}: artifact hash registry absent")
    for relative, expected in hashes.items():
        target = run / str(relative)
        try:
            target.resolve().relative_to(run.resolve())
        except ValueError as exc:
            raise InsightReportError(f"{run_id}: unsafe artifact path {relative}") from exc
        if target.is_symlink() or not target.is_file() or sha256_file(target) != expected:
            raise InsightReportError(f"{run_id}: missing/changed artifact {relative}")
    required = tuple(map(str, specification["required_files"]))
    missing = [name for name in required if not (run / name).is_file()]
    unbound = [name for name in required if name != "RUN_MANIFEST.json" and name not in hashes]
    if missing or unbound:
        raise InsightReportError(f"{run_id}: missing={missing}; not hash-bound={unbound}")
    return {"run_id": run_id, "status": completed["status"],
            "completed_sha256": sha256_file(completed_path),
            "run_manifest_sha256": manifest_hash, "artifact_count": len(hashes)}


def resolve_and_verify_runs(root: Path, stamp: str,
                            protocol: dict[str, Any]) -> tuple[dict[str, Path], dict[str, Any]]:
    if not re.fullmatch(r"[a-z0-9_.-]+", stamp or ""):
        raise InsightReportError("Explicit safe stamp required; latest fallback forbidden")
    runs, verification = {}, {}
    for experiment in EXPERIMENTS:
        run = root / "outputs" / f"{experiment}_{stamp}"
        verification[experiment] = _verify_run(
            run, experiment, stamp, protocol["experiments"][experiment])
        runs[experiment] = run
    return runs, verification


def _read(run: Path, name: str, required: Iterable[str] = ()) -> pd.DataFrame:
    frame = pd.read_csv(run / name)
    missing = sorted(set(required) - set(frame))
    if missing:
        raise InsightReportError(f"{run.name}/{name}: missing columns {missing}")
    return frame


def _metric_rows(frame: pd.DataFrame, experiment: str, analysis: str,
                 source_file: str, protocol: dict[str, Any], **metadata) -> list[dict]:
    rows = []
    id_columns = [c for c in ("cohort", "registry", "endpoint_variant", "repeat",
                              "repeat_seed", "seed", "fold", "k") if c in frame]
    for record in frame.to_dict(orient="records"):
        model = str(record.get("model_id", record.get("baseline", "")))
        for metric in protocol["metrics"]:
            source = metric
            if metric == "auprc_average_precision_positive" and source not in record:
                source = "auprc"
            if metric == "accuracy" and source not in record:
                source = "accuracy_at_0.5"
            value = pd.to_numeric(record.get(source), errors="coerce")
            if not np.isfinite(value):
                continue
            row = {"experiment": experiment, "analysis": analysis, "model_id": model,
                   "metric": metric, "value": float(value),
                   "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
                   "source_file": source_file}
            row.update({c: record.get(c) for c in id_columns}); row.update(metadata)
            rows.append(row)
    return rows


def _prediction_metrics(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    rows = []
    for column in [c for c in frame if c.startswith("p_")]:
        p = frame[column].to_numpy(float)
        rows.append({"cohort": cohort, "model_id": column[2:],
                     **_metrics(frame.label.to_numpy(int), p, p)})
    return pd.DataFrame(rows)


def collect_metrics(runs: dict[str, Path], protocol: dict[str, Any]) -> pd.DataFrame:
    rows = []
    a = _read(runs[EXPERIMENTS[0]], "model_cohort_predictions.csv", ("cohort", "label"))
    for cohort, block in a.groupby("cohort", sort=False):
        rows += _metric_rows(_prediction_metrics(block, str(cohort)), "A",
                             "model_cohort_predictions", "model_cohort_predictions.csv", protocol)
    rows += _metric_rows(_read(runs[EXPERIMENTS[0]], "simple_baseline_metrics.csv", ("baseline",)),
                         "A", "dataset_construction_baselines", "simple_baseline_metrics.csv",
                         protocol, cohort="d1_train_oof")
    b = runs[EXPERIMENTS[1]]
    rows += _metric_rows(_read(b, "per_repeat_pooled_metrics.csv", ("seed", "model_id")),
                         "B", "repeat_pooled_oof", "per_repeat_pooled_metrics.csv", protocol)
    rows += _metric_rows(_read(b, "per_fold_metrics.csv", ("seed", "fold", "model_id")),
                         "B", "repeat_fold", "per_fold_metrics.csv", protocol)
    c = runs[EXPERIMENTS[2]]
    rows += _metric_rows(_read(c, "pooled_grouped_metrics.csv", ("registry", "model_id")),
                         "C", "registry_pooled_oof", "pooled_grouped_metrics.csv", protocol)
    rows += _metric_rows(_read(c, "per_fold_grouped_metrics.csv", ("registry", "fold", "model_id")),
                         "C", "registry_fold", "per_fold_grouped_metrics.csv", protocol)
    d = runs[EXPERIMENTS[3]]
    for filename, analysis, meta in (
        ("fixed_panel_pooled_metrics.csv", "endpoint_pooled_oof", {"cohort": "agextend_oof"}),
        ("fixed_panel_fold_metrics.csv", "endpoint_fold_oof", {"cohort": "agextend_oof"}),
        ("d1_nonoverlap_sensitivity_metrics.csv", "d1_nonoverlap_sensitivity", {}),
        ("challenge_metrics.csv", "retrospective_challenge", {}),
        ("agextend_simple_baseline_metrics.csv", "endpoint_baselines", {}),
    ):
        rows += _metric_rows(_read(d, filename), "D", analysis, filename, protocol, **meta)
    e = runs[EXPERIMENTS[4]]
    for filename, analysis, meta in (
        ("fixed_panel_pooled_metrics.csv", "endpoint_pooled_oof", {"cohort": "drugage_celegans_oof"}),
        ("fixed_panel_fold_metrics.csv", "endpoint_fold_oof", {"cohort": "drugage_celegans_oof"}),
        ("endpoint_sensitivity_metrics.csv", "endpoint_variant_sensitivity", {}),
        ("repeated5x5_metrics.csv", "repeated_5x5", {}),
        ("publication_grouped_metrics.csv", "publication_grouped", {}),
        ("drugage_simple_baseline_metrics.csv", "endpoint_baselines", {}),
    ):
        rows += _metric_rows(_read(e, filename), "E", analysis, filename, protocol, **meta)
    result = pd.DataFrame(rows)
    if result.empty: raise InsightReportError("No metric rows produced")
    columns = ["experiment", "analysis", "cohort", "registry", "endpoint_variant",
               "repeat", "repeat_seed", "seed", "fold", "k", "model_id", "metric", "value",
               "direction", "source_file"]
    for c in columns:
        if c not in result: result[c] = np.nan
    return result[columns].sort_values(["experiment", "analysis", "model_id", "metric",
                                        "seed", "fold"], kind="stable", na_position="last")


def _rank_frame(frame: pd.DataFrame, experiment: str, analysis: str,
                regime: tuple[str, ...], protocol: dict[str, Any], source: str) -> pd.DataFrame:
    rows = []
    for keys, block in frame.groupby(list(regime), dropna=False, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,); meta = dict(zip(regime, keys))
        for metric in protocol["primary_metrics"]:
            if metric not in block: continue
            ranked = block[["model_id", metric]].copy()
            ranked["rank"] = ranked[metric].rank(ascending=metric in LOWER_IS_BETTER, method="min")
            for r in ranked.to_dict(orient="records"):
                rows.append({"experiment": experiment, "analysis": analysis, **meta,
                             "metric": metric, "model_id": r["model_id"],
                             "value": r[metric], "rank": r["rank"], "source_file": source})
    return pd.DataFrame(rows)


def collect_ranks(runs: dict[str, Path], protocol: dict[str, Any]) -> pd.DataFrame:
    frames = []
    a = _read(runs[EXPERIMENTS[0]], "model_cohort_predictions.csv", ("cohort", "label"))
    a_metrics = pd.concat(
        [_prediction_metrics(block, str(cohort))
         for cohort, block in a.groupby("cohort", sort=False)], ignore_index=True)
    frames.append(_rank_frame(a_metrics, "A", "model_cohort_predictions", ("cohort",),
                              protocol, "model_cohort_predictions.csv"))
    b = _read(runs[EXPERIMENTS[1]], "per_repeat_pooled_metrics.csv", ("seed", "model_id"))
    frames.append(_rank_frame(b, "B", "repeat_pooled_oof", ("seed",), protocol,
                              "per_repeat_pooled_metrics.csv"))
    c = _read(runs[EXPERIMENTS[2]], "pooled_grouped_metrics.csv", ("registry", "model_id"))
    frames.append(_rank_frame(c, "C", "registry_pooled_oof", ("registry",), protocol,
                              "pooled_grouped_metrics.csv"))
    for exp, run, filename, analysis, cohort in (
        ("D", runs[EXPERIMENTS[3]], "fixed_panel_pooled_metrics.csv", "endpoint_pooled_oof", "agextend_oof"),
        ("D", runs[EXPERIMENTS[3]], "challenge_metrics.csv", "retrospective_challenge", "agextend_challenge"),
        ("E", runs[EXPERIMENTS[4]], "fixed_panel_pooled_metrics.csv", "endpoint_pooled_oof", "drugage_celegans_oof"),
        ("E", runs[EXPERIMENTS[4]], "publication_grouped_metrics.csv", "publication_grouped", "drugage_celegans_publication_grouped"),
    ):
        frame = _read(run, filename, ("model_id",)).copy()
        if "cohort" not in frame: frame["cohort"] = cohort
        frames.append(_rank_frame(frame, exp, analysis, ("cohort",), protocol, filename))
    # Preserve endpoint sensitivity/stability ranks rather than reducing D/E to
    # their primary pooled table. These remain descriptive and cannot select models.
    for exp, run, filename, analysis, regime, fallback in (
        ("D", runs[EXPERIMENTS[3]], "d1_nonoverlap_sensitivity_metrics.csv",
         "d1_nonoverlap_sensitivity", "cohort", "agextend_oof_no_d1_overlap"),
        ("E", runs[EXPERIMENTS[4]], "endpoint_sensitivity_metrics.csv",
         "endpoint_variant_sensitivity", "endpoint_variant", "primary"),
        ("E", runs[EXPERIMENTS[4]], "repeated5x5_metrics.csv",
         "repeated_5x5", "repeat_seed", "pooled"),
    ):
        frame = _read(run, filename, ("model_id",)).copy()
        if regime not in frame:
            frame[regime] = fallback
        frames.append(_rank_frame(frame, exp, analysis, (regime,), protocol, filename))
    result = pd.concat(frames, ignore_index=True, sort=False)
    columns = ["experiment", "analysis", "cohort", "registry", "endpoint_variant",
               "repeat_seed", "seed", "metric",
               "model_id", "value", "rank", "source_file"]
    for c in columns:
        if c not in result: result[c] = np.nan
    if result.empty: raise InsightReportError("No rank rows produced")
    return result[columns].sort_values(["experiment", "analysis", "metric", "cohort",
                                        "registry", "seed", "rank"], kind="stable", na_position="last")


def _effects(run: Path, experiment: str, files: dict[str, tuple]) -> list[dict]:
    rows = []
    for filename, (kind, effect, low, high) in files.items():
        frame = _read(run, filename, ("cohort", "model_id"))
        if "term" in frame: frame = frame[frame.term == "standardized_qed"]
        for r in frame.to_dict(orient="records"):
            # Logistic effects stay entirely on the log-odds (coefficient) scale.
            # The source files' ``or_ci_*`` columns are on the odds-ratio scale and
            # must never be paired with a coefficient in the integrated table.
            if kind.startswith("qed_logistic"):
                beta = pd.to_numeric(r.get("coefficient"), errors="coerce")
                se = pd.to_numeric(r.get("standard_error"), errors="coerce")
                ci_low = float(beta - 1.959963984540054 * se) if np.isfinite(beta) and np.isfinite(se) else np.nan
                ci_high = float(beta + 1.959963984540054 * se) if np.isfinite(beta) and np.isfinite(se) else np.nan
            else:
                ci_low = r.get(low, np.nan) if low else np.nan
                ci_high = r.get(high, np.nan) if high else np.nan
            rows.append({"experiment": experiment, "cohort": r["cohort"],
                         "model_id": r["model_id"], "effect_kind": kind,
                         "effect": r.get(effect, np.nan),
                         "ci_low": ci_low, "ci_high": ci_high,
                         "p_value": r.get("p_value", np.nan),
                         "q_holm_within_cohort": r.get("q_holm_within_cohort", np.nan),
                         "n": r.get("n_positive", r.get("n", np.nan)),
                         "status": r.get("status", ""), "source_file": filename})
    return rows


def collect_qed_effects(runs: dict[str, Path]) -> pd.DataFrame:
    kinds = {
        "bias": ("spearman_rho", "spearman_rho", "rho_ci_low", "rho_ci_high"),
        "trend": ("cochran_armitage_trend", "z_statistic", None, None),
        "logistic": ("qed_logistic_unadjusted", "coefficient", "or_ci_low", "or_ci_high"),
        "adjusted": ("qed_logistic_similarity_adjusted", "coefficient", "or_ci_low", "or_ci_high")}
    rows = _effects(runs[EXPERIMENTS[0]], "A", {
        "positive_only_qed_associations.csv": kinds["bias"],
        "qed_trend_tests.csv": kinds["trend"], "qed_logistic_models.csv": kinds["logistic"],
        "qed_adjusted_models.csv": kinds["adjusted"]})
    for exp, run, prefix in (("D", runs[EXPERIMENTS[3]], "agextend"),
                             ("E", runs[EXPERIMENTS[4]], "drugage")):
        rows += _effects(run, exp, {
            f"{prefix}_qed_bias.csv": kinds["bias"],
            f"{prefix}_qed_trend_tests.csv": kinds["trend"],
            f"{prefix}_qed_logistic_models.csv": kinds["logistic"],
            f"{prefix}_qed_adjusted_models.csv": kinds["adjusted"]})
    result = pd.DataFrame(rows)
    if result.empty: raise InsightReportError("No QED effects produced")
    for c in ("effect", "ci_low", "ci_high", "p_value", "q_holm_within_cohort", "n"):
        result[c] = pd.to_numeric(result[c], errors="coerce")
    return result.sort_values(["experiment", "cohort", "effect_kind", "model_id"], kind="stable")


def _family_support(qed: pd.DataFrame, cohort: str, protocol: dict[str, Any]) -> dict:
    cfg = protocol["bias_decision"]; alpha = float(cfg["holm_alpha"])
    block = qed[(qed.cohort == cohort) & (qed.effect_kind == "spearman_rho")]
    evidence = []
    for family, model in cfg["family_representatives"].items():
        found = block[block.model_id == model]
        if found.empty:
            evidence.append({"family": family, "model_id": model, "present": False,
                             "negative": False, "holm_supported": False}); continue
        r = found.iloc[0]; negative = bool(r.effect < 0)
        supported = bool(negative and r.ci_high < 0 and r.q_holm_within_cohort < alpha)
        evidence.append({"family": family, "model_id": model, "present": True,
                         "effect": float(r.effect), "ci_low": float(r.ci_low),
                         "ci_high": float(r.ci_high), "q_holm": float(r.q_holm_within_cohort),
                         "negative": negative, "holm_supported": supported})
    return {"cohort": cohort, "families": evidence,
            "negative_families": sum(v["negative"] for v in evidence),
            "holm_supported_families": sum(v["holm_supported"] for v in evidence),
            "required_families": int(cfg["min_distinct_families"])}


def _claim(claim_id, text, verdict, primary, replication, contradictory, cohorts,
           models, effect, ci, multiplicity, limitations, allowed, prohibited) -> dict:
    return dict(zip(EVIDENCE_COLUMNS, (claim_id, text, primary, replication, contradictory,
        cohorts, models, effect, ci, multiplicity, limitations, allowed, prohibited, verdict)))


def adjudicate(metrics: pd.DataFrame, ranks: pd.DataFrame, qed: pd.DataFrame,
               runs: dict[str, Path], protocol: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    claims, rows = {}, []
    rcfg = protocol["ranking_decision"]
    b = ranks[(ranks.experiment == "B") & (ranks.analysis == "repeat_pooled_oof")]
    unstable, shares = [], {}
    for metric, block in b.groupby("metric"):
        winners = block[block["rank"] == 1].groupby("seed").model_id.first()
        share = max(Counter(winners).values()) / len(winners); shares[metric] = share
        if 1 - share >= float(rcfg["min_repeat_top_change_fraction"]): unstable.append(metric)
    c = ranks[(ranks.experiment == "C") & (ranks.analysis == "registry_pooled_oof")]
    random, grouped = c[c.registry == rcfg["random_registry"]], c[c.registry != rcfg["random_registry"]]
    changed = comparisons = 0
    for (registry, metric), block in grouped.groupby(["registry", "metric"]):
        ref = random[random.metric == metric]
        if len(ref):
            comparisons += 1
            changed += int(set(block.loc[block["rank"] == 1, "model_id"]) !=
                           set(ref.loc[ref["rank"] == 1, "model_id"]))
    repeated = len(unstable) >= int(rcfg["min_metrics_with_repeat_swapping"])
    grouped_changed = comparisons and changed/comparisons >= float(rcfg["min_grouped_top_change_fraction"])
    verdict = ("PARTITION_AND_EVALUATION_REGIME_SENSITIVE" if repeated and grouped_changed else
               "REPEATED_PARTITION_SENSITIVE_ONLY" if repeated else "REPEATED_RANKS_STABLE")
    claims["claim_1"] = {"verdict": verdict, "unstable_metrics": unstable,
                         "modal_winner_shares": shares, "grouped_changes": [changed, comparisons]}
    rank_wording = {
        "PARTITION_AND_EVALUATION_REGIME_SENSITIVE":
            "Model ordering was sensitive to both repeated partitions and chemistry-aware validation regimes.",
        "REPEATED_PARTITION_SENSITIVE_ONLY":
            "Model ordering was sensitive to repeated random partitions but not materially changed by chemistry-aware grouping.",
        "REPEATED_RANKS_STABLE":
            "Model ordering was stable under the prespecified repeated-partition criterion.",
    }[verdict]
    rows.append(_claim("claim_1", "Model-family rankings are partition-sensitive.", verdict,
        f"{len(unstable)}/{len(rcfg['metrics'])} metrics met the prespecified repeat-swapping rule.",
        f"Grouped CV changed top model in {changed}/{comparisons} registry-metric comparisons.",
        "Stable metrics: " + ", ".join(sorted(set(rcfg["metrics"])-set(unstable))),
        "D1 train (324)", "fixed ten-model panel", json.dumps(shares, sort_keys=True),
        "Compound-cluster CIs are retained in Experiment B; folds are not independent.",
        "Prespecified descriptive rank rules; no fold-level p values.",
        "Repeated folds reuse compounds and model ranks are correlated.",
        rank_wording,
        "One split proves that one model family is superior."))

    bcfg = protocol["bias_decision"]
    d1 = _family_support(qed, bcfg["d1_primary_cohort"], protocol)
    endpoint = {name: _family_support(qed, cohort, protocol)
                for name, cohort in bcfg["endpoint_primary_cohorts"].items()}
    d1_multi = d1["holm_supported_families"] >= d1["required_families"]
    endpoint_rep = any(v["holm_supported_families"] >= v["required_families"] for v in endpoint.values())
    directional = d1["negative_families"] >= d1["required_families"]
    bias_verdict = ("MULTI_FAMILY_D1_BIAS_REPLICATED_ENDPOINT_ALIGNED" if d1_multi and endpoint_rep else
                    "MULTI_FAMILY_D1_ONLY" if d1_multi else
                    "HETEROGENEOUS_DIRECTIONAL_EVIDENCE" if directional else "BROAD_BIAS_NOT_SUPPORTED")
    claims["claim_2"] = {"verdict": bias_verdict, "d1": d1, "endpoint": endpoint,
                         "endpoint_replication": bool(endpoint_rep)}
    bias_wording = {
        "MULTI_FAMILY_D1_BIAS_REPLICATED_ENDPOINT_ALIGNED":
            "Reduced sensitivity for high-QED positives was observed across multiple model families and replicated in endpoint-aligned benchmarks.",
        "MULTI_FAMILY_D1_ONLY":
            "Reduced sensitivity for high-QED positives was observed across multiple model families on D1 but was not replicated endpoint-aligned.",
        "HETEROGENEOUS_DIRECTIONAL_EVIDENCE":
            "The evaluated classifiers showed heterogeneous directional sensitivity across QED strata.",
        "BROAD_BIAS_NOT_SUPPORTED":
            "The prespecified evidence did not support a broad multi-family QED-bias claim.",
    }[bias_verdict]
    rows.append(_claim("claim_2", "Structure-based classifiers show reduced sensitivity for high-QED positives.",
        bias_verdict,
        f"D1 OOF: {d1['holm_supported_families']}/{len(d1['families'])} prespecified families met negative rho, CI<0, Holm q<{bcfg['holm_alpha']}.",
        "; ".join(f"{k}: {v['holm_supported_families']}/{len(v['families'])} families" for k,v in endpoint.items()),
        "All nonsupporting families remain in all_qed_effects.csv.",
        bcfg["d1_primary_cohort"]+"; "+"; ".join(v["cohort"] for v in endpoint.values()),
        ", ".join(f"{k}={v}" for k,v in bcfg["family_representatives"].items()),
        "; ".join(f"{v['model_id']} rho={v.get('effect',np.nan):.3f}" for v in d1["families"] if v["present"]),
        "Bootstrap 95% CIs per effect.", f"Holm within cohort/family, alpha={bcfg['holm_alpha']}.",
        "Observational endpoint-defined associations do not establish QED causality.",
        bias_wording,
        "QED causally reduces geroprotective activity or clinical efficacy."))

    alpha=float(bcfg["holm_alpha"]); family_models=set(bcfg["family_representatives"].values())
    u=qed[(qed.experiment=="A")&(qed.cohort==bcfg["d1_primary_cohort"])&(qed.effect_kind=="qed_logistic_unadjusted")]
    a=qed[(qed.experiment=="A")&(qed.cohort==bcfg["d1_primary_cohort"])&(qed.effect_kind=="qed_logistic_similarity_adjusted")]
    merged=u[u.model_id.isin(family_models)].merge(a[a.model_id.isin(family_models)], on=["cohort","model_id"], suffixes=("_u","_a"))
    before=merged[(merged.effect_u<0)&(merged.q_holm_within_cohort_u<alpha)]
    after=before[(before.effect_a<0)&(before.q_holm_within_cohort_a<alpha)]
    sim_verdict=("UNRESOLVED_NO_HOLM_SUPPORTED_UNADJUSTED_EFFECT" if not len(before) else
                 "ASSOCIATION_NOT_RETAINED_AFTER_MEASURED_SIMILARITY_ADJUSTMENT" if not len(after) else
                 "ASSOCIATION_RETAINED_AFTER_MEASURED_SIMILARITY_ADJUSTMENT" if len(after)==len(before) else
                 "MIXED_ATTENUATION_AFTER_MEASURED_SIMILARITY_ADJUSTMENT")
    claims["claim_3"]={"verdict":sim_verdict,"supported_before":len(before),"supported_after":len(after)}
    similarity_wording = {
        "UNRESOLVED_NO_HOLM_SUPPORTED_UNADJUSTED_EFFECT":
            "The similarity-adjustment question remained unresolved because no prespecified unadjusted effect met the Holm criterion.",
        "ASSOCIATION_NOT_RETAINED_AFTER_MEASURED_SIMILARITY_ADJUSTMENT":
            "The Holm-supported QED associations were not retained after adjustment for the measured similarity proxy.",
        "ASSOCIATION_RETAINED_AFTER_MEASURED_SIMILARITY_ADJUSTMENT":
            "All Holm-supported QED associations were retained after adjustment for the measured similarity proxy.",
        "MIXED_ATTENUATION_AFTER_MEASURED_SIMILARITY_ADJUSTMENT":
            "Similarity adjustment attenuated some, but not all, Holm-supported QED associations.",
    }[sim_verdict]
    rows.append(_claim("claim_3", "The QED association is or is not explained by similarity to D1 train chemistry.", sim_verdict,
        f"{len(after)}/{len(before)} Holm-supported negative effects remained after adjustment.",
        "Endpoint adjusted effects are reported but do not replace D1 evidence.",
        "Discordant adjusted/unadjusted models remain visible.", bcfg["d1_primary_cohort"],
        ", ".join(sorted(family_models)), f"supported before={len(before)}, after={len(after)}",
        "Per-model OR intervals.", f"Holm q<{alpha}; raw p is not adjudicative.",
        "Maximum Morgan Tanimoto is one proxy, not all chemical similarity.",
        similarity_wording,
        "Similarity causally explains the association or exhausts chemical familiarity."))

    arun=runs[EXPERIMENTS[0]]; base=_read(arun,"simple_baseline_metrics.csv",("baseline",)); perm=_read(arun,"label_permutation_controls.csv",("auroc",))
    basic=base[base.baseline=="basic_properties"].iloc[0]; null95=float(perm.auroc.quantile(.975))
    panel=float(metrics[(metrics.experiment=="B")&(metrics.analysis=="repeat_pooled_oof")&(metrics.metric=="auroc")].groupby("model_id").value.mean().median())
    detectable=float(basic.auroc)>null95; substantial=detectable and float(basic.auroc)>=panel
    base_verdict="SUBSTANTIAL_PHYSICOCHEMICAL_SIGNAL" if substantial else "DETECTABLE_BUT_BELOW_TYPICAL_FIXED_MODEL" if detectable else "NO_SIGNAL_ABOVE_PERMUTATION_CONTROL"
    claims["claim_4"]={"verdict":base_verdict,"basic_auroc":float(basic.auroc),"permutation_97_5":null95,"panel_median":panel}
    baseline_wording = {
        "SUBSTANTIAL_PHYSICOCHEMICAL_SIGNAL":
            "Basic properties captured operational-label signal comparable to the typical fixed-model panel.",
        "DETECTABLE_BUT_BELOW_TYPICAL_FIXED_MODEL":
            "Basic properties captured detectable operational-label structure but remained below the typical fixed-model panel.",
        "NO_SIGNAL_ABOVE_PERMUTATION_CONTROL":
            "The basic-property baseline did not exceed the prespecified permutation control.",
    }[base_verdict]
    rows.append(_claim("claim_4", "A simple physicochemical baseline captures a substantial part of operational labels.", base_verdict,
        f"Basic-property OOF AUROC={basic.auroc:.3f}; permutation 97.5th={null95:.3f}.",
        f"Median repeated-CV panel AUROC={panel:.3f}.", "Below-panel baseline performance is explicit.",
        "D1 train OOF", "basic-property logistic baseline and fixed panel",
        f"AUROC-null boundary={float(basic.auroc)-null95:+.3f}", "Empirical 100-permutation null.",
        "Prespecified sanity control, not model selection.",
        "Separability may reflect class construction, not biology.",
        baseline_wording,
        "Basic properties biologically explain geroprotection."))

    delta=_read(runs[EXPERIMENTS[2]],"random_vs_grouped_deltas.csv",("registry","model_id","metric","delta_grouped_minus_random"))
    primary=delta[delta.metric.isin(protocol["primary_metrics"])].copy()
    primary["material"]=primary.apply(lambda r: abs(float(r.delta_grouped_minus_random))>=float(protocol["grouped_decision"]["practical_delta"][r.metric]),axis=1)
    material=float(primary.material.mean()); rank_fraction=changed/comparisons if comparisons else np.nan
    group_verdict="CHEMISTRY_GROUPING_MATERIALLY_CHANGES_EVALUATION" if material>=float(protocol["grouped_decision"]["min_material_fraction"]) or grouped_changed else "LITTLE_MATERIAL_CHANGE_UNDER_GROUPING"
    claims["claim_5"]={"verdict":group_verdict,"material_fraction":material,"top_rank_change_fraction":rank_fraction}
    grouping_wording = (
        "Chemistry-aware grouping materially changed the evaluation under the prespecified decision rule."
        if group_verdict == "CHEMISTRY_GROUPING_MATERIALLY_CHANGES_EVALUATION" else
        "Chemistry-aware grouping produced little material change under the prespecified decision rule."
    )
    rows.append(_claim("claim_5", "Chemistry-aware grouping changes or does not change evaluation.", group_verdict,
        f"{material:.1%} model-metric-registry deltas exceeded prespecified practical bounds.",
        f"Top identity changed {changed}/{comparisons} times.", "Small/favorable deltas remain visible.",
        "D1 random, scaffold, similarity-component CV", "fixed ten-model panel",
        f"material fraction={material:.3f}; rank-change fraction={rank_fraction:.3f}",
        "Descriptive regime comparison; folds are dependent.", "Practical thresholds, no significance claim.",
        "Singleton scaffolds/group balance can limit grouped-CV severity.",
        grouping_wording,
        "Grouped CV proves broad out-of-domain generalization."))

    def endpoint_claim(cid, exp, cohort, name, replication_analysis):
        block=metrics[(metrics.experiment==exp)&(metrics.analysis=="endpoint_pooled_oof")]
        gb4=block[block.model_id=="gb4_equal"].set_index("metric").value
        endpoint_run = runs[EXPERIMENTS[3] if exp == "D" else EXPERIMENTS[4]]
        endpoint_predictions = _read(
            endpoint_run, "fixed_panel_oof_predictions.csv", ("label",))
        labels = endpoint_predictions["label"].to_numpy(int)
        if len(labels) == 0 or not set(np.unique(labels)).issubset({0, 1}):
            raise InsightReportError(f"{exp}: endpoint labels are empty or nonbinary")
        prevalence=float(labels.mean())
        auroc=float(gb4.get("auroc",np.nan)); auprc=float(gb4.get("auprc_average_precision_positive",np.nan))
        above=bool(auroc>.5 and auprc>prevalence)
        rep=metrics[(metrics.experiment==exp)&(metrics.analysis==replication_analysis)&(metrics.model_id=="gb4_equal")]
        rep_auc=rep[rep.metric=="auroc"].value
        replication_available=bool(len(rep_auc))
        replicated=bool(replication_available and rep_auc.mean()>.5)
        replication_text=(
            f"{replication_analysis} AUROC above chance={replicated}."
            if replication_available else
            f"{replication_analysis} was not estimable under the prespecified registry; no AUROC was produced."
        )
        verdict="DIRECTIONAL_ENDPOINT_ARCHITECTURE_TRANSPORT" if above and replicated else "PRIMARY_ENDPOINT_SIGNAL_WITHOUT_SENSITIVITY_REPLICATION" if above else "ENDPOINT_ARCHITECTURE_TRANSPORT_NOT_SUPPORTED"
        payload={"verdict":verdict,"gb4_auroc":auroc,"gb4_auprc":auprc,"prevalence":prevalence,"sensitivity_replication_available":replication_available,"sensitivity_replication":replicated}
        row=_claim(cid,f"The fixed architecture transports to the {name} operational endpoint.",verdict,
            f"GB4 OOF AUROC={auroc:.3f}, AUPRC={auprc:.3f}, prevalence={prevalence:.3f}.",
            replication_text,
            "All components/sensitivities retained; GB4 need not win.",cohort,"GB4 architecture and components",
            f"AUROC-0.5={auroc-.5:+.3f}; AUPRC-prevalence={auprc-prevalence:+.3f}",
            "Fold/repeat distributions; no absolute AUPRC cutoff.","Descriptive benchmark, not superiority.",
            "Endpoint-aligned retraining is not locked-D1 external validation.",
            "Directional architecture transport." if above else "Transport not supported for this endpoint.",
            "Locked D1 external validation, SOTA, or clinical prediction.")
        return payload,row
    claims["claim_6"],r6=endpoint_claim("claim_6","D","AgeXtend OOF and challenge","AgeXtend","d1_nonoverlap_sensitivity")
    claims["claim_7"],r7=endpoint_claim("claim_7","E","DrugAge Build 5 C. elegans","DrugAge C. elegans","publication_grouped")
    rows += [r6,r7]

    title_verdict=("BROAD_TITLE_SUPPORTED" if d1_multi and endpoint_rep else
                   "NARROW_TITLE_TO_D1_STYLE_BENCHMARK" if d1_multi else
                   "QUALIFY_TITLE_AS_HETEROGENEOUS_SENSITIVITY" if directional else "BROAD_TITLE_NOT_SUPPORTED")
    title=("Structure-based geroprotector classifiers are biased against drug-like chemistry: a benchmarking and explainability analysis" if title_verdict=="BROAD_TITLE_SUPPORTED" else
           "D1-style structure-based geroprotector classifiers show reduced sensitivity across drug-likeness strata" if d1_multi else
           "Structure-based geroprotector classifiers show heterogeneous sensitivity across drug-likeness strata" if directional else
           "Use a neutral benchmarking title without a broad bias assertion")
    claims["claim_8"]={"verdict":title_verdict,"recommended_title":title,"d1_multi_family":bool(d1_multi),"endpoint_replication":bool(endpoint_rep)}
    rows.append(_claim("claim_8", "The broad title is supported, needs qualification, or should be narrowed.", title_verdict,
        f"D1 multi-family Holm criterion={d1_multi}.",f"Endpoint-aligned D/E replication={endpoint_rep}.",
        "Retrospective D1-model stress tests cannot satisfy endpoint replication.",
        "D1, AgeXtend, DrugAge C. elegans OOF","prespecified family representatives; blends excluded from family count",
        f"D1 supported families={d1['holm_supported_families']}; endpoint replication={endpoint_rep}",
        "Negative bootstrap CIs and Holm q required.",f"Holm q<{alpha}.",
        "The title remains empirical, not causal or clinical.",title,
        "Keep 'are biased' based only on D1 or retrospective stress tests."))
    evidence=pd.DataFrame(rows,columns=EVIDENCE_COLUMNS)
    if evidence.shape!=(8,14) or tuple(evidence)!=EVIDENCE_COLUMNS:
        raise InsightReportError("Evidence matrix differs from exact 8-row/14-column contract")
    return evidence,claims


def _render(tmp: Path,evidence: pd.DataFrame,claims: dict,verification:dict) -> None:
    main=["# Integrated manuscript-ready results","","Generated by the prespecified decision tree; the manuscript was not edited.",""]
    for r in evidence.itertuples(index=False):
        main += [f"## {r.claim_id}: {r.verdict}","",r.allowed_wording,"",f"Primary: {r.primary_evidence}","",f"Replication: {r.replication_evidence}",""]
    (tmp/"recommended_main_text_results.md").write_text("\n".join(main)+"\n")
    supp=["# Recommended supplementary reporting","","Report the complete family, null and contradictory results.",""]+[f"- **{r.claim_id} ({r.verdict})**: {r.limitations}" for r in evidence.itertuples(index=False)]
    (tmp/"recommended_supplementary_results.md").write_text("\n".join(supp)+"\n")
    c=claims["claim_8"]
    (tmp/"recommended_title_and_claims.md").write_text(f"# Title and claims recommendation\n\nVerdict: **{c['verdict']}**\n\nRecommended wording: **{c['recommended_title']}**\n\nEndpoint-aligned replication is mandatory for a broad title. The manuscript was not edited.\n")
    risks=["# Logical risk register","","- D1 held-out, repeated, and grouped folds are not independent datasets.","- Repeated folds reuse compounds; fold-level p values are not independent evidence.","- D1-model DrugAge/AgeXtend analyses are retrospective stress tests, not prospective external validation.","- D/E test architecture transport after retraining, not validation of the locked D1 model.","- AgeXtend challenge binary metrics are unstable with very few negatives.","- Blocked historical reproductions were not approximated.","- QED/similarity associations are observational and noncausal.","- AUPRC is prevalence-dependent.","- Negative and contradictory evidence remains sealed.","","## Verified exact stamped runs",""]
    risks += [f"- `{k}`: `{v['run_id']}` ({v['status']}; {v['artifact_count']} verified artifacts)" for k,v in verification.items()]
    (tmp/"logical_risk_register.md").write_text("\n".join(risks)+"\n")


def run(*,root:Path,config_path:Path,run_id:str,stamp:str)->Path:
    if not re.fullmatch(r"insight_suite_report_[a-z0-9_.-]+",run_id): raise InsightReportError("Invalid report run_id")
    base_run_id=f"insight_suite_report_{stamp}"
    allowed_run_ids={
        base_run_id:"initial",
        f"{base_run_id}_corrected":"corrected_prevalence_field_v1",
        f"{base_run_id}_corrected_v2":
            "corrected_prevalence_and_infeasible_replication_wording_v2",
    }
    if run_id not in allowed_run_ids:
        raise InsightReportError(
            "Report must use the exact A-E stamp; only locked reporting-only "
            "correction revisions are permitted")
    root=root.resolve(); destination=root/"outputs"/run_id
    if destination.exists() or destination.is_symlink(): raise InsightReportError(f"Run exists: {destination}")
    protocol,protocol_sha=_load_protocol(config_path.resolve())
    runs,verification=resolve_and_verify_runs(root,stamp,protocol)
    metrics=collect_metrics(runs,protocol); ranks=collect_ranks(runs,protocol); qed=collect_qed_effects(runs)
    evidence,claims=adjudicate(metrics,ranks,qed,runs,protocol)
    destination.parent.mkdir(parents=True,exist_ok=True); tmp=Path(tempfile.mkdtemp(prefix=".insightreport.work-",dir=destination.parent))
    try:
        _write_csv(tmp/"all_experiment_metrics_long.csv",metrics); _write_csv(tmp/"all_experiment_model_ranks.csv",ranks); _write_csv(tmp/"all_qed_effects.csv",qed); _write_csv(tmp/"claim_evidence_matrix.csv",evidence)
        atomic_write_json(tmp/"claims_allowed.json",claims); _render(tmp,evidence,claims,verification)
        atomic_write_json(tmp/"RUN_MANIFEST.json",{"schema_version":f"{SCHEMA}.run_manifest.v1","run_id":run_id,"stamp":stamp,"report_revision":allowed_run_ids[run_id],"protocol_sha256":protocol_sha,"source_code_sha256":sha256_file(Path(__file__)),"resolved_runs":verification,"input_binding_sha256":canonical_sha256(verification),"output_rows":{"metrics":len(metrics),"ranks":len(ranks),"qed":len(qed),"claims":len(evidence)},"holm_adjusted_evidence_used":True,"arbitrary_absolute_auprc_transport_cutoff_used":False,"manuscript_edited":False,"existing_runs_modified":False,"runtime":{"python":platform.python_version(),"platform":platform.platform()}})
        atomic_write_json(tmp/"COMPLETED.json",{"schema_version":f"{SCHEMA}.completed.v1","status":"COMPLETE","run_id":run_id,"run_manifest_sha256":sha256_file(tmp/"RUN_MANIFEST.json"),"artifact_hashes":{str(p.relative_to(tmp)):sha256_file(p) for p in sorted(tmp.rglob("*")) if p.is_file() and p.name!="COMPLETED.json"}})
        os.replace(tmp,destination)
    finally:
        if tmp.exists(): shutil.rmtree(tmp)
    return destination


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--root",type=Path,required=True); p.add_argument("--config",type=Path,required=True); p.add_argument("--run-id",required=True); p.add_argument("--stamp",required=True); a=p.parse_args(argv)
    result=run(root=a.root,config_path=a.config,run_id=a.run_id,stamp=a.stamp); print(json.dumps({"run":str(result),"status":"COMPLETE"},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
