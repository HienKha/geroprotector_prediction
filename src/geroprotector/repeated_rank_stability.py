"""Experiment B: how much does the model ranking depend on the random partition?

Ten repeats of stratified five-fold cross-validation over the 324 D1 training
compounds, seeds 42 through 51, with the fixed ten-model panel.  Every compound
receives exactly one out-of-fold prediction per repeat and ten in total.

This is a stability study.  Section 6.1 of `prespecified insight-analysis protocol` forbids using its
output to replace the locked GB4 blend or to select a new manuscript model, and
nothing here reads a D1 test label or an external outcome.

Seed 42 reproduces the fold structure of every sealed cross-fitted run in this
project, so before seeds 43..51 are attempted the rebuilt seed-42 streams are
compared with the sealed ones under tolerances declared in the protocol.  The
run stops if any tolerance fails; no tolerance is relaxed afterwards.

Because the same 324 compounds recur in all ten repeats, the 50 folds are not 50
independent observations.  Paired intervals therefore come from a compound-cluster
bootstrap: a compound ID is resampled with all ten of its repeat predictions
attached, the repeat-level metric is recomputed, and the paired difference is
averaged over repeats.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import tempfile
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from geroprotector.study_common import bind_checkpoint_directory, md_table
from geroprotector.d1_training import load_d1_train
from geroprotector.model_panel import (
    BASE_MODELS, BLENDS, PANEL, Panel, add_blends, run_folds)
from geroprotector.fixed_blend_paper405 import _features
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, sha256_file
from geroprotector.nine_ml_featuresets import _metrics


class RankStabilityError(RuntimeError):
    """Raised when a contract, a parity check or a leakage guard fails."""


SCHEMA = "geroprotector.repeated_rank_stability"
PRIMARY_METRICS = ("auprc_average_precision_positive", "auroc", "accuracy", "mcc",
                   "macro_f1", "brier", "cohen_kappa")
SECONDARY_METRICS = ("recall_sensitivity", "specificity")
LOWER_IS_BETTER = ("brier",)
SEALED_COLUMN = {
    "paper_svm": ("quad_blend_20260822/cv5_train_oof_predictions.csv", "probability_paper_svm"),
    "tanimoto_svc": ("quad_blend_20260822/cv5_train_oof_predictions.csv", "probability_tanimoto_svc"),
    "tabpfn_v2": ("quad_blend_20260822/cv5_train_oof_predictions.csv", "probability_tabpfn_v2"),
    "tabfm": ("quad_blend_20260822/cv5_train_oof_predictions.csv", "probability_tabfm"),
    "catboost_full": ("nineml_cv5_full_20260823/cv5_train_oof_predictions.csv", "probability_catboost"),
    "xgboost_full": ("nineml_cv5_full_20260823/cv5_train_oof_predictions.csv", "probability_xgboost"),
    "lightgbm_full": ("nineml_cv5_full_20260823/cv5_train_oof_predictions.csv", "probability_lightgbm"),
    "tabm_full": ("screeningblend_altmodels_20260819/d1_train_oof_predictions.csv", "probability_tabm"),
}


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "geroprotector.repeated_rank_stability.protocol.v1":
        raise RankStabilityError("Unknown protocol schema")
    if payload["data_boundary"]["d1_test_labels_loaded"] is not False:
        raise RankStabilityError("Protocol permits loading D1 test labels")
    return payload, sha256_file(path)


def _repeat_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    return _metrics(y, probability, probability)


def _run_repeat(panel: Panel, features, labels, paper_row_indices, seed, checkpoint_dir):
    """Five fold jobs for one seed, checkpointed so a crash does not lose GPU work."""
    checkpoint = checkpoint_dir / f"repeat_seed{seed}.csv"
    if checkpoint.is_file():
        print(f"  seed {seed}: resuming from checkpoint", flush=True)
        return pd.read_csv(checkpoint)
    y_train = np.asarray(labels, dtype=int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
    local = np.arange(len(y_train), dtype=int)
    rows = np.full(len(y_train), -1, dtype=int)
    folds = []
    for fold, (relative_fit, relative_validation) in enumerate(
            splitter.split(local, y_train)):
        folds.append((relative_fit, relative_validation))
        rows[relative_validation] = fold
    if (rows < 0).any():
        raise RankStabilityError(f"seed {seed}: fold assignment incomplete")
    jobs = folds
    columns_absolute = run_folds(
        panel, features, labels, jobs,
        # Hold model RNG fixed across partition repeats. This isolates the stated
        # scientific factor, random partitioning, while reproducing the sealed
        # seed-42 convention exactly.
        [42 + k for k in range(len(jobs))],
        n_rows=len(labels), tag=f"seed {seed} fold")
    columns = columns_absolute
    blended = add_blends(columns)
    frame = pd.DataFrame({"seed": int(seed),
                          "paper_row_index": np.asarray(paper_row_indices, int),
                          "fold": rows, "label": y_train,
                          **{f"p_{k}": v for k, v in blended.items()}})
    frame.to_csv(checkpoint, index=False, lineterminator="\n")
    return frame


def _seed42_parity(root: Path, frame: pd.DataFrame, tolerances: dict) -> pd.DataFrame:
    rows = []
    reference_cache: dict[str, pd.DataFrame] = {}
    ordered = frame.sort_values("paper_row_index").reset_index(drop=True)
    for model in BASE_MODELS:
        relative, column = SEALED_COLUMN[model]
        if relative not in reference_cache:
            reference_cache[relative] = pd.read_csv(root / "outputs" / relative) \
                .sort_values("paper_row_index").reset_index(drop=True)
        reference = reference_cache[relative]
        if not np.array_equal(reference["paper_row_index"].to_numpy(),
                              ordered["paper_row_index"].to_numpy()):
            raise RankStabilityError(f"{model}: sealed stream covers different rows")
        rebuilt = ordered[f"p_{model}"].to_numpy(float)
        sealed = reference[column].to_numpy(float)
        deviation = float(np.max(np.abs(rebuilt - sealed)))
        rho = float(stats.spearmanr(rebuilt, sealed).statistic)
        agreement = float(np.mean((rebuilt >= 0.5) == (sealed >= 0.5)))
        criterion = "max_abs_deviation"
        tolerance = float(tolerances[model])
        passed = bool(deviation <= tolerance)
        rows.append({
            "model_id": model, "reference": f"{relative}::{column}",
            "n": int(len(sealed)), "max_abs_deviation": deviation,
            "mean_abs_deviation": float(np.mean(np.abs(rebuilt - sealed))),
            "spearman_rho_vs_sealed": rho,
            "decision_agreement_at_0.5": agreement,
            "criterion": criterion, "declared_tolerance": json.dumps(tolerance),
            "passed": passed})
    return pd.DataFrame(rows)


def _ranks(frame: pd.DataFrame, metric: str) -> pd.Series:
    ascending = metric in LOWER_IS_BETTER
    return frame.groupby("seed")[metric].rank(ascending=ascending, method="min")


def validate_seed42(*, root: Path, config_path: Path) -> pd.DataFrame:
    """Run the blocking seed-42 replay without creating a final result run."""
    protocol, _protocol_sha = _load_protocol(config_path)
    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    frame, _audit = load_d1_train(root)
    labels = frame["label"].to_numpy(int)
    features = _features(frame, frame["smiles"].astype(str).tolist(), fixed_protocol)
    panel = Panel(root)
    with tempfile.TemporaryDirectory(prefix="seed42-parity-") as temporary:
        rebuilt = _run_repeat(
            panel, features, labels, frame["paper_row_index"].to_numpy(int), 42,
            Path(temporary))
    parity = _seed42_parity(root, rebuilt, protocol["seed42_parity"]["tolerances"])
    if not bool(parity["passed"].all()):
        raise RankStabilityError("Corrected seed-42 parity failed:\n" + parity.to_string(index=False))
    return parity


def run(*, root: Path, config_path: Path, run_id: str,
        seeds: list[int] | None = None) -> Path:
    if not re.fullmatch(r"repeated_rank_stability_[a-z0-9_.-]+", run_id):
        raise RankStabilityError("RUN_ID must start with repeated_rank_stability_")
    root = root.resolve()
    protocol, protocol_sha = _load_protocol(config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise RankStabilityError(f"Run directory already exists: {destination}")
    declared_seeds = list(protocol["folds"]["seeds"])
    seeds = list(seeds or declared_seeds)
    if seeds != declared_seeds:
        raise RankStabilityError(
            "The final Experiment B run must use exactly the ten prespecified seeds; "
            "use a separate smoke-test entry point rather than sealing a subset")
    if seeds[0] != 42:
        raise RankStabilityError("Seed 42 must run first so parity can gate the rest")

    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    frame, audit = load_d1_train(root)
    labels = frame["label"].to_numpy(int)
    paper_row_indices = frame["paper_row_index"].to_numpy(int)
    features = _features(frame, frame["smiles"].astype(str).tolist(), fixed_protocol)
    split_sha = audit["paper_split_sha256"]
    print(f"D1 train {len(frame)} rows, {int(labels.sum())} positive; "
          "held-out labels were not loaded into this module", flush=True)

    checkpoint_dir = root / "outputs" / f".{run_id}.work"
    checkpoint_contract = {
        "run_id": run_id,
        "protocol_sha256": protocol_sha,
        "train_boundary_inputs": {
            "split_registry_sha256": audit["split_registry"]["sha256"],
            "train_oof_sha256": audit["train_oof_label_source"]["sha256"],
        },
        "paper_split_sha256": split_sha,
        "seeds": seeds,
        "source_sha256": {
            name: sha256_file(root / "src" / "geroprotector" / name)
            for name in ("repeated_rank_stability.py", "model_panel.py", "study_common.py",
                         "d1_training.py")
        },
        "model_protocol_sha256": {
            name: sha256_file(root / "configs" / name)
            for name in ("fixed_blend_paper405.yaml", "traditional_paper405.yaml",
                         "screening_blend_tabfm_protocol.yaml",
                         "screening_blend_altmodels_protocol.yaml")
        },
    }
    checkpoint_binding_sha = bind_checkpoint_directory(
        checkpoint_dir, checkpoint_contract, error_cls=RankStabilityError)
    panel = Panel(root)

    repeats, parity = [], None
    for seed in seeds:
        repeats.append(_run_repeat(panel, features, labels, paper_row_indices, seed,
                                   checkpoint_dir))
        if seed == 42:
            parity = _seed42_parity(root, repeats[0],
                                    protocol["seed42_parity"]["tolerances"])
            print(parity.to_string(index=False), flush=True)
            if not bool(parity["passed"].all()):
                raise RankStabilityError(
                    "Seed-42 parity with the sealed OOF streams failed; seeds 43..51 "
                    "are not run and no tolerance is relaxed.\n"
                    + parity.to_string(index=False))
            print("  seed-42 parity PASSED; continuing to the remaining seeds",
                  flush=True)

    predictions = pd.concat(repeats, ignore_index=True)
    # every compound must have exactly one OOF prediction per repeat
    counts = predictions.groupby(["seed", "paper_row_index"]).size()
    if counts.min() != 1 or counts.max() != 1:
        raise RankStabilityError("A compound has more than one OOF prediction in a repeat")
    if predictions.groupby("paper_row_index").size().nunique() != 1:
        raise RankStabilityError("Compounds do not all carry the same number of repeats")

    per_fold, per_repeat = [], []
    for seed, block in predictions.groupby("seed"):
        for model in PANEL:
            probability = block[f"p_{model}"].to_numpy(float)
            y = block["label"].to_numpy(int)
            per_repeat.append({"seed": int(seed), "model_id": model,
                               **_repeat_metrics(y, probability)})
            for fold in sorted(block["fold"].unique()):
                mask = (block["fold"] == fold).to_numpy()
                per_fold.append({"seed": int(seed), "fold": int(fold), "model_id": model,
                                 **_repeat_metrics(y[mask], probability[mask])})
    per_fold = pd.DataFrame(per_fold)
    per_repeat = pd.DataFrame(per_repeat)

    # ---- rank distribution -------------------------------------------------
    rank_rows = []
    for metric in PRIMARY_METRICS:
        per_repeat[f"rank_{metric}"] = _ranks(per_repeat, metric)
        for model in PANEL:
            values = per_repeat.loc[per_repeat.model_id == model, metric]
            ranks = per_repeat.loc[per_repeat.model_id == model, f"rank_{metric}"]
            rank_rows.append({
                "metric": metric, "model_id": model, "repeats": int(len(values)),
                "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                "median": float(values.median()),
                "iqr_low": float(values.quantile(0.25)),
                "iqr_high": float(values.quantile(0.75)),
                "median_rank": float(ranks.median()),
                "rank_iqr_low": float(ranks.quantile(0.25)),
                "rank_iqr_high": float(ranks.quantile(0.75)),
                "best_rank": float(ranks.min()), "worst_rank": float(ranks.max()),
                "fraction_ranked_first": float((ranks == 1).mean()),
                "fraction_top_three": float((ranks <= 3).mean())})
    rank_distribution = pd.DataFrame(rank_rows)

    # ---- pairwise wins and rank-correlation across repeats -----------------
    win_rows = []
    for metric in PRIMARY_METRICS:
        pivot = per_repeat.pivot(index="seed", columns="model_id", values=metric)
        for a, b in combinations(PANEL, 2):
            better = (pivot[a] < pivot[b]) if metric in LOWER_IS_BETTER else (pivot[a] > pivot[b])
            win_rows.append({"metric": metric, "model_a": a, "model_b": b,
                             "a_wins": int(better.sum()),
                             "b_wins": int((~better & (pivot[a] != pivot[b])).sum()),
                             "ties": int((pivot[a] == pivot[b]).sum()),
                             "mean_difference_a_minus_b": float((pivot[a] - pivot[b]).mean())})
    pairwise = pd.DataFrame(win_rows)

    correlation_rows = []
    for metric in PRIMARY_METRICS:
        pivot = per_repeat.pivot(index="seed", columns="model_id", values=f"rank_{metric}")
        for a, b in combinations(sorted(pivot.index), 2):
            correlation_rows.append({
                "metric": metric, "seed_a": int(a), "seed_b": int(b),
                "spearman_rho": float(stats.spearmanr(pivot.loc[a], pivot.loc[b]).statistic)})
    rank_correlation = pd.DataFrame(correlation_rows)

    # ---- compound-cluster bootstrap ----------------------------------------
    bootstrap = _cluster_bootstrap(predictions, protocol["bootstrap"])

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".rankstab.work-", dir=destination.parent))
    try:
        registry = predictions[["seed", "paper_row_index", "fold", "label"]].copy()
        _write_csv(tmp / "split_registry.csv", registry)
        _write_csv(tmp / "per_row_repeated_oof_predictions.csv", predictions)
        _write_csv(tmp / "per_fold_metrics.csv", per_fold)
        _write_csv(tmp / "per_repeat_pooled_metrics.csv", per_repeat)
        _write_csv(tmp / "rank_distribution.csv", rank_distribution)
        _write_csv(tmp / "pairwise_win_matrix.csv", pairwise)
        _write_csv(tmp / "rank_correlation_across_repeats.csv", rank_correlation)
        _write_csv(tmp / "compound_cluster_bootstrap_differences.csv", bootstrap)
        _write_csv(tmp / "seed42_parity_checks.csv", parity)

        lines = ["# Experiment B -- repeated model-rank stability (D1 train only)", "",
                 f"run_id: `{run_id}`  |  {len(seeds)} repeats x 5 folds  |  "
                 f"threshold fixed at 0.5", "",
                 "The 50 folds are not 50 independent observations: the same 324 "
                 "compounds recur in every repeat. Paired intervals use a "
                 "compound-cluster bootstrap.", "",
                 "## Seed-42 parity with the sealed OOF streams", "",
                 md_table(parity, "{:.3e}"), ""]
        for metric in PRIMARY_METRICS:
            piece = rank_distribution[rank_distribution.metric == metric][
                ["model_id", "mean", "sd", "median", "iqr_low", "iqr_high",
                 "median_rank", "rank_iqr_low", "rank_iqr_high",
                 "fraction_ranked_first", "fraction_top_three"]]
            direction = " (lower is better)" if metric in LOWER_IS_BETTER else ""
            lines += [f"## {metric}{direction}", "",
                      md_table(piece.reset_index(drop=True), "{:.4f}"), ""]
        lines += ["## Rank correlation across repeats", "",
                  md_table(rank_correlation.groupby("metric")["spearman_rho"]
                           .agg(["mean", "min", "max"]).reset_index(), "{:.3f}"), ""]
        (tmp / "rank_stability_summary.md").write_text("\n".join(lines) + "\n")

        atomic_write_json(tmp / "RUN_MANIFEST.json", {
            "schema_version": f"{SCHEMA}.run_manifest.v1", "run_id": run_id,
            "protocol_sha256": protocol_sha, "paper_split_sha256": split_sha,
            "checkpoint_binding_sha256": checkpoint_binding_sha,
            "seeds": seeds, "folds_per_seed": 5, "fold_jobs": len(seeds) * 5,
            "component_seed_rule": "42 + fold_index, fixed across partition repeats",
            "panel": list(PANEL),
            "blend_definitions": {k: list(v) for k, v in BLENDS.items()},
            "resolved_model_settings": panel.resolved_settings,
            "fixed_threshold": 0.5, "threshold_is_tuned": False,
            "seed42_parity": parity.to_dict(orient="records"),
            "rows": "D1 train only (324)",
            "d1_test_labels_loaded": False,
            "external_outcomes_loaded": False,
            "test_or_external_labels_used_in_fit_selection_or_threshold": False,
            "data_audit": audit,
            "runtime": {"python": platform.python_version(),
                        "platform": platform.platform()},
            "existing_runs_modified": False})
        atomic_write_json(tmp / "COMPLETED.json", {
            "schema_version": f"{SCHEMA}.completed.v1", "status": "COMPLETE",
            "run_id": run_id,
            "run_manifest_sha256": sha256_file(tmp / "RUN_MANIFEST.json"),
            "artifact_hashes": {str(p.relative_to(tmp)): sha256_file(p)
                                for p in sorted(tmp.rglob("*"))
                                if p.is_file() and p.name != "COMPLETED.json"}})
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    print(json.dumps({"run": str(destination), "status": "COMPLETE"}, indent=2))
    return destination


def _cluster_bootstrap(predictions: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Resample compound IDs, carrying all repeats of a sampled compound together."""
    n_resamples = int(settings["n_resamples"])
    rng = np.random.default_rng(int(settings["seed"]))
    compounds = np.sort(predictions["paper_row_index"].unique())
    index = {c: i for i, c in enumerate(compounds)}
    seeds = np.sort(predictions["seed"].unique())
    # (compound, repeat) tensors so a resample is a single fancy-index
    label = np.full(len(compounds), -1)
    probability = {m: np.full((len(compounds), len(seeds)), np.nan) for m in PANEL}
    for s, seed in enumerate(seeds):
        block = predictions[predictions.seed == seed]
        positions = np.array([index[c] for c in block["paper_row_index"]])
        label[positions] = block["label"].to_numpy(int)
        for model in PANEL:
            probability[model][positions, s] = block[f"p_{model}"].to_numpy(float)

    from sklearn.metrics import (average_precision_score, cohen_kappa_score, f1_score,
                                 matthews_corrcoef, roc_auc_score)

    def metric_value(name, y, p):
        if name == "auprc_average_precision_positive":
            return average_precision_score(y, p)
        if name == "auroc":
            return roc_auc_score(y, p)
        if name == "mcc":
            return matthews_corrcoef(y, (p >= 0.5).astype(int))
        if name == "brier":
            return np.mean((p - y) ** 2)
        if name == "accuracy":
            return np.mean((p >= 0.5).astype(int) == y)
        if name == "macro_f1":
            return f1_score(y, (p >= 0.5).astype(int), average="macro", zero_division=0)
        if name == "cohen_kappa":
            return cohen_kappa_score(y, (p >= 0.5).astype(int))
        raise ValueError(name)

    metrics = PRIMARY_METRICS
    draws = rng.integers(0, len(compounds), size=(n_resamples, len(compounds)))
    rows = []
    for metric in metrics:
        pooled = {m: np.full((n_resamples,), np.nan) for m in PANEL}
        for b in range(n_resamples):
            idx = draws[b]
            y = label[idx]
            if len(np.unique(y)) < 2:
                continue
            for model in PANEL:
                values = [metric_value(metric, y, probability[model][idx, s])
                          for s in range(len(seeds))]
                pooled[model][b] = float(np.mean(values))
        for a, b_model in combinations(PANEL, 2):
            difference = pooled[a] - pooled[b_model]
            difference = difference[np.isfinite(difference)]
            if len(difference) < 100:
                continue
            rows.append({
                "metric": metric, "model_a": a, "model_b": b_model,
                "mean_difference": float(np.mean(difference)),
                "ci_low": float(np.percentile(difference, 2.5)),
                "ci_high": float(np.percentile(difference, 97.5)),
                "excludes_zero": bool(np.percentile(difference, 2.5) > 0 or
                                      np.percentile(difference, 97.5) < 0),
                "n_resamples": int(len(difference)),
                "unit": "compound_cluster_all_repeats_together"})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parity-only", action="store_true")
    a = parser.parse_args(argv)
    if a.parity_only:
        parity = validate_seed42(root=a.root.resolve(), config_path=a.config.resolve())
        print(parity.to_string(index=False))
        return 0
    run(root=a.root, config_path=a.config, run_id=a.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
