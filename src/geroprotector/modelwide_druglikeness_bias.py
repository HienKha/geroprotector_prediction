"""Experiment A: is reduced sensitivity for drug-like chemistry model-wide?

The manuscript title uses the plural noun `classifiers`.  That word is only
defensible if the negative QED association is a property of structure-based
classifiers in general rather than of one ensemble.  This module tests it on a
prespecified ten-model panel across four cohorts, reusing sealed prediction
streams throughout and refitting nothing.

Panel (section 3 of `prespecified insight-analysis protocol`):
    paper_svm, tanimoto_svc, tabpfn_v2, tabfm, catboost_full, xgboost_full,
    lightgbm_full, tabm_full, blend3_equal, gb4_equal

Cohorts (section 5.2):
    d1_train_oof (324, the primary development-partition analysis),
    d1_test (81), drugage, agextend -- the last three are post-lock descriptive
    replications and are never pooled with the first.

Every analysis is restricted to true positives under that cohort's operational
endpoint, because the question is retrieval bias, not calibration.  Threshold is
fixed at 0.5 for every model and every cohort.  No outcome is used to choose a
model, a covariate, a stratum boundary or a test.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from geroprotector.chemistry.standardize import standardize_smiles
from geroprotector.study_common import (
    BASIC_PROPERTIES,
    QED_STRATA,
    cliffs_delta,
    cochran_armitage,
    holm,
    logistic_fit,
    max_tanimoto,
    md_table,
    molecular_variables,
    morgan_generator,
    morgan_matrix,
    qed_stratum,
    spearman_bootstrap,
    standardized_mean_difference,
    wilson,
)
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.traditional_paper405 import (
    _read_sources,
    paper_split_indices,
)


class ModelwideBiasError(RuntimeError):
    """Raised when a contract, an alignment check or a leakage guard fails."""


SCHEMA = "geroprotector.modelwide_druglikeness_bias"
THRESHOLD = 0.5
BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260825
PERMUTATIONS = 100
PERMUTATION_SEED = 20260825

BASE_MODELS = ("paper_svm", "tanimoto_svc", "tabpfn_v2", "tabfm", "catboost_full",
               "xgboost_full", "lightgbm_full", "tabm_full")
BLENDS = {
    "blend3_equal": ("paper_svm", "tabpfn_v2", "tabfm"),
    "gb4_equal": ("paper_svm", "tanimoto_svc", "tabpfn_v2", "tabfm"),
}
PANEL = BASE_MODELS + tuple(BLENDS)
COHORTS = ("d1_train_oof", "d1_test", "drugage", "agextend")
# One representative per model family for the figure.
FIGURE_MODELS = ("paper_svm", "tanimoto_svc", "tabpfn_v2", "catboost_full", "gb4_equal")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def _read(root: Path, relative: str, **kwargs) -> pd.DataFrame:
    path = root / "outputs" / relative
    if not path.is_file():
        raise ModelwideBiasError(f"Required sealed input is missing: {relative}")
    return pd.read_csv(path, **kwargs)


def _load_bias_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != f"{SCHEMA}.protocol.v1":
        raise ModelwideBiasError("Unknown model-wide bias protocol schema")
    if protocol.get("panel") != list(PANEL) or float(protocol["threshold"]["fixed"]) != 0.5:
        raise ModelwideBiasError("Protocol changes the fixed panel or threshold")
    contract = protocol.get("structure_contract", {})
    if contract.get("model_input_structure") != "source_smiles" or \
       contract.get("primary_qed") != "standardized_parent":
        raise ModelwideBiasError("Unsupported or ambiguous structure contract")
    return protocol, sha256_file(path)


# ------------------------------------------------------------ cohort assembly

def _d1_context(root: Path, positive: Path, negative: Path):
    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    frame, audit = _read_sources(positive.resolve(), negative.resolve(), traditional)
    train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    smiles, _ = _validated_raw_smiles(frame)
    return frame, np.asarray(smiles, dtype=object), train_indices, test_indices, split_sha256, audit


def _train_oof_probabilities(root: Path) -> pd.DataFrame:
    """Fold-aligned OOF probabilities for the eight base models on D1 train."""
    quad = _read(root, "quad_blend_20260822/cv5_train_oof_predictions.csv")
    gbm = _read(root, "nineml_cv5_full_20260823/cv5_train_oof_predictions.csv")
    alt = _read(root, "screeningblend_altmodels_20260819/d1_train_oof_predictions.csv")

    if sorted(quad["paper_row_index"]) != sorted(gbm["paper_row_index"]) or \
       sorted(quad["paper_row_index"]) != sorted(alt["paper_row_index"]):
        raise ModelwideBiasError("D1 train OOF streams cover different rows")
    quad = quad.sort_values("paper_row_index").reset_index(drop=True)
    gbm = gbm.sort_values("paper_row_index").reset_index(drop=True)
    alt = alt.sort_values("paper_row_index").reset_index(drop=True)
    # Both cross-fitted runs record the fold index; they must be the identical
    # StratifiedKFold(5, shuffle=True, random_state=42) partition or the streams
    # are not poolable.
    if not np.array_equal(quad["fold"].to_numpy(), gbm["fold"].to_numpy()):
        raise ModelwideBiasError("quad_blend and nineml_cv5_full use different folds")
    if not np.array_equal(quad["label"].to_numpy(), gbm["label"].to_numpy()) or \
       not np.array_equal(quad["label"].to_numpy(), alt["label"].to_numpy()):
        raise ModelwideBiasError("D1 train OOF label vectors disagree")
    out = pd.DataFrame({
        "paper_row_index": quad["paper_row_index"].to_numpy(int),
        "fold": quad["fold"].to_numpy(int),
        "label": quad["label"].to_numpy(int),
        "p_paper_svm": quad["probability_paper_svm"],
        "p_tanimoto_svc": quad["probability_tanimoto_svc"],
        "p_tabpfn_v2": quad["probability_tabpfn_v2"],
        "p_tabfm": quad["probability_tabfm"],
        "p_catboost_full": gbm["probability_catboost"],
        "p_xgboost_full": gbm["probability_xgboost"],
        "p_lightgbm_full": gbm["probability_lightgbm"],
        "p_tabm_full": alt["probability_tabm"],
    })
    return out


def _test_probabilities(root: Path) -> pd.DataFrame:
    quad = _read(root, "quad_blend_20260822/predictions_d1_test.csv")
    full = _read(root, "nineml_full_scaled_20260823/predictions.csv")
    alt = _read(root, "screeningblend_altmodels_20260819/d1_test_predictions.csv")
    quad = quad.sort_values("paper_row_index").reset_index(drop=True)
    alt = alt.sort_values("paper_row_index").reset_index(drop=True)
    if not np.array_equal(quad["paper_row_index"].to_numpy(), alt["paper_row_index"].to_numpy()):
        raise ModelwideBiasError("D1 test streams cover different rows")
    out = pd.DataFrame({
        "paper_row_index": quad["paper_row_index"].to_numpy(int),
        "label": quad["label"].to_numpy(int),
        "p_paper_svm": quad["probability_paper_svm"],
        "p_tanimoto_svc": quad["probability_tanimoto_svc"],
        "p_tabpfn_v2": quad["probability_tabpfn_v2"],
        "p_tabfm": quad["probability_tabfm"],
        "p_tabm_full": alt["probability_tabm"],
    })
    for model_id, column in (("catboost", "p_catboost_full"), ("xgboost", "p_xgboost_full"),
                             ("lightgbm", "p_lightgbm_full")):
        piece = full[full["model_id"] == model_id].set_index("paper_row_index")
        out[column] = piece.loc[out["paper_row_index"], "probability"].to_numpy()
    return out


def _external_probabilities(root: Path, cohort: str) -> pd.DataFrame:
    base = _read(root, f"screeningblend_tabfm_20260821/external_predictions_{cohort}.csv")
    alt = _read(root, f"screeningblend_altmodels_20260819/external_predictions_{cohort}.csv",
                usecols=["external_id", "probability_tabm"])
    gbm = _read(root, f"gbm_external_raw_20260826/external_predictions_{cohort}.csv")
    merged = base.merge(alt, on="external_id", how="left", validate="one_to_one")
    merged = merged.merge(gbm, on="external_id", how="left", validate="one_to_one")
    if merged["probability_tabm"].isna().any():
        raise ModelwideBiasError(f"{cohort}: tabm stream does not cover every compound")
    if merged["probability_catboost_full"].isna().any():
        raise ModelwideBiasError(f"{cohort}: GBM stream does not cover every compound")
    out = pd.DataFrame({
        "external_id": merged["external_id"],
        "source_smiles": merged["source_smiles"],
        "standardized_parent_smiles": merged["standardized_parent_smiles"],
        "label": pd.to_numeric(merged["label"], errors="coerce"),
        "sealed_max_train_tanimoto": merged["maximum_tanimoto_to_fitted_train"],
        "p_paper_svm": merged["probability_paper_svm"],
        "p_tanimoto_svc": merged["probability_tanimoto_svc"],
        "p_tabpfn_v2": merged["probability_tabpfn_v2"],
        "p_tabfm": merged["probability_tabfm"],
        "p_tabm_full": merged["probability_tabm"],
        "p_catboost_full": merged["probability_catboost_full"],
        "p_xgboost_full": merged["probability_xgboost_full"],
        "p_lightgbm_full": merged["probability_lightgbm_full"],
    })
    return out


def _add_blends(frame: pd.DataFrame) -> pd.DataFrame:
    for name, components in BLENDS.items():
        frame[f"p_{name}"] = frame[[f"p_{c}" for c in components]].mean(axis=1)
    for model in PANEL:
        frame[f"d_{model}"] = (frame[f"p_{model}"] >= THRESHOLD).astype(int)
    return frame


def build_compound_table(root: Path, positive: Path, negative: Path) -> tuple[pd.DataFrame, dict]:
    frame, smiles, train_indices, test_indices, split_sha256, audit = \
        _d1_context(root, positive, negative)
    generator = morgan_generator()
    d1_bits = morgan_matrix(list(smiles), generator)
    d1_parent_smiles = np.asarray(
        [standardize_smiles(value).standardized_parent_smiles for value in smiles],
        dtype=object)
    d1_parent_bits = morgan_matrix(list(d1_parent_smiles), generator)
    d1_properties = molecular_variables(list(d1_parent_smiles))
    d1_raw_properties = molecular_variables(list(smiles))

    pieces, diagnostics = [], {}

    # --- D1 train OOF -------------------------------------------------------
    oof = _train_oof_probabilities(root)
    position = {int(r): i for i, r in enumerate(train_indices)}
    rows = np.array([position[int(r)] for r in oof["paper_row_index"]])
    absolute = np.asarray(train_indices)[rows]
    block = oof.copy()
    block["compound_id"] = [f"d1_{int(r)}" for r in block["paper_row_index"]]
    block["cohort"] = "d1_train_oof"
    block["smiles"] = smiles[absolute]
    block["source_smiles"] = smiles[absolute]
    block["standardized_parent_smiles"] = d1_parent_smiles[absolute]
    for column in d1_properties.columns:
        block[column] = d1_properties[column].to_numpy()[absolute]
    block["qed_source_smiles_sensitivity"] = d1_raw_properties["qed"].to_numpy()[absolute]
    # Fold-local applicability: for an OOF row, the "active fitted training set" is
    # its own fold's fit rows, never the whole partition.
    similarity = np.full(len(block), np.nan)
    for fold in sorted(block["fold"].unique()):
        validation = (block["fold"] == fold).to_numpy()
        fit_rows = absolute[~validation]
        similarity[validation] = max_tanimoto(d1_bits[absolute[validation]], d1_bits[fit_rows])
    block["max_train_tanimoto"] = similarity
    block["max_train_tanimoto_scope"] = "fold_fit_rows"
    parent_similarity = np.full(len(block), np.nan)
    for fold in sorted(block["fold"].unique()):
        validation = (block["fold"] == fold).to_numpy()
        fit_rows = absolute[~validation]
        parent_similarity[validation] = max_tanimoto(
            d1_parent_bits[absolute[validation]], d1_parent_bits[fit_rows])
    block["max_train_tanimoto_parent_identity"] = parent_similarity
    pieces.append(block)

    # --- D1 held-out test ---------------------------------------------------
    test = _test_probabilities(root)
    block = test.copy()
    block["compound_id"] = [f"d1_{int(r)}" for r in block["paper_row_index"]]
    block["cohort"] = "d1_test"
    block["fold"] = -1
    block["smiles"] = smiles[np.asarray(block["paper_row_index"], int)]
    block["source_smiles"] = block["smiles"]
    block["standardized_parent_smiles"] = d1_parent_smiles[
        np.asarray(block["paper_row_index"], int)]
    for column in d1_properties.columns:
        block[column] = d1_properties[column].to_numpy()[np.asarray(block["paper_row_index"], int)]
    block["qed_source_smiles_sensitivity"] = d1_raw_properties["qed"].to_numpy()[
        np.asarray(block["paper_row_index"], int)]
    block["max_train_tanimoto"] = max_tanimoto(
        d1_bits[np.asarray(block["paper_row_index"], int)], d1_bits[np.asarray(train_indices)])
    block["max_train_tanimoto_scope"] = "all_324_train_rows"
    block["max_train_tanimoto_parent_identity"] = max_tanimoto(
        d1_parent_bits[np.asarray(block["paper_row_index"], int)],
        d1_parent_bits[np.asarray(train_indices)])
    pieces.append(block)

    # --- external cohorts ---------------------------------------------------
    for cohort in ("drugage", "agextend"):
        block = _external_probabilities(root, cohort)
        block["compound_id"] = [f"{cohort}_{v}" for v in block["external_id"]]
        block["cohort"] = cohort
        block["fold"] = -1
        block["paper_row_index"] = -1
        block["smiles"] = block["standardized_parent_smiles"]
        properties = molecular_variables(block["standardized_parent_smiles"].astype(str).tolist())
        raw_properties = molecular_variables(block["source_smiles"].astype(str).tolist())
        for column in properties.columns:
            block[column] = properties[column].to_numpy()
        block["qed_source_smiles_sensitivity"] = raw_properties["qed"].to_numpy()
        bits = morgan_matrix(block["source_smiles"].astype(str).tolist(), generator)
        parent_bits = morgan_matrix(
            block["standardized_parent_smiles"].astype(str).tolist(), generator)
        block["max_train_tanimoto"] = max_tanimoto(
            bits, d1_bits[np.asarray(train_indices)])
        block["max_train_tanimoto_scope"] = "all_324_train_rows"
        block["max_train_tanimoto_parent_identity"] = max_tanimoto(
            parent_bits, d1_parent_bits[np.asarray(train_indices)])
        # Cross-check the independently recomputed similarity against the sealed value.
        deviation = float(np.nanmax(np.abs(
            block["max_train_tanimoto"] - block["sealed_max_train_tanimoto"])))
        diagnostics[f"{cohort}_max_tanimoto_vs_sealed"] = deviation
        if deviation > 1e-6:
            raise ModelwideBiasError(
                f"{cohort}: model-input Tanimoto differs from the sealed source-smiles "
                f"value by {deviation:.6g}")
        pieces.append(block.drop(columns=["sealed_max_train_tanimoto"]))

    table = pd.concat(pieces, ignore_index=True, sort=False)
    table = _add_blends(table)
    table["qed_stratum"] = qed_stratum(table["qed"].to_numpy(float))
    table["excluded_unparsable"] = ~table["parsable"].astype(bool)
    table["excluded_no_qed"] = ~np.isfinite(table["qed"].to_numpy(float))
    ordered = (["cohort", "compound_id", "paper_row_index", "external_id", "fold", "label",
                "smiles", "source_smiles", "standardized_parent_smiles", "qed",
                "qed_source_smiles_sensitivity", "qed_stratum"] + list(BASIC_PROPERTIES[1:]) +
               ["lipinski_violations", "scaffold", "is_acyclic", "max_train_tanimoto",
                "max_train_tanimoto_parent_identity",
                "max_train_tanimoto_scope", "parsable", "excluded_unparsable",
                "excluded_no_qed"] +
               [f"p_{m}" for m in PANEL] + [f"d_{m}" for m in PANEL])
    for column in ordered:
        if column not in table.columns:
            table[column] = np.nan
    diagnostics["paper_split_sha256"] = split_sha256
    diagnostics["d1_data_audit"] = audit
    return table[ordered], diagnostics


# -------------------------------------------------------------------- analyses

def positive_associations(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in COHORTS:
        block = table[(table["cohort"] == cohort) & (table["label"] == 1)]
        block = block[np.isfinite(block["qed"].to_numpy(float))]
        for model in PANEL:
            result = spearman_bootstrap(block["qed"].to_numpy(float),
                                        block[f"p_{model}"].to_numpy(float),
                                        n_resamples=BOOTSTRAP, seed=BOOTSTRAP_SEED)
            rows.append({"cohort": cohort, "model_id": model,
                         "n_positive": result["n"], "spearman_rho": result["rho"],
                         "p_value": result["p_value"], "rho_ci_low": result["ci_low"],
                         "rho_ci_high": result["ci_high"],
                         "bootstrap_resamples": BOOTSTRAP,
                         "bootstrap_valid": result["bootstrap_valid"]})
    frame = pd.DataFrame(rows)
    frame["q_holm_within_cohort"] = np.nan
    for cohort in COHORTS:
        mask = frame["cohort"] == cohort
        frame.loc[mask, "q_holm_within_cohort"] = holm(frame.loc[mask, "p_value"])
    return frame


def stratum_recall(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in COHORTS:
        block = table[(table["cohort"] == cohort) & (table["label"] == 1)]
        for model in PANEL:
            for _low, _high, name in QED_STRATA:
                piece = block[block["qed_stratum"] == name]
                n = int(len(piece))
                k = int(piece[f"d_{model}"].sum()) if n else 0
                low, high = wilson(k, n)
                rows.append({"cohort": cohort, "model_id": model, "qed_stratum": name,
                             "n_positive": n, "retrieved_at_0.5": k,
                             "recall": (k / n) if n else np.nan,
                             "recall_ci_low": low, "recall_ci_high": high})
    return pd.DataFrame(rows)


def trend_tests(recall: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in COHORTS:
        for model in PANEL:
            piece = recall[(recall["cohort"] == cohort) & (recall["model_id"] == model)]
            piece = piece.set_index("qed_stratum").loc[[s[2] for s in QED_STRATA]]
            result = cochran_armitage(piece["retrieved_at_0.5"].astype(int).tolist(),
                                      piece["n_positive"].astype(int).tolist())
            first = piece["recall"].dropna()
            rows.append({"cohort": cohort, "model_id": model,
                         "z_statistic": result["statistic"], "p_value": result["p_value"],
                         "strata_used": result["strata_used"],
                         "n_positive_total": int(piece["n_positive"].sum()),
                         "recall_lowest_stratum": float(first.iloc[0]) if len(first) else np.nan,
                         "recall_highest_stratum": float(first.iloc[-1]) if len(first) else np.nan})
    frame = pd.DataFrame(rows)
    frame["q_holm_within_cohort"] = np.nan
    for cohort in COHORTS:
        mask = frame["cohort"] == cohort
        frame.loc[mask, "q_holm_within_cohort"] = holm(frame.loc[mask, "p_value"])
    return frame


def _logistic_row(cohort: str, model: str, design: np.ndarray, y: np.ndarray,
                  names: list[str], sd: float) -> list[dict]:
    if len(y) < 10 or len(np.unique(y)) < 2:
        return [{"cohort": cohort, "model_id": model, "term": name,
                 "coefficient": np.nan, "standard_error": np.nan, "z": np.nan,
                 "p_value": np.nan, "odds_ratio_per_sd": np.nan,
                 "or_ci_low": np.nan, "or_ci_high": np.nan, "n": int(len(y)),
                 "n_retrieved": int(y.sum()), "converged": False,
                 "separation": bool(len(np.unique(y)) < 2), "qed_sd": sd,
                 "status": "degenerate_outcome"} for name in names]
    fit = logistic_fit(design, y)
    rows = []
    for position, name in enumerate(names):
        beta = fit["coefficients"][position]
        se = fit["standard_errors"][position]
        rows.append({"cohort": cohort, "model_id": model, "term": name,
                     "coefficient": float(beta), "standard_error": float(se),
                     "z": float(fit["z"][position]), "p_value": float(fit["p_values"][position]),
                     "odds_ratio_per_sd": float(np.exp(beta)) if name != "intercept" else np.nan,
                     "or_ci_low": float(np.exp(beta - 1.959963984540054 * se))
                                  if name != "intercept" and np.isfinite(se) else np.nan,
                     "or_ci_high": float(np.exp(beta + 1.959963984540054 * se))
                                   if name != "intercept" and np.isfinite(se) else np.nan,
                     "n": fit["n"], "n_retrieved": fit["n_positive"],
                     "converged": fit["converged"], "separation": fit["separation"],
                     "qed_sd": sd,
                     "status": ("separation" if fit["separation"] else
                                "converged" if fit["converged"] else "not_converged")})
    return rows


def logistic_models(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    simple, adjusted = [], []
    for cohort in COHORTS:
        block = table[(table["cohort"] == cohort) & (table["label"] == 1)].copy()
        block = block[np.isfinite(block["qed"].to_numpy(float))]
        qed = block["qed"].to_numpy(float)
        sd = float(qed.std(ddof=1)) if len(qed) > 1 else np.nan
        z_qed = (qed - qed.mean()) / sd if np.isfinite(sd) and sd > 0 else np.full(len(qed), np.nan)
        tanimoto = block["max_train_tanimoto"].to_numpy(float)
        for model in PANEL:
            y = block[f"d_{model}"].to_numpy(float)
            simple += _logistic_row(cohort, model, z_qed, y,
                                    ["intercept", "standardized_qed"], sd)
            keep = np.isfinite(z_qed) & np.isfinite(tanimoto)
            adjusted += _logistic_row(
                cohort, model, np.column_stack([z_qed[keep], tanimoto[keep]]), y[keep],
                ["intercept", "standardized_qed", "max_train_tanimoto"], sd)
    simple_frame, adjusted_frame = pd.DataFrame(simple), pd.DataFrame(adjusted)
    for frame in (simple_frame, adjusted_frame):
        frame["q_holm_within_cohort"] = np.nan
        for cohort in COHORTS:
            mask = (frame["cohort"] == cohort) & (frame["term"] == "standardized_qed")
            frame.loc[mask, "q_holm_within_cohort"] = holm(frame.loc[mask, "p_value"])
    return simple_frame, adjusted_frame


# --------------------------------------------------- dataset-construction controls

def fold_local_baselines(table: pd.DataFrame, rng: np.random.Generator, *,
                         cohort: str = "d1_train_oof",
                         permutations: int = PERMUTATIONS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Section 5.6 controls, fit only within the active cohort's training folds."""
    block = table[table["cohort"] == cohort].copy()
    block = block[np.isfinite(block["qed"].to_numpy(float))]
    y = block["label"].to_numpy(float)
    folds = block["fold"].to_numpy(int)
    panels = {"qed_only": ["qed"], "basic_properties": list(BASIC_PROPERTIES)}
    rows, permutation_rows = [], []
    for name, columns in panels.items():
        x = block[columns].to_numpy(float)
        oof = np.full(len(y), np.nan)
        for fold in np.unique(folds):
            validation = folds == fold
            mu = x[~validation].mean(axis=0)
            sigma = np.where(x[~validation].std(axis=0, ddof=0) > 0,
                             x[~validation].std(axis=0, ddof=0), 1.0)
            fit = logistic_fit((x[~validation] - mu) / sigma, y[~validation])
            design = np.column_stack([np.ones(int(validation.sum())),
                                      (x[validation] - mu) / sigma])
            oof[validation] = 1.0 / (1.0 + np.exp(-np.clip(design @ fit["coefficients"], -35, 35)))
        rows.append({"baseline": name, "n": int(len(y)), "prevalence": float(y.mean()),
                     "auroc": float(_auroc(y, oof)),
                     "auprc": float(_ap(y, oof)),
                     "accuracy_at_0.5": float(((oof >= 0.5).astype(int) == y).mean()),
                     "brier": float(np.mean((oof - y) ** 2))})
        # label-permutation control on the cheap baseline only
        if name == "qed_only":
            for permutation in range(int(permutations)):
                shuffled = rng.permutation(y)
                oof_p = np.full(len(y), np.nan)
                for fold in np.unique(folds):
                    validation = folds == fold
                    mu = x[~validation].mean(axis=0)
                    sigma = np.where(x[~validation].std(axis=0, ddof=0) > 0,
                                     x[~validation].std(axis=0, ddof=0), 1.0)
                    fit = logistic_fit((x[~validation] - mu) / sigma, shuffled[~validation])
                    design = np.column_stack([np.ones(int(validation.sum())),
                                              (x[validation] - mu) / sigma])
                    oof_p[validation] = 1.0 / (1.0 + np.exp(
                        -np.clip(design @ fit["coefficients"], -35, 35)))
                permutation_rows.append({"baseline": name, "permutation": permutation,
                                         "auroc": float(_auroc(shuffled, oof_p)),
                                         "auprc": float(_ap(shuffled, oof_p)),
                                         "prevalence": float(shuffled.mean())})
    # prevalence-only probability baseline
    oof = np.full(len(y), np.nan)
    for fold in np.unique(folds):
        validation = folds == fold
        oof[validation] = y[~validation].mean()
    rows.append({"baseline": "prevalence_only", "n": int(len(y)),
                 "prevalence": float(y.mean()), "auroc": float(_auroc(y, oof)),
                 "auprc": float(_ap(y, oof)),
                 "accuracy_at_0.5": float(((oof >= 0.5).astype(int) == y).mean()),
                 "brier": float(np.mean((oof - y) ** 2))})
    baselines = pd.DataFrame(rows)
    baselines.insert(0, "cohort", cohort)
    permutation_frame = pd.DataFrame(permutation_rows)
    if not permutation_frame.empty:
        permutation_frame.insert(0, "cohort", cohort)
    return baselines, permutation_frame


_fold_local_baselines = fold_local_baselines


def _auroc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def _ap(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def class_shifts(table: pd.DataFrame, *, cohorts: tuple[str, ...] | None = None) -> pd.DataFrame:
    rows = []
    for cohort in (cohorts or COHORTS):
        block = table[table["cohort"] == cohort]
        positive = block[block["label"] == 1]
        negative = block[block["label"] == 0]
        if len(negative) < 3:
            continue
        for prop in BASIC_PROPERTIES:
            a = positive[prop].to_numpy(float)
            b = negative[prop].to_numpy(float)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            ks = stats.ks_2samp(a, b) if len(a) and len(b) else None
            rows.append({"cohort": cohort, "property": prop,
                         "n_positive": int(len(a)), "n_negative": int(len(b)),
                         "mean_positive": float(a.mean()) if len(a) else np.nan,
                         "mean_negative": float(b.mean()) if len(b) else np.nan,
                         "standardized_mean_difference": standardized_mean_difference(a, b),
                         "cliffs_delta": cliffs_delta(a, b),
                         "ks_statistic": float(ks.statistic) if ks else np.nan,
                         "ks_p_value": float(ks.pvalue) if ks else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- figure

def _figure(table: pd.DataFrame, associations: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 6.5, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42})
    titles = {"d1_train_oof": "D1 train OOF", "d1_test": "D1 test",
              "drugage": "DrugAge", "agextend": "AgeXtend"}
    fig, axes = plt.subplots(len(COHORTS), len(FIGURE_MODELS),
                             figsize=(7.2, 6.4), sharex=True, sharey=True)
    for r, cohort in enumerate(COHORTS):
        block = table[(table["cohort"] == cohort) & (table["label"] == 1)]
        block = block[np.isfinite(block["qed"].to_numpy(float))]
        for c, model in enumerate(FIGURE_MODELS):
            ax = axes[r, c]
            x = block["qed"].to_numpy(float)
            y = block[f"p_{model}"].to_numpy(float)
            ax.scatter(x, y, s=3.0, alpha=0.35, linewidths=0, color="#0072B2")
            if len(x) >= 4 and len(np.unique(x)) > 1:
                coefficients = np.polyfit(x, y, 1)
                grid = np.linspace(x.min(), x.max(), 40)
                ax.plot(grid, np.polyval(coefficients, grid), lw=1.1, color="#D55E00")
            ax.axhline(THRESHOLD, lw=0.5, ls=":", color="#4D4D4D")
            row = associations[(associations["cohort"] == cohort) &
                               (associations["model_id"] == model)]
            if len(row):
                row = row.iloc[0]
                ax.set_title(f"$\\rho$={row.spearman_rho:+.2f} "
                             f"[{row.rho_ci_low:+.2f},{row.rho_ci_high:+.2f}]\n"
                             f"n={int(row.n_positive)}", fontsize=5.6, pad=2)
            if r == 0:
                ax.text(0.5, 1.42, model, transform=ax.transAxes, ha="center",
                        fontsize=7, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{titles[cohort]}\npredicted probability", fontsize=6.2)
            if r == len(COHORTS) - 1:
                ax.set_xlabel("QED")
            ax.set_ylim(-0.02, 1.02)
    fig.subplots_adjust(hspace=0.55, wspace=0.16, top=0.90)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------- run

def run(*, root: Path, config_path: Path, positive: Path, negative: Path, run_id: str) -> Path:
    if not re.fullmatch(r"modelwide_druglikeness_bias_[a-z0-9_.-]+", run_id):
        raise ModelwideBiasError("RUN_ID must start with modelwide_druglikeness_bias_")
    root = root.resolve()
    protocol, protocol_sha = _load_bias_protocol(config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise ModelwideBiasError(f"Run directory already exists: {destination}")

    table, diagnostics = build_compound_table(root, positive, negative)
    print(f"compound table {table.shape}; "
          + ", ".join(f"{c}={int((table['cohort'] == c).sum())}" for c in COHORTS), flush=True)

    associations = positive_associations(table)
    print("  spearman associations done", flush=True)
    recall = stratum_recall(table)
    trends = trend_tests(recall)
    simple, adjusted = logistic_models(table)
    print("  logistic models done", flush=True)
    rng = np.random.default_rng(PERMUTATION_SEED)
    baselines, permutations = _fold_local_baselines(table, rng)
    shifts = class_shifts(table)
    print("  controls done", flush=True)

    summary_rows = []
    for cohort in COHORTS:
        for model in PANEL:
            a = associations[(associations.cohort == cohort) & (associations.model_id == model)].iloc[0]
            t = trends[(trends.cohort == cohort) & (trends.model_id == model)].iloc[0]
            s = simple[(simple.cohort == cohort) & (simple.model_id == model) &
                       (simple.term == "standardized_qed")].iloc[0]
            j = adjusted[(adjusted.cohort == cohort) & (adjusted.model_id == model) &
                         (adjusted.term == "standardized_qed")].iloc[0]
            summary_rows.append({
                "cohort": cohort, "model_id": model, "n_positive": a.n_positive,
                "spearman_rho": a.spearman_rho, "rho_ci_low": a.rho_ci_low,
                "rho_ci_high": a.rho_ci_high, "rho_q_holm": a.q_holm_within_cohort,
                "trend_z": t.z_statistic, "trend_q_holm": t.q_holm_within_cohort,
                "recall_low_qed": t.recall_lowest_stratum,
                "recall_high_qed": t.recall_highest_stratum,
                "or_per_sd_qed": s.odds_ratio_per_sd, "or_ci_low": s.or_ci_low,
                "or_ci_high": s.or_ci_high, "or_status": s.status,
                "adjusted_or_per_sd_qed": j.odds_ratio_per_sd,
                "adjusted_or_ci_low": j.or_ci_low, "adjusted_or_ci_high": j.or_ci_high,
                "adjusted_status": j.status})
    summary = pd.DataFrame(summary_rows)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".mwbias.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "model_cohort_predictions.csv", table)
        _write_csv(tmp / "positive_only_qed_associations.csv", associations)
        _write_csv(tmp / "qed_stratum_recall.csv", recall)
        _write_csv(tmp / "qed_logistic_models.csv", simple)
        _write_csv(tmp / "qed_adjusted_models.csv", adjusted)
        _write_csv(tmp / "qed_trend_tests.csv", trends)
        _write_csv(tmp / "basic_property_class_shifts.csv", shifts)
        _write_csv(tmp / "simple_baseline_metrics.csv", baselines)
        _write_csv(tmp / "label_permutation_controls.csv", permutations)
        _write_csv(tmp / "modelwide_bias_summary.csv", summary)
        _figure(table, associations, tmp / "modelwide_bias_by_family.pdf")

        lines = ["# Experiment A -- model-wide drug-likeness bias audit", "",
                 f"run_id: `{run_id}`  |  threshold fixed at {THRESHOLD}  |  "
                 f"{BOOTSTRAP} bootstrap resamples", "",
                 "Within-positive analyses only. Holm correction is applied across the "
                 "ten models within each cohort and inferential family.", ""]
        for cohort in COHORTS:
            piece = summary[summary.cohort == cohort][
                ["model_id", "n_positive", "spearman_rho", "rho_ci_low", "rho_ci_high",
                 "rho_q_holm", "trend_z", "trend_q_holm", "recall_low_qed",
                 "recall_high_qed", "or_per_sd_qed", "adjusted_or_per_sd_qed"]]
            lines += [f"## {cohort}", "", md_table(piece.reset_index(drop=True), "{:.3f}"), ""]
        lines += ["## Dataset-construction controls (D1 train OOF, fold-local)", "",
                  md_table(baselines, "{:.4f}"), "",
                  "### Label-permutation control (100 fixed permutations, QED-only baseline)",
                  "",
                  f"- AUROC  mean {permutations['auroc'].mean():.4f}  "
                  f"sd {permutations['auroc'].std(ddof=1):.4f}",
                  f"- AUPRC  mean {permutations['auprc'].mean():.4f}  "
                  f"sd {permutations['auprc'].std(ddof=1):.4f}  "
                  f"(fold prevalence {float((table.cohort == 'd1_train_oof').pipe(lambda m: table.loc[m, 'label'].mean())):.4f})",
                  "", "## Class construction shifts", "", md_table(shifts, "{:.3f}"), ""]
        (tmp / "modelwide_bias_summary.md").write_text("\n".join(lines) + "\n")

        atomic_write_json(tmp / "RUN_MANIFEST.json", {
            "schema_version": f"{SCHEMA}.run_manifest.v1", "run_id": run_id,
            "protocol_sha256": protocol_sha,
            "structure_contract": protocol["structure_contract"],
            "panel": list(PANEL), "cohorts": list(COHORTS),
            "blend_definitions": {k: list(v) for k, v in BLENDS.items()},
            "fixed_threshold": THRESHOLD, "threshold_is_tuned": False,
            "qed_strata": [list(s) for s in QED_STRATA],
            "bootstrap": {"n_resamples": BOOTSTRAP, "seed": BOOTSTRAP_SEED,
                          "unit": "compound"},
            "permutations": {"n": PERMUTATIONS, "seed": PERMUTATION_SEED},
            "multiplicity": "holm within cohort x inferential family",
            "sealed_inputs": {name: sha256_file(root / "outputs" / name) for name in [
                "quad_blend_20260822/cv5_train_oof_predictions.csv",
                "quad_blend_20260822/predictions_d1_test.csv",
                "nineml_cv5_full_20260823/cv5_train_oof_predictions.csv",
                "nineml_full_scaled_20260823/predictions.csv",
                "screeningblend_altmodels_20260819/d1_train_oof_predictions.csv",
                "screeningblend_altmodels_20260819/d1_test_predictions.csv",
                "screeningblend_tabfm_20260821/external_predictions_drugage.csv",
                "screeningblend_tabfm_20260821/external_predictions_agextend.csv",
                "gbm_external_raw_20260826/external_predictions_drugage.csv",
                "gbm_external_raw_20260826/external_predictions_agextend.csv"]},
            "alignment_diagnostics": diagnostics,
            "models_refitted_in_this_run": False,
            "test_or_external_labels_used_in_fit_selection_or_threshold": False,
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
    print(json.dumps({"run": str(destination), "status": "COMPLETE"}, indent=2))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    a = parser.parse_args(argv)
    run(root=a.root, config_path=a.config, positive=a.positive, negative=a.negative,
        run_id=a.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
