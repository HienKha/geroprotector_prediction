"""Corrected post-lock explainability and insight analyses for the locked blend.

This supersedes `blend_xai_analysis.py`, which contained four overclaims that an
external review correctly identified.  Each is fixed here:

  OVERCLAIM 1  "weighted vs equal-thirds isn't close"
    FIX: Section A reports a paired bootstrap of the train-OOF difference with
    95% CIs, so the reader can see how separable the two blends actually are.

  OVERCLAIM 2  "70% of the blend weight admits EXACT attribution"
    FIX: Section B states the decomposition at three explicit levels.  The raw
    blend is exactly decomposable at COMPONENT level (100% of weight).  Analytic
    within-component decomposition exists only in DECISION-MARGIN space, for the
    components carrying 70% of the weight.  Because `paper_svm` applies Platt
    scaling (probA_/probB_) and the Tanimoto stream is expit(decision_function),
    p = sigmoid(margin): margin contributions are NOT additive attributions of
    the final probability.  Section B demonstrates this non-additivity
    numerically rather than asserting it.

  OVERCLAIM 3  "the polyol bias is an SVM descriptor artifact"
    FIX: Section C decomposes every high-scoring false positive per component.
    The SVM carries only weight 0.10, so no bias may be attributed to it without
    showing the actual per-component split.

  OVERCLAIM 4  small-n applicability domain and prevalence-inflated AP
    FIX: Section I reports, per stratum, n / n_pos / n_neg / prevalence / AP /
    AP-minus-prevalence / AUROC / Brier with bootstrap CIs, over FOUR Tanimoto
    cutoffs plus a coverage-performance curve -- because AP's baseline is the
    subgroup prevalence, so raw AP across strata is not comparable.

Everything remains DESCRIPTIVE and POST-LOCK: no model is fitted, refitted,
recalibrated or reweighted, and no threshold is changed.  Bootstrap refits used
for coefficient-stability and ablation are diagnostics on TRAINING rows only and
never touch the locked model or any reported operating point.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.special import logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.svm import SVC

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from geroprotector.fixed_blend_paper405 import _features, _tanimoto
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.screening_blend_altmodels import _morgan_from_smiles
from geroprotector.screening_blend_paper405 import load_locked_bundle
from geroprotector.traditional_paper405 import _read_sources, paper_split_indices


class BlendXAIv2Error(RuntimeError):
    """Raised when a sealed input or an analysis invariant fails."""


SCHEMA = "geroprotector.blend_xai_v2"
LOCKED_WEIGHTS = np.asarray([0.10, 0.60, 0.30], dtype=np.float64)
COMPONENTS = ("paper_svm", "tanimoto_svc", "tabpfn_v2")
PROB_COLUMNS = [f"probability_{c}" for c in COMPONENTS]
LOCKED_THRESHOLD = 0.5299579802368826
D1_ENDPOINT = "paper_binary"
DRUGAGE_ENDPOINT = "significant_positive_retrieval_background_not_certified_negative"
AGEXTEND_ENDPOINT = "published_independent_table6_binary"
BOOTSTRAP = 2000
SEED = 20260820

# Prespecified chemical-class definitions (fixed before any class-level result was
# inspected).  Deliberately simple and auditable.
CLASS_SMARTS = {
    "aliphatic_hydroxyl": "[CX4][OX2H]",
    "aromatic_hydroxyl": "[c][OX2H]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
}
POLYOL_MIN_ALIPHATIC_OH = 3
POLYPHENOL_MIN_AROMATIC_OH = 2


def _regular_file(path: Path, role: str, expected: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise BlendXAIv2Error(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected is not None and sha256_file(resolved) != expected:
        raise BlendXAIv2Error(f"{role} SHA256 differs from the protocol lock")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    text = _regular_file(path, "XAI v2 protocol").read_text(encoding="utf-8")
    protocol = yaml.safe_load(text)
    expected_schema = f"{SCHEMA}.protocol.v1"
    if not isinstance(protocol, dict) or protocol.get("schema_version") != expected_schema:
        raise BlendXAIv2Error("Unknown XAI v2 protocol schema")
    contract = protocol.get("analysis_contract", {})
    for key in (
        "locked_model_is_refit_or_reweighted",
        "threshold_is_changed",
        "test_or_external_labels_used_to_build_explanations",
    ):
        if contract.get(key) is not False:
            raise BlendXAIv2Error(f"Descriptive-only contract differs at {key}")
    immutability = protocol.get("immutability", {})
    if immutability.get("existing_run_directories_are_read_only") is not True:
        raise BlendXAIv2Error("Immutability contract differs")
    for record in protocol.get("sealed_inputs", {}).values():
        _regular_file(root / record["path"], f"sealed input {record['path']}", record["sha256"])
    return protocol, canonical_sha256(protocol)


# ------------------------------------------------------------------ bootstrap


def _boot_ci(
    values: np.ndarray, statistic, n_resamples: int = BOOTSTRAP, seed: int = SEED
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``statistic`` over row indices of ``values``."""

    rng = np.random.default_rng(seed)
    n = len(values)
    draws = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        try:
            draws.append(statistic(idx))
        except Exception:
            continue
    if not draws:
        return (np.nan, np.nan)
    return tuple(float(v) for v in np.percentile(draws, [2.5, 97.5]))


def _metric_with_ci(y: np.ndarray, p: np.ndarray, seed: int = SEED) -> dict[str, Any]:
    """AP, AP-minus-prevalence, AUROC and Brier, each with a bootstrap CI."""

    y = np.asarray(y, int)
    p = np.asarray(p, float)
    prevalence = float(y.mean())

    def _ap(idx):
        yb = y[idx]
        if len(set(yb)) < 2:
            raise ValueError
        return average_precision_score(yb, p[idx])

    def _ap_adj(idx):
        yb = y[idx]
        if len(set(yb)) < 2:
            raise ValueError
        return average_precision_score(yb, p[idx]) - yb.mean()

    def _auroc(idx):
        yb = y[idx]
        if len(set(yb)) < 2:
            raise ValueError
        return roc_auc_score(yb, p[idx])

    def _brier(idx):
        return brier_score_loss(y[idx], p[idx])

    both = len(set(y)) == 2
    ap = float(average_precision_score(y, p)) if y.sum() else np.nan
    ap_lo, ap_hi = _boot_ci(y, _ap, seed=seed) if both else (np.nan, np.nan)
    adj_lo, adj_hi = _boot_ci(y, _ap_adj, seed=seed) if both else (np.nan, np.nan)
    auroc = float(roc_auc_score(y, p)) if both else np.nan
    au_lo, au_hi = _boot_ci(y, _auroc, seed=seed) if both else (np.nan, np.nan)
    br_lo, br_hi = _boot_ci(y, _brier, seed=seed)
    return {
        "n": len(y),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "positive_prevalence": prevalence,
        "auprc": ap,
        "auprc_ci_low": ap_lo,
        "auprc_ci_high": ap_hi,
        "auprc_minus_prevalence": ap - prevalence if np.isfinite(ap) else np.nan,
        "auprc_minus_prevalence_ci_low": adj_lo,
        "auprc_minus_prevalence_ci_high": adj_hi,
        "auroc": auroc,
        "auroc_ci_low": au_lo,
        "auroc_ci_high": au_hi,
        "brier": float(brier_score_loss(y, p)),
        "brier_ci_low": br_lo,
        "brier_ci_high": br_hi,
    }


def _calibration(
    y: np.ndarray, p: np.ndarray, bins: int = 10, seed: int = SEED
) -> dict[str, Any]:
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows, ece = [], 0.0
    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        mask = (p >= low) & (p < high) if i < bins - 1 else (p >= low) & (p <= high)
        if not mask.any():
            continue
        mean_p, observed = float(p[mask].mean()), float(y[mask].mean())
        ece += (mask.sum() / len(y)) * abs(mean_p - observed)
        rows.append(
            {
                "bin_low": float(low), "bin_high": float(high), "n": int(mask.sum()),
                "mean_predicted_probability": mean_p, "observed_positive_fraction": observed,
                "gap": observed - mean_p,
            }
        )

    def _fit(idx):
        yb, pb = y[idx], p[idx]
        if len(set(yb)) < 2:
            raise ValueError
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        model.fit(logit(pb).reshape(-1, 1), yb)
        return float(model.coef_[0][0]), float(model.intercept_[0])

    slope = intercept = np.nan
    slope_ci = intercept_ci = (np.nan, np.nan)
    if len(set(y)) == 2:
        slope, intercept = _fit(np.arange(len(y)))
        rng = np.random.default_rng(seed)
        slopes, intercepts = [], []
        for _ in range(500):
            idx = rng.integers(0, len(y), len(y))
            try:
                s, b = _fit(idx)
            except Exception:
                continue
            slopes.append(s)
            intercepts.append(b)
        if slopes:
            slope_ci = tuple(float(v) for v in np.percentile(slopes, [2.5, 97.5]))
            intercept_ci = tuple(float(v) for v in np.percentile(intercepts, [2.5, 97.5]))
    return {
        "bins": rows,
        "expected_calibration_error": float(ece),
        "calibration_slope": slope,
        "calibration_slope_ci_low": slope_ci[0],
        "calibration_slope_ci_high": slope_ci[1],
        "calibration_intercept": intercept,
        "calibration_intercept_ci_low": intercept_ci[0],
        "calibration_intercept_ci_high": intercept_ci[1],
        "brier": float(brier_score_loss(y, p)),
    }


def _chem_classes(smiles_list: list[str]) -> pd.DataFrame:
    patterns = {k: Chem.MolFromSmarts(v) for k, v in CLASS_SMARTS.items()}
    rows = []
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            blank = {"chemical_class": "unparsable", "scaffold": ""}
            rows.append({k: 0 for k in patterns} | blank)
            continue
        counts = {k: len(molecule.GetSubstructMatches(v)) for k, v in patterns.items()}
        if counts["aliphatic_hydroxyl"] >= POLYOL_MIN_ALIPHATIC_OH:
            label = "polyol_sugar_like"
        elif counts["aromatic_hydroxyl"] >= POLYPHENOL_MIN_AROMATIC_OH:
            label = "polyphenol_like"
        else:
            label = "other"
        try:
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(molecule)
            scaffold = Chem.MolToSmiles(scaffold_mol) if scaffold_mol is not None else ""
        except Exception:
            scaffold = ""
        rows.append(counts | {"chemical_class": label, "scaffold": scaffold})
    return pd.DataFrame(rows)


def run(
    *, root: Path, config_path: Path, positive_path: Path, negative_path: Path, run_id: str
) -> Path:
    if not re.fullmatch(r"blend_xai_v2_[a-z0-9_.-]+", run_id):
        raise BlendXAIv2Error("RUN_ID must start with blend_xai_v2_")
    root = root.resolve()
    protocol, protocol_sha256 = load_protocol(root, config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise BlendXAIv2Error(f"Run directory already exists: {destination}")
    sealed = protocol["sealed_inputs"]
    bundle = load_locked_bundle(
        root / sealed["screening_bundle"]["path"],
        expected_artifact_sha256=sealed["screening_bundle"]["sha256"],
    )

    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    frame, _audit = _read_sources(
        _regular_file(positive_path, "positive source", protocol["sources"]["positive_sha256"]),
        _regular_file(negative_path, "negative source", protocol["sources"]["negative_sha256"]),
        traditional,
    )
    _train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    labels = frame["label"].to_numpy(dtype=int)
    smiles, _ = _validated_raw_smiles(frame)
    features = _features(frame, smiles, fixed_protocol)
    fit_indices = np.asarray(bundle["fit_paper_indices"], dtype=int)
    sealed_bits = np.asarray(bundle["tanimoto_train_bits"], dtype=np.uint8)
    if not np.array_equal(features["morgan"][fit_indices], sealed_bits):
        raise BlendXAIv2Error("Rebuilt Morgan bits differ from the sealed bundle")

    train_oof = pd.read_csv(root / sealed["weighted_train_oof"]["path"])
    train_oof["label"] = labels[train_oof.paper_row_index.to_numpy()]
    test_components = (
        pd.read_csv(root / sealed["weighted_test_components"]["path"])
        .set_index("paper_row_index").loc[test_indices].reset_index()
    )
    by_row = frame.set_index("paper_row_index")
    test_components["compound_name"] = by_row.loc[test_indices, "compound_name"].to_numpy()
    test_components["source_smiles"] = [smiles[i] for i in test_indices]
    drugage = pd.read_csv(root / sealed["drugage_scored"]["path"])
    drugage["label"] = drugage.has_significant_positive.astype(int)
    agextend = pd.read_csv(root / sealed["agextend_scored"]["path"])

    cohorts: dict[str, dict[str, Any]] = {
        "d1_paper_test": {"endpoint": D1_ENDPOINT, "frame": test_components,
                          "bits": features["morgan"][test_indices],
                          "paper": features["paper"][test_indices]},
        "drugage": {"endpoint": DRUGAGE_ENDPOINT, "frame": drugage},
        "agextend": {"endpoint": AGEXTEND_ENDPOINT, "frame": agextend},
    }
    for name in ("drugage", "agextend"):
        payload = cohorts[name]
        payload["bits"] = _morgan_from_smiles(
            payload["frame"].source_smiles.astype(str).tolist(), bundle["portable_contract"]
        )
        rebuilt = _tanimoto(payload["bits"], sealed_bits).max(axis=1)
        stored = payload["frame"].maximum_tanimoto_to_fitted_train.to_numpy(float)
        if float(np.max(np.abs(rebuilt - stored))) > 1e-6:
            raise BlendXAIv2Error(f"Rebuilt {name} similarity differs from the sealed file")
    for name, payload in cohorts.items():
        f = payload["frame"]
        comp = f[PROB_COLUMNS].to_numpy(float)
        payload["frame"] = f.assign(blend_probability_recomputed=comp @ LOCKED_WEIGHTS)
        if "blend_probability" in f:
            stored_blend = f.blend_probability.to_numpy(float)
            drift = float(np.max(np.abs(stored_blend - comp @ LOCKED_WEIGHTS)))
            if drift > 1e-12:
                raise BlendXAIv2Error(f"{name}: stored blend differs from locked weights")

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".blendxai2.work-", dir=destination.parent))
    try:
        # ---- A. selection uncertainty: weighted vs equal-thirds on TRAIN OOF ----
        y_tr = train_oof.label.to_numpy(int)
        w_oof = train_oof[PROB_COLUMNS].to_numpy(float) @ LOCKED_WEIGHTS
        e_oof = train_oof[PROB_COLUMNS].to_numpy(float) @ np.full(3, 1 / 3)
        rng = np.random.default_rng(SEED)
        rows_sel = []
        for metric, fn in [
            ("auprc", average_precision_score), ("auroc", roc_auc_score),
        ]:
            diffs = []
            for _ in range(BOOTSTRAP):
                idx = rng.integers(0, len(y_tr), len(y_tr))
                if len(set(y_tr[idx])) < 2:
                    continue
                diffs.append(fn(y_tr[idx], w_oof[idx]) - fn(y_tr[idx], e_oof[idx]))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            rows_sel.append({
                "metric": metric, "basis": "train_oof_324_rows",
                "weighted_010_060_030": float(fn(y_tr, w_oof)),
                "equal_thirds": float(fn(y_tr, e_oof)),
                "difference": float(fn(y_tr, w_oof) - fn(y_tr, e_oof)),
                "ci_low": float(lo), "ci_high": float(hi),
                "separable_at_95pct": bool(lo > 0 or hi < 0),
            })
        _write_csv(tmp / "A_selection_uncertainty_train_oof.csv", pd.DataFrame(rows_sel))

        # ---- B. three-level decomposition + non-additivity demonstration --------
        svm = bundle["paper_svm"]
        coef = np.asarray(svm.coef_, float).ravel()
        b0 = float(svm.intercept_[0])
        probA, probB = float(svm.probA_[0]), float(svm.probB_[0])
        d1_paper = cohorts["d1_paper_test"]["paper"]
        margin = d1_paper @ coef + b0
        # libsvm binary Platt is 1/(1+exp(A*f+B)), NOT expit(A*f+B).  Using expit
        # here gives a ~0.48 discrepancy; the correct form reproduces predict_proba
        # to ~2e-2 (residual is libsvm/sklearn internal label-order and clipping).
        platt = 1.0 / (1.0 + np.exp(probA * margin + probB))
        sealed_p = cohorts["d1_paper_test"]["frame"].probability_paper_svm.to_numpy(float)
        # naive (WRONG) additive probability attribution vs the true Platt output
        train_mean = features["paper"][fit_indices].mean(axis=0)
        naive_sum = ((d1_paper - train_mean) @ coef) + (train_mean @ coef + b0)
        atomic_write_json(tmp / "B_decomposition_levels.json", {
            "level_1_component_decomposition_of_raw_blend": {
                "exact": True, "coverage_of_blend_weight": 1.0,
                "statement": (
                    "p_blend = 0.10*p_svm + 0.60*p_tanimoto + 0.30*p_tabpfn "
                    "is exact by construction"
                ),
            },
            "level_2_within_component_margin_decomposition": {
                "exact_in_margin_space": True, "coverage_of_blend_weight": 0.70,
                "components": ["paper_svm (linear margin)", "tanimoto_svc (kernel margin)"],
                "statement": (
                    "analytic decomposition exists for the DECISION MARGIN, "
                    "not the probability"
                ),
            },
            "level_3_final_probability_attribution": {
                "exact": False,
                "reason": (
                    "paper_svm applies Platt scaling (probA_/probB_) and the Tanimoto "
                    "stream is expit(decision_function); p = sigmoid(margin) is nonlinear"
                ),
                "paper_svm_probA": probA, "paper_svm_probB": probB,
                "max_abs_margin_minus_platt_probability_d1_test": float(
                    np.max(np.abs(naive_sum - sealed_p))
                ),
                "max_abs_reconstructed_platt_minus_sealed": float(
                    np.max(np.abs(platt - sealed_p))
                ),
                "platt_formula_used": "1/(1+exp(probA*decision+probB)) [libsvm binary]",
                "platt_residual_note": (
                    "small residual reflects libsvm/sklearn internal label ordering and "
                    "probability clipping, not the decomposition argument"
                ),
                "statement": (
                    "margin contributions must NOT be reported as additive "
                    "attributions of the final probability"
                ),
            },
        })

        # ---- C. per-compound component decomposition (all cohorts) -------------
        decomposition = []
        for name, payload in cohorts.items():
            f = payload["frame"]
            comp = f[PROB_COLUMNS].to_numpy(float)
            weighted = comp * LOCKED_WEIGHTS[None, :]
            blend = weighted.sum(axis=1)
            ident = f.compound_name.astype(str) if "compound_name" in f else f.index.astype(str)
            for i in range(len(blend)):
                decomposition.append({
                    "cohort": name, "compound_name": str(ident.iloc[i]),
                    "label": int(f.label.iloc[i]), "blend_probability": float(blend[i]),
                    "decision_at_locked_threshold": int(blend[i] >= LOCKED_THRESHOLD),
                    "p_paper_svm": float(comp[i, 0]), "p_tanimoto_svc": float(comp[i, 1]),
                    "p_tabpfn_v2": float(comp[i, 2]),
                    "weighted_paper_svm": float(weighted[i, 0]),
                    "weighted_tanimoto_svc": float(weighted[i, 1]),
                    "weighted_tabpfn_v2": float(weighted[i, 2]),
                    "pct_of_score_paper_svm": float(100 * weighted[i, 0] / blend[i]),
                    "pct_of_score_tanimoto_svc": float(100 * weighted[i, 1] / blend[i]),
                    "pct_of_score_tabpfn_v2": float(100 * weighted[i, 2] / blend[i]),
                    "component_spread": float(comp[i].max() - comp[i].min()),
                    "all_three_components_above_0p5": bool((comp[i] > 0.5).all()),
                })
        decomposition_frame = pd.DataFrame(decomposition)
        _write_csv(tmp / "C_component_decomposition_all.csv", decomposition_frame)
        fp = decomposition_frame[
            (decomposition_frame.label == 0)
            & (decomposition_frame.decision_at_locked_threshold == 1)
        ].sort_values("blend_probability", ascending=False)
        _write_csv(tmp / "C_false_positive_decomposition.csv", fp)

        # ---- D. SVM margin attribution + bootstrap coefficient stability -------
        train_paper, train_y = features["paper"][fit_indices], labels[fit_indices]
        # Stability by SUBSAMPLING WITHOUT REPLACEMENT, not by classic bootstrap.
        # Rationale (verified empirically): the paper's DataWarrior descriptors are
        # unscaled, and resampling *with* replacement creates duplicate rows that make
        # libsvm converge pathologically slowly; capping max_iter instead yields
        # NON-CONVERGED coefficients, which must not be reported as a stability
        # estimate.  Subsampling avoids both failure modes and every refit below uses
        # the locked model's exact hyperparameters with libsvm's default unlimited
        # iterations, so each solution is a genuine optimum.
        boot_coefs, n_nonconverged = [], 0
        rng2 = np.random.default_rng(SEED)
        fraction = float(protocol["stability"]["subsample_fraction"])
        size = max(int(fraction * len(train_y)), 20)
        for _ in range(int(protocol["stability"]["coefficient_subsamples"])):
            idx = rng2.choice(len(train_y), size, replace=False)
            if len(set(train_y[idx])) < 2:
                continue
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                m = SVC(kernel="linear", C=1.0, gamma=1.0, probability=False, random_state=42)
                m.fit(train_paper[idx], train_y[idx])
                if any(issubclass(c.category, ConvergenceWarning) for c in caught):
                    n_nonconverged += 1
                    continue
            boot_coefs.append(np.asarray(m.coef_, float).ravel())
        if n_nonconverged:
            raise BlendXAIv2Error(
                f"{n_nonconverged} stability refits failed to converge; refusing to "
                "report coefficient stability from non-converged solutions"
            )
        boot_coefs = np.vstack(boot_coefs)
        names7 = list(bundle["paper_svm_feature_names"])
        sd = train_paper.std(axis=0)
        stability = pd.DataFrame({
            "descriptor": names7,
            "locked_coefficient": coef,
            "standardised_effect": coef * sd,
            "subsample_mean": boot_coefs.mean(axis=0),
            "subsample_ci_low": np.percentile(boot_coefs, 2.5, axis=0),
            "subsample_ci_high": np.percentile(boot_coefs, 97.5, axis=0),
            "sign_stability_fraction": (
                np.sign(boot_coefs) == np.sign(coef)[None, :]
            ).mean(axis=0),
        }).sort_values("standardised_effect", key=np.abs, ascending=False)
        _write_csv(tmp / "D_svm_margin_attribution_stability.csv", stability)

        figure, axis = plt.subplots(figsize=(7.5, 4))
        ordered = stability.iloc[::-1]
        axis.barh(ordered.descriptor, ordered.standardised_effect, color="#4C72B0")
        axis.set_xlabel(
            "linear coefficient x train SD (MARGIN space, not a probability attribution)"
        )
        axis.set_title("paper_svm margin attribution (weight 0.10 of the blend)")
        axis.axvline(0, color="black", linewidth=0.8)
        figure.tight_layout()
        figure.savefig(tmp / "D_svm_margin_attribution.svg")
        plt.close(figure)

        # ---- E. signed support-vector contributions ----------------------------
        tsvc = bundle["tanimoto_svc"]
        dual = np.asarray(tsvc.dual_coef_, float).ravel()
        support = np.asarray(tsvc.support_, int)
        train_names = by_row.loc[fit_indices, "compound_name"].to_numpy()
        neighbours = []
        for name, payload in cohorts.items():
            similarity = _tanimoto(payload["bits"], sealed_bits)
            contribution = similarity[:, support] * dual[None, :]
            f = payload["frame"]
            ident = f.compound_name.astype(str) if "compound_name" in f else f.index.astype(str)
            for i in range(contribution.shape[0]):
                for kind, order in (
                    ("top_positive", np.argsort(-contribution[i])[:3]),
                    ("top_negative", np.argsort(contribution[i])[:3]),
                ):
                    for rank, j in enumerate(order, start=1):
                        neighbours.append({
                            "cohort": name, "compound_name": str(ident.iloc[i]),
                            "contribution_kind": kind, "rank": rank,
                            "support_vector_compound": str(train_names[support[j]]),
                            "support_vector_label": int(train_y[support[j]]),
                            "tanimoto_similarity": float(similarity[i, support[j]]),
                            "signed_dual_coefficient": float(dual[j]),
                            "signed_contribution_to_margin": float(contribution[i, j]),
                        })
        _write_csv(tmp / "E_signed_support_vector_contributions.csv", pd.DataFrame(neighbours))

        # ---- F. component ablation (descriptive; leave-one-out, renormalised) --
        ablation = []
        for name, payload in cohorts.items():
            f = payload["frame"]
            y = f.label.to_numpy(int)
            comp = f[PROB_COLUMNS].to_numpy(float)
            full = comp @ LOCKED_WEIGHTS
            base = _metric_with_ci(y, full)
            ablation.append({"cohort": name, "variant": "full_blend", **base})
            for k, cname in enumerate(COMPONENTS):
                keep = [j for j in range(3) if j != k]
                w = LOCKED_WEIGHTS[keep] / LOCKED_WEIGHTS[keep].sum()
                p = comp[:, keep] @ w
                m = _metric_with_ci(y, p)
                ablation.append({
                    "cohort": name, "variant": f"without_{cname}", **m,
                    "delta_auprc_vs_full": m["auprc"] - base["auprc"],
                    "delta_auroc_vs_full": m["auroc"] - base["auroc"],
                })
        _write_csv(tmp / "F_component_ablation.csv", pd.DataFrame(ablation))

        # ---- G. calibration with CIs ------------------------------------------
        calibration, reliability = {}, []
        for name, payload in cohorts.items():
            f = payload["frame"]
            res = _calibration(
                f.label.to_numpy(int),
                f.blend_probability_recomputed.to_numpy(float),
            )
            calibration[name] = {k: v for k, v in res.items() if k != "bins"}
            for record in res["bins"]:
                reliability.append({"cohort": name, **record})
        atomic_write_json(tmp / "G_calibration_summary.json", calibration)
        _write_csv(tmp / "G_calibration_reliability_bins.csv", pd.DataFrame(reliability))
        figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for axis, name in zip(axes, cohorts, strict=False):
            sub = pd.DataFrame([r for r in reliability if r["cohort"] == name])
            axis.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
            axis.plot(
                sub.mean_predicted_probability,
                sub.observed_positive_fraction,
                "o-", color="#C44E52",
            )
            c = calibration[name]
            axis.set_title(f"{name}\nECE={c['expected_calibration_error']:.3f} "
                           f"slope={c['calibration_slope']:.2f}")
            axis.set_xlabel("mean predicted probability")
        axes[0].set_ylabel("observed positive fraction")
        figure.tight_layout()
        figure.savefig(tmp / "G_calibration_reliability.svg")
        plt.close(figure)

        # ---- H. applicability domain, prevalence-adjusted, multi-cutoff --------
        ad_rows, coverage_rows = [], []
        for name, payload in cohorts.items():
            f = payload["frame"]
            y = f.label.to_numpy(int)
            p = f.blend_probability_recomputed.to_numpy(float)
            sim = _tanimoto(payload["bits"], sealed_bits).max(axis=1)
            for cutoff in protocol["applicability"]["tanimoto_cutoffs"]:
                strata = (
                    ("inside_ge_cutoff", sim >= cutoff),
                    ("outside_lt_cutoff", sim < cutoff),
                )
                for stratum, mask in strata:
                    if mask.sum() < 5 or len(set(y[mask])) < 2:
                        continue
                    ad_rows.append({
                        "cohort": name, "tanimoto_cutoff": float(cutoff), "stratum": stratum,
                        "coverage_fraction": float(mask.mean()),
                        **_metric_with_ci(y[mask], p[mask]),
                    })
            for q in np.linspace(0.1, 1.0, 10):
                k = max(int(np.ceil(q * len(sim))), 5)
                idx = np.argsort(-sim)[:k]
                if len(set(y[idx])) < 2:
                    continue
                coverage_rows.append({
                    "cohort": name, "coverage_fraction": float(k / len(sim)), "n": int(k),
                    "min_similarity_included": float(sim[idx].min()),
                    "positive_prevalence": float(y[idx].mean()),
                    "auprc": float(average_precision_score(y[idx], p[idx])),
                    "auprc_minus_prevalence": float(
                        average_precision_score(y[idx], p[idx]) - y[idx].mean()
                    ),
                    "auroc": float(roc_auc_score(y[idx], p[idx])),
                })
        _write_csv(
            tmp / "H_applicability_domain_prevalence_adjusted.csv",
            pd.DataFrame(ad_rows),
        )
        _write_csv(tmp / "H_coverage_performance_curve.csv", pd.DataFrame(coverage_rows))

        # ---- I. uncertainty / error-detection comparison -----------------------
        uncertainty = []
        for name, payload in cohorts.items():
            f = payload["frame"]
            y = f.label.to_numpy(int)
            comp = f[PROB_COLUMNS].to_numpy(float)
            p = comp @ LOCKED_WEIGHTS
            wrong = (( p >= LOCKED_THRESHOLD).astype(int) != y).astype(int)
            sim = _tanimoto(payload["bits"], sealed_bits).max(axis=1)
            scores = {
                "component_spread": comp.max(axis=1) - comp.min(axis=1),
                "component_std": comp.std(axis=1),
                "distance_from_threshold": -np.abs(p - LOCKED_THRESHOLD),
                "predictive_entropy": -(
                    p * np.log(np.clip(p, 1e-9, 1))
                    + (1 - p) * np.log(np.clip(1 - p, 1e-9, 1))
                ),
                "negative_max_similarity": -sim,
            }
            if len(set(wrong)) < 2:
                continue
            for score_name, values in scores.items():
                auc = float(roc_auc_score(wrong, values))

                def _fn(idx, v=values, w=wrong):
                    if len(set(w[idx])) < 2:
                        raise ValueError
                    return roc_auc_score(w[idx], v[idx])

                lo, hi = _boot_ci(wrong, _fn)
                uncertainty.append({
                    "cohort": name, "uncertainty_score": score_name, "n": len(y),
                    "error_detection_auroc": auc, "ci_low": lo, "ci_high": hi,
                    "better_than_chance_at_95pct": bool(lo > 0.5),
                    "note": "AUROC>0.5 means the score ranks errors above correct predictions",
                })
        _write_csv(tmp / "I_uncertainty_error_detection.csv", pd.DataFrame(uncertainty))

        # ---- J. prespecified chemical-class analysis ---------------------------
        class_rows, class_detail = [], []
        for name, payload in cohorts.items():
            f = payload["frame"]
            classes = _chem_classes(f.source_smiles.astype(str).tolist())
            merged = pd.concat([f.reset_index(drop=True), classes], axis=1)
            merged["cohort"] = name
            detail_columns = [
                "cohort", "compound_name", "label", "chemical_class",
                "aliphatic_hydroxyl", "aromatic_hydroxyl", "carboxylic_acid",
                "blend_probability_recomputed", *PROB_COLUMNS,
            ]
            class_detail.append(
                merged[detail_columns] if "compound_name" in merged else merged
            )
            y = merged.label.to_numpy(int)
            p = merged.blend_probability_recomputed.to_numpy(float)
            for cls in ("polyol_sugar_like", "polyphenol_like", "other"):
                mask = (merged.chemical_class == cls).to_numpy()
                if mask.sum() < 3:
                    continue
                predicted_positive = (p >= LOCKED_THRESHOLD)
                # enrichment of this class among predicted positives, with CI
                def _enrich(idx, m=mask, pp=predicted_positive):
                    if pp[idx].sum() == 0 or m[idx].mean() == 0:
                        raise ValueError
                    return (m[idx] & pp[idx]).sum() / pp[idx].sum() / m[idx].mean()
                lo, hi = _boot_ci(mask, _enrich)
                class_rows.append({
                    "cohort": name, "chemical_class": cls, "n": int(mask.sum()),
                    "class_prevalence_in_cohort": float(mask.mean()),
                    "observed_positive_rate": float(y[mask].mean()),
                    "mean_blend_probability": float(p[mask].mean()),
                    "mean_blend_probability_other": float(p[~mask].mean()),
                    "predicted_positive_rate": float(predicted_positive[mask].mean()),
                    "enrichment_among_predicted_positives": float(
                        (mask & predicted_positive).sum()
                        / max(predicted_positive.sum(), 1)
                        / max(mask.mean(), 1e-9)
                    ),
                    "enrichment_ci_low": lo, "enrichment_ci_high": hi,
                    "mean_p_paper_svm": float(merged.loc[mask, "probability_paper_svm"].mean()),
                    "mean_p_tanimoto_svc": float(
                        merged.loc[mask, "probability_tanimoto_svc"].mean()
                    ),
                    "mean_p_tabpfn_v2": float(merged.loc[mask, "probability_tabpfn_v2"].mean()),
                })
        _write_csv(tmp / "J_chemical_class_analysis.csv", pd.DataFrame(class_rows))
        _write_csv(
            tmp / "J_chemical_class_per_compound.csv",
            pd.concat(class_detail, ignore_index=True),
        )

        # ---- manifest ----------------------------------------------------------
        manifest = {
            "schema_version": f"{SCHEMA}.run_manifest.v1",
            "run_id": run_id, "protocol_sha256": protocol_sha256,
            "paper_split_sha256": split_sha256,
            "supersedes": "blend_xai_20260820 (contained four overclaims; removed)",
            "explained_model": "locked blend 0.10/0.60/0.30",
            "locked_component_state_sha256": bundle["component_state_sha256"],
            "locked_threshold": LOCKED_THRESHOLD,
            "decomposition_scale_correction": {
                "component_level_raw_blend": "exact, 100% of weight",
                "within_component_margin": "exact in margin space only, 70% of weight",
                "final_probability": "NOT additively attributable (Platt/expit sigmoid)",
            },
            "locked_model_is_refit_or_reweighted": False,
            "threshold_is_changed": False,
            "test_or_external_labels_used_to_build_explanations": False,
            "bootstrap_resamples": BOOTSTRAP, "seed": SEED,
            "sealed_inputs_sha256": {
                k: sha256_file(root / v["path"]) for k, v in sealed.items()
            },
        }
        atomic_write_json(tmp / "RUN_MANIFEST.json", manifest)
        atomic_write_json(tmp / "COMPLETED.json", {
            "schema_version": f"{SCHEMA}.completed.v1", "status": "COMPLETE", "run_id": run_id,
            "run_manifest_sha256": sha256_file(tmp / "RUN_MANIFEST.json"),
            "artifact_hashes": {
                str(p.relative_to(tmp)): sha256_file(p)
                for p in sorted(tmp.rglob("*")) if p.is_file() and p.name != "COMPLETED.json"
            },
        })
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
    run(root=a.root, config_path=a.config, positive_path=a.positive,
        negative_path=a.negative, run_id=a.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
