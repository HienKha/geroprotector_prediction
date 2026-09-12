"""Post-lock explainability for the equal-thirds blends, TabFM vs TabPFN-v2.

`blend_xai_v2` explained the locked 0.10/0.60/0.30 TabPFN-v2 blend.  This module
runs the same suite (sections A-J) on the equal-thirds SVM/Tanimoto/TabFM blend,
and -- so the comparison is not confounded by the weight vector -- on the
equal-thirds SVM/Tanimoto/TabPFN-v2 blend as well.  Section K then scores the
two head to head on insight-quality criteria fixed in the protocol before any
result was produced.

Every correction made in v2 is carried over verbatim:

  * three-level decomposition, with the final probability explicitly NOT
    additively attributable (Platt / expit sigmoids);
  * the libsvm binary Platt form 1/(1+exp(A*f+B)), not expit(A*f+B);
  * coefficient stability by subsampling WITHOUT replacement, with any
    non-converged refit raising rather than being reported;
  * applicability domain reported prevalence-adjusted, multi-cutoff, with CIs,
    because AP's baseline is the subgroup prevalence.

One scale correction is specific to v3 and is stated rather than glossed: moving
from 0.10/0.60/0.30 to equal thirds LOWERS the analytically decomposable share
of the blend weight from 0.70 to 0.6667, because the Tanimoto slot -- which has
an exact kernel-margin decomposition -- lost weight to the foundation model.
Equal thirds buys accuracy parity, not more explainability.

Everything is DESCRIPTIVE and POST-LOCK: no model is fitted, refitted,
recalibrated or reweighted, and no threshold is moved.  Thresholds come from the
324-row D1 train OOF only; no test or external label enters any explanation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)
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


class BlendXAIv3Error(RuntimeError):
    """Raised when a sealed input or an analysis invariant fails."""


SCHEMA = "geroprotector.blend_xai_v3"
EQUAL_THIRDS = np.full(3, 1.0 / 3.0)
D1_ENDPOINT = "paper_binary"
DRUGAGE_ENDPOINT = "significant_positive_retrieval_background_not_certified_negative"
AGEXTEND_ENDPOINT = "published_independent_table6_binary"
BOOTSTRAP = 2000
SEED = 20260821

CLASS_SMARTS = {
    "aliphatic_hydroxyl": "[CX4][OX2H]",
    "aromatic_hydroxyl": "[c][OX2H]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
}
POLYOL_MIN_ALIPHATIC_OH = 3
POLYPHENOL_MIN_AROMATIC_OH = 2

COHORT_ORDER = ("d1_paper_test", "drugage", "agextend")


def _regular_file(path: Path, role: str, expected: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise BlendXAIv3Error(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected is not None and sha256_file(resolved) != expected:
        raise BlendXAIv3Error(f"{role} SHA256 differs from the protocol lock")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    text = _regular_file(path, "xAI v3 protocol").read_text(encoding="utf-8")
    protocol = yaml.safe_load(text)
    if not isinstance(protocol, dict) or protocol.get("schema_version") != f"{SCHEMA}.protocol.v1":
        raise BlendXAIv3Error("Unknown xAI v3 protocol schema")
    contract = protocol.get("analysis_contract", {})
    for key in (
        "locked_model_is_refit_or_reweighted",
        "threshold_is_changed",
        "test_or_external_labels_used_to_build_explanations",
    ):
        if contract.get(key) is not False:
            raise BlendXAIv3Error(f"Analysis contract differs at {key}")
    for name, spec in protocol["blends"].items():
        if not np.allclose(spec["weights"], EQUAL_THIRDS, rtol=0.0, atol=1e-15):
            raise BlendXAIv3Error(f"Blend {name} is not equal thirds")
    if protocol.get("immutability", {}).get("existing_run_directories_are_read_only") is not True:
        raise BlendXAIv3Error("Immutability contract differs")
    for record in protocol.get("sealed_inputs", {}).values():
        _regular_file(root / record["path"], f"sealed input {record['path']}", record["sha256"])
    return protocol, canonical_sha256(protocol)


# ------------------------------------------------------------------ statistics


def _boot_ci(values, statistic, n_resamples: int = BOOTSTRAP, seed: int = SEED):
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
        return (float("nan"), float("nan"))
    return tuple(float(v) for v in np.percentile(draws, [2.5, 97.5]))


def _metric_with_ci(y: np.ndarray, p: np.ndarray, seed: int = SEED) -> dict[str, Any]:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    prevalence = float(y.mean())
    both = len(set(y)) == 2

    def _guard(fn):
        def inner(idx):
            yb = y[idx]
            if len(set(yb)) < 2:
                raise ValueError
            return fn(yb, p[idx])
        return inner

    ap = float(average_precision_score(y, p)) if y.sum() else float("nan")
    ap_lo, ap_hi = _boot_ci(y, _guard(average_precision_score), seed=seed) if both else (np.nan,)*2
    adj_lo, adj_hi = (
        _boot_ci(y, _guard(lambda a, b: average_precision_score(a, b) - a.mean()), seed=seed)
        if both else (np.nan,) * 2
    )
    auroc = float(roc_auc_score(y, p)) if both else float("nan")
    au_lo, au_hi = _boot_ci(y, _guard(roc_auc_score), seed=seed) if both else (np.nan,) * 2
    br_lo, br_hi = _boot_ci(y, lambda idx: brier_score_loss(y[idx], p[idx]), seed=seed)
    return {
        "n": len(y), "n_positive": int(y.sum()), "n_negative": int((1 - y).sum()),
        "positive_prevalence": prevalence,
        "auprc": ap, "auprc_ci_low": ap_lo, "auprc_ci_high": ap_hi,
        "auprc_minus_prevalence": ap - prevalence if np.isfinite(ap) else float("nan"),
        "auprc_minus_prevalence_ci_low": adj_lo, "auprc_minus_prevalence_ci_high": adj_hi,
        "auroc": auroc, "auroc_ci_low": au_lo, "auroc_ci_high": au_hi,
        "brier": float(brier_score_loss(y, p)), "brier_ci_low": br_lo, "brier_ci_high": br_hi,
    }


def _calibration(y: np.ndarray, p: np.ndarray, bins: int = 10, seed: int = SEED) -> dict[str, Any]:
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
        rows.append({"bin_low": float(low), "bin_high": float(high), "n": int(mask.sum()),
                     "mean_predicted_probability": mean_p,
                     "observed_positive_fraction": observed, "gap": observed - mean_p})

    def _fit(idx):
        yb, pb = y[idx], p[idx]
        if len(set(yb)) < 2:
            raise ValueError
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        model.fit(logit(pb).reshape(-1, 1), yb)
        return float(model.coef_[0][0]), float(model.intercept_[0])

    slope = intercept = float("nan")
    slope_ci = intercept_ci = (float("nan"), float("nan"))
    if len(set(y)) == 2:
        slope, intercept = _fit(np.arange(len(y)))
        rng = np.random.default_rng(seed)
        slopes, intercepts = [], []
        for _ in range(500):
            try:
                s, b = _fit(rng.integers(0, len(y), len(y)))
            except Exception:
                continue
            slopes.append(s)
            intercepts.append(b)
        if slopes:
            slope_ci = tuple(float(v) for v in np.percentile(slopes, [2.5, 97.5]))
            intercept_ci = tuple(float(v) for v in np.percentile(intercepts, [2.5, 97.5]))
    return {
        "bins": rows, "expected_calibration_error": float(ece),
        "calibration_slope": slope, "calibration_slope_ci_low": slope_ci[0],
        "calibration_slope_ci_high": slope_ci[1], "calibration_intercept": intercept,
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
            rows.append({k: 0 for k in patterns} | {"chemical_class": "unparsable", "scaffold": ""})
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


def _mcc_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    # Full precision, unrounded: the sealed runs select over the exact predicted
    # probabilities, so rounding here would break threshold parity with them.
    candidates = np.unique(np.concatenate([np.asarray(p, float), [0.5]]))
    best, chosen = -2.0, 0.5
    for t in candidates:
        m = matthews_corrcoef(y, (p >= t).astype(int))
        if m > best:
            best, chosen = float(m), float(t)
    return chosen, best


# ------------------------------------------------------------------------ run


def run(
    *, root: Path, config_path: Path, positive_path: Path, negative_path: Path, run_id: str
) -> Path:
    if not re.fullmatch(r"blend_xai_v3_[a-z0-9_.-]+", run_id):
        raise BlendXAIv3Error("RUN_ID must start with blend_xai_v3_")
    root = root.resolve()
    protocol, protocol_sha256 = load_protocol(root, config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise BlendXAIv3Error(f"Run directory already exists: {destination}")
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
    train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    labels = frame["label"].to_numpy(dtype=int)
    smiles, _ = _validated_raw_smiles(frame)
    features = _features(frame, smiles, fixed_protocol)
    fit_indices = np.asarray(bundle["fit_paper_indices"], dtype=int)
    sealed_bits = np.asarray(bundle["tanimoto_train_bits"], dtype=np.uint8)
    if not np.array_equal(features["morgan"][fit_indices], sealed_bits):
        raise BlendXAIv3Error("Rebuilt Morgan bits differ from the sealed bundle")
    by_row = frame.set_index("paper_row_index")

    # ---- assemble both blends' component streams, row-aligned -----------------
    tabpfn_oof = pd.read_csv(root / sealed["weighted_train_oof"]["path"])
    tabpfn_oof = tabpfn_oof.set_index("paper_row_index").loc[train_indices].reset_index()
    tabfm_oof = pd.read_csv(root / sealed["tabfm_train_oof"]["path"])
    tabfm_oof = tabfm_oof.set_index("paper_row_index").loc[train_indices].reset_index()
    y_train = labels[train_indices]

    tabpfn_test = pd.read_csv(root / sealed["weighted_test_components"]["path"])
    tabpfn_test = tabpfn_test.set_index("paper_row_index").loc[test_indices].reset_index()
    tabfm_test = pd.read_csv(root / sealed["tabfm_test"]["path"])
    tabfm_test = tabfm_test.set_index("paper_row_index").loc[test_indices].reset_index()
    for f in (tabpfn_test, tabfm_test):
        f["compound_name"] = by_row.loc[test_indices, "compound_name"].to_numpy()
        f["source_smiles"] = [smiles[i] for i in test_indices]
        f["label"] = labels[test_indices]

    drugage_pfn = pd.read_csv(root / sealed["drugage_scored"]["path"])
    drugage_pfn["label"] = drugage_pfn.has_significant_positive.astype(int)
    agextend_pfn = pd.read_csv(root / sealed["agextend_scored"]["path"])
    drugage_fm = pd.read_csv(root / sealed["tabfm_drugage"]["path"])
    agextend_fm = pd.read_csv(root / sealed["tabfm_agextend"]["path"])
    for a, b, name in ((drugage_pfn, drugage_fm, "drugage"), (agextend_pfn, agextend_fm, "agextend")):
        if not (a.external_id.to_numpy() == b.external_id.to_numpy()).all():
            raise BlendXAIv3Error(f"{name}: TabFM and sealed external row order differ")
        for column in ("probability_paper_svm", "probability_tanimoto_svc"):
            if not np.allclose(a[column], b[column], rtol=0.0, atol=1e-12):
                raise BlendXAIv3Error(f"{name}: shared component {column} differs between runs")

    BLENDS = {
        "eqthirds_tabfm": {
            "slot3": "tabfm",
            "cohorts": {"d1_paper_test": tabfm_test, "drugage": drugage_fm, "agextend": agextend_fm},
            "train_oof": tabfm_oof,
        },
        "eqthirds_tabpfnv2": {
            "slot3": "tabpfn_v2",
            "cohorts": {"d1_paper_test": tabpfn_test, "drugage": drugage_pfn,
                        "agextend": agextend_pfn},
            "train_oof": tabpfn_oof,
        },
    }
    endpoints = {"d1_paper_test": D1_ENDPOINT, "drugage": DRUGAGE_ENDPOINT,
                 "agextend": AGEXTEND_ENDPOINT}

    # shared Morgan bits and similarity per cohort (identical across both blends)
    bits = {"d1_paper_test": features["morgan"][test_indices]}
    for name, f in (("drugage", drugage_pfn), ("agextend", agextend_pfn)):
        bits[name] = _morgan_from_smiles(
            f.source_smiles.astype(str).tolist(), bundle["portable_contract"]
        )
        rebuilt = _tanimoto(bits[name], sealed_bits).max(axis=1)
        stored = f.maximum_tanimoto_to_fitted_train.to_numpy(float)
        if float(np.max(np.abs(rebuilt - stored))) > 1e-6:
            raise BlendXAIv3Error(f"Rebuilt {name} similarity differs from the sealed file")
    similarity_max = {c: _tanimoto(bits[c], sealed_bits).max(axis=1) for c in COHORT_ORDER}

    # ---- thresholds: 324-row train OOF MCC only -------------------------------
    sealed_thresholds = pd.read_csv(root / sealed["tabfm_thresholds"]["path"])
    thresholds, threshold_rows = {}, []
    for name, spec in BLENDS.items():
        columns = ["probability_paper_svm", "probability_tanimoto_svc",
                   f"probability_{spec['slot3']}"]
        spec["columns"] = columns
        oof = spec["train_oof"][columns].to_numpy(float) @ EQUAL_THIRDS
        spec["train_oof_blend"] = oof
        value, mcc = _mcc_threshold(y_train, oof)
        expected = protocol["blends"][name].get("expected_train_oof_mcc_threshold")
        if expected is not None and abs(value - float(expected)) > 1e-12:
            raise BlendXAIv3Error(
                f"{name}: re-derived threshold {value!r} differs from the sealed value {expected!r}"
            )
        thresholds[name] = value
        threshold_rows.append({"blend": name, "threshold": value,
                               "threshold_source": "full_324_d1_train_cross_fitted_oof_mcc",
                               "oof_mcc_at_selection": mcc,
                               "matches_sealed_run": expected is not None,
                               "test_or_external_labels_used": False})
        print(f"{name}: train-OOF MCC threshold = {value:.16f} (MCC {mcc:.4f})", flush=True)
    if not np.isclose(thresholds["eqthirds_tabfm"],
                      float(sealed_thresholds.set_index("model")
                            .loc["blend_equal_thirds_tabfm", "threshold"]), atol=1e-12):
        raise BlendXAIv3Error("TabFM blend threshold parity against the sealed run failed")

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".blendxai3.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "blend_thresholds.csv", pd.DataFrame(threshold_rows))

        # ================= A. selection uncertainty, TabFM vs TabPFN ===========
        # Same question v2 asked of weighted-vs-equal-thirds, now asked of the two
        # slot-3 models under identical weights, on TRAIN OOF only.
        rng = np.random.default_rng(SEED)
        fm_oof = BLENDS["eqthirds_tabfm"]["train_oof_blend"]
        pfn_oof = BLENDS["eqthirds_tabpfnv2"]["train_oof_blend"]
        rows_sel = []
        for metric, fn in (("auprc", average_precision_score), ("auroc", roc_auc_score)):
            diffs = []
            for _ in range(BOOTSTRAP):
                idx = rng.integers(0, len(y_train), len(y_train))
                if len(set(y_train[idx])) < 2:
                    continue
                diffs.append(fn(y_train[idx], fm_oof[idx]) - fn(y_train[idx], pfn_oof[idx]))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            rows_sel.append({
                "metric": metric, "basis": "train_oof_324_rows",
                "eqthirds_tabfm": float(fn(y_train, fm_oof)),
                "eqthirds_tabpfnv2": float(fn(y_train, pfn_oof)),
                "difference_tabfm_minus_tabpfnv2": float(
                    fn(y_train, fm_oof) - fn(y_train, pfn_oof)),
                "ci_low": float(lo), "ci_high": float(hi),
                "separable_at_95pct": bool(lo > 0 or hi < 0),
            })
        _write_csv(tmp / "A_selection_uncertainty_train_oof.csv", pd.DataFrame(rows_sel))

        # ================= B. three-level decomposition ========================
        svm = bundle["paper_svm"]
        coef = np.asarray(svm.coef_, float).ravel()
        b0 = float(svm.intercept_[0])
        probA, probB = float(svm.probA_[0]), float(svm.probB_[0])
        d1_paper = features["paper"][test_indices]
        margin = d1_paper @ coef + b0
        platt = 1.0 / (1.0 + np.exp(probA * margin + probB))
        sealed_p = tabfm_test.probability_paper_svm.to_numpy(float)
        train_mean = features["paper"][fit_indices].mean(axis=0)
        naive_sum = ((d1_paper - train_mean) @ coef) + (train_mean @ coef + b0)
        decomposable_weight = float(EQUAL_THIRDS[0] + EQUAL_THIRDS[1])
        atomic_write_json(tmp / "B_decomposition_levels.json", {
            "level_1_component_decomposition_of_raw_blend": {
                "exact": True, "coverage_of_blend_weight": 1.0,
                "statement": "p_blend = (p_svm + p_tanimoto + p_slot3)/3 is exact by construction",
            },
            "level_2_within_component_margin_decomposition": {
                "exact_in_margin_space": True,
                "coverage_of_blend_weight": decomposable_weight,
                "components": ["paper_svm (linear margin)", "tanimoto_svc (kernel margin)"],
                "comparison_to_locked_010_060_030_blend": {
                    "locked_blend_coverage": 0.70,
                    "equal_thirds_coverage": decomposable_weight,
                    "change": decomposable_weight - 0.70,
                    "interpretation": (
                        "equal thirds DECREASES the analytically decomposable share, "
                        "because the Tanimoto slot lost weight to the foundation model"
                    ),
                },
                "statement": (
                    "analytic decomposition exists for the DECISION MARGIN, not the probability"
                ),
            },
            "level_3_final_probability_attribution": {
                "exact": False,
                "reason": (
                    "paper_svm applies Platt scaling (probA_/probB_) and the Tanimoto stream is "
                    "expit(decision_function); p = sigmoid(margin) is nonlinear"
                ),
                "paper_svm_probA": probA, "paper_svm_probB": probB,
                "max_abs_margin_minus_platt_probability_d1_test": float(
                    np.max(np.abs(naive_sum - sealed_p))),
                "max_abs_reconstructed_platt_minus_sealed": float(np.max(np.abs(platt - sealed_p))),
                "platt_formula_used": "1/(1+exp(probA*decision+probB)) [libsvm binary]",
                "slot3_applies_platt": {"tabfm": False, "tabpfn_v2": False},
                "slot3_note": (
                    "neither foundation model applies Platt scaling; both emit a softmax "
                    "posterior directly, so slot 3 has no margin-space decomposition at all"
                ),
                "statement": (
                    "margin contributions must NOT be reported as additive attributions "
                    "of the final probability"
                ),
            },
        })

        # ================= C / F / G / H / I / J, per blend ====================
        decomposition, ablation, reliability, ad_rows = [], [], [], []
        coverage_rows, uncertainty, class_rows, class_detail = [], [], [], []
        calibration: dict[str, Any] = {}
        redundancy = []
        for blend_name, spec in BLENDS.items():
            slot3 = spec["slot3"]
            columns = spec["columns"]
            threshold = thresholds[blend_name]
            calibration[blend_name] = {}
            for cohort in COHORT_ORDER:
                f = spec["cohorts"][cohort].reset_index(drop=True)
                y = f.label.to_numpy(int)
                comp = f[columns].to_numpy(float)
                blend = comp @ EQUAL_THIRDS
                ident = (f.compound_name.astype(str) if "compound_name" in f
                         else pd.Series(f.index.astype(str)))

                # ---- C. per-compound component decomposition -----------------
                weighted = comp * EQUAL_THIRDS[None, :]
                for i in range(len(blend)):
                    decomposition.append({
                        "blend": blend_name, "cohort": cohort,
                        "compound_name": str(ident.iloc[i]), "label": int(y[i]),
                        "blend_probability": float(blend[i]),
                        "decision_at_blend_threshold": int(blend[i] >= threshold),
                        "p_paper_svm": float(comp[i, 0]),
                        "p_tanimoto_svc": float(comp[i, 1]),
                        "p_slot3": float(comp[i, 2]), "slot3_model": slot3,
                        "weighted_paper_svm": float(weighted[i, 0]),
                        "weighted_tanimoto_svc": float(weighted[i, 1]),
                        "weighted_slot3": float(weighted[i, 2]),
                        "pct_of_score_paper_svm": float(100 * weighted[i, 0] / blend[i]),
                        "pct_of_score_tanimoto_svc": float(100 * weighted[i, 1] / blend[i]),
                        "pct_of_score_slot3": float(100 * weighted[i, 2] / blend[i]),
                        "component_spread": float(comp[i].max() - comp[i].min()),
                        "all_three_components_above_0p5": bool((comp[i] > 0.5).all()),
                    })

                # ---- component redundancy ------------------------------------
                correlation = np.corrcoef(comp, rowvar=False)
                redundancy.append({
                    "blend": blend_name, "cohort": cohort, "n": len(y),
                    "r_svm_tanimoto": float(correlation[0, 1]),
                    "r_svm_slot3": float(correlation[0, 2]),
                    "r_tanimoto_slot3": float(correlation[1, 2]),
                    "mean_abs_pairwise_r": float(np.mean(np.abs(
                        [correlation[0, 1], correlation[0, 2], correlation[1, 2]]))),
                    "mean_component_spread": float((comp.max(axis=1) - comp.min(axis=1)).mean()),
                })

                # ---- F. leave-one-out ablation --------------------------------
                base = _metric_with_ci(y, blend)
                ablation.append({"blend": blend_name, "cohort": cohort,
                                 "variant": "full_blend", **base})
                for k, cname in enumerate(("paper_svm", "tanimoto_svc", slot3)):
                    keep = [j for j in range(3) if j != k]
                    p = comp[:, keep] @ np.full(2, 0.5)
                    m = _metric_with_ci(y, p)
                    ablation.append({
                        "blend": blend_name, "cohort": cohort, "variant": f"without_{cname}", **m,
                        "delta_auprc_vs_full": m["auprc"] - base["auprc"],
                        "delta_auroc_vs_full": m["auroc"] - base["auroc"],
                    })

                # ---- G. calibration ------------------------------------------
                res = _calibration(y, blend)
                calibration[blend_name][cohort] = {k: v for k, v in res.items() if k != "bins"}
                for record in res["bins"]:
                    reliability.append({"blend": blend_name, "cohort": cohort, **record})

                # ---- H. applicability domain ---------------------------------
                sim = similarity_max[cohort]
                for cutoff in protocol["applicability"]["tanimoto_cutoffs"]:
                    for stratum, mask in (("inside_ge_cutoff", sim >= cutoff),
                                          ("outside_lt_cutoff", sim < cutoff)):
                        if mask.sum() < 5 or len(set(y[mask])) < 2:
                            continue
                        ad_rows.append({
                            "blend": blend_name, "cohort": cohort,
                            "tanimoto_cutoff": float(cutoff), "stratum": stratum,
                            "coverage_fraction": float(mask.mean()),
                            **_metric_with_ci(y[mask], blend[mask]),
                        })
                for q in np.linspace(0.1, 1.0, 10):
                    k = max(int(np.ceil(q * len(sim))), 5)
                    idx = np.argsort(-sim)[:k]
                    if len(set(y[idx])) < 2:
                        continue
                    coverage_rows.append({
                        "blend": blend_name, "cohort": cohort,
                        "coverage_fraction": float(k / len(sim)), "n": int(k),
                        "min_similarity_included": float(sim[idx].min()),
                        "positive_prevalence": float(y[idx].mean()),
                        "auprc": float(average_precision_score(y[idx], blend[idx])),
                        "auprc_minus_prevalence": float(
                            average_precision_score(y[idx], blend[idx]) - y[idx].mean()),
                        "auroc": float(roc_auc_score(y[idx], blend[idx])),
                    })

                # ---- I. uncertainty / error detection -------------------------
                wrong = ((blend >= threshold).astype(int) != y).astype(int)
                if len(set(wrong)) == 2:
                    scores = {
                        "component_spread": comp.max(axis=1) - comp.min(axis=1),
                        "component_std": comp.std(axis=1),
                        "distance_from_threshold": -np.abs(blend - threshold),
                        "predictive_entropy": -(
                            blend * np.log(np.clip(blend, 1e-9, 1))
                            + (1 - blend) * np.log(np.clip(1 - blend, 1e-9, 1))),
                        "negative_max_similarity": -sim,
                    }
                    for score_name, values in scores.items():
                        def _fn(idx, v=values, w=wrong):
                            if len(set(w[idx])) < 2:
                                raise ValueError
                            return roc_auc_score(w[idx], v[idx])
                        lo, hi = _boot_ci(wrong, _fn)
                        uncertainty.append({
                            "blend": blend_name, "cohort": cohort,
                            "uncertainty_score": score_name, "n": len(y),
                            "n_errors": int(wrong.sum()),
                            "error_detection_auroc": float(roc_auc_score(wrong, values)),
                            "ci_low": lo, "ci_high": hi,
                            "better_than_chance_at_95pct": bool(lo > 0.5),
                        })

                # ---- J. chemical classes --------------------------------------
                classes = _chem_classes(f.source_smiles.astype(str).tolist())
                merged = pd.concat([f, classes], axis=1)
                merged["blend"] = blend_name
                merged["cohort"] = cohort
                merged["blend_probability"] = blend
                class_detail.append(merged[[
                    "blend", "cohort", "compound_name", "label", "chemical_class",
                    "aliphatic_hydroxyl", "aromatic_hydroxyl", "carboxylic_acid",
                    "blend_probability", *columns]])
                predicted_positive = blend >= threshold
                for cls in ("polyol_sugar_like", "polyphenol_like", "other"):
                    mask = (merged.chemical_class == cls).to_numpy()
                    if mask.sum() < 3:
                        continue

                    def _enrich(idx, m=mask, pp=predicted_positive):
                        if pp[idx].sum() == 0 or m[idx].mean() == 0:
                            raise ValueError
                        return (m[idx] & pp[idx]).sum() / pp[idx].sum() / m[idx].mean()
                    lo, hi = _boot_ci(mask, _enrich)
                    class_rows.append({
                        "blend": blend_name, "cohort": cohort, "chemical_class": cls,
                        "n": int(mask.sum()), "class_prevalence_in_cohort": float(mask.mean()),
                        "observed_positive_rate": float(y[mask].mean()),
                        "mean_blend_probability": float(blend[mask].mean()),
                        "mean_blend_probability_other": float(blend[~mask].mean()),
                        "predicted_positive_rate": float(predicted_positive[mask].mean()),
                        "enrichment_among_predicted_positives": float(
                            (mask & predicted_positive).sum()
                            / max(predicted_positive.sum(), 1) / max(mask.mean(), 1e-9)),
                        "enrichment_ci_low": lo, "enrichment_ci_high": hi,
                        "mean_p_paper_svm": float(comp[mask, 0].mean()),
                        "mean_p_tanimoto_svc": float(comp[mask, 1].mean()),
                        "mean_p_slot3": float(comp[mask, 2].mean()),
                    })
                print(f"  {blend_name} / {cohort}: sections C,F,G,H,I,J done", flush=True)

        decomposition_frame = pd.DataFrame(decomposition)
        _write_csv(tmp / "C_component_decomposition_all.csv", decomposition_frame)
        _write_csv(tmp / "C_false_positive_decomposition.csv", decomposition_frame[
            (decomposition_frame.label == 0)
            & (decomposition_frame.decision_at_blend_threshold == 1)
        ].sort_values(["blend", "blend_probability"], ascending=[True, False]))
        _write_csv(tmp / "F_component_ablation.csv", pd.DataFrame(ablation))
        atomic_write_json(tmp / "G_calibration_summary.json", calibration)
        _write_csv(tmp / "G_calibration_reliability_bins.csv", pd.DataFrame(reliability))
        _write_csv(tmp / "H_applicability_domain_prevalence_adjusted.csv", pd.DataFrame(ad_rows))
        _write_csv(tmp / "H_coverage_performance_curve.csv", pd.DataFrame(coverage_rows))
        _write_csv(tmp / "I_uncertainty_error_detection.csv", pd.DataFrame(uncertainty))
        _write_csv(tmp / "J_chemical_class_analysis.csv", pd.DataFrame(class_rows))
        _write_csv(tmp / "J_chemical_class_per_compound.csv",
                   pd.concat(class_detail, ignore_index=True))
        _write_csv(tmp / "K_component_redundancy.csv", pd.DataFrame(redundancy))

        # ================= D. SVM margin attribution (shared) ==================
        train_paper, train_y = features["paper"][fit_indices], labels[fit_indices]
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
            raise BlendXAIv3Error(
                f"{n_nonconverged} stability refits failed to converge; refusing to report "
                "coefficient stability from non-converged solutions"
            )
        boot_coefs = np.vstack(boot_coefs)
        names7 = list(bundle["paper_svm_feature_names"])
        sd = train_paper.std(axis=0)
        stability = pd.DataFrame({
            "descriptor": names7, "locked_coefficient": coef,
            "standardised_effect": coef * sd,
            "subsample_mean": boot_coefs.mean(axis=0),
            "subsample_ci_low": np.percentile(boot_coefs, 2.5, axis=0),
            "subsample_ci_high": np.percentile(boot_coefs, 97.5, axis=0),
            "sign_stability_fraction": (
                np.sign(boot_coefs) == np.sign(coef)[None, :]).mean(axis=0),
            "blend_weight_carried_equal_thirds": float(EQUAL_THIRDS[0]),
            "blend_weight_carried_locked_010_060_030": 0.10,
        }).sort_values("standardised_effect", key=np.abs, ascending=False)
        _write_csv(tmp / "D_svm_margin_attribution_stability.csv", stability)
        figure, axis = plt.subplots(figsize=(7.5, 4))
        ordered = stability.iloc[::-1]
        axis.barh(ordered.descriptor, ordered.standardised_effect, color="#4C72B0")
        axis.set_xlabel(
            "linear coefficient x train SD (MARGIN space, not a probability attribution)")
        axis.set_title("paper_svm margin attribution (weight 1/3 under equal thirds)")
        axis.axvline(0, color="black", linewidth=0.8)
        figure.tight_layout()
        figure.savefig(tmp / "D_svm_margin_attribution.svg")
        plt.close(figure)

        # ================= E. signed support-vector contributions (shared) =====
        tsvc = bundle["tanimoto_svc"]
        dual = np.asarray(tsvc.dual_coef_, float).ravel()
        support = np.asarray(tsvc.support_, int)
        train_names = by_row.loc[fit_indices, "compound_name"].to_numpy()
        neighbours = []
        for cohort in COHORT_ORDER:
            f = BLENDS["eqthirds_tabfm"]["cohorts"][cohort].reset_index(drop=True)
            ident = (f.compound_name.astype(str) if "compound_name" in f
                     else pd.Series(f.index.astype(str)))
            sim_matrix = _tanimoto(bits[cohort], sealed_bits)
            contribution = sim_matrix[:, support] * dual[None, :]
            for i in range(contribution.shape[0]):
                for kind, order in (("top_positive", np.argsort(-contribution[i])[:3]),
                                    ("top_negative", np.argsort(contribution[i])[:3])):
                    for rank, j in enumerate(order, start=1):
                        neighbours.append({
                            "cohort": cohort, "compound_name": str(ident.iloc[i]),
                            "contribution_kind": kind, "rank": rank,
                            "support_vector_compound": str(train_names[support[j]]),
                            "support_vector_label": int(train_y[support[j]]),
                            "tanimoto_similarity": float(sim_matrix[i, support[j]]),
                            "signed_dual_coefficient": float(dual[j]),
                            "signed_contribution_to_margin": float(contribution[i, j]),
                        })
        _write_csv(tmp / "E_signed_support_vector_contributions.csv", pd.DataFrame(neighbours))

        # ================= K. head-to-head insight comparison ==================
        ablation_frame = pd.DataFrame(ablation)
        uncertainty_frame = pd.DataFrame(uncertainty)
        class_frame = pd.DataFrame(class_rows)
        redundancy_frame = pd.DataFrame(redundancy)
        ad_frame = pd.DataFrame(ad_rows)
        verdicts = []

        def _verdict(criterion, cohort, fm, pfn, higher_is_better, detail=""):
            if not (np.isfinite(fm) and np.isfinite(pfn)):
                winner = "undetermined"
            elif abs(fm - pfn) < 1e-12:
                winner = "tie"
            else:
                winner = ("eqthirds_tabfm" if (fm > pfn) == higher_is_better
                          else "eqthirds_tabpfnv2")
            verdicts.append({
                "criterion": criterion, "cohort": cohort, "eqthirds_tabfm": fm,
                "eqthirds_tabpfnv2": pfn, "difference_tabfm_minus_tabpfnv2": fm - pfn,
                "higher_is_better": higher_is_better, "winner": winner, "detail": detail,
            })

        for cohort in COHORT_ORDER:
            g = ablation_frame[(ablation_frame.cohort == cohort)
                               & (ablation_frame.variant == "full_blend")].set_index("blend")
            _verdict("discrimination_auprc", cohort, float(g.loc["eqthirds_tabfm", "auprc"]),
                     float(g.loc["eqthirds_tabpfnv2", "auprc"]), True)
            _verdict("discrimination_auroc", cohort, float(g.loc["eqthirds_tabfm", "auroc"]),
                     float(g.loc["eqthirds_tabpfnv2", "auroc"]), True)
            c_fm = calibration["eqthirds_tabfm"][cohort]
            c_pfn = calibration["eqthirds_tabpfnv2"][cohort]
            _verdict("calibration_ece", cohort, c_fm["expected_calibration_error"],
                     c_pfn["expected_calibration_error"], False)
            _verdict("calibration_slope_distance_from_1", cohort,
                     abs(c_fm["calibration_slope"] - 1.0),
                     abs(c_pfn["calibration_slope"] - 1.0), False)
            _verdict("brier", cohort, c_fm["brier"], c_pfn["brier"], False)
            u = uncertainty_frame[uncertainty_frame.cohort == cohort]
            if len(u):
                fm_u = u[u.blend == "eqthirds_tabfm"]
                pfn_u = u[u.blend == "eqthirds_tabpfnv2"]
                _verdict("best_error_detection_auroc", cohort,
                         float(fm_u.error_detection_auroc.max()) if len(fm_u) else np.nan,
                         float(pfn_u.error_detection_auroc.max()) if len(pfn_u) else np.nan,
                         True, "max over the five uncertainty scores")
                _verdict("n_uncertainty_scores_beating_chance", cohort,
                         float(fm_u.better_than_chance_at_95pct.sum()) if len(fm_u) else np.nan,
                         float(pfn_u.better_than_chance_at_95pct.sum()) if len(pfn_u) else np.nan,
                         True, "95% CI lower bound above 0.5")
            r = redundancy_frame[redundancy_frame.cohort == cohort].set_index("blend")
            _verdict("component_redundancy_mean_abs_r", cohort,
                     float(r.loc["eqthirds_tabfm", "mean_abs_pairwise_r"]),
                     float(r.loc["eqthirds_tabpfnv2", "mean_abs_pairwise_r"]), False,
                     "lower correlation => components carry more distinct information")
            a = ablation_frame[(ablation_frame.cohort == cohort)
                               & (ablation_frame.variant.str.startswith("without_"))]
            _verdict("max_unique_component_contribution_auprc", cohort,
                     float(-a[a.blend == "eqthirds_tabfm"].delta_auprc_vs_full.min()),
                     float(-a[a.blend == "eqthirds_tabpfnv2"].delta_auprc_vs_full.min()), True,
                     "largest AP drop when a single component is removed")
            for cutoff in (0.40,):
                s = ad_frame[(ad_frame.cohort == cohort) & (ad_frame.tanimoto_cutoff == cutoff)]
                def _gap(blend):
                    sub = s[s.blend == blend].set_index("stratum")
                    if {"inside_ge_cutoff", "outside_lt_cutoff"} - set(sub.index):
                        return float("nan")
                    return float(sub.loc["inside_ge_cutoff", "auprc_minus_prevalence"]
                                 - sub.loc["outside_lt_cutoff", "auprc_minus_prevalence"])
                _verdict(f"applicability_domain_signal_at_{cutoff}", cohort,
                         _gap("eqthirds_tabfm"), _gap("eqthirds_tabpfnv2"), True,
                         "inside-minus-outside AP-above-prevalence; larger => AD is actionable")
            cf = class_frame[(class_frame.cohort == cohort)
                             & (class_frame.chemical_class == "polyol_sugar_like")]
            if len(cf) == 2:
                cfi = cf.set_index("blend")
                _verdict("polyol_enrichment_distance_from_1", cohort,
                         abs(float(cfi.loc["eqthirds_tabfm",
                                           "enrichment_among_predicted_positives"]) - 1.0),
                         abs(float(cfi.loc["eqthirds_tabpfnv2",
                                           "enrichment_among_predicted_positives"]) - 1.0),
                         False, "closer to 1 => less unexplained chemical-class bias")
        _verdict("analytic_attribution_coverage_of_blend_weight", "all",
                 decomposable_weight, decomposable_weight, True,
                 "identical: both blends are equal thirds with the same two analytic components")
        verdict_frame = pd.DataFrame(verdicts)
        _write_csv(tmp / "K_insight_comparison.csv", verdict_frame)
        tally = (verdict_frame[verdict_frame.winner.isin(
            ["eqthirds_tabfm", "eqthirds_tabpfnv2"])]
            .groupby(["criterion", "winner"]).size().unstack(fill_value=0).reset_index())
        _write_csv(tmp / "K_insight_comparison_tally.csv", tally)

        # ---- figure: reliability, both blends ---------------------------------
        figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        reliability_frame = pd.DataFrame(reliability)
        for axis, cohort in zip(axes, COHORT_ORDER, strict=False):
            axis.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
            for blend_name, colour in (("eqthirds_tabfm", "#C44E52"),
                                       ("eqthirds_tabpfnv2", "#4C72B0")):
                sub = reliability_frame[(reliability_frame.cohort == cohort)
                                        & (reliability_frame.blend == blend_name)]
                axis.plot(sub.mean_predicted_probability, sub.observed_positive_fraction,
                          "o-", color=colour, label=blend_name)
            e1 = calibration["eqthirds_tabfm"][cohort]["expected_calibration_error"]
            e2 = calibration["eqthirds_tabpfnv2"][cohort]["expected_calibration_error"]
            axis.set_title(f"{cohort}\nECE TabFM={e1:.3f} / TabPFN={e2:.3f}")
            axis.set_xlabel("mean predicted probability")
        axes[0].set_ylabel("observed positive fraction")
        axes[0].legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(tmp / "G_calibration_reliability.svg")
        plt.close(figure)

        # ---- manifest ---------------------------------------------------------
        manifest = {
            "schema_version": f"{SCHEMA}.run_manifest.v1",
            "run_id": run_id, "protocol_sha256": protocol_sha256,
            "paper_split_sha256": split_sha256,
            "relationship_to_blend_xai_v2": (
                "blend_xai_v2 explained the locked 0.10/0.60/0.30 TabPFN-v2 blend and remains "
                "valid and untouched. v3 explains the equal-thirds TabFM blend and runs the "
                "equal-thirds TabPFN-v2 blend alongside it so the slot-3 comparison is not "
                "confounded by the weight vector."
            ),
            "explained_blends": {
                k: {"components": ["paper_svm", "tanimoto_svc", v["slot3"]],
                    "weights": EQUAL_THIRDS.tolist(), "threshold": thresholds[k]}
                for k, v in BLENDS.items()},
            "locked_component_state_sha256": bundle["component_state_sha256"],
            "decomposition_scale_correction": {
                "component_level_raw_blend": "exact, 100% of weight",
                "within_component_margin": (
                    f"exact in margin space only, {decomposable_weight:.4f} of weight "
                    "(DOWN from 0.70 in the 0.10/0.60/0.30 blend)"),
                "final_probability": "NOT additively attributable (Platt/expit sigmoid)",
            },
            "locked_model_is_refit_or_reweighted": False,
            "threshold_is_changed": False,
            "test_or_external_labels_used_to_build_explanations": False,
            "thresholds_derived_from": "full_324_d1_train_cross_fitted_oof_mcc",
            "tabfm_threshold_parity_with_sealed_run": True,
            "external_shared_component_parity_atol": 1e-12,
            "bootstrap_resamples": BOOTSTRAP, "seed": SEED,
            "runtime": {"python": platform.python_version(), "platform": platform.platform()},
            "sealed_inputs_sha256": {k: sha256_file(root / v["path"]) for k, v in sealed.items()},
            "existing_runs_modified": False,
        }
        atomic_write_json(tmp / "RUN_MANIFEST.json", manifest)
        atomic_write_json(tmp / "COMPLETED.json", {
            "schema_version": f"{SCHEMA}.completed.v1", "status": "COMPLETE", "run_id": run_id,
            "run_manifest_sha256": sha256_file(tmp / "RUN_MANIFEST.json"),
            "artifact_hashes": {
                str(p.relative_to(tmp)): sha256_file(p)
                for p in sorted(tmp.rglob("*")) if p.is_file() and p.name != "COMPLETED.json"},
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
