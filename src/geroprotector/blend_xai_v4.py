"""Head-to-head explainability: the three-component blend versus the four-component blend.

The ablation (gb4_ablation_*) showed that dropping the Tanimoto component improves
most point estimates on D1 test and DrugAge but not under cross-validation. This
module asks whether that difference is real and whether it is supported by the
insight analyses, comparing

    TRIPLE  = (paper_svm + tabpfn_v2 + tabfm) / 3
    QUAD    = (paper_svm + tanimoto_svc + tabpfn_v2 + tabfm) / 4

on all four cohorts at a decision threshold fixed at 0.5.

Nothing is refitted. Both models are equal-weight averages of components already
fitted on the 324 D1 training rows and scored once, so both are locked models with
prespecified weights. The Tanimoto similarity used for the applicability-domain
section is a property of chemical space, not a model component, so it is available
for both models alike.

Sections
  A  head-to-head paired bootstrap, every cohort x every metric
  B  per-compound component decomposition and the compounds whose decision flips
  C  component redundancy (pairwise correlation, mean disagreement)
  D  leave-one-out component contribution within each model
  E  calibration (ECE, slope, intercept, Brier, reliability bins)
  F  error detection from five uncertainty scores
  G  applicability domain, prevalence-adjusted, multi-cutoff
  H  prespecified chemical-class enrichment
  I  verdict tally over the prespecified criteria

Every criterion and its direction of preference is fixed in the protocol before
any result is produced. No test or external label is used to choose a model, a
weight or a threshold.
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
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from geroprotector.blend_xai_v3 import _chem_classes
from geroprotector.fixed_blend_paper405 import _features, _tanimoto
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.screening_blend_altmodels import _morgan_from_smiles
from geroprotector.screening_blend_paper405 import load_locked_bundle
from geroprotector.traditional_paper405 import _read_sources, paper_split_indices


class BlendXAIv4Error(RuntimeError):
    """Raised when a sealed input, a parity proof or a contract fails."""


SCHEMA = "geroprotector.blend_xai_v4"
COMPONENTS = ("paper_svm", "tanimoto_svc", "tabpfn_v2", "tabfm")
TRIPLE_IDX = (0, 2, 3)
QUAD_IDX = (0, 1, 2, 3)
MODELS = {"triple_no_tanimoto": TRIPLE_IDX, "quad_with_tanimoto": QUAD_IDX}
THRESHOLD = 0.5
COHORTS = ("cv5_train_oof", "d1_test", "drugage", "agextend")
METRICS = ("Accuracy", "Sensitivity", "Specificity", "Kappa", "AUROC", "AUPRC",
           "Brier", "MCC", "MacroF1")
LOWER_BETTER = {"Brier"}
BOOTSTRAP = 10000
SEED = 20260822
PARITY_ATOL = 1e-12


def _regular_file(path: Path, role: str, expected: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise BlendXAIv4Error(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected is not None and sha256_file(resolved) != expected:
        raise BlendXAIv4Error(f"{role} SHA256 differs from the protocol lock")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = yaml.safe_load(_regular_file(path, "xAI v4 protocol").read_text("utf-8"))
    if not isinstance(protocol, dict) or protocol.get("schema_version") != f"{SCHEMA}.protocol.v1":
        raise BlendXAIv4Error("Unknown xAI v4 protocol schema")
    contract = protocol["contract"]
    for key in ("any_model_is_refitted", "threshold_is_selected_or_tuned",
                "test_or_external_labels_used_to_choose_model_weight_or_threshold"):
        if contract.get(key) is not False:
            raise BlendXAIv4Error(f"Contract differs at {key}")
    if float(contract["fixed_threshold"]) != THRESHOLD:
        raise BlendXAIv4Error("Threshold contract differs")
    for record in protocol["sealed_inputs"].values():
        _regular_file(root / record["path"], f"sealed input {record['path']}", record["sha256"])
    return protocol, canonical_sha256(protocol)


# ------------------------------------------------------------------- metrics


def _point(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    d = (p >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, d, labels=[0, 1]).ravel()
    return {"Accuracy": accuracy_score(y, d),
            "Sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "Specificity": tn / (tn + fp) if tn + fp else np.nan,
            "Kappa": cohen_kappa_score(y, d), "AUROC": roc_auc_score(y, p),
            "AUPRC": average_precision_score(y, p), "Brier": brier_score_loss(y, p),
            "MCC": matthews_corrcoef(y, d),
            "MacroF1": f1_score(y, d, average="macro", zero_division=0)}


def _auroc_w(y, p, W):
    o = np.argsort(p, kind="mergesort")
    ys, ps, Ws = y[o].astype(float), p[o], W[:, o]
    ends = np.flatnonzero(np.r_[np.diff(ps) != 0, True])
    cpos, cneg = np.cumsum(Ws * ys, 1)[:, ends], np.cumsum(Ws * (1 - ys), 1)[:, ends]
    gpos, gneg = np.diff(cpos, axis=1, prepend=0.0), np.diff(cneg, axis=1, prepend=0.0)
    den = cpos[:, -1] * cneg[:, -1]
    return np.divide((gpos * ((cneg - gneg) + 0.5 * gneg)).sum(1), den,
                     out=np.full(len(W), np.nan), where=den > 0)


def _ap_w(y, p, W):
    o = np.argsort(-p, kind="mergesort")
    ys, ps, Ws = y[o].astype(float), p[o], W[:, o]
    tp, tot = np.cumsum(Ws * ys, 1), np.cumsum(Ws, 1)
    ends = np.flatnonzero(np.r_[np.diff(ps) != 0, True])
    tpe, tote = tp[:, ends], tot[:, ends]
    prec = np.divide(tpe, tote, out=np.zeros_like(tpe), where=tote > 0)
    npos = tp[:, -1]
    return np.divide((np.diff(tpe, axis=1, prepend=0.0) * prec).sum(1), npos,
                     out=np.full(len(W), np.nan), where=npos > 0)


def _bundle_w(y, p, W):
    d = (p >= THRESHOLD).astype(int)
    tp = W @ ((y == 1) & (d == 1)).astype(float)
    fp = W @ ((y == 0) & (d == 1)).astype(float)
    fn = W @ ((y == 1) & (d == 0)).astype(float)
    tn = W @ ((y == 0) & (d == 0)).astype(float)
    n = W.sum(1)
    safe = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)  # noqa: E731
    acc = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n)
    f1p, f1n = safe(2 * tp, 2 * tp + fp + fn), safe(2 * tn, 2 * tn + fn + fp)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"Accuracy": acc, "Sensitivity": safe(tp, tp + fn), "Specificity": safe(tn, tn + fp),
            "Kappa": safe(acc - pe, 1 - pe), "AUROC": _auroc_w(y, p, W), "AUPRC": _ap_w(y, p, W),
            "Brier": (W @ ((p - y) ** 2)) / n, "MCC": safe(tp * tn - fp * fn, den),
            "MacroF1": (f1p + f1n) / 2}


def _holm(p):
    p = np.asarray(p, float); o = np.argsort(p); a = np.empty_like(p); r = 0.0
    for i, ix in enumerate(o):
        r = max(r, (len(p) - i) * p[ix]); a[ix] = min(r, 1.0)
    return a


def _calibration(y, p, bins=10, seed=SEED):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows, ece = [], 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        mp, obs = float(p[mask].mean()), float(y[mask].mean())
        ece += (mask.sum() / len(y)) * abs(mp - obs)
        rows.append({"bin_low": float(lo), "bin_high": float(hi), "n": int(mask.sum()),
                     "mean_predicted": mp, "observed_fraction": obs, "gap": obs - mp})

    def fit(idx):
        yb, pb = y[idx], p[idx]
        if len(set(yb)) < 2:
            raise ValueError
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        m.fit(logit(pb).reshape(-1, 1), yb)
        return float(m.coef_[0][0]), float(m.intercept_[0])

    slope, intercept = fit(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(500):
        try:
            draws.append(fit(rng.integers(0, len(y), len(y))))
        except Exception:
            continue
    lo_s, hi_s = (np.percentile([d[0] for d in draws], [2.5, 97.5]) if draws else (np.nan,) * 2)
    return {"bins": rows, "ece": float(ece), "slope": slope, "slope_ci_low": float(lo_s),
            "slope_ci_high": float(hi_s), "intercept": intercept,
            "brier": float(brier_score_loss(y, p))}


# ----------------------------------------------------------------------- run


def run(*, root: Path, config_path: Path, positive_path: Path, negative_path: Path,
        run_id: str) -> Path:
    if not re.fullmatch(r"blend_xai_v4_[a-z0-9_.-]+", run_id):
        raise BlendXAIv4Error("RUN_ID must start with blend_xai_v4_")
    root = root.resolve()
    protocol, protocol_sha256 = load_protocol(root, config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise BlendXAIv4Error(f"Run directory already exists: {destination}")
    sealed = protocol["sealed_inputs"]
    columns = [f"probability_{c}" for c in COMPONENTS]

    bundle = load_locked_bundle(root / sealed["screening_bundle"]["path"],
                                expected_artifact_sha256=sealed["screening_bundle"]["sha256"])
    sealed_bits = np.asarray(bundle["tanimoto_train_bits"], dtype=np.uint8)

    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    frame, _audit = _read_sources(
        _regular_file(positive_path, "positive source", protocol["sources"]["positive_sha256"]),
        _regular_file(negative_path, "negative source", protocol["sources"]["negative_sha256"]),
        traditional)
    train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    labels = frame["label"].to_numpy(int)
    smiles, _ = _validated_raw_smiles(frame)
    features = _features(frame, smiles, fixed_protocol)
    by_row = frame.set_index("paper_row_index")

    data: dict[str, dict[str, Any]] = {}
    oof = pd.read_csv(root / sealed["quad_train_oof"]["path"])
    data["cv5_train_oof"] = {
        "y": oof.label.to_numpy(int), "panel": oof[columns].to_numpy(float),
        "bits": features["morgan"][train_indices],
        "smiles": [smiles[i] for i in train_indices],
        "name": by_row.loc[train_indices, "compound_name"].astype(str).to_numpy()}
    test = pd.read_csv(root / sealed["quad_d1_test"]["path"])
    data["d1_test"] = {
        "y": test.label.to_numpy(int), "panel": test[columns].to_numpy(float),
        "bits": features["morgan"][test_indices],
        "smiles": [smiles[i] for i in test_indices],
        "name": by_row.loc[test_indices, "compound_name"].astype(str).to_numpy()}
    for cohort in ("drugage", "agextend"):
        d = pd.read_csv(root / sealed[f"quad_{cohort}"]["path"])
        bits = _morgan_from_smiles(d.source_smiles.astype(str).tolist(),
                                   bundle["portable_contract"])
        data[cohort] = {"y": d.label.to_numpy(int), "panel": d[columns].to_numpy(float),
                        "bits": bits, "smiles": d.source_smiles.astype(str).tolist(),
                        "name": d.compound_name.astype(str).to_numpy()}
    for cohort, payload in data.items():
        payload["similarity"] = _tanimoto(payload["bits"], sealed_bits).max(axis=1)
        for key, idx in MODELS.items():
            payload[key] = payload["panel"][:, list(idx)] @ np.full(len(idx), 1 / len(idx))
        print(f"{cohort}: n={len(payload['y'])}, pos={int(payload['y'].sum())}", flush=True)

    # parity against the sealed quad column
    drift = float(np.max(np.abs(
        data["d1_test"]["quad_with_tanimoto"] - test.blend_eq_quarters.to_numpy(float))))
    if drift > PARITY_ATOL:
        raise BlendXAIv4Error(f"Rebuilt quad blend differs from the sealed column: {drift!r}")
    print(f"parity of rebuilt quad vs sealed column: {drift:.3e}", flush=True)

    rng = np.random.default_rng(SEED)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".blendxai4.work-", dir=destination.parent))
    try:
        # ---- A. head-to-head -------------------------------------------------
        head, calib_rows, reliability, unc, ad_rows, cls_rows = [], [], [], [], [], []
        redundancy, loo, flips, decomposition = [], [], [], []
        calibration: dict[str, dict[str, Any]] = {}
        for cohort in COHORTS:
            payload = data[cohort]
            y, n = payload["y"], len(payload["y"])
            W = rng.multinomial(n, np.full(n, 1 / n), size=BOOTSTRAP).astype(float)
            W = W[(W @ (y == 1).astype(float) > 0) & (W @ (y == 0).astype(float) > 0)]
            boot = {k: _bundle_w(y, payload[k], W) for k in MODELS}
            point = {k: _point(y, payload[k]) for k in MODELS}
            one = np.ones((1, n))
            for k in MODELS:
                chk = _bundle_w(y, payload[k], one)
                for m in METRICS:
                    assert abs(chk[m][0] - point[k][m]) < 1e-9, (cohort, k, m)
            raw = {}
            for metric in METRICS:
                d = boot["triple_no_tanimoto"][metric] - boot["quad_with_tanimoto"][metric]
                if metric in LOWER_BETTER:
                    d = -d                      # positive => triple better
                lo, hi = np.percentile(d, [2.5, 97.5])
                raw[metric] = min(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0)
                delta = point["triple_no_tanimoto"][metric] - point["quad_with_tanimoto"][metric]
                head.append({"cohort": cohort, "n": n, "metric": metric,
                             "triple": point["triple_no_tanimoto"][metric],
                             "quad": point["quad_with_tanimoto"][metric],
                             "delta_favouring_triple": -delta if metric in LOWER_BETTER else delta,
                             "ci_low": lo, "ci_high": hi, "p": raw[metric]})
            for metric, q in zip(METRICS, _holm([raw[m] for m in METRICS])):
                for r in head:
                    if r["cohort"] == cohort and r["metric"] == metric:
                        r["q"] = q

            # ---- B. decomposition + decision flips ---------------------------
            for key, idx in MODELS.items():
                weighted = payload["panel"][:, list(idx)] * (1 / len(idx))
                blend = payload[key]
                for i in range(n):
                    decomposition.append({
                        "cohort": cohort, "model": key, "compound_name": payload["name"][i],
                        "label": int(y[i]), "blend_probability": float(blend[i]),
                        "decision": int(blend[i] >= THRESHOLD),
                        **{f"weighted_{COMPONENTS[c]}": float(weighted[i, j])
                           for j, c in enumerate(idx)},
                        "component_spread": float(payload["panel"][i, list(idx)].max()
                                                  - payload["panel"][i, list(idx)].min())})
            dt = (payload["triple_no_tanimoto"] >= THRESHOLD).astype(int)
            dq = (payload["quad_with_tanimoto"] >= THRESHOLD).astype(int)
            mask = dt != dq
            for i in np.flatnonzero(mask):
                flips.append({
                    "cohort": cohort, "compound_name": payload["name"][i], "label": int(y[i]),
                    "p_triple": float(payload["triple_no_tanimoto"][i]),
                    "p_quad": float(payload["quad_with_tanimoto"][i]),
                    "triple_decision": int(dt[i]), "quad_decision": int(dq[i]),
                    "triple_correct": bool(dt[i] == y[i]), "quad_correct": bool(dq[i] == y[i]),
                    "p_tanimoto": float(payload["panel"][i, 1]),
                    "max_tanimoto_to_train": float(payload["similarity"][i])})

            # ---- C. redundancy ----------------------------------------------
            for key, idx in MODELS.items():
                sub = payload["panel"][:, list(idx)]
                corr = np.corrcoef(sub, rowvar=False)
                off = corr[np.triu_indices(len(idx), k=1)]
                redundancy.append({
                    "cohort": cohort, "model": key, "n_components": len(idx),
                    "mean_abs_pairwise_r": float(np.mean(np.abs(off))),
                    "max_abs_pairwise_r": float(np.max(np.abs(off))),
                    "mean_component_spread": float((sub.max(1) - sub.min(1)).mean())})

            # ---- D. leave-one-out -------------------------------------------
            for key, idx in MODELS.items():
                base = _point(y, payload[key])
                for drop in idx:
                    keep = [c for c in idx if c != drop]
                    p = payload["panel"][:, keep] @ np.full(len(keep), 1 / len(keep))
                    m = _point(y, p)
                    loo.append({"cohort": cohort, "model": key,
                                "component_removed": COMPONENTS[drop],
                                **{f"delta_{k}": (base[k] - m[k]) * (-1 if k in LOWER_BETTER else 1)
                                   for k in METRICS}})

            # ---- E. calibration ---------------------------------------------
            calibration.setdefault(cohort, {})
            for key in MODELS:
                res = _calibration(y, payload[key])
                calibration[cohort][key] = {k: v for k, v in res.items() if k != "bins"}
                calib_rows.append({"cohort": cohort, "model": key,
                                   **{k: v for k, v in res.items() if k != "bins"}})
                for b in res["bins"]:
                    reliability.append({"cohort": cohort, "model": key, **b})

            # ---- F. error detection -----------------------------------------
            for key, idx in MODELS.items():
                p = payload[key]
                wrong = ((p >= THRESHOLD).astype(int) != y).astype(int)
                if len(set(wrong)) < 2:
                    continue
                sub = payload["panel"][:, list(idx)]
                scores = {"component_spread": sub.max(1) - sub.min(1),
                          "component_std": sub.std(1),
                          "distance_from_threshold": -np.abs(p - THRESHOLD),
                          "predictive_entropy": -(p * np.log(np.clip(p, 1e-9, 1))
                                                  + (1 - p) * np.log(np.clip(1 - p, 1e-9, 1))),
                          "negative_max_similarity": -payload["similarity"]}
                for sname, values in scores.items():
                    auc = float(roc_auc_score(wrong, values))
                    draws = []
                    for _ in range(2000):
                        i = rng.integers(0, n, n)
                        if len(set(wrong[i])) < 2:
                            continue
                        draws.append(roc_auc_score(wrong[i], values[i]))
                    lo, hi = np.percentile(draws, [2.5, 97.5])
                    unc.append({"cohort": cohort, "model": key, "uncertainty_score": sname,
                                "n_errors": int(wrong.sum()), "error_detection_auroc": auc,
                                "ci_low": float(lo), "ci_high": float(hi),
                                "beats_chance": bool(lo > 0.5)})

            # ---- G. applicability domain -------------------------------------
            for key in MODELS:
                p, sim = payload[key], payload["similarity"]
                for cutoff in protocol["applicability"]["tanimoto_cutoffs"]:
                    for stratum, m in (("inside_ge_cutoff", sim >= cutoff),
                                       ("outside_lt_cutoff", sim < cutoff)):
                        if m.sum() < 5 or len(set(y[m])) < 2:
                            continue
                        ap = average_precision_score(y[m], p[m])
                        ad_rows.append({"cohort": cohort, "model": key,
                                        "tanimoto_cutoff": float(cutoff), "stratum": stratum,
                                        "n": int(m.sum()), "coverage": float(m.mean()),
                                        "prevalence": float(y[m].mean()), "auprc": float(ap),
                                        "auprc_minus_prevalence": float(ap - y[m].mean()),
                                        "auroc": float(roc_auc_score(y[m], p[m]))})

            # ---- H. chemical classes -----------------------------------------
            classes = _chem_classes(payload["smiles"])
            for key in MODELS:
                p = payload[key]
                predicted = p >= THRESHOLD
                for cls in ("polyol_sugar_like", "polyphenol_like", "other"):
                    m = (classes.chemical_class == cls).to_numpy()
                    if m.sum() < 3:
                        continue
                    cls_rows.append({
                        "cohort": cohort, "model": key, "chemical_class": cls,
                        "n": int(m.sum()), "observed_positive_rate": float(y[m].mean()),
                        "cohort_prevalence": float(y.mean()),
                        "label_justified_enrichment": float(y[m].mean() / max(y.mean(), 1e-9)),
                        "mean_blend_probability": float(p[m].mean()),
                        "predicted_positive_rate": float(predicted[m].mean()),
                        "enrichment_among_predicted_positives": float(
                            (m & predicted).sum() / max(predicted.sum(), 1)
                            / max(m.mean(), 1e-9))})
            print(f"  {cohort}: sections A-H done", flush=True)

        head_frame = pd.DataFrame(head)
        _write_csv(tmp / "A_head_to_head.csv", head_frame)
        _write_csv(tmp / "B_component_decomposition.csv", pd.DataFrame(decomposition))
        _write_csv(tmp / "B_decision_flips.csv", pd.DataFrame(flips))
        _write_csv(tmp / "C_component_redundancy.csv", pd.DataFrame(redundancy))
        _write_csv(tmp / "D_leave_one_out.csv", pd.DataFrame(loo))
        _write_csv(tmp / "E_calibration_summary.csv", pd.DataFrame(calib_rows))
        _write_csv(tmp / "E_reliability_bins.csv", pd.DataFrame(reliability))
        _write_csv(tmp / "F_error_detection.csv", pd.DataFrame(unc))
        _write_csv(tmp / "G_applicability_domain.csv", pd.DataFrame(ad_rows))
        _write_csv(tmp / "H_chemical_class.csv", pd.DataFrame(cls_rows))

        # ---- I. verdict ------------------------------------------------------
        verdicts = []
        for cohort in COHORTS:
            for metric in METRICS:
                r = head_frame[(head_frame.cohort == cohort)
                               & (head_frame.metric == metric)].iloc[0]
                verdicts.append({"cohort": cohort, "criterion": f"discrimination_{metric}",
                                 "triple": r.triple, "quad": r.quad,
                                 "winner": ("triple" if r.delta_favouring_triple > 0
                                            else "quad" if r.delta_favouring_triple < 0 else "tie"),
                                 "significant": bool(r.q < 0.05)})
            c = calibration[cohort]
            for name, lower in (("ece", True), ("brier", True)):
                t, q = c["triple_no_tanimoto"][name], c["quad_with_tanimoto"][name]
                verdicts.append({"cohort": cohort, "criterion": f"calibration_{name}",
                                 "triple": t, "quad": q,
                                 "winner": "triple" if (t < q) == lower else "quad",
                                 "significant": False})
            t = abs(c["triple_no_tanimoto"]["slope"] - 1)
            q = abs(c["quad_with_tanimoto"]["slope"] - 1)
            verdicts.append({"cohort": cohort, "criterion": "calibration_slope_distance_from_1",
                             "triple": t, "quad": q,
                             "winner": "triple" if t < q else "quad", "significant": False})
            u = pd.DataFrame(unc)
            u = u[u.cohort == cohort]
            if len(u):
                t = u[u.model == "triple_no_tanimoto"].error_detection_auroc.max()
                q = u[u.model == "quad_with_tanimoto"].error_detection_auroc.max()
                verdicts.append({"cohort": cohort, "criterion": "best_error_detection_auroc",
                                 "triple": t, "quad": q,
                                 "winner": "triple" if t > q else "quad", "significant": False})
            rr = pd.DataFrame(redundancy)
            rr = rr[rr.cohort == cohort].set_index("model")
            t = rr.loc["triple_no_tanimoto", "mean_abs_pairwise_r"]
            q = rr.loc["quad_with_tanimoto", "mean_abs_pairwise_r"]
            verdicts.append({"cohort": cohort, "criterion": "component_redundancy",
                             "triple": t, "quad": q,
                             "winner": "triple" if t < q else "quad", "significant": False})
        verdict_frame = pd.DataFrame(verdicts)
        _write_csv(tmp / "I_verdicts.csv", verdict_frame)
        tally = verdict_frame.groupby(["cohort", "winner"]).size().unstack(fill_value=0)
        _write_csv(tmp / "I_verdict_tally.csv", tally.reset_index())

        manifest = {
            "schema_version": f"{SCHEMA}.run_manifest.v1", "run_id": run_id,
            "protocol_sha256": protocol_sha256, "paper_split_sha256": split_sha256,
            "compared_models": {
                "triple_no_tanimoto": ["paper_svm", "tabpfn_v2", "tabfm"],
                "quad_with_tanimoto": list(COMPONENTS)},
            "weights": "equal within each model", "fixed_threshold": THRESHOLD,
            "any_model_is_refitted": False, "threshold_is_selected_or_tuned": False,
            "test_or_external_labels_used_to_choose_model_weight_or_threshold": False,
            "rebuilt_quad_vs_sealed_max_abs_difference": drift,
            "bootstrap_resamples": BOOTSTRAP, "seed": SEED,
            "runtime": {"python": platform.python_version(), "platform": platform.platform()},
            "sealed_inputs_sha256": {k: sha256_file(root / v["path"])
                                     for k, v in sealed.items()},
            "existing_runs_modified": False,
        }
        atomic_write_json(tmp / "RUN_MANIFEST.json", manifest)
        atomic_write_json(tmp / "COMPLETED.json", {
            "schema_version": f"{SCHEMA}.completed.v1", "status": "COMPLETE", "run_id": run_id,
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
    run(root=a.root, config_path=a.config, positive_path=a.positive,
        negative_path=a.negative, run_id=a.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
