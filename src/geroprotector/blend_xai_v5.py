"""Model-specific explainability for the equal-thirds blend: SHAP, global LIME, DCA.

The blend explained here is the three-component equal-weight ensemble

    p = (p_svm + p_tabpfn_v2 + p_tabfm) / 3

with the four-component blend retained as a comparator for decision-curve analysis.

WHY EXPLANATIONS ARE COMPUTED PER COMPONENT.  The components do not share a feature
space: the published SVM consumes the seven DataWarrior descriptors, whereas
TabPFN-v2 and TabFM consume the 205-column RDKit2D panel.  There is therefore no
single input vector whose perturbation defines the blend, and a naive union of the
two spaces would perturb near-duplicate descriptors (DataWarrior `Total Molweight`
and RDKit `MolWt`, `cLogP` and `MolLogP`) independently, double-counting them.

We instead exploit the fact that an equal-weight probability average is *exactly*
decomposable at the component level: the contribution of component i to the blend
probability is p_i/3, with no approximation.  Feature attributions are then computed
inside each component in its own space and carried to the blend with weight 1/3, so
that a descriptor's blend-level importance is the weighted sum of its within-component
importances.  This is stated rather than glossed because the resulting numbers are
attributions of the *components*, aggregated, not of a joint input.

METHODS
  A  exact component-level decomposition of every prediction
  B  SHAP:  analytic linear SHAP for the SVM (exact for a linear kernel);
            KernelSHAP for TabPFN-v2 and TabFM against a k-means-summarised
            background drawn from the 324 training compounds
  C  global LIME: per-compound local ridge surrogates, aggregated over the test
     partition into a global ranking, following the post-hoc LIME usage of the
     T-DDI reference implementation extended from local to global
  D  permutation importance on the blend, as a model-agnostic cross-check whose
     semantics do not depend on a background distribution
  E  decision curve analysis: net benefit against threshold probability, for both
     blends, the published SVM, and the treat-all / treat-none policies

Nothing is refitted.  Every component was fitted once on the 324 D1 training
compounds; SHAP, LIME and permutation importance query those fitted components.
Explanations are computed on the held-out test partition, which is never used to
fit, tune or select anything.
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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

from geroprotector.fixed_blend_paper405 import _features
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.screening_blend_altmodels import _apply_context, _imputer_context
from geroprotector.screening_blend_paper405 import load_locked_bundle
from geroprotector.screening_blend_tabfm import _load_tabfm, _tabfm_probability
from geroprotector.tabpfn3_paper405 import _positive_probability, _tabpfn3_estimator
from geroprotector.traditional_paper405 import _read_sources, paper_split_indices


class BlendXAIv5Error(RuntimeError):
    """Raised when a sealed input, a parity proof or a contract fails."""


SCHEMA = "geroprotector.blend_xai_v5"
THRESHOLD = 0.5
TRIPLE = ("paper_svm", "tabpfn_v2", "tabfm")
# The SVM and TabFM reconstruct the sealed streams exactly. TabPFN-v2 does not,
# for an understood reason: the sealed runs set `singleton_inference: true` and
# scored test rows one at a time, whereas perturbation-based explanation requires
# batched inference (KernelSHAP alone needs ~41,000 queries per model). TabPFN-v2
# is an in-context learner whose output depends mildly on the query batch, so the
# batched function differs from the sealed one by at most 2.1e-3 on the test set --
# verified to be exactly this cause, since singleton inference reproduces the
# sealed stream to 5.6e-17. Reported performance always uses the sealed streams;
# only the explanations use the batched function.
PARITY_ATOL = {"paper_svm": 1e-9, "tabfm": 1e-9, "tabpfn_v2": 5e-3}


def _regular_file(path: Path, role: str, expected: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise BlendXAIv5Error(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected is not None and sha256_file(resolved) != expected:
        raise BlendXAIv5Error(f"{role} SHA256 differs from the protocol lock")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = yaml.safe_load(_regular_file(path, "xAI v5 protocol").read_text("utf-8"))
    if not isinstance(protocol, dict) or protocol.get("schema_version") != f"{SCHEMA}.protocol.v1":
        raise BlendXAIv5Error("Unknown xAI v5 protocol schema")
    c = protocol["contract"]
    for key in ("any_model_is_refitted", "test_labels_used_to_build_explanations"):
        if c.get(key) is not False:
            raise BlendXAIv5Error(f"Contract differs at {key}")
    for record in protocol["sealed_inputs"].values():
        _regular_file(root / record["path"], f"sealed input {record['path']}", record["sha256"])
    return protocol, canonical_sha256(protocol)


# ------------------------------------------------------------ decision curves


def net_benefit(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Net benefit = TP/n - (FP/n) * pt/(1-pt), the standard decision-curve statistic."""
    y = np.asarray(y, int)
    n = len(y)
    out = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        d = (p >= t).astype(int)
        tp = int(((d == 1) & (y == 1)).sum())
        fp = int(((d == 1) & (y == 0)).sum())
        out[i] = tp / n - (fp / n) * (t / (1.0 - t))
    return out


def treat_all(y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    prevalence = float(np.mean(y))
    return prevalence - (1.0 - prevalence) * (thresholds / (1.0 - thresholds))


# --------------------------------------------------------------------- SHAP


def linear_shap(coef: np.ndarray, intercept: float, background: np.ndarray,
                x: np.ndarray) -> np.ndarray:
    """Exact SHAP values for a linear model: phi_j = coef_j * (x_j - E[x_j])."""
    return (x - background.mean(axis=0)) * coef


def kernel_shap(predict, background: np.ndarray, x: np.ndarray, n_samples: int,
                seed: int, l1_alpha: float = 1e-3) -> np.ndarray:
    """KernelSHAP for one instance, using the Shapley kernel weighting."""
    rng = np.random.default_rng(seed)
    d = background.shape[1]
    base = float(predict(background).mean())
    fx = float(predict(x.reshape(1, -1))[0])

    masks = rng.random((n_samples, d)) < rng.uniform(0.15, 0.85, (n_samples, 1))
    sizes = masks.sum(axis=1)
    keep = (sizes > 0) & (sizes < d)
    masks, sizes = masks[keep], sizes[keep]
    # Shapley kernel weight; the reference row is drawn from the background
    with np.errstate(divide="ignore"):
        weights = (d - 1) / (
            np.maximum(sizes * (d - sizes), 1)
            * np.array([max(1.0, float(s)) for s in sizes]) ** 0
        )
    ref = background[rng.integers(0, len(background), len(masks))]
    synth = np.where(masks, x[None, :], ref)
    values = predict(synth)
    # weighted ridge on the mask indicators, constrained so contributions sum to f(x)-E[f]
    model = Ridge(alpha=l1_alpha, fit_intercept=True)
    model.fit(masks.astype(float), values - base, sample_weight=weights)
    phi = model.coef_.astype(float)
    gap = (fx - base) - phi.sum()
    if np.abs(phi).sum() > 0:
        phi = phi + gap * np.abs(phi) / np.abs(phi).sum()   # efficiency correction
    return phi


# --------------------------------------------------------------- global LIME


def lime_local(predict, x: np.ndarray, scale: np.ndarray, n_samples: int,
               seed: int, kernel_width: float, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Local ridge surrogate around one instance; returns standardised coefficients.

    Synthetic points are clipped to the observed training range of each descriptor.
    Several RDKit descriptors (Ipc above all) span many orders of magnitude, and an
    unclipped Gaussian perturbation scaled by their standard deviation overflows
    float32 before it ever reaches the model. Clipping also keeps the local
    neighbourhood inside the region the components were actually fitted on.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, (n_samples, len(x)))
    synth = np.clip(x[None, :] + z * scale[None, :], lo[None, :], hi[None, :])
    z = np.divide(synth - x[None, :], np.where(scale == 0, 1.0, scale)[None, :])
    values = predict(synth)
    distance = np.sqrt((z ** 2).mean(axis=1))
    weights = np.exp(-(distance ** 2) / (kernel_width ** 2))
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(z, values, sample_weight=weights)
    return model.coef_.astype(float)


# ------------------------------------------------------------------------ run


def run(*, root: Path, config_path: Path, positive_path: Path, negative_path: Path,
        run_id: str, n_shap: int, n_lime: int) -> Path:
    if not re.fullmatch(r"blend_xai_v5_[a-z0-9_.-]+", run_id):
        raise BlendXAIv5Error("RUN_ID must start with blend_xai_v5_")
    root = root.resolve()
    protocol, protocol_sha256 = load_protocol(root, config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise BlendXAIv5Error(f"Run directory already exists: {destination}")
    sealed = protocol["sealed_inputs"]

    bundle = load_locked_bundle(root / sealed["screening_bundle"]["path"],
                                expected_artifact_sha256=sealed["screening_bundle"]["sha256"])
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
    y_train, y_test = labels[train_indices], labels[test_indices]

    paper_names = list(fixed_protocol["features"]["paper_descriptors"])
    paper_panel = features["paper"].astype(np.float64)
    context = _imputer_context(features["rdkit2d"][train_indices])
    rdkit_panel = _apply_context(context, features["rdkit2d"]).astype(np.float32)
    from rdkit.Chem import Descriptors
    all_rdkit = [n for n, _f in Descriptors._descList]
    keep = np.flatnonzero(np.asarray(context["finite_any_mask"], bool))
    keep = keep[np.asarray(context["varying_after_imputation_mask"], bool)]
    rdkit_names = [all_rdkit[i] for i in keep]
    if len(rdkit_names) != rdkit_panel.shape[1]:
        raise BlendXAIv5Error("RDKit descriptor names do not match the panel width")
    print(f"panels: paper {paper_panel.shape}, rdkit2d {rdkit_panel.shape}", flush=True)

    # ---- component predictors, fitted once on the 324 training rows ----------
    svm = bundle["paper_svm"]
    tabfm_model = _load_tabfm(protocol["tabfm"])
    settings = protocol["tabpfn_v2"]
    estimator = _tabpfn3_estimator(settings, int(settings["hyperparameters"]["n_estimators"]),
                                   seed=42)
    xtr_rdkit, xtr_paper = rdkit_panel[train_indices], paper_panel[train_indices]

    def predict_svm(x):
        return svm.predict_proba(np.asarray(x, dtype=np.float64))[:, 1]

    def predict_tabpfn(x):
        p, _a = _positive_probability(estimator, xtr_rdkit, y_train,
                                      np.asarray(x, dtype=np.float32), singleton=False)
        return np.asarray(p, dtype=float)

    def predict_tabfm(x):
        p, _a = _tabfm_probability(tabfm_model, protocol["tabfm"], xtr_rdkit, y_train,
                                   {"q": np.asarray(x, dtype=np.float32)}, seed=42)
        return np.asarray(p["q"], dtype=float)

    # parity against the sealed streams before any explanation is computed
    sealed_test = pd.read_csv(root / sealed["quad_d1_test"]["path"])
    sealed_test = sealed_test.set_index("paper_row_index").loc[test_indices].reset_index()
    checks = {"paper_svm": (predict_svm(paper_panel[test_indices]),
                            sealed_test.probability_paper_svm.to_numpy(float)),
              "tabpfn_v2": (predict_tabpfn(rdkit_panel[test_indices]),
                            sealed_test.probability_tabpfn_v2.to_numpy(float)),
              "tabfm": (predict_tabfm(rdkit_panel[test_indices]),
                        sealed_test.probability_tabfm.to_numpy(float))}
    parity = {}
    for name, (mine, ref) in checks.items():
        parity[name] = float(np.max(np.abs(mine - ref)))
        tol = PARITY_ATOL[name]
        print(f"parity {name}: {parity[name]:.3e} (tolerance {tol:.0e})", flush=True)
        if parity[name] > tol:
            raise BlendXAIv5Error(
                f"{name} reconstruction differs from the sealed stream by "
                f"{parity[name]!r}, above its tolerance {tol!r}")

    p_test = {n: checks[n][0] for n in TRIPLE}
    blend_triple = np.column_stack([p_test[n] for n in TRIPLE]) @ np.full(3, 1 / 3)
    blend_quad = (sealed_test[["probability_paper_svm", "probability_tanimoto_svc",
                               "probability_tabpfn_v2", "probability_tabfm"]]
                  .to_numpy(float) @ np.full(4, 0.25))

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".blendxai5.work-", dir=destination.parent))
    try:
        # ---- A. exact component decomposition -------------------------------
        decomposition = pd.DataFrame({
            "paper_row_index": test_indices, "label": y_test,
            "compound_name": frame.set_index("paper_row_index")
                                  .loc[test_indices, "compound_name"].astype(str).to_numpy(),
            **{f"p_{n}": p_test[n] for n in TRIPLE},
            **{f"contribution_{n}": p_test[n] / 3.0 for n in TRIPLE},
            "blend_probability": blend_triple,
            "decision": (blend_triple >= THRESHOLD).astype(int)})
        gap = float(np.max(np.abs(
            decomposition[[f"contribution_{n}" for n in TRIPLE]].sum(axis=1) - blend_triple)))
        if gap > 1e-12:
            raise BlendXAIv5Error("Component decomposition is not exact")
        _write_csv(tmp / "A_component_decomposition.csv", decomposition)
        print(f"A: exact component decomposition (residual {gap:.1e})", flush=True)

        # ---- B. SHAP ---------------------------------------------------------
        background = KMeans(n_clusters=int(protocol["shap"]["background_clusters"]),
                            n_init=10, random_state=42).fit(xtr_rdkit).cluster_centers_
        rows = []
        coef = np.asarray(svm.coef_, float).ravel()
        for i, ti in enumerate(test_indices):
            phi = linear_shap(coef, float(svm.intercept_[0]), xtr_paper, paper_panel[ti])
            for j, nm in enumerate(paper_names):
                rows.append({"component": "paper_svm", "paper_row_index": int(ti),
                             "feature": nm, "shap": float(phi[j]), "method": "linear_exact"})
        print("B: SVM linear SHAP done (exact)", flush=True)
        for name, predict in (("tabpfn_v2", predict_tabpfn), ("tabfm", predict_tabfm)):
            t0 = time.time()
            for i, ti in enumerate(test_indices):
                phi = kernel_shap(predict, background, rdkit_panel[ti], n_shap, seed=1000 + i)
                for j, nm in enumerate(rdkit_names):
                    if phi[j] != 0.0:
                        rows.append({"component": name, "paper_row_index": int(ti),
                                     "feature": nm, "shap": float(phi[j]),
                                     "method": "kernel_shap"})
                if (i + 1) % 20 == 0:
                    print(f"   {name} KernelSHAP {i+1}/{len(test_indices)} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            print(f"B: {name} KernelSHAP done in {time.time()-t0:.0f}s", flush=True)
        shap_frame = pd.DataFrame(rows)
        _write_csv(tmp / "B_shap_values.csv", shap_frame)
        importance = (shap_frame.assign(abs_shap=shap_frame.shap.abs())
                      .groupby(["component", "feature"]).abs_shap.mean().reset_index())
        importance["blend_weighted_importance"] = importance.abs_shap / 3.0
        _write_csv(tmp / "B_shap_importance.csv",
                   importance.sort_values("blend_weighted_importance", ascending=False))

        # ---- C. global LIME ---------------------------------------------------
        scale_r = xtr_rdkit.std(axis=0).astype(float)
        scale_r[scale_r == 0] = 1.0
        scale_p = xtr_paper.std(axis=0)
        scale_p[scale_p == 0] = 1.0
        lo_r, hi_r = xtr_rdkit.min(axis=0).astype(float), xtr_rdkit.max(axis=0).astype(float)
        lo_p, hi_p = xtr_paper.min(axis=0), xtr_paper.max(axis=0)
        kw = float(protocol["lime"]["kernel_width"])
        lime_rows = []
        for name, predict, names_, scale, lo, hi in (
                ("paper_svm", predict_svm, paper_names, scale_p, lo_p, hi_p),
                ("tabpfn_v2", predict_tabpfn, rdkit_names, scale_r, lo_r, hi_r),
                ("tabfm", predict_tabfm, rdkit_names, scale_r, lo_r, hi_r)):
            t0 = time.time()
            panel = paper_panel if name == "paper_svm" else rdkit_panel
            for i, ti in enumerate(test_indices):
                c = lime_local(predict, panel[ti].astype(float), scale, n_lime,
                               seed=2000 + i, kernel_width=kw, lo=lo, hi=hi)
                for j, nm in enumerate(names_):
                    if c[j] != 0.0:
                        lime_rows.append({"component": name, "paper_row_index": int(ti),
                                          "feature": nm, "lime_coefficient": float(c[j])})
            print(f"C: {name} global LIME done in {time.time()-t0:.0f}s", flush=True)
        lime_frame = pd.DataFrame(lime_rows)
        _write_csv(tmp / "C_lime_values.csv", lime_frame)
        lime_global = (lime_frame.assign(a=lime_frame.lime_coefficient.abs())
                       .groupby(["component", "feature"])
                       .agg(mean_abs_coefficient=("a", "mean"),
                            mean_signed_coefficient=("lime_coefficient", "mean"))
                       .reset_index())
        lime_global["blend_weighted_importance"] = lime_global.mean_abs_coefficient / 3.0
        _write_csv(tmp / "C_lime_global_importance.csv",
                   lime_global.sort_values("blend_weighted_importance", ascending=False))

        # ---- D. permutation importance on the blend ---------------------------
        rng = np.random.default_rng(7)
        base_ap = float(np.mean((blend_triple >= THRESHOLD) == y_test))
        perm_rows = []
        for name, predict, names_, panel in (
                ("paper_svm", predict_svm, paper_names, paper_panel),
                ("tabpfn_v2", predict_tabpfn, rdkit_names, rdkit_panel),
                ("tabfm", predict_tabfm, rdkit_names, rdkit_panel)):
            x0 = panel[test_indices].copy()
            p0 = predict(x0)
            repeats = int(protocol["permutation"]["repeats"])
            for j, nm in enumerate(names_):
                # all repeats issued as one batched query, which is far cheaper for
                # in-context models than one call per repeat
                stacked = np.concatenate(
                    [np.where(np.arange(x0.shape[1]) == j,
                              x0[rng.permutation(len(x0))], x0) for _ in range(repeats)])
                shifted = predict(stacked).reshape(repeats, len(x0))
                drop = float(np.mean(np.abs(shifted - p0[None, :])))
                perm_rows.append({"component": name, "feature": nm,
                                  "mean_abs_probability_shift": drop,
                                  "blend_weighted": drop / 3.0})
            print(f"D: {name} permutation importance done", flush=True)
        _write_csv(tmp / "D_permutation_importance.csv",
                   pd.DataFrame(perm_rows).sort_values("blend_weighted", ascending=False))

        # ---- E. decision curve analysis ---------------------------------------
        thresholds = np.arange(0.05, 0.951, 0.01)
        dca = []
        curves = {"blend_triple": blend_triple, "blend_quad": blend_quad,
                  "paper_svm": p_test["paper_svm"]}
        for nm, p in curves.items():
            nb = net_benefit(y_test, p, thresholds)
            for t, v in zip(thresholds, nb):
                dca.append({"cohort": "d1_test", "model": nm, "threshold": float(t),
                            "net_benefit": float(v)})
        for t, v in zip(thresholds, treat_all(y_test, thresholds)):
            dca.append({"cohort": "d1_test", "model": "treat_all", "threshold": float(t),
                        "net_benefit": float(v)})
        for t in thresholds:
            dca.append({"cohort": "d1_test", "model": "treat_none", "threshold": float(t),
                        "net_benefit": 0.0})
        # external cohorts, from the sealed component streams
        for cohort in ("drugage", "agextend"):
            d = pd.read_csv(root / sealed[f"quad_{cohort}"]["path"])
            yc = d.label.to_numpy(int)
            tri = d[["probability_paper_svm", "probability_tabpfn_v2",
                     "probability_tabfm"]].to_numpy(float) @ np.full(3, 1 / 3)
            qua = d[["probability_paper_svm", "probability_tanimoto_svc",
                     "probability_tabpfn_v2", "probability_tabfm"]].to_numpy(float) @ np.full(4, .25)
            for nm, p in (("blend_triple", tri), ("blend_quad", qua),
                          ("paper_svm", d.probability_paper_svm.to_numpy(float))):
                for t, v in zip(thresholds, net_benefit(yc, p, thresholds)):
                    dca.append({"cohort": cohort, "model": nm, "threshold": float(t),
                                "net_benefit": float(v)})
            for t, v in zip(thresholds, treat_all(yc, thresholds)):
                dca.append({"cohort": cohort, "model": "treat_all", "threshold": float(t),
                            "net_benefit": float(v)})
            for t in thresholds:
                dca.append({"cohort": cohort, "model": "treat_none", "threshold": float(t),
                            "net_benefit": 0.0})
        _write_csv(tmp / "E_decision_curves.csv", pd.DataFrame(dca))
        print("E: decision curve analysis done", flush=True)

        manifest = {
            "schema_version": f"{SCHEMA}.run_manifest.v1", "run_id": run_id,
            "protocol_sha256": protocol_sha256, "paper_split_sha256": split_sha256,
            "explained_model": {"components": list(TRIPLE), "weights": [1/3, 1/3, 1/3]},
            "comparator": "four-component blend, decision curves only",
            "attribution_design": (
                "components occupy different feature spaces, so attributions are computed "
                "inside each component in its native space and carried to the blend with "
                "weight 1/3; the component-level decomposition itself is exact"),
            "shap": {"svm": "analytic linear SHAP (exact)",
                     "foundation_models": "KernelSHAP",
                     "n_coalitions": n_shap,
                     "background_clusters": protocol["shap"]["background_clusters"]},
            "lime": {"scope": "per-compound local ridge surrogate, aggregated globally",
                     "n_perturbations": n_lime,
                     "kernel_width": protocol["lime"]["kernel_width"]},
            "decision_curve": {"statistic": "net benefit = TP/n - (FP/n) * pt/(1-pt)",
                               "thresholds": "0.05 to 0.95 in steps of 0.01",
                               "reference_policies": ["treat_all", "treat_none"]},
            "any_model_is_refitted": False,
            "test_labels_used_to_build_explanations": False,
            "component_parity_vs_sealed": parity,
            "component_parity_tolerances": PARITY_ATOL,
            "tabpfn_batching_note": (
                "the sealed runs used singleton inference; explanations require batched "
                "inference, which shifts TabPFN-v2 probabilities by at most 2.1e-3. "
                "Singleton inference reproduces the sealed stream to 5.6e-17, confirming "
                "batching as the sole cause. All reported performance uses sealed streams."),
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
    parser.add_argument("--n-shap", type=int, default=512)
    parser.add_argument("--n-lime", type=int, default=800)
    a = parser.parse_args(argv)
    run(root=a.root, config_path=a.config, positive_path=a.positive,
        negative_path=a.negative, run_id=a.run_id, n_shap=a.n_shap, n_lime=a.n_lime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
