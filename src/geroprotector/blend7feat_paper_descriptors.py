"""Equal-thirds blends where slot 3 sees ONLY the paper's 7 DataWarrior descriptors.

The question this answers: does the tabular foundation/deep model in slot 3
actually need the 205-feature RDKit2D panel, or do the seven descriptors the
published SVM uses already carry the signal?

Design (confirmed with the user before running):

  * blend stays SVM + Tanimoto + <model>, weights 1/3 each -- unchanged;
  * `paper_svm` keeps its 7 DataWarrior descriptors -- unchanged;
  * `tanimoto_svc` keeps Morgan r2/2048 binary fingerprints -- unchanged,
    because a Tanimoto/Jaccard kernel is defined on binary sets and is not
    meaningful over seven continuous descriptors;
  * ONLY slot 3 changes: TabPFN-v2 / TabFM / BiSHop / TabM are refitted on the
    seven descriptors instead of the RDKit2D panel.

So the single manipulated variable is the slot-3 feature panel, and every
number here is directly comparable to the existing equal-thirds tables.

Per-model preprocessing is held identical to how each model was run before, so
the panel is the only thing that differs:
  * TabPFN-v2 and TabFM: fold-local imputation only; both apply their own
    internal preprocessing, exactly as in the sealed runs.
  * BiSHop and TabM: fold-local imputation + fit-set-only QuantileTransformer +
    3-seed ensemble, exactly as in `screening_blend_altmodels`.
(On these seven descriptors the imputation/variance step is effectively the
identity -- all 7 are finite and non-constant, so all 7 are retained.)

Two evaluations, both produced here:
  1. the locked 80/20 paper split -- fit on 324, score the 81 held-out rows;
  2. 5-fold cross-validation on the 324 training rows only
     (StratifiedKFold(5, shuffle=True, random_state=42) -- the same folds every
     cross-fitted stream in this repository uses).

The chemistry components are NOT refitted: their OOF and test streams are read
from the sealed runs and proven byte-identical before use.  No external cohort
is touched.
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
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from geroprotector.fixed_blend_paper405 import _features, _selected_component_predictions
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.screening_blend_altmodels import (
    _alt_probabilities,
    _apply_context,
    _imputer_context,
)
from geroprotector.screening_blend_tabfm import _load_tabfm, _tabfm_probability
from geroprotector.tabpfn3_paper405 import _positive_probability, _tabpfn3_estimator
from geroprotector.traditional_paper405 import _read_sources, paper_split_indices


class Blend7FeatError(RuntimeError):
    """Raised when a sealed input, a parity proof or a contract fails."""


SCHEMA = "geroprotector.blend7feat_paper_descriptors"
CHEMISTRY = ("paper_svm", "tanimoto_svc")
EQUAL_THIRDS = np.full(3, 1.0 / 3.0)
SLOT3 = ("tabpfn_v2", "tabfm", "bishop", "tabm")
BOOTSTRAP = 10000
SEED = 20260821


def _regular_file(path: Path, role: str, expected: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise Blend7FeatError(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected is not None and sha256_file(resolved) != expected:
        raise Blend7FeatError(f"{role} SHA256 differs from the protocol lock")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    text = _regular_file(path, "7-feature blend protocol").read_text(encoding="utf-8")
    protocol = yaml.safe_load(text)
    expected_schema = f"{SCHEMA}.protocol.v1"
    if not isinstance(protocol, dict) or protocol.get("schema_version") != expected_schema:
        raise Blend7FeatError("Unknown 7-feature blend protocol schema")
    design = protocol.get("design", {})
    if (
        design.get("slot3_feature_panel") != "paper_7_datawarrior_descriptors"
        or design.get("tanimoto_keeps_morgan_fingerprints") is not True
        or design.get("paper_svm_unchanged") is not True
        or not np.allclose(design.get("weights"), EQUAL_THIRDS, rtol=0.0, atol=1e-15)
    ):
        raise Blend7FeatError("Design contract differs")
    contract = protocol.get("contract", {})
    if contract.get("chemistry_components_are_not_refitted") is not True:
        raise Blend7FeatError("Chemistry-parity contract differs")
    if contract.get("external_cohorts_used") is not False:
        raise Blend7FeatError("External cohorts must not be used here")
    immutability = protocol.get("immutability", {})
    if immutability.get("existing_run_directories_are_read_only") is not True:
        raise Blend7FeatError("Immutability contract differs")
    for record in protocol.get("sealed_inputs", {}).values():
        _regular_file(root / record["path"], f"sealed input {record['path']}", record["sha256"])
    return protocol, canonical_sha256(protocol)


# ------------------------------------------------------------------ metrics


def _all_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    d = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, d, labels=[0, 1]).ravel()
    both = len(set(y)) == 2
    return {
        "n": len(y),
        "accuracy": float(accuracy_score(y, d)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "cohen_kappa": float(cohen_kappa_score(y, d)),
        "f1_positive": float(f1_score(y, d, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y, d, average="macro", zero_division=0)),
        "auprc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
        "auroc": float(roc_auc_score(y, p)) if both else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
    }


DECISION_METRICS = ("accuracy", "specificity", "sensitivity", "cohen_kappa",
                    "f1_positive", "f1_macro")
RANKING_METRICS = ("auprc", "auroc", "brier")


def _holm(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _metric_value(name: str, y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    d = (p >= threshold).astype(int)
    if name == "accuracy":
        return float(accuracy_score(y, d))
    if name in ("specificity", "sensitivity"):
        tn, fp, fn, tp = confusion_matrix(y, d, labels=[0, 1]).ravel()
        if name == "specificity":
            return float(tn / (tn + fp)) if (tn + fp) else float("nan")
        return float(tp / (tp + fn)) if (tp + fn) else float("nan")
    if name == "cohen_kappa":
        return float(cohen_kappa_score(y, d))
    if name == "f1_positive":
        return float(f1_score(y, d, pos_label=1, zero_division=0))
    if name == "f1_macro":
        return float(f1_score(y, d, average="macro", zero_division=0))
    if name == "auprc":
        return float(average_precision_score(y, p))
    if name == "auroc":
        return float(roc_auc_score(y, p))
    if name == "brier":
        return float(brier_score_loss(y, p))
    raise Blend7FeatError(f"Unknown metric {name}")


def _compare(
    cohort: str, y: np.ndarray,
    challenger: tuple[str, np.ndarray, float],
    baseline: tuple[str, np.ndarray, float],
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Paired bootstrap CI + p for every metric, plus McNemar on the decisions."""

    name_a, pa, ta = challenger
    name_b, pb, tb = baseline
    metrics = list(DECISION_METRICS) + list(RANKING_METRICS)
    rng = np.random.default_rng(seed)
    n = len(y)
    draws: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        for m in metrics:
            draws[m].append(
                _metric_value(m, y[idx], pa[idx], ta) - _metric_value(m, y[idx], pb[idx], tb)
            )

    staged, raw_p = [], []
    for m in metrics:
        observed = _metric_value(m, y, pa, ta) - _metric_value(m, y, pb, tb)
        sample = np.asarray(draws[m], dtype=float)
        low, high = np.percentile(sample, [2.5, 97.5])
        share = (float(np.mean(sample <= 0.0)) if observed > 0
                 else float(np.mean(sample >= 0.0)))
        p_value = min(1.0, 2.0 * max(share, 1.0 / max(len(sample), 1)))
        raw_p.append(p_value)
        staged.append({
            "cohort": cohort, "challenger": name_a, "baseline": name_b, "metric": m,
            "challenger_value": _metric_value(m, y, pa, ta),
            "baseline_value": _metric_value(m, y, pb, tb),
            "difference": float(observed),
            "ci95_low": float(low), "ci95_high": float(high),
            "ci_excludes_zero": bool(low > 0 or high < 0),
            "p_raw": float(p_value),
        })
    for record, adjusted in zip(staged, _holm(raw_p), strict=True):
        record["p_holm"] = float(adjusted)
        record["significant_holm_0p05"] = bool(adjusted < 0.05)

    da, db = (pa >= ta).astype(int), (pb >= tb).astype(int)
    ca, cb = (da == y), (db == y)
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    discordant = only_a + only_b
    p_mcnemar = (1.0 if discordant == 0 else
                 float(binomtest(only_a, discordant, 0.5, alternative="two-sided").pvalue))
    mcnemar = {
        "cohort": cohort, "challenger": name_a, "baseline": name_b,
        "challenger_only_correct": only_a, "baseline_only_correct": only_b,
        "discordant_pairs": discordant, "mcnemar_exact_p": p_mcnemar,
        "significant_0p05": bool(p_mcnemar < 0.05),
    }
    return staged, mcnemar


# ---------------------------------------------------------------------- run


def _slot3(
    name: str, root: Path, protocol: dict[str, Any],
    x_fit_raw: np.ndarray, y_fit: np.ndarray, targets_raw: dict[str, np.ndarray],
    tabfm_model, seed: int,
) -> dict[str, np.ndarray]:
    """Slot-3 probabilities on the 7-descriptor panel, per-model preprocessing preserved."""

    if name in ("bishop", "tabm"):
        # identical pipeline to screening_blend_altmodels: imputer + QuantileTransformer
        # + 3-seed ensemble.  Only the incoming panel differs.
        probabilities, _extra = _alt_probabilities(
            root, name, protocol["altmodels"], x_fit_raw, y_fit, targets_raw
        )
        return probabilities

    # TabPFN-v2 / TabFM: fold-local imputation only; each applies its own internal
    # preprocessing, exactly as in the sealed runs.
    context = _imputer_context(x_fit_raw)
    x_fit = _apply_context(context, x_fit_raw)
    targets = {k: _apply_context(context, v) for k, v in targets_raw.items()}

    if name == "tabpfn_v2":
        settings = protocol["tabpfn_v2"]
        n_estimators = int(settings["hyperparameters"]["n_estimators"])
        estimator = _tabpfn3_estimator(settings, n_estimators, seed=seed)
        output = {}
        for key, matrix in targets.items():
            probability, _audit = _positive_probability(
                estimator, x_fit, y_fit, matrix, singleton=False
            )
            output[key] = probability
        del estimator
        return output

    if name == "tabfm":
        probabilities, _audit = _tabfm_probability(
            tabfm_model, protocol["tabfm"], x_fit, y_fit, targets, seed=seed
        )
        return probabilities
    raise Blend7FeatError(f"Unknown slot-3 model {name}")


def run(
    *, root: Path, config_path: Path, positive_path: Path, negative_path: Path, run_id: str
) -> Path:
    if not re.fullmatch(r"blend7feat_[a-z0-9_.-]+", run_id):
        raise Blend7FeatError("RUN_ID must start with blend7feat_")
    root = root.resolve()
    protocol, protocol_sha256 = load_protocol(root, config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise Blend7FeatError(f"Run directory already exists: {destination}")
    sealed = protocol["sealed_inputs"]

    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    frame, _audit = _read_sources(
        _regular_file(positive_path, "positive source", protocol["sources"]["positive_sha256"]),
        _regular_file(negative_path, "negative source", protocol["sources"]["negative_sha256"]),
        traditional,
    )
    train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    if split_sha256 != protocol["paper_split"]["expected_assignment_sha256"]:
        raise Blend7FeatError("Paper split differs from the sealed assignment")
    labels = frame["label"].to_numpy(dtype=int)
    smiles, _ = _validated_raw_smiles(frame)
    features = _features(frame, smiles, fixed_protocol)
    paper_panel = features["paper"]
    if paper_panel.shape[1] != 7:
        raise Blend7FeatError("Paper descriptor panel is not 7 columns")

    y_train = labels[train_indices]
    y_test = labels[test_indices]

    # -- chemistry streams from the sealed runs, parity-checked ------------------
    atol = float(protocol["contract"]["parity_atol"])
    sealed_oof = pd.read_csv(root / sealed["weighted_train_oof"]["path"])
    sealed_oof = sealed_oof.set_index("paper_row_index").loc[train_indices].reset_index()
    sealed_test = pd.read_csv(root / sealed["weighted_test_components"]["path"])
    sealed_test = sealed_test.set_index("paper_row_index").loc[test_indices].reset_index()

    settings = fixed_protocol["components"]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_split = list(folds.split(train_indices, y_train))
    fold_id = np.full(len(train_indices), -1, dtype=int)
    chem_oof = {name: np.full(len(train_indices), np.nan) for name in CHEMISTRY}
    for fold, (relative_fit, relative_validation) in enumerate(fold_split):
        fold_id[relative_validation] = fold
        chemistry, _models, _a = _selected_component_predictions(
            features, labels, train_indices[relative_fit], train_indices[relative_validation],
            settings, seed=42 + fold, requested=CHEMISTRY,
        )
        for name in CHEMISTRY:
            chem_oof[name][relative_validation] = chemistry[name]
        print(f"chemistry OOF fold {fold + 1}/5", flush=True)
    parity = {}
    for name in CHEMISTRY:
        drift = float(np.max(np.abs(
            chem_oof[name] - sealed_oof[f"probability_{name}"].to_numpy(float)
        )))
        if drift > atol:
            raise Blend7FeatError(f"Rebuilt OOF {name} differs from the sealed run")
        parity[f"oof_{name}"] = drift

    chem_test, _models, _a = _selected_component_predictions(
        features, labels, train_indices, test_indices, settings, seed=42, requested=CHEMISTRY,
    )
    for name in CHEMISTRY:
        drift = float(np.max(np.abs(
            chem_test[name] - sealed_test[f"probability_{name}"].to_numpy(float)
        )))
        if drift > atol:
            raise Blend7FeatError(f"Rebuilt test {name} differs from the sealed run")
        parity[f"test_{name}"] = drift

    # -- slot 3 on the 7-descriptor panel ----------------------------------------
    tabfm_model = _load_tabfm(protocol["tabfm"])
    print("TabFM weights loaded", flush=True)

    oof_slot3: dict[str, np.ndarray] = {}
    test_slot3: dict[str, np.ndarray] = {}
    for name in SLOT3:
        stream = np.full(len(train_indices), np.nan)
        for fold, (relative_fit, relative_validation) in enumerate(fold_split):
            sub_fit = train_indices[relative_fit]
            sub_validation = train_indices[relative_validation]
            out = _slot3(
                name, root, protocol,
                paper_panel[sub_fit], labels[sub_fit],
                {"validation": paper_panel[sub_validation]},
                tabfm_model, seed=42 + fold,
            )
            stream[relative_validation] = out["validation"]
            print(f"[{name}] OOF fold {fold + 1}/5", flush=True)
        if not np.isfinite(stream).all():
            raise Blend7FeatError(f"{name} OOF is incomplete")
        oof_slot3[name] = stream

        out = _slot3(
            name, root, protocol,
            paper_panel[train_indices], y_train,
            {"d1_test": paper_panel[test_indices]},
            tabfm_model, seed=42,
        )
        test_slot3[name] = out["d1_test"]
        print(f"[{name}] full 324-row fit -> scored 81 test rows", flush=True)

    # -- blends -------------------------------------------------------------------
    def _blend(svm, tani, third):
        return np.column_stack([svm, tani, third]) @ EQUAL_THIRDS

    oof_blend = {n: _blend(chem_oof["paper_svm"], chem_oof["tanimoto_svc"], oof_slot3[n])
                 for n in SLOT3}
    test_blend = {n: _blend(chem_test["paper_svm"], chem_test["tanimoto_svc"], test_slot3[n])
                  for n in SLOT3}

    # -- metrics at the common fixed 0.5 rule ------------------------------------
    THRESHOLD = 0.5
    metric_rows = []
    for cohort, y, streams in (
        ("d1_test_80_20", y_test,
         {"paper_svm": chem_test["paper_svm"], "tanimoto_svc": chem_test["tanimoto_svc"],
          **{f"slot3_{n}": test_slot3[n] for n in SLOT3},
          **{f"blend_eq_thirds_{n}_7feat": test_blend[n] for n in SLOT3}}),
        ("cv5_train_oof", y_train,
         {"paper_svm": chem_oof["paper_svm"], "tanimoto_svc": chem_oof["tanimoto_svc"],
          **{f"slot3_{n}": oof_slot3[n] for n in SLOT3},
          **{f"blend_eq_thirds_{n}_7feat": oof_blend[n] for n in SLOT3}}),
    ):
        for model_name, probability in streams.items():
            metric_rows.append({
                "cohort": cohort, "model": model_name, "threshold_rule": "fixed_0p5",
                **_all_metrics(y, probability, THRESHOLD),
            })
    metrics = pd.DataFrame(metric_rows)

    # per-fold CV metrics
    fold_rows = []
    for name in SLOT3:
        for fold in range(5):
            mask = fold_id == fold
            fold_rows.append({
                "model": f"blend_eq_thirds_{name}_7feat", "fold": fold,
                **_all_metrics(y_train[mask], oof_blend[name][mask], THRESHOLD),
            })
    for fold in range(5):
        mask = fold_id == fold
        fold_rows.append({
            "model": "paper_svm", "fold": fold,
            **_all_metrics(y_train[mask], chem_oof["paper_svm"][mask], THRESHOLD),
        })
    per_fold = pd.DataFrame(fold_rows)

    # -- significance vs the paper SVM -------------------------------------------
    comparisons, mcnemars = [], []
    for cohort, y, blends, svm in (
        ("d1_test_80_20", y_test, test_blend, chem_test["paper_svm"]),
        ("cv5_train_oof", y_train, oof_blend, chem_oof["paper_svm"]),
    ):
        for name in SLOT3:
            staged, mcnemar = _compare(
                cohort, y,
                (f"blend_eq_thirds_{name}_7feat", blends[name], THRESHOLD),
                ("paper_svm", svm, THRESHOLD),
            )
            comparisons.extend(staged)
            mcnemars.append(mcnemar)
            print(f"significance [{cohort}] {name} vs paper_svm done", flush=True)
    significance = pd.DataFrame(comparisons)
    mcnemar_frame = pd.DataFrame(mcnemars)

    predictions = pd.DataFrame({
        "paper_row_index": train_indices, "fold": fold_id, "label": y_train,
        "probability_paper_svm": chem_oof["paper_svm"],
        "probability_tanimoto_svc": chem_oof["tanimoto_svc"],
        **{f"probability_slot3_{n}": oof_slot3[n] for n in SLOT3},
        **{f"blend_eq_thirds_{n}_7feat": oof_blend[n] for n in SLOT3},
    })
    test_predictions = pd.DataFrame({
        "paper_row_index": test_indices, "label": y_test,
        "probability_paper_svm": chem_test["paper_svm"],
        "probability_tanimoto_svc": chem_test["tanimoto_svc"],
        **{f"probability_slot3_{n}": test_slot3[n] for n in SLOT3},
        **{f"blend_eq_thirds_{n}_7feat": test_blend[n] for n in SLOT3},
    })

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".b7f.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "metrics_all.csv", metrics)
        _write_csv(tmp / "cv5_per_fold_metrics.csv", per_fold)
        _write_csv(tmp / "significance_vs_paper_svm.csv", significance)
        _write_csv(tmp / "mcnemar_vs_paper_svm.csv", mcnemar_frame)
        _write_csv(tmp / "cv5_train_oof_predictions.csv", predictions)
        _write_csv(tmp / "d1_test_predictions.csv", test_predictions)

        cols = [("model", "Model"), ("accuracy", "Acc"), ("specificity", "Spec"),
                ("sensitivity", "Sens"), ("cohen_kappa", "Kappa"),
                ("f1_positive", "F1+"), ("f1_macro", "F1mac"),
                ("auprc", "AP+"), ("auroc", "AUROC"), ("brier", "Brier")]
        lines = [
            "# Equal-thirds blends with slot 3 restricted to the paper's 7 descriptors",
            "",
            "Only slot 3 changed: TabPFN-v2 / TabFM / BiSHop / TabM now see the seven",
            "DataWarrior descriptors instead of the 205-feature RDKit2D panel. paper_svm and",
            "tanimoto_svc are unchanged (Tanimoto keeps its Morgan fingerprints -- a Tanimoto",
            "kernel is undefined over continuous descriptors). Weights remain 1/3 each.",
            "",
            "All metrics at a common fixed 0.5 threshold. Chemistry components were not",
            "refitted; their streams are proven identical to the sealed runs.",
            "",
        ]
        for cohort, title in (("d1_test_80_20", "D1 held-out test, 80/20 split (n=81)"),
                              ("cv5_train_oof", "5-fold CV, D1 train only (n=324)")):
            sub = metrics[metrics.cohort == cohort].sort_values("auprc", ascending=False)
            lines += [f"## {title}", "",
                      "| " + " | ".join(c[1] for c in cols) + " |",
                      "|" + "---|" * len(cols)]
            for row in sub.itertuples(index=False):
                cells = []
                for key, _label in cols:
                    v = getattr(row, key)
                    cells.append(v if isinstance(v, str)
                                 else ("nan" if not np.isfinite(v) else f"{v:.4f}"))
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        (tmp / "summary.md").write_text("\n".join(lines), encoding="utf-8")

        manifest = {
            "schema_version": f"{SCHEMA}.run_manifest.v1",
            "run_id": run_id, "protocol_sha256": protocol_sha256,
            "paper_split_sha256": split_sha256,
            "design": {
                "slot3_feature_panel": "paper 7 DataWarrior descriptors",
                "slot3_models": list(SLOT3),
                "paper_svm_panel": "paper 7 DataWarrior descriptors (unchanged)",
                "tanimoto_panel": "Morgan r2/2048 binary (unchanged)",
                "weights": EQUAL_THIRDS.tolist(),
                "single_manipulated_variable": "slot-3 feature panel",
            },
            "evaluations": ["d1_test_80_20 (n=81)", "cv5_train_oof (n=324)"],
            "threshold_rule": "fixed 0.5 for every model (common, no selection)",
            "bootstrap": {"resamples": BOOTSTRAP, "seed": SEED, "unit": "compound"},
            "multiplicity": "Holm-Bonferroni across the 9 metrics per comparison",
            "chemistry_parity_max_abs_difference": parity,
            "chemistry_components_are_not_refitted": True,
            "external_cohorts_used": False,
            "runtime": {"python": platform.python_version(), "platform": platform.platform()},
            "sealed_inputs_sha256": {
                k: sha256_file(root / v["path"]) for k, v in sealed.items()
            },
            "existing_runs_modified": False,
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
