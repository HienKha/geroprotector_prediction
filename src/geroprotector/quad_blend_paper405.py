"""Four-component blend: paper SVM + Tanimoto + TabPFN-v2 + TabFM, threshold fixed at 0.5.

Two weight sets are evaluated:

  A. EQUAL QUARTERS  -- [0.25, 0.25, 0.25, 0.25], prespecified, no data used.
  B. OOF-OPTIMISED   -- searched on the 324-row D1 train cross-fitted OOF only,
                        over the 0.01-step simplex.  Three objectives are
                        searched and all three are reported, so the choice of
                        objective is visible rather than hidden:
                          B1 MCC at 0.5      (primary: matches the fixed threshold)
                          B2 AUPRC           (threshold-free)
                          B3 macro-F1 at 0.5

THRESHOLD.  Fixed at 0.5 for every model, every cohort, every table.  No
threshold is selected, tuned or moved anywhere in this module -- that is the
whole point of pinning it.  The paper SVM is additionally reported at its own
published operating point (decision_function >= 0, i.e. p >= 0.5098302195333542)
because that is literally what the source paper's notebook does, and forcing it
onto a foreign threshold would misrepresent the baseline.

WHY NO MODEL IS REFITTED.  Every component stream this experiment needs already
exists, and all of them are fold-aligned:

  * weightedblend405 built the paper_svm / tanimoto_svc / tabpfn_v2 train OOF
    with StratifiedKFold(5, shuffle=True, random_state=42) on train_indices;
  * screeningblend_tabfm built the tabfm train OOF with the same folds
    (fixed_blend_paper405.yaml: cross_fitted_oof_folds 5, seed 42);
  * the same runs scored D1 test and both externals from components fitted on
    all 324 training rows.

So the 5-fold CV here is a genuine cross-fitted CV, and the lock/test/external
evaluation is a genuine fit-on-train/score-once evaluation.  This module only
re-weights and re-thresholds streams that already exist; it fits nothing, and it
touches no GPU.  Every reuse is hash-verified and every shared component stream
is proven identical across the two source runs before use.

LEAKAGE.  Weights are searched on the 324 train OOF rows only.  Reporting the
searched weights' performance on those same rows is optimistic, so a NESTED
variant is also reported: the weight search is repeated inside each CV fold on
the other four folds and applied to the held-out fold.  The nested row is the
honest cross-validation estimate.  No test or external label is used to choose
weights, objectives or thresholds anywhere.
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract
from geroprotector.traditional_paper405 import paper_split_indices


class QuadBlendError(RuntimeError):
    """Raised when a sealed input, a parity proof or a contract fails."""


SCHEMA = "geroprotector.quad_blend_paper405"
COMPONENTS = ("paper_svm", "tanimoto_svc", "tabpfn_v2", "tabfm")
COLUMNS = [f"probability_{c}" for c in COMPONENTS]
FIXED_THRESHOLD = 0.5
PAPER_SVM_NATIVE_THRESHOLD = 0.5098302195333542
EQUAL_QUARTERS = np.full(4, 0.25)
PARITY_ATOL = 1e-12
COHORTS = ("d1_test", "drugage", "agextend")


def _regular_file(path: Path, role: str, expected: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise QuadBlendError(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected is not None and sha256_file(resolved) != expected:
        raise QuadBlendError(f"{role} SHA256 differs from the protocol lock")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = yaml.safe_load(_regular_file(path, "quad-blend protocol").read_text("utf-8"))
    if not isinstance(protocol, dict) or protocol.get("schema_version") != f"{SCHEMA}.protocol.v1":
        raise QuadBlendError("Unknown quad-blend protocol schema")
    contract = protocol["contract"]
    if float(contract["fixed_threshold"]) != FIXED_THRESHOLD:
        raise QuadBlendError("Threshold contract differs: this module pins 0.5")
    for key in ("threshold_is_selected_or_tuned", "any_model_is_refitted",
                "test_or_external_labels_used_for_weights_or_threshold"):
        if contract.get(key) is not False:
            raise QuadBlendError(f"Contract differs at {key}")
    if list(protocol["blend"]["components"]) != list(COMPONENTS):
        raise QuadBlendError("Component contract differs")
    if protocol.get("immutability", {}).get("existing_run_directories_are_read_only") is not True:
        raise QuadBlendError("Immutability contract differs")
    for record in protocol["sealed_inputs"].values():
        _regular_file(root / record["path"], f"sealed input {record['path']}", record["sha256"])
    return protocol, canonical_sha256(protocol)


# ------------------------------------------------------------- vectorised search


def _simplex_grid(step: float) -> np.ndarray:
    """All 4-component weight vectors on the simplex at the given step."""

    n = int(round(1.0 / step))
    if abs(n * step - 1.0) > 1e-12:
        raise QuadBlendError("Grid step must divide 1 exactly")
    points = [
        (a, b, c, n - a - b - c)
        for a in range(n + 1)
        for b in range(n - a + 1)
        for c in range(n - a - b + 1)
    ]
    return np.asarray(points, dtype=float) / n


def _counts(y: np.ndarray, decisions: np.ndarray) -> tuple[np.ndarray, ...]:
    """tp/fp/fn/tn per column of a (n_rows, n_weightvectors) decision matrix."""

    positive = (y == 1).astype(float)
    negative = 1.0 - positive
    tp = positive @ decisions
    fp = negative @ decisions
    return tp, fp, positive.sum() - tp, negative.sum() - fp


def _mcc_vec(y: np.ndarray, decisions: np.ndarray) -> np.ndarray:
    tp, fp, fn, tn = _counts(y, decisions)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.divide(tp * tn - fp * fn, denominator,
                     out=np.zeros_like(tp), where=denominator > 0)


def _macro_f1_vec(y: np.ndarray, decisions: np.ndarray) -> np.ndarray:
    tp, fp, fn, tn = _counts(y, decisions)
    positive = np.divide(2 * tp, 2 * tp + fp + fn,
                         out=np.zeros_like(tp), where=2 * tp + fp + fn > 0)
    negative = np.divide(2 * tn, 2 * tn + fn + fp,
                         out=np.zeros_like(tn), where=2 * tn + fn + fp > 0)
    return (positive + negative) / 2.0


def _ap_vec(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Average precision per column of a (n_rows, n_weightvectors) score matrix."""

    order = np.argsort(-scores, axis=0, kind="stable")
    ys = y[order].astype(float)
    tp = np.cumsum(ys, axis=0)
    ranks = np.arange(1, scores.shape[0] + 1, dtype=float)[:, None]
    precision = tp / ranks
    total = tp[-1]
    return np.divide((precision * ys).sum(axis=0), total,
                     out=np.zeros_like(total), where=total > 0)


def search_weights(
    y: np.ndarray, panel: np.ndarray, grid: np.ndarray, objective: str, chunk: int = 4096
) -> tuple[np.ndarray, float]:
    """Best weight vector on `grid` for `objective`, evaluated on these rows only."""

    best_value, best_weights = -np.inf, None
    for start in range(0, len(grid), chunk):
        block = grid[start:start + chunk]
        scores = panel @ block.T
        if objective == "auprc":
            values = _ap_vec(y, scores)
        else:
            decisions = (scores >= FIXED_THRESHOLD).astype(float)
            values = (_mcc_vec(y, decisions) if objective == "mcc"
                      else _macro_f1_vec(y, decisions))
        index = int(np.argmax(values))
        if values[index] > best_value:
            best_value, best_weights = float(values[index]), block[index].copy()
    return best_weights, best_value


# ------------------------------------------------------------------- reporting


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    d = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, d, labels=[0, 1]).ravel()
    both = len(set(y)) == 2
    return {
        "n": len(y), "n_positive": int(y.sum()), "threshold": float(threshold),
        # --- the source paper's own metric set ---
        "accuracy": float(accuracy_score(y, d)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "cohen_kappa": float(cohen_kappa_score(y, d)),
        # --- additional metrics ---
        "auprc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
        "auroc": float(roc_auc_score(y, p)) if both else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "mcc": float(matthews_corrcoef(y, d)),
        "macro_f1": float(f1_score(y, d, average="macro", zero_division=0)),
        "f1_positive": float(f1_score(y, d, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, d)),
        "precision_positive": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "npv": float(tn / (tn + fn)) if (tn + fn) else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _plain_markdown(frame: pd.DataFrame, digits: int = 4) -> list[str]:
    header = list(frame.columns)
    lines = ["| " + " | ".join(str(h) for h in header) + " |",
             "|" + "---|" * len(header)]
    for row in frame.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, (int, np.integer)):
                cells.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                cells.append("nan" if not np.isfinite(value) else f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


# ------------------------------------------------------------------------ run


def run(*, root: Path, config_path: Path, run_id: str) -> Path:
    if not re.fullmatch(r"quad_blend_[a-z0-9_.-]+", run_id):
        raise QuadBlendError("RUN_ID must start with quad_blend_")
    root = root.resolve()
    protocol, protocol_sha256 = load_protocol(root, config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise QuadBlendError(f"Run directory already exists: {destination}")
    sealed = protocol["sealed_inputs"]

    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    if split_sha256 != protocol["paper_split"]["expected_assignment_sha256"]:
        raise QuadBlendError("Paper split differs from the sealed assignment")

    # ---- train OOF panel: four fold-aligned cross-fitted streams --------------
    pfn = pd.read_csv(root / sealed["weighted_train_oof"]["path"])
    pfn = pfn.set_index("paper_row_index").loc[train_indices].reset_index()
    fm = pd.read_csv(root / sealed["tabfm_train_oof"]["path"])
    fm = fm.set_index("paper_row_index").loc[train_indices].reset_index()
    for column in ("probability_paper_svm", "probability_tanimoto_svc"):
        drift = float(np.max(np.abs(pfn[column].to_numpy(float) - fm[column].to_numpy(float))))
        if drift > PARITY_ATOL:
            raise QuadBlendError(
                f"train OOF: shared component {column} differs between the two source "
                f"runs by {drift!r}; the streams are not fold-aligned"
            )
    y_train = pfn.label.to_numpy(int) if "label" in pfn else fm.label.to_numpy(int)
    if not np.array_equal(y_train, fm.label.to_numpy(int)):
        raise QuadBlendError("train OOF labels differ between the two source runs")
    oof_panel = np.column_stack([
        pfn.probability_paper_svm.to_numpy(float),
        pfn.probability_tanimoto_svc.to_numpy(float),
        pfn.probability_tabpfn_v2.to_numpy(float),
        fm.probability_tabfm.to_numpy(float),
    ])
    print(f"train OOF panel {oof_panel.shape}, parity of shared streams <= {PARITY_ATOL}",
          flush=True)

    # ---- fold identity, proven to match every source run ---------------------
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_split = list(folds.split(train_indices, y_train))
    fold_id = np.full(len(train_indices), -1, dtype=int)
    for fold, (_fit, validation) in enumerate(fold_split):
        fold_id[validation] = fold
    if (fold_id < 0).any():
        raise QuadBlendError("Fold assignment is incomplete")

    # ---- scored cohorts: components fitted on all 324 train rows -------------
    test_pfn = pd.read_csv(root / sealed["weighted_test_components"]["path"])
    test_pfn = test_pfn.set_index("paper_row_index").loc[test_indices].reset_index()
    test_fm = pd.read_csv(root / sealed["tabfm_test"]["path"])
    test_fm = test_fm.set_index("paper_row_index").loc[test_indices].reset_index()
    panels: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    frames: dict[str, pd.DataFrame] = {}

    for column in ("probability_paper_svm", "probability_tanimoto_svc"):
        drift = float(np.max(np.abs(
            test_pfn[column].to_numpy(float) - test_fm[column].to_numpy(float))))
        if drift > PARITY_ATOL:
            raise QuadBlendError(f"d1_test: shared component {column} differs by {drift!r}")
    labels["d1_test"] = test_pfn.label.to_numpy(int)
    panels["d1_test"] = np.column_stack([
        test_pfn.probability_paper_svm.to_numpy(float),
        test_pfn.probability_tanimoto_svc.to_numpy(float),
        test_pfn.probability_tabpfn_v2.to_numpy(float),
        test_fm.probability_tabfm.to_numpy(float),
    ])
    frames["d1_test"] = pd.DataFrame({"paper_row_index": test_indices})

    for cohort, pfn_key, fm_key, label_column in (
        ("drugage", "drugage_scored", "tabfm_drugage", "has_significant_positive"),
        ("agextend", "agextend_scored", "tabfm_agextend", None),
    ):
        a = pd.read_csv(root / sealed[pfn_key]["path"])
        b = pd.read_csv(root / sealed[fm_key]["path"])
        if not (a.external_id.to_numpy() == b.external_id.to_numpy()).all():
            raise QuadBlendError(f"{cohort}: external row order differs between source runs")
        for column in ("probability_paper_svm", "probability_tanimoto_svc",
                       "probability_tabpfn_v2"):
            if column in b:
                drift = float(np.max(np.abs(a[column].to_numpy(float)
                                            - b[column].to_numpy(float))))
                if drift > PARITY_ATOL:
                    raise QuadBlendError(f"{cohort}: shared component {column} differs "
                                         f"by {drift!r}")
        labels[cohort] = (a[label_column].astype(int).to_numpy() if label_column
                          else a.label.to_numpy(int))
        panels[cohort] = np.column_stack([
            a.probability_paper_svm.to_numpy(float),
            a.probability_tanimoto_svc.to_numpy(float),
            a.probability_tabpfn_v2.to_numpy(float),
            b.probability_tabfm.to_numpy(float),
        ])
        frames[cohort] = a[["external_id", "compound_name", "source_smiles"]].copy()
        print(f"{cohort}: {panels[cohort].shape[0]} rows, "
              f"{int(labels[cohort].sum())} positive", flush=True)

    # ---- weight sets ---------------------------------------------------------
    step = float(protocol["weight_search"]["grid_step"])
    grid = _simplex_grid(step)
    print(f"simplex grid: {len(grid)} weight vectors at step {step}", flush=True)
    weight_sets: dict[str, np.ndarray] = {"eq_quarters": EQUAL_QUARTERS.copy()}
    search_rows = [{
        "weight_set": "eq_quarters", "objective": "none_prespecified",
        "searched": False, "oof_objective_value": float("nan"),
        **{f"w_{c}": float(v) for c, v in zip(COMPONENTS, EQUAL_QUARTERS, strict=True)},
    }]
    for name, objective in (("oof_mcc", "mcc"), ("oof_auprc", "auprc"),
                            ("oof_macro_f1", "macro_f1")):
        weights, value = search_weights(y_train, oof_panel, grid, objective)
        weight_sets[name] = weights
        search_rows.append({
            "weight_set": name, "objective": objective, "searched": True,
            "search_basis": "324_row_d1_train_cross_fitted_oof_only",
            "oof_objective_value": value,
            **{f"w_{c}": float(v) for c, v in zip(COMPONENTS, weights, strict=True)},
        })
        print(f"{name}: weights {np.round(weights, 3)} -> OOF {objective} {value:.4f}",
              flush=True)

    # ---- nested weight search: honest CV estimate ----------------------------
    nested_scores: dict[str, np.ndarray] = {}
    nested_rows = []
    for name, objective in (("oof_mcc", "mcc"), ("oof_auprc", "auprc"),
                            ("oof_macro_f1", "macro_f1")):
        stream = np.full(len(train_indices), np.nan)
        for fold, (fit, validation) in enumerate(fold_split):
            weights, value = search_weights(
                y_train[fit], oof_panel[fit], grid, objective)
            stream[validation] = oof_panel[validation] @ weights
            nested_rows.append({
                "weight_set": name, "objective": objective, "fold": fold,
                "inner_objective_value": value,
                **{f"w_{c}": float(v) for c, v in zip(COMPONENTS, weights, strict=True)},
            })
        nested_scores[name] = stream
        print(f"nested {name}: 5 inner searches done", flush=True)

    # ---- evaluation ----------------------------------------------------------
    rows = []
    svm_oof = oof_panel[:, 0]

    def add(cohort, model, y, p, threshold=FIXED_THRESHOLD, note=""):
        rows.append({"cohort": cohort, "model": model, "note": note,
                     **_metrics(y, p, threshold)})

    # baselines on train OOF
    add("cv5_train_oof", "paper_svm", y_train, svm_oof, note="fixed_0.5")
    add("cv5_train_oof", "paper_svm_published_point", y_train, svm_oof,
        PAPER_SVM_NATIVE_THRESHOLD, note="decision_function>=0")
    for slot, index in (("tabpfn_v2", 2), ("tabfm", 3)):
        add("cv5_train_oof", f"eq_thirds_{slot}", y_train,
            oof_panel[:, [0, 1, index]] @ np.full(3, 1 / 3), note="3-component reference")
    for name, weights in weight_sets.items():
        add("cv5_train_oof", f"quad_{name}", y_train, oof_panel @ weights,
            note="weights searched on these same rows" if name != "eq_quarters"
                 else "prespecified")
    for name, stream in nested_scores.items():
        add("cv5_train_oof", f"quad_{name}_NESTED", y_train, stream,
            note="weights re-searched inside each fold; honest CV estimate")

    # per-fold detail
    fold_rows = []
    fold_streams = {f"quad_{k}": oof_panel @ v for k, v in weight_sets.items()}
    fold_streams |= {f"quad_{k}_NESTED": v for k, v in nested_scores.items()}
    fold_streams["paper_svm"] = svm_oof
    fold_streams["eq_thirds_tabfm"] = oof_panel[:, [0, 1, 3]] @ np.full(3, 1 / 3)
    fold_streams["eq_thirds_tabpfn_v2"] = oof_panel[:, [0, 1, 2]] @ np.full(3, 1 / 3)
    for model, stream in fold_streams.items():
        for fold in range(5):
            mask = fold_id == fold
            fold_rows.append({"model": model, "fold": fold,
                              **_metrics(y_train[mask], stream[mask], FIXED_THRESHOLD)})

    # locked evaluation on the held-out test set and both externals
    for cohort in COHORTS:
        y, panel = labels[cohort], panels[cohort]
        add(cohort, "paper_svm", y, panel[:, 0], note="fixed_0.5")
        add(cohort, "paper_svm_published_point", y, panel[:, 0],
            PAPER_SVM_NATIVE_THRESHOLD, note="decision_function>=0")
        for slot, index in (("tabpfn_v2", 2), ("tabfm", 3)):
            add(cohort, f"eq_thirds_{slot}", y, panel[:, [0, 1, index]] @ np.full(3, 1 / 3),
                note="3-component reference")
        for name, weights in weight_sets.items():
            add(cohort, f"quad_{name}", y, panel @ weights,
                note="weights fixed from train OOF; scored once")
        frames[cohort] = frames[cohort].assign(
            label=y,
            **{f"probability_{c}": panel[:, i] for i, c in enumerate(COMPONENTS)},
            **{f"blend_{name}": panel @ weights for name, weights in weight_sets.items()},
        )

    metrics = pd.DataFrame(rows)
    per_fold = pd.DataFrame(fold_rows)
    fold_summary = per_fold.groupby("model").agg(
        **{f"{m}_{s}": (m, s) for m in ("accuracy", "cohen_kappa", "mcc", "macro_f1", "auprc")
           for s in ("mean", "std")}).reset_index()

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".quadblend.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "weight_sets.csv", pd.DataFrame(search_rows))
        _write_csv(tmp / "nested_fold_weights.csv", pd.DataFrame(nested_rows))
        _write_csv(tmp / "metrics_all.csv", metrics)
        _write_csv(tmp / "cv5_per_fold_metrics.csv", per_fold)
        _write_csv(tmp / "cv5_fold_mean_sd.csv", fold_summary)
        _write_csv(tmp / "cv5_train_oof_predictions.csv", pd.DataFrame({
            "paper_row_index": train_indices, "fold": fold_id, "label": y_train,
            **{f"probability_{c}": oof_panel[:, i] for i, c in enumerate(COMPONENTS)},
            **{f"blend_{k}": oof_panel @ v for k, v in weight_sets.items()},
            **{f"blend_{k}_nested": v for k, v in nested_scores.items()},
        }))
        for cohort in COHORTS:
            _write_csv(tmp / f"predictions_{cohort}.csv", frames[cohort])

        show = ["model", "n", "accuracy", "specificity", "sensitivity", "cohen_kappa",
                "auprc", "auroc", "brier", "mcc", "macro_f1"]
        lines = [
            "# Four-component blend: paper SVM + Tanimoto + TabPFN-v2 + TabFM",
            "", "**Threshold fixed at 0.5 everywhere.** No threshold is selected or moved. "
            "The paper SVM is also shown at its published operating point "
            "(`decision_function >= 0`), which is what the source paper's notebook does.",
            "", "No model is refitted: every component stream already existed and all four "
            "are fold-aligned (StratifiedKFold 5, shuffle, seed 42 on train_indices). "
            "This run only re-weights them.", "",
            "## Weight sets", "",
        ]
        lines += _plain_markdown(pd.DataFrame(search_rows)[
            ["weight_set", "objective", "oof_objective_value",
             *[f"w_{c}" for c in COMPONENTS]]])
        for cohort, title in (
            ("cv5_train_oof", "5-fold CV on D1 train (n=324), threshold 0.5"),
            ("d1_test", "D1 held-out test (n=81), train->lock->score once, threshold 0.5"),
            ("drugage", "DrugAge external (n=446), threshold 0.5"),
            ("agextend", "AgeXtend external (n=69), threshold 0.5"),
        ):
            sub = metrics[metrics.cohort == cohort].sort_values("mcc", ascending=False)
            lines += ["", f"## {title}", ""] + _plain_markdown(sub[show])
        lines += ["", "## Per-fold mean +/- SD (5-fold CV, threshold 0.5)", ""]
        lines += _plain_markdown(fold_summary[
            ["model", "accuracy_mean", "accuracy_std", "cohen_kappa_mean", "cohen_kappa_std",
             "mcc_mean", "mcc_std", "macro_f1_mean", "macro_f1_std"]])
        lines += ["", "**Read the NESTED rows for the honest cross-validation estimate.** "
                  "The plain `quad_oof_*` rows report weights searched on the very rows "
                  "they are scored on, so they are optimistic by construction."]
        (tmp / "summary.md").write_text("\n".join(lines), encoding="utf-8")

        manifest = {
            "schema_version": f"{SCHEMA}.run_manifest.v1",
            "run_id": run_id, "protocol_sha256": protocol_sha256,
            "paper_split_sha256": split_sha256,
            "components": list(COMPONENTS),
            "fixed_threshold": FIXED_THRESHOLD,
            "threshold_is_selected_or_tuned": False,
            "paper_svm_published_operating_point": PAPER_SVM_NATIVE_THRESHOLD,
            "any_model_is_refitted": False,
            "gpu_used": False,
            "reuse_justification": (
                "weightedblend405 (paper_svm/tanimoto_svc/tabpfn_v2) and "
                "screeningblend_tabfm (tabfm) built their train OOF with the identical "
                "StratifiedKFold(5, shuffle=True, random_state=42) on train_indices, and "
                "scored d1_test/drugage/agextend from components fitted on all 324 train "
                "rows. Shared component streams were proven identical across both runs to "
                f"atol {PARITY_ATOL}."
            ),
            "weight_search": {
                "grid_step": step, "n_weight_vectors": int(len(grid)),
                "basis": "324_row_d1_train_cross_fitted_oof_only",
                "objectives": ["mcc@0.5", "auprc", "macro_f1@0.5"],
                "selected": {k: v.tolist() for k, v in weight_sets.items()},
                "nested_variant_reported": True,
            },
            "test_or_external_labels_used_for_weights_or_threshold": False,
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
                                if p.is_file() and p.name != "COMPLETED.json"},
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
    parser.add_argument("--run-id", required=True)
    a = parser.parse_args(argv)
    run(root=a.root, config_path=a.config, run_id=a.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
