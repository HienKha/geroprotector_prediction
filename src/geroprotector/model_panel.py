"""The fixed ten-model panel, fitted fold-locally on the D1 training partition.

Experiments B and C differ only in how the folds are constructed -- repeated
random `StratifiedKFold` for B, chemistry-aware groups for C -- so the fitting
contract itself lives here once and is shared.

Every learned transformation is fitted inside the fold:

  * RDKit2D median imputation and the zero-variance filter (`_imputer_context`)
    are refitted on the fold's fit rows;
  * the TabM quantile transformer is refitted on those rows;
  * the Tanimoto kernel is built from fit rows and the validation kernel is
    evaluated only against fit rows;
  * TabPFN-v2 and TabFM are conditioned only on fit rows;
  * the paper SVM sees the seven DataWarrior descriptors unscaled, exactly as
    published.

The parameters are never typed from memory: they are resolved from the sealed
`fixed_blend_paper405.yaml`, `traditional_paper405.yaml`,
`screening_blend_tabfm_protocol.yaml` and
`screening_blend_altmodels_protocol.yaml` and echoed into the caller's manifest.

The 81 held-out D1 test rows are never passed to this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from geroprotector.fixed_blend_paper405 import (
    _fit_imputer,
    _fit_tanimoto,
    _tabpfn_probability,
)
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.screening_blend_altmodels import _alt_probabilities
from geroprotector.traditional_paper405 import _load_protocol, _models, _positive_score

BASE_MODELS = ("paper_svm", "tanimoto_svc", "tabpfn_v2", "tabfm", "catboost_full",
               "xgboost_full", "lightgbm_full", "tabm_full")
BLENDS = {
    "blend3_equal": ("paper_svm", "tabpfn_v2", "tabfm"),
    "gb4_equal": ("paper_svm", "tanimoto_svc", "tabpfn_v2", "tabfm"),
}
PANEL = BASE_MODELS + tuple(BLENDS)
GBM_SOURCE = {"catboost_full": "catboost", "xgboost_full": "xgboost",
              "lightgbm_full": "lightgbm"}


class PanelError(RuntimeError):
    """Raised when a fold contract or a leakage guard fails."""


class Panel:
    """Holds the resolved settings and the loaded foundation-model checkpoints."""

    def __init__(self, root: Path, *, tabpfn_seed: int = 42):
        self.root = Path(root).resolve()
        self.fixed_protocol, self.fixed_sha = load_fixed_protocol(
            self.root / "configs" / "fixed_blend_paper405.yaml")
        self.traditional_protocol, self.traditional_sha = _load_protocol(
            self.root / "configs" / "traditional_paper405.yaml")
        self.tabfm_protocol = yaml.safe_load(
            (self.root / "configs" / "screening_blend_tabfm_protocol.yaml").read_text())
        self.altmodels_protocol = yaml.safe_load(
            (self.root / "configs" / "screening_blend_altmodels_protocol.yaml").read_text())
        self.tabpfn_seed = int(tabpfn_seed)
        self._tabfm_model = None

    # -------------------------------------------------------------- settings
    @property
    def resolved_settings(self) -> dict[str, Any]:
        return {
            "paper_svm": self.fixed_protocol["components"]["paper_svm"],
            "tanimoto_svc": self.fixed_protocol["components"]["tanimoto_svc"],
            "tabpfn_v2": {k: v for k, v in self.fixed_protocol["components"]["tabpfn_v2"].items()},
            "tabfm": self.tabfm_protocol["tabfm"],
            "tabm": self.altmodels_protocol["alt_models"]["tabm"],
            "alt_model_training": self.altmodels_protocol["alt_model_training"],
            "gbm": {v: self.traditional_protocol["models"][v] for v in GBM_SOURCE.values()},
            "foundation_model_seed_rule": (
                "component_seed = 42 + fold_index for every partition repeat; "
                "algorithm RNG is held fixed so repeated CV isolates partition variation"),
            "config_sha256": {"fixed_blend_paper405.yaml": self.fixed_sha,
                              "traditional_paper405.yaml": self.traditional_sha},
        }

    def _tabfm(self):
        if self._tabfm_model is None:
            from geroprotector.screening_blend_tabfm import _load_tabfm
            self._tabfm_model = _load_tabfm(self.tabfm_protocol["tabfm"])
        return self._tabfm_model

    def reload_tabfm(self):
        """Drop and reload TabFM before its consecutive fold pass."""
        self._tabfm_model = None
        return self._tabfm()

    # ------------------------------------------------------------------ fold
    def fold_probabilities(self, features: dict[str, np.ndarray], labels: np.ndarray,
                           fit: np.ndarray, target: np.ndarray, *,
                           models: Sequence[str] = BASE_MODELS,
                           component_seed: int | None = None,
                           verbose: bool = True) -> dict[str, np.ndarray]:
        """Fit `models` on `fit` rows and score `target` rows. Indices are absolute.

        `component_seed` is the random_state handed to the in-context foundation
        models. Every sealed cross-fitted run uses `42 + fold_index`. Repeated
        partition experiments keep that rule fixed across repeats so they vary the
        partition, not both the partition and model RNG.
        """
        seed = self.tabpfn_seed if component_seed is None else int(component_seed)
        unknown = set(models) - set(BASE_MODELS)
        if unknown:
            raise PanelError(f"Unknown model ids: {sorted(unknown)}")
        if np.intersect1d(fit, target).size:
            raise PanelError("Fit and target row sets overlap")
        if len(np.unique(labels[fit])) < 2:
            raise PanelError("A fold's fit rows contain a single class")
        components = self.fixed_protocol["components"]
        out: dict[str, np.ndarray] = {}

        if "paper_svm" in models:
            from sklearn.svm import SVC
            settings = components["paper_svm"]
            estimator = SVC(kernel=settings["kernel"], C=float(settings["C"]),
                            gamma=float(settings["gamma"]),
                            probability=bool(settings["probability"]),
                            random_state=int(settings["random_state"]))
            estimator.fit(features["paper"][fit], labels[fit])
            out["paper_svm"] = np.asarray(
                estimator.predict_proba(features["paper"][target])[:, 1], dtype=float)

        if "tanimoto_svc" in models:
            probability, _model = _fit_tanimoto(
                features["morgan"][fit], labels[fit], features["morgan"][target],
                components["tanimoto_svc"])
            out["tanimoto_svc"] = probability

        need_panel = {"tabpfn_v2", "tabfm", "catboost_full", "xgboost_full",
                      "lightgbm_full", "tabm_full"} & set(models)
        if need_panel:
            # Fold-local imputation + variance filter, fitted on fit rows only.
            fit_panel, target_panel, _audit = _fit_imputer(
                features["rdkit2d"][fit], features["rdkit2d"][target])

        if "tabpfn_v2" in models:
            probability, _audit = _tabpfn_probability(
                fit_panel, labels[fit], target_panel, components["tabpfn_v2"], seed)
            out["tabpfn_v2"] = probability

        if "tabfm" in models:
            from geroprotector.screening_blend_tabfm import _tabfm_probability
            probability, _audit = _tabfm_probability(
                self._tabfm(), self.tabfm_protocol["tabfm"], fit_panel,
                labels[fit], {"target": target_panel}, seed)
            out["tabfm"] = probability["target"]

        for model_id, source in GBM_SOURCE.items():
            if model_id not in models:
                continue
            estimator = _models(self.traditional_protocol)[source]
            estimator.fit(fit_panel.astype(np.float64), labels[fit])
            _score, probability = _positive_score(
                estimator, target_panel.astype(np.float64), model_id=source)
            out[model_id] = np.asarray(probability, dtype=float)

        if "tabm_full" in models:
            probabilities, _bundle = _alt_probabilities(
                self.root, "tabm", self.altmodels_protocol,
                features["rdkit2d"][fit], labels[fit],
                {"target": features["rdkit2d"][target]})
            out["tabm_full"] = probabilities["target"]

        for name, values in out.items():
            if len(values) != len(target) or not np.isfinite(values).all() or \
               ((values < 0) | (values > 1)).any():
                raise PanelError(f"{name} returned invalid probabilities")
        if verbose:
            print("      " + "  ".join(f"{k}:ok" for k in models), flush=True)
        return out


def add_blends(columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Equal-weight blends, computed from fold-aligned component probabilities."""
    out = dict(columns)
    for name, parts in BLENDS.items():
        out[name] = np.mean(np.stack([columns[p] for p in parts]), axis=0)
    return out


def run_folds(panel: "Panel", features: dict[str, np.ndarray], labels: np.ndarray,
              folds: Sequence[tuple[np.ndarray, np.ndarray]],
              component_seeds: Sequence[int], *, n_rows: int,
              tag: str = "fold") -> dict[str, np.ndarray]:
    """Out-of-fold probabilities for the eight base models over `folds`.

    Two passes. The first fits everything except TabFM. The second reloads the
    TabFM checkpoint and runs its folds consecutively, matching the sealed run's
    execution order. Positions are relative: `folds` holds
    (fit, target) index pairs into the rows being cross-validated, and
    `component_seeds[k]` is the foundation-model seed for fold k.
    """
    import time

    deterministic = tuple(m for m in BASE_MODELS if m != "tabfm")
    columns = {m: np.full(n_rows, np.nan) for m in BASE_MODELS}
    for index, (fit, target) in enumerate(folds):
        started = time.time()
        probabilities = panel.fold_probabilities(
            features, labels, fit, target, models=deterministic,
            component_seed=int(component_seeds[index]), verbose=False)
        for model, values in probabilities.items():
            columns[model][target] = values
        print(f"  {tag} {index + 1}/{len(folds)} pass 1 in {time.time() - started:.1f}s",
              flush=True)
    panel.reload_tabfm()
    for index, (fit, target) in enumerate(folds):
        started = time.time()
        probabilities = panel.fold_probabilities(
            features, labels, fit, target, models=("tabfm",),
            component_seed=int(component_seeds[index]), verbose=False)
        columns["tabfm"][target] = probabilities["tabfm"]
        print(f"  {tag} {index + 1}/{len(folds)} tabfm in {time.time() - started:.1f}s",
              flush=True)
    covered = np.zeros(n_rows, dtype=bool)
    for _fit, target in folds:
        covered[target] = True
    for model, values in columns.items():
        if not np.isfinite(values[covered]).all():
            raise PanelError(f"{model} out-of-fold stream is incomplete")
    return columns
