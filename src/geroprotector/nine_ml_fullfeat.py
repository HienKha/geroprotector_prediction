"""The nine ML algorithms on the FULL RDKit2D descriptor panel, not the paper's seven.

Companion to `nine_ml_featuresets` (which ran the paper's seven descriptors) and
`nine_ml_cv5`. The split, hyperparameters, threshold and evaluation are identical;
only the feature panel changes, so the two tables are directly comparable.

FEATURE PANEL.  217 RDKit2D descriptors computed from SMILES, reduced to 205 by a
median imputation and zero-variance filter whose medians and mask are fitted on the
324 training rows alone and then applied unchanged to the 81 held-out rows.

THE SVM NEEDS SCALING HERE, AND THAT IS A DEVIATION WORTH STATING.  The published
SVM is specified with `preprocessing: none`, which is well posed on the seven
DataWarrior descriptors because they are on comparable scales. It is not well posed
on 205 RDKit2D descriptors, whose magnitudes span many orders (molecular weight
against fractional descriptors against `Ipc`). Unscaled, libsvm does not converge in
any practical time. This module therefore runs the SVM twice on the full panel:

  * `svm_original_paper`        -- kernel, C, gamma and random_state exactly as
                                   published, but with a StandardScaler fitted on
                                   training rows only;
  * `svm_original_paper_unscaled` -- literally as published, with a hard iteration
                                   cap, recording whether it converged.

Reporting both is the honest option: the scaled result is the meaningful one, and
the unscaled attempt documents why a deviation was necessary rather than asserting
it. If the unscaled fit does not converge, its metrics are recorded and flagged as
coming from a non-converged solution, never presented as a fair comparison.

The 81 held-out test rows enter no fit, no transform and no threshold decision, and
the threshold is fixed at 0.5 for every algorithm.
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

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from geroprotector.fixed_blend_paper405 import _features
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.nine_ml_featuresets import FIXED_THRESHOLD, MODEL_ORDER, _metrics
from geroprotector.screening_blend_altmodels import _apply_context, _imputer_context
from geroprotector.traditional_paper405 import (
    _load_protocol,
    _models,
    _positive_score,
    _read_sources,
    paper_split_indices,
)


class NineMLFullError(RuntimeError):
    """Raised when a contract or a leakage guard fails."""


SCHEMA = "geroprotector.nine_ml_fullfeat"
UNSCALED_MAX_ITER = 2_000_000


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def run(*, root: Path, config_path: Path, positive_path: Path, negative_path: Path,
        run_id: str, skip_unscaled: bool) -> Path:
    if not re.fullmatch(r"nineml_full_[a-z0-9_.-]+", run_id):
        raise NineMLFullError("RUN_ID must start with nineml_full_")
    root = root.resolve()
    protocol, protocol_sha256 = _load_protocol(config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise NineMLFullError(f"Run directory already exists: {destination}")

    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    frame, audit = _read_sources(positive_path.resolve(), negative_path.resolve(), traditional)
    train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    labels = frame["label"].to_numpy(int)
    y_train, y_test = labels[train_indices], labels[test_indices]
    smiles, _ = _validated_raw_smiles(frame)
    raw = _features(frame, smiles, fixed_protocol)["rdkit2d"]

    context = _imputer_context(raw[train_indices])        # TRAIN ROWS ONLY
    panel = _apply_context(context, raw).astype(np.float64)
    from rdkit.Chem import Descriptors
    names_all = [n for n, _f in Descriptors._descList]
    keep = np.flatnonzero(np.asarray(context["finite_any_mask"], bool))
    keep = keep[np.asarray(context["varying_after_imputation_mask"], bool)]
    names = [names_all[i] for i in keep]
    if len(names) != panel.shape[1]:
        raise NineMLFullError("Descriptor names do not match the panel width")
    x_train, x_test = panel[train_indices], panel[test_indices]
    print(f"full panel {raw.shape[1]} -> {panel.shape[1]} after train-only imputation "
          f"and variance filter; train {x_train.shape}, test {x_test.shape}", flush=True)

    settings = protocol["models"]["svm_original_paper"]
    models = _models(protocol)
    # Linear regression is scale-sensitive and must be scaled on this panel. Unscaled,
    # the design matrix has condition number ~2.8e51 (RDKit `Ipc` reaches 6.3e35 while
    # other columns are order 1), lstsq's rank cutoff zeroes every coefficient, and the
    # model collapses to an intercept-only constant. That is a numerical failure, not a
    # result, so the scaled form is used. kNN and logistic regression already carry a
    # train-only scaler; the tree ensembles are scale-invariant and are left unscaled.
    models["linear_regression"] = Pipeline([
        ("scaler", StandardScaler()), ("model", LinearRegression())])
    # published hyperparameters, but scaled -- the deviation documented in the docstring
    models["svm_original_paper"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel=settings["kernel"], C=float(settings["C"]),
                      gamma=float(settings["gamma"]),
                      probability=bool(settings["probability"]),
                      random_state=int(settings["random_state"])))])

    metric_rows, prediction_rows = [], []
    order = list(MODEL_ORDER)
    convergence = {}
    for model_id in order:
        model = models[model_id]
        model.fit(x_train, y_train)
        score, probability = _positive_score(model, x_test, model_id=model_id)
        metric_rows.append({"model_id": model_id, "feature_set": "full_rdkit2d",
                            "n_features": int(x_train.shape[1]),
                            "preprocessing": ("standard_scaler_train_only"
                                              if model_id in ("svm_original_paper", "knn",
                                                              "logistic_regression",
                                                              "linear_regression")
                                              else "none"),
                            **_metrics(y_test, score, probability)})
        for position, row in enumerate(test_indices):
            prediction_rows.append({
                "model_id": model_id, "paper_row_index": int(row),
                "label": int(y_test[position]), "ranking_score": float(score[position]),
                "probability": float(probability[position])})
        print(f"  {model_id:22s} AP+ {metric_rows[-1]['auprc_average_precision_positive']:.4f}"
              f"  MCC {metric_rows[-1]['mcc']:.4f}", flush=True)

    # the literal published specification, with a cap, recorded either way
    if not skip_unscaled:
        print(f"  attempting the unscaled published SVM (max_iter={UNSCALED_MAX_ITER})...",
              flush=True)
        failure = None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                raw_svm = SVC(kernel=settings["kernel"], C=float(settings["C"]),
                              gamma=float(settings["gamma"]), probability=True,
                              random_state=int(settings["random_state"]),
                              max_iter=UNSCALED_MAX_ITER)
                raw_svm.fit(x_train, y_train)
                converged = not any(issubclass(c.category, ConvergenceWarning)
                                    for c in caught)
        except Exception as error:            # sklearn refuses outright on this panel
            failure, converged = f"{type(error).__name__}: {error}", False
        convergence["svm_original_paper_unscaled"] = {
            "fitted": failure is None, "converged": bool(converged),
            "max_iter": UNSCALED_MAX_ITER, "failure": failure}
        if failure is None:
            score, probability = _positive_score(
                raw_svm, x_test, model_id="svm_original_paper_unscaled")
            metric_rows.append({"model_id": "svm_original_paper_unscaled",
                                "feature_set": "full_rdkit2d",
                                "n_features": int(x_train.shape[1]),
                                "preprocessing": "none_as_published",
                                "converged": bool(converged),
                                **_metrics(y_test, score, probability)})
            for position, row in enumerate(test_indices):
                prediction_rows.append({
                    "model_id": "svm_original_paper_unscaled",
                    "paper_row_index": int(row), "label": int(y_test[position]),
                    "ranking_score": float(score[position]),
                    "probability": float(probability[position])})
            print(f"  unscaled published SVM converged: {converged}", flush=True)
        else:
            # No metrics recorded: an estimator that could not be fitted has no
            # performance to report, and inventing one would misrepresent the baseline.
            print(f"  unscaled published SVM could NOT be fitted -> {failure}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".ninefull.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "metrics.csv", metrics)
        _write_csv(tmp / "predictions.csv", pd.DataFrame(prediction_rows))
        _write_csv(tmp / "feature_list.csv",
                   pd.DataFrame({"index": range(len(names)), "feature": names}))
        atomic_write_json(tmp / "RUN_MANIFEST.json", {
            "schema_version": f"{SCHEMA}.run_manifest.v1", "run_id": run_id,
            "protocol_sha256": protocol_sha256, "paper_split_sha256": split_sha256,
            "feature_set": "full RDKit2D panel",
            "n_features_raw": int(raw.shape[1]), "n_features_used": int(panel.shape[1]),
            "imputation_and_variance_mask_fitted_on": "train rows only",
            "models": order, "model_settings": protocol["models"],
            "linear_regression_deviation": (
                "scaled with a train-only StandardScaler; unscaled OLS on this panel is "
                "rank-deficient (condition number ~2.8e51) and collapses to an "
                "intercept-only constant prediction"),
            "svm_deviation": (
                "the published SVM specifies no preprocessing, which is not well posed on "
                "205 RDKit2D descriptors spanning many orders of magnitude; a StandardScaler "
                "fitted on training rows only was therefore added, and the literal unscaled "
                "specification is reported separately with its convergence status"),
            "svm_convergence": convergence,
            "fixed_threshold": FIXED_THRESHOLD, "threshold_is_tuned": False,
            "test_rows_used_in_any_fit_or_transform": False,
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
    print(json.dumps({"run": str(destination), "status": "COMPLETE"}, indent=2))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--skip-unscaled", action="store_true")
    a = parser.parse_args(argv)
    run(root=a.root, config_path=a.config, positive_path=a.positive,
        negative_path=a.negative, run_id=a.run_id, skip_unscaled=a.skip_unscaled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
