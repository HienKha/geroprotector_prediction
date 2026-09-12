"""Stage A0: the only prediction streams Experiment A is missing.

`prespecified insight-analysis protocol` section 5.3 requires CatBoost, XGBoost and LightGBM
probabilities on the DrugAge and AgeXtend cohorts.  Those three algorithms have
been evaluated on the full RDKit2D panel for D1 test (`nineml_full_scaled_20260823`)
and for 5-fold train OOF (`nineml_cv5_full_20260823`), but never on the external
cohorts.

The contract of section 5.3 is followed literally:

  1. refit the exact frozen configuration on the 324 D1 training compounds only;
  2. reproduce the existing D1 held-out probability stream within a tolerance that
     is declared in the protocol *before* the comparison;
  3. stop if parity fails -- no parameter is adjusted to obtain parity;
  4. only then score the already curated external structures, using no outcome;
  5. write everything to this new run and nothing else.

The corrected external structures are read from the exact `source_smiles` strings
used by every other component stream in Experiment A. Molecular identity
descriptors may use standardized parents, but model-family comparisons must not
mix prediction inputs within a cohort.
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

import numpy as np
import pandas as pd

from geroprotector.fixed_blend_paper405 import _features
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.screening_blend_altmodels import (
    _apply_context,
    _imputer_context,
    _rdkit2d_from_smiles,
)
from geroprotector.traditional_paper405 import (
    _load_protocol,
    _models,
    _positive_score,
    _read_sources,
    paper_split_indices,
)


class GBMExternalError(RuntimeError):
    """Raised when a contract or a parity guard fails."""


SCHEMA = "geroprotector.gbm_external_predictions"
MODELS = ("catboost", "xgboost", "lightgbm")
PARITY_RUN = "nineml_full_scaled_20260823"
COHORTS = {
    "drugage": "screeningblend_tabfm_20260821/external_predictions_drugage.csv",
    "agextend": "screeningblend_tabfm_20260821/external_predictions_agextend.csv",
}
# Declared before the comparison is made.  Both stages fit the identical estimator
# on the identical matrix, so the only expected difference is float replay noise.
PARITY_ATOL = 1e-10


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def run(*, root: Path, config_path: Path, positive_path: Path, negative_path: Path,
        run_id: str) -> Path:
    if not re.fullmatch(r"gbm_external_[a-z0-9_.-]+", run_id):
        raise GBMExternalError("RUN_ID must start with gbm_external_")
    root = root.resolve()
    protocol, protocol_sha256 = _load_protocol(config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise GBMExternalError(f"Run directory already exists: {destination}")

    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    frame, audit = _read_sources(positive_path.resolve(), negative_path.resolve(), traditional)
    train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    labels = frame["label"].to_numpy(int)
    y_train = labels[train_indices]
    smiles, _ = _validated_raw_smiles(frame)
    raw = _features(frame, smiles, fixed_protocol)["rdkit2d"]

    context = _imputer_context(raw[train_indices])          # TRAIN ROWS ONLY
    panel = _apply_context(context, raw).astype(np.float64)
    x_train, x_test = panel[train_indices], panel[test_indices]
    from rdkit.Chem import Descriptors
    descriptor_names = [n for n, _f in Descriptors._descList]
    print(f"full panel {raw.shape[1]} -> {panel.shape[1]}; train {x_train.shape}",
          flush=True)

    fitted = {}
    for model_id in MODELS:
        model = _models(protocol)[model_id]
        model.fit(x_train, y_train)
        fitted[model_id] = model
        print(f"  fitted {model_id}", flush=True)

    # ---- step 2/3: parity against the sealed D1 held-out stream -----------------
    sealed = pd.read_csv(root / "outputs" / PARITY_RUN / "predictions.csv")
    parity_rows = []
    for model_id in MODELS:
        reference = sealed[sealed["model_id"] == model_id].copy()
        if reference.empty:
            raise GBMExternalError(f"No sealed D1 test stream for {model_id}")
        reference = reference.set_index("paper_row_index").loc[list(test_indices)]
        _score, probability = _positive_score(fitted[model_id], x_test, model_id=model_id)
        deviation = float(np.max(np.abs(probability - reference["probability"].to_numpy())))
        parity_rows.append({"model_id": model_id, "cohort": "d1_test",
                            "reference_run": PARITY_RUN, "n": int(len(reference)),
                            "max_abs_deviation": deviation, "tolerance": PARITY_ATOL,
                            "passed": bool(deviation <= PARITY_ATOL)})
        print(f"  parity {model_id:10s} max|dp| = {deviation:.3e}", flush=True)
    parity = pd.DataFrame(parity_rows)
    if not bool(parity["passed"].all()):
        raise GBMExternalError(
            "Parity with the sealed D1 held-out stream failed; refusing to score "
            f"external cohorts.\n{parity.to_string(index=False)}")

    # ---- step 4: score the curated external structures, outcome-blind ----------
    external = {}
    for cohort, relative in COHORTS.items():
        source = pd.read_csv(root / "outputs" / relative,
                             usecols=["external_id", "source_smiles"])
        raw_external = _rdkit2d_from_smiles(
            source["source_smiles"].astype(str).tolist(), descriptor_names)
        x_external = _apply_context(context, raw_external).astype(np.float64)
        if x_external.shape[1] != x_train.shape[1]:
            raise GBMExternalError(f"{cohort}: external panel width differs from train")
        out = pd.DataFrame({"external_id": source["external_id"]})
        for model_id in MODELS:
            score, probability = _positive_score(
                fitted[model_id], x_external, model_id=model_id)
            out[f"probability_{model_id}_full"] = probability
            out[f"score_{model_id}_full"] = score
        external[cohort] = out
        print(f"  scored {cohort}: {out.shape}", flush=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".gbmext.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "parity_checks.csv", parity)
        for cohort, out in external.items():
            _write_csv(tmp / f"external_predictions_{cohort}.csv", out)
        atomic_write_json(tmp / "RUN_MANIFEST.json", {
            "schema_version": f"{SCHEMA}.run_manifest.v1", "run_id": run_id,
            "protocol_sha256": protocol_sha256, "paper_split_sha256": split_sha256,
            "models": list(MODELS),
            "model_settings": {m: protocol["models"][m] for m in MODELS},
            "feature_set": "full RDKit2D panel, train-only imputation and variance filter",
            "n_features": int(x_train.shape[1]),
            "fitted_on": "D1 train only (324 rows)",
            "parity_reference_run": PARITY_RUN,
            "parity_tolerance_declared_before_comparison": PARITY_ATOL,
            "parity_max_abs_deviation": {r["model_id"]: r["max_abs_deviation"]
                                         for r in parity_rows},
            "external_structure_source": COHORTS,
            "external_columns_read": ["external_id", "source_smiles"],
            "external_model_input_representation": "source_smiles_aligned_across_all_components",
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
