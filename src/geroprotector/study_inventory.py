"""Stage 0 of the insight-analysis suite: a strictly read-only inventory.

Section 2 of `prespecified insight-analysis protocol` requires an audit *before* any new experiment
code runs.  This module opens every sealed input the later stages depend on,
records its hash, re-derives the D1 split, enumerates which model x cohort
prediction streams already exist, and writes the result to a new immutable run
directory.  It never writes outside that directory and never fits a model.

The one thing it deliberately does NOT do is repair anything.  If the package
manifest is stale, or a sealed artifact hash disagrees with its own
COMPLETED.json, the fact is recorded and the exit status turns non-zero.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract
from geroprotector.traditional_paper405 import (
    _load_protocol,
    _read_sources,
    paper_split_indices,
)

SCHEMA = "geroprotector.study_inventory"

# Every sealed run a later stage reads.  `role` documents why it is needed.
SEALED_RUNS = {
    "weightedblend405_20260817": "equal-weight anchor; full-train OOF components",
    "screeningblend405_20260817": "paper_svm / tanimoto_svc / tabpfn_v2 streams",
    "screeningblend_external_20260817": "DrugAge + AgeXtend cohort construction",
    "screeningblend_tabfm_20260821": "tabfm streams, all four cohorts",
    "screeningblend_altmodels_20260819": "bishop / tabm / tabnet streams",
    "quad_blend_20260822": "GB4 equal-quarters streams",
    "nineml_paper7_20260822": "nine ML, seven descriptors, D1 test",
    "nineml_cv5_20260822": "nine ML, seven descriptors, 5-fold OOF",
    "nineml_full_scaled_20260823": "nine ML, full RDKit2D, D1 test (corrected)",
    "nineml_cv5_full_20260823": "GBM full-panel 5-fold OOF",
    "blend_xai_v4_20260822": "triple vs quad explainability",
    "blend_xai_v5_20260823": "SHAP / LIME / DCA",
    "blend_insight_20260823": "QED strata, scaffold, evidence strength",
    "ablation_catboost_20260823": "candidate ablation grid",
    "ablation_lightgbm_20260823": "candidate ablation grid",
    "ablation_xgboost_20260823": "candidate ablation grid",
    "ablation_tabm_20260823": "candidate ablation grid",
    "ablation_bishop_20260823": "candidate ablation grid",
}

# The ten-model panel of section 3, and where each stream lives per cohort.
# `None` marks a gap that a later stage has to fill under a parity contract.
PANEL = ("paper_svm", "tanimoto_svc", "tabpfn_v2", "tabfm", "catboost_full",
         "xgboost_full", "lightgbm_full", "tabm_full", "blend3_equal", "gb4_equal")

STREAM_MAP = {
    ("paper_svm", "d1_train_oof"): ("screeningblend_tabfm_20260821/d1_train_oof_predictions.csv", "probability_paper_svm"),
    ("paper_svm", "d1_test"): ("screeningblend_tabfm_20260821/d1_test_predictions.csv", "probability_paper_svm"),
    ("paper_svm", "drugage"): ("screeningblend_tabfm_20260821/external_predictions_drugage.csv", "probability_paper_svm"),
    ("paper_svm", "agextend"): ("screeningblend_tabfm_20260821/external_predictions_agextend.csv", "probability_paper_svm"),
    ("tanimoto_svc", "d1_train_oof"): ("screeningblend_tabfm_20260821/d1_train_oof_predictions.csv", "probability_tanimoto_svc"),
    ("tanimoto_svc", "d1_test"): ("screeningblend_tabfm_20260821/d1_test_predictions.csv", "probability_tanimoto_svc"),
    ("tanimoto_svc", "drugage"): ("screeningblend_tabfm_20260821/external_predictions_drugage.csv", "probability_tanimoto_svc"),
    ("tanimoto_svc", "agextend"): ("screeningblend_tabfm_20260821/external_predictions_agextend.csv", "probability_tanimoto_svc"),
    ("tabpfn_v2", "d1_train_oof"): ("quad_blend_20260822/cv5_train_oof_predictions.csv", "probability_tabpfn_v2"),
    ("tabpfn_v2", "d1_test"): ("quad_blend_20260822/predictions_d1_test.csv", "probability_tabpfn_v2"),
    ("tabpfn_v2", "drugage"): ("screeningblend_tabfm_20260821/external_predictions_drugage.csv", "probability_tabpfn_v2"),
    ("tabpfn_v2", "agextend"): ("screeningblend_tabfm_20260821/external_predictions_agextend.csv", "probability_tabpfn_v2"),
    ("tabfm", "d1_train_oof"): ("screeningblend_tabfm_20260821/d1_train_oof_predictions.csv", "probability_tabfm"),
    ("tabfm", "d1_test"): ("screeningblend_tabfm_20260821/d1_test_predictions.csv", "probability_tabfm"),
    ("tabfm", "drugage"): ("screeningblend_tabfm_20260821/external_predictions_drugage.csv", "probability_tabfm"),
    ("tabfm", "agextend"): ("screeningblend_tabfm_20260821/external_predictions_agextend.csv", "probability_tabfm"),
    ("tabm_full", "d1_train_oof"): ("screeningblend_altmodels_20260819/d1_train_oof_predictions.csv", "probability_tabm"),
    ("tabm_full", "d1_test"): ("screeningblend_altmodels_20260819/d1_test_predictions.csv", "probability_tabm"),
    ("tabm_full", "drugage"): ("screeningblend_altmodels_20260819/external_predictions_drugage.csv", "probability_tabm"),
    ("tabm_full", "agextend"): ("screeningblend_altmodels_20260819/external_predictions_agextend.csv", "probability_tabm"),
    ("catboost_full", "d1_train_oof"): ("nineml_cv5_full_20260823/cv5_train_oof_predictions.csv", "probability_catboost"),
    ("catboost_full", "d1_test"): ("nineml_full_scaled_20260823/predictions.csv", "model_id==catboost"),
    ("catboost_full", "drugage"): (None, None),
    ("catboost_full", "agextend"): (None, None),
    ("xgboost_full", "d1_train_oof"): ("nineml_cv5_full_20260823/cv5_train_oof_predictions.csv", "probability_xgboost"),
    ("xgboost_full", "d1_test"): ("nineml_full_scaled_20260823/predictions.csv", "model_id==xgboost"),
    ("xgboost_full", "drugage"): (None, None),
    ("xgboost_full", "agextend"): (None, None),
    ("lightgbm_full", "d1_train_oof"): ("nineml_cv5_full_20260823/cv5_train_oof_predictions.csv", "probability_lightgbm"),
    ("lightgbm_full", "d1_test"): ("nineml_full_scaled_20260823/predictions.csv", "model_id==lightgbm"),
    ("lightgbm_full", "drugage"): (None, None),
    ("lightgbm_full", "agextend"): (None, None),
    ("blend3_equal", "*"): ("derived", "mean(paper_svm, tabpfn_v2, tabfm)"),
    ("gb4_equal", "*"): ("derived", "mean(paper_svm, tanimoto_svc, tabpfn_v2, tabfm)"),
}


def _md(frame: pd.DataFrame) -> str:
    """Plain GitHub-flavoured markdown table; avoids a `tabulate` dependency."""
    cols = [str(c) for c in frame.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in rec]
            for rec in frame.itertuples(index=False, name=None)]
    widths = [max(len(cols[i]), *(len(r[i]) for r in rows)) if rows else len(cols[i])
              for i in range(len(cols))]
    def line(cells):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    out = [line(cols), "| " + " | ".join("-" * w for w in widths) + " |"]
    out += [line(r) for r in rows]
    return "\n".join(out)


def _git_status(root: Path) -> dict:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                             capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return {"repository": False, "stderr": out.stderr.strip()[:400]}
        return {"repository": True, "dirty_paths": out.stdout.strip().splitlines()}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"repository": False, "error": repr(exc)}


def _verify_sealed(root: Path) -> tuple[pd.DataFrame, list[str]]:
    """Re-hash every artifact of every sealed run against its COMPLETED.json."""
    rows, problems = [], []
    for run_id, role in SEALED_RUNS.items():
        run = root / "outputs" / run_id
        if not run.is_dir():
            problems.append(f"missing sealed run: {run_id}")
            rows.append({"run_id": run_id, "role": role, "present": False,
                         "completed_schema": "", "artifacts": 0, "hash_mismatches": -1, "status": "MISSING"})
            continue
        completed = run / "COMPLETED.json"
        if not completed.is_file():
            problems.append(f"{run_id}: no COMPLETED.json")
            rows.append({"run_id": run_id, "role": role, "present": True,
                         "completed_schema": "", "artifacts": 0, "hash_mismatches": -1, "status": "NO_COMPLETED"})
            continue
        payload = json.loads(completed.read_text())
        recorded = payload.get("artifact_hashes") or payload.get("artifacts") or {}
        schema = "artifact_hashes"
        if not recorded:
            # Runs sealed before the artifact_hashes convention record individual
            # `<stem>_sha256` keys instead.  Resolve each to a real file.
            schema = "legacy_stem_sha256"
            for key, value in payload.items():
                if not (key.endswith("_sha256") and isinstance(value, str)):
                    continue
                stem = key[: -len("_sha256")]
                hits = [c for c in run.rglob("*")
                        if c.is_file() and c.stem == stem]
                if len(hits) == 1:
                    recorded[str(hits[0].relative_to(run))] = value
                else:
                    problems.append(f"{run_id}: cannot resolve legacy key {key} "
                                    f"({len(hits)} candidate files)")
        mismatch = 0
        for relative, expected in recorded.items():
            target = run / relative
            if not target.is_file():
                mismatch += 1
                problems.append(f"{run_id}: artifact absent {relative}")
                continue
            if sha256_file(target) != expected:
                mismatch += 1
                problems.append(f"{run_id}: hash mismatch {relative}")
        rows.append({"run_id": run_id, "role": role, "present": True,
                     "completed_schema": schema,
                     "artifacts": len(recorded), "hash_mismatches": mismatch,
                     "status": "OK" if mismatch == 0 else "MISMATCH"})
    return pd.DataFrame(rows), problems


def _cohort_counts(root: Path, positive: Path, negative: Path) -> tuple[pd.DataFrame, dict]:
    fixed_protocol, fixed_sha = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    frame, audit = _read_sources(positive.resolve(), negative.resolve(), traditional)
    train_indices, test_indices, split_sha = paper_split_indices(traditional)
    labels = frame["label"].to_numpy(int)
    rows = [
        {"cohort": "d1_all", "n": len(labels), "positive": int(labels.sum()),
         "negative": int((labels == 0).sum())},
        {"cohort": "d1_train", "n": len(train_indices),
         "positive": int(labels[train_indices].sum()),
         "negative": int((labels[train_indices] == 0).sum())},
        {"cohort": "d1_test", "n": len(test_indices),
         "positive": int(labels[test_indices].sum()),
         "negative": int((labels[test_indices] == 0).sum())},
    ]
    for cohort, relative in (("drugage", "external_predictions_drugage.csv"),
                             ("agextend", "external_predictions_agextend.csv")):
        path = root / "outputs" / "screeningblend_tabfm_20260821" / relative
        ext = pd.read_csv(path)
        lab = pd.to_numeric(ext["label"], errors="coerce")
        rows.append({"cohort": cohort, "n": int(len(ext)),
                     "positive": int((lab == 1).sum()), "negative": int((lab == 0).sum())})
    meta = {"paper_split_sha256": split_sha, "fixed_protocol_sha256": fixed_sha,
            "data_audit": audit}
    return pd.DataFrame(rows), meta


def _stream_matrix(root: Path) -> tuple[pd.DataFrame, list[str]]:
    rows, gaps = [], []
    cohorts = ("d1_train_oof", "d1_test", "drugage", "agextend")
    for model in PANEL:
        for cohort in cohorts:
            key = (model, cohort) if (model, cohort) in STREAM_MAP else (model, "*")
            relative, column = STREAM_MAP.get(key, (None, None))
            if relative == "derived":
                rows.append({"model_id": model, "cohort": cohort, "source": "derived",
                             "column": column, "available": True, "n_rows": -1})
                continue
            if relative is None:
                gaps.append(f"{model} / {cohort}")
                rows.append({"model_id": model, "cohort": cohort, "source": "",
                             "column": "", "available": False, "n_rows": 0})
                continue
            path = root / "outputs" / relative
            available, n_rows = False, 0
            if path.is_file():
                head = pd.read_csv(path, nrows=0)
                if column.startswith("model_id=="):
                    wanted = column.split("==", 1)[1]
                    full = pd.read_csv(path, usecols=["model_id"])
                    n_rows = int((full["model_id"] == wanted).sum())
                    available = n_rows > 0
                else:
                    available = column in head.columns
                    if available:
                        n_rows = int(len(pd.read_csv(path, usecols=[column])))
            if not available:
                gaps.append(f"{model} / {cohort} (column {column} not in {relative})")
            rows.append({"model_id": model, "cohort": cohort, "source": relative,
                         "column": column, "available": available, "n_rows": n_rows})
    return pd.DataFrame(rows), gaps


def _packages() -> dict:
    import importlib
    versions = {}
    for name in ("numpy", "pandas", "sklearn", "scipy", "torch", "rdkit", "tabpfn",
                 "tabfm", "xgboost", "lightgbm", "catboost", "shap", "lime"):
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"UNAVAILABLE ({type(exc).__name__})"
    return versions


def run(*, root: Path, positive: Path, negative: Path, run_id: str) -> tuple[Path, list[str]]:
    root = root.resolve()
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise SystemExit(f"Run directory already exists: {destination}")

    sealed, problems = _verify_sealed(root)
    counts, meta = _cohort_counts(root, positive, negative)
    matrix, gaps = _stream_matrix(root)
    packages = _packages()

    manifest_stale = None
    manifest_script = root / "scripts" / "freeze_package_manifest.py"
    if manifest_script.is_file():
        out = subprocess.run([sys.executable, str(manifest_script), "--check"],
                             cwd=root, capture_output=True, text=True)
        manifest_stale = {"returncode": out.returncode,
                          "stdout": out.stdout.strip()[:2000]}
        if out.returncode != 0:
            problems.append(f"freeze_package_manifest --check failed pre-existing: "
                            f"{out.stdout.strip()[:200]}")

    checkpoints = {}
    checkpoint_path = Path(os.environ.get(
        "TABPFN_V2_CHECKPOINT", "checkpoints/tabpfn-v2-classifier.ckpt"))
    for label, path in (("tabpfn_v2", checkpoint_path),):
        checkpoints[label] = sha256_file(path) if path.is_file() else "ABSENT"

    external = {}
    for label, path in (
            ("drugage_build5", Path(os.environ.get(
                "DRUGAGE_SOURCE_DIR", "external_data/drugage_build5"))),
            ("agextend_2024", Path(os.environ.get(
                "AGEXTEND_SOURCE_DIR", "external_data/agextend_2024")))):
        if path.is_dir():
            external[label] = {p.name: sha256_file(p) for p in sorted(path.iterdir())
                               if p.is_file()}
        else:
            external[label] = "ABSENT"
            problems.append(f"external source directory absent: {path}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".inventory.work-", dir=destination.parent))
    try:
        sealed.to_csv(tmp / "sealed_run_verification.csv", index=False, lineterminator="\n")
        counts.to_csv(tmp / "cohort_counts.csv", index=False, lineterminator="\n")
        matrix.to_csv(tmp / "prediction_stream_matrix.csv", index=False, lineterminator="\n")
        atomic_write_json(tmp / "inventory.json", {
            "schema_version": f"{SCHEMA}.inventory.v1",
            "git": _git_status(root),
            "paper_split_sha256": meta["paper_split_sha256"],
            "fixed_protocol_sha256": meta["fixed_protocol_sha256"],
            "d1_data_audit": meta["data_audit"],
            "source_files": {
                "positive": {"path": str(positive), "sha256": sha256_file(positive)},
                "negative": {"path": str(negative), "sha256": sha256_file(negative)}},
            "package_versions": packages,
            "model_checkpoints": checkpoints,
            "external_source_hashes": external,
            "package_manifest_check": manifest_stale,
            "prediction_stream_gaps": gaps,
            "problems": problems,
            "runtime": {"python": platform.python_version(),
                        "platform": platform.platform(),
                        "executable": sys.executable},
            "existing_runs_modified": False,
            "wrote_outside_run_directory": False})
        lines = ["# Insight-analysis suite -- Stage 0 inventory", "",
                 f"run_id: `{run_id}`", "",
                 "## Cohort counts", "", _md(counts), "",
                 "## Sealed run verification", "",
                 _md(sealed), "",
                 "## Prediction stream availability", "",
                 _md(matrix), "",
                 "## Gaps that later stages must fill", ""]
        lines += [f"- {g}" for g in gaps] or ["- none"]
        lines += ["", "## Problems recorded", ""]
        lines += [f"- {p}" for p in problems] or ["- none"]
        (tmp / "inventory_report.md").write_text("\n".join(lines) + "\n")
        atomic_write_json(tmp / "COMPLETED.json", {
            "schema_version": f"{SCHEMA}.completed.v1",
            "status": "COMPLETE_WITH_PROBLEMS" if problems else "COMPLETE",
            "run_id": run_id,
            "artifact_hashes": {str(p.relative_to(tmp)): sha256_file(p)
                                for p in sorted(tmp.rglob("*"))
                                if p.is_file() and p.name != "COMPLETED.json"}})
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return destination, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    a = parser.parse_args(argv)
    destination, problems = run(root=a.root, positive=a.positive,
                                negative=a.negative, run_id=a.run_id)
    print(json.dumps({"run": str(destination),
                      "problems": problems}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
