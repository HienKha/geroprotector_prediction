"""Where the blend's errors concentrate: evidence strength, drug-likeness, scaffolds.

Three stratified error analyses of the three-component blend (published SVM,
TabPFN-v2, TabFM), with the four-component blend reported alongside. The decision
threshold is fixed at 0.5 and nothing is refitted; all probabilities are read from
the sealed component streams.

A  EVIDENCE STRENGTH (DrugAge only).  DrugAge records how many independent
   publications and how many individual observations support each compound's
   lifespan effect. If the blend's false negatives concentrate on positives backed
   by a single publication, that is evidence about label quality rather than about
   the model. Recall is therefore reported per evidence stratum, with Wilson
   intervals and a Cochran-Armitage trend test across ordered strata.

   The obvious confounder is that well-studied compounds might simply resemble the
   training set more closely. That is measured, not assumed: mean maximum Tanimoto
   similarity to the training compounds is reported per stratum, and a logistic
   model of correctness on evidence strength adjusted for similarity is fitted
   alongside the unadjusted one.

B  DRUG-LIKENESS.  QED and Lipinski rule-of-five violations are computed from the
   structures and used as a coarse pharmacokinetic proxy (oral absorption and
   permeability). Accuracy, recall and specificity are reported per stratum.

C  MURCKO SCAFFOLDS.  Bemis-Murcko frameworks are computed for every cohort. Two
   questions are asked: whether errors cluster within scaffold families, and how
   much scaffold overlap exists between the training partition and each evaluation
   cohort -- the second bearing directly on whether random splitting of a chemical
   benchmark measures interpolation among analogues rather than extrapolation.

No test or external label is used to fit, tune or select anything; the labels enter
only as the outcome being stratified.
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
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from sklearn.linear_model import LogisticRegression

from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.traditional_paper405 import _read_sources, paper_split_indices

RDLogger.DisableLog("rdApp.*")


class BlendInsightError(RuntimeError):
    """Raised when a sealed input or a contract fails."""


SCHEMA = "geroprotector.blend_insight_v1"
THRESHOLD = 0.5
TRIPLE = ["probability_paper_svm", "probability_tabpfn_v2", "probability_tabfm"]
QUAD = ["probability_paper_svm", "probability_tanimoto_svc",
        "probability_tabpfn_v2", "probability_tabfm"]


def _regular_file(path: Path, role: str, expected: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise BlendInsightError(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    from geroprotector.hashing import sha256_file
    if expected is not None and sha256_file(resolved) != expected:
        raise BlendInsightError(f"{role} SHA256 differs from the protocol lock")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(centre - half), float(centre + half))


def cochran_armitage(successes: list[int], totals: list[int],
                     scores: list[float]) -> tuple[float, float]:
    """Cochran-Armitage test for trend in a proportion across ordered strata."""
    successes = np.asarray(successes, float)
    totals = np.asarray(totals, float)
    scores = np.asarray(scores, float)
    n = totals.sum()
    p = successes.sum() / n
    numerator = float(np.sum(scores * (successes - totals * p)))
    mean_score = float(np.sum(totals * scores) / n)
    variance = float(p * (1 - p) * np.sum(totals * (scores - mean_score) ** 2))
    if variance <= 0:
        return (float("nan"), float("nan"))
    z = numerator / np.sqrt(variance)
    return (float(z), float(2 * stats.norm.sf(abs(z))))


def descriptors_from_smiles(smiles: list[str]) -> pd.DataFrame:
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            rows.append({"qed": np.nan, "mol_weight": np.nan, "clogp": np.nan,
                         "h_donors": np.nan, "h_acceptors": np.nan,
                         "lipinski_violations": np.nan, "scaffold": "",
                         "parsable": False})
            continue
        mw = Descriptors.MolWt(m)
        logp = Crippen.MolLogP(m)
        hbd = Lipinski.NumHDonors(m)
        hba = Lipinski.NumHAcceptors(m)
        violations = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
        try:
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(m)
            scaffold = Chem.MolToSmiles(scaffold_mol) if scaffold_mol is not None else ""
        except Exception:
            scaffold = ""
        try:
            qed = float(QED.qed(m))
        except Exception:
            qed = np.nan
        rows.append({"qed": qed, "mol_weight": float(mw), "clogp": float(logp),
                     "h_donors": int(hbd), "h_acceptors": int(hba),
                     "lipinski_violations": violations, "scaffold": scaffold,
                     "parsable": True})
    return pd.DataFrame(rows)


def stratum_metrics(y: np.ndarray, d: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    y, d = y[mask], d[mask]
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    tp = int(((y == 1) & (d == 1)).sum())
    tn = int(((y == 0) & (d == 0)).sum())
    correct = int((y == d).sum())
    rec_lo, rec_hi = wilson(tp, pos) if pos else (np.nan, np.nan)
    return {"n": int(mask.sum()), "n_positive": pos, "n_negative": neg,
            "accuracy": correct / max(len(y), 1),
            "recall": tp / pos if pos else float("nan"),
            "recall_ci_low": rec_lo, "recall_ci_high": rec_hi,
            "specificity": tn / neg if neg else float("nan")}


def run(*, root: Path, config_path: Path, positive_path: Path,
        negative_path: Path, run_id: str) -> Path:
    if not re.fullmatch(r"blend_insight_[a-z0-9_.-]+", run_id):
        raise BlendInsightError("RUN_ID must start with blend_insight_")
    root = root.resolve()
    from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
    protocol = yaml.safe_load(
        _regular_file(config_path, "insight protocol").read_text("utf-8"))
    if protocol.get("schema_version") != f"{SCHEMA}.protocol.v1":
        raise BlendInsightError("Unknown insight protocol schema")
    if protocol["contract"].get("any_model_is_refitted") is not False:
        raise BlendInsightError("Contract differs")
    protocol_sha256 = canonical_sha256(protocol)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise BlendInsightError(f"Run directory already exists: {destination}")
    sealed = protocol["sealed_inputs"]
    for record in sealed.values():
        _regular_file(root / record["path"], record["path"], record["sha256"])

    quad = root / sealed["quad_drugage"]["path"]
    drugage = pd.read_csv(quad)
    raw = pd.read_csv(root / sealed["drugage_raw"]["path"])
    if not (raw.external_id.to_numpy() == drugage.external_id.to_numpy()).all():
        raise BlendInsightError("DrugAge row order differs between sealed files")
    drugage["publication_count"] = raw.publication_count.to_numpy()
    drugage["observation_count"] = raw.observation_count.to_numpy()
    drugage["max_tanimoto"] = raw.maximum_tanimoto_to_fitted_train.to_numpy()

    cohorts: dict[str, pd.DataFrame] = {"drugage": drugage}
    for name, key in (("d1_test", "quad_d1_test"), ("agextend", "quad_agextend")):
        cohorts[name] = pd.read_csv(root / sealed[key]["path"])
    for name, frame in cohorts.items():
        frame["p_triple"] = frame[TRIPLE].to_numpy(float) @ np.full(3, 1 / 3)
        frame["p_quad"] = frame[QUAD].to_numpy(float) @ np.full(4, 0.25)
        for m in ("triple", "quad"):
            frame[f"d_{m}"] = (frame[f"p_{m}"] >= THRESHOLD).astype(int)
        print(f"{name}: n={len(frame)}, positives={int(frame.label.sum())}", flush=True)

    # The sealed D1 prediction files carry no SMILES column, so structures are
    # rebuilt from the same raw sources and the same split every other module uses.
    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    source_frame, _audit = _read_sources(
        _regular_file(positive_path, "positive source",
                      protocol["sources"]["positive_sha256"]),
        _regular_file(negative_path, "negative source",
                      protocol["sources"]["negative_sha256"]), traditional)
    train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    if split_sha256 != protocol["paper_split"]["expected_assignment_sha256"]:
        raise BlendInsightError("Paper split differs from the sealed assignment")
    all_smiles, _ = _validated_raw_smiles(source_frame)
    all_smiles = np.asarray(all_smiles, dtype=object)
    if not np.array_equal(cohorts["d1_test"].paper_row_index.to_numpy(), test_indices):
        raise BlendInsightError("D1 test row order differs from the paper split")
    cohorts["d1_test"]["source_smiles"] = all_smiles[test_indices]
    d1_train_smiles = all_smiles[train_indices]

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".insight.work-", dir=destination.parent))
    try:
        # ================= A. evidence strength (DrugAge) ====================
        d = cohorts["drugage"]
        y, pos_mask = d.label.to_numpy(int), d.label.to_numpy(int) == 1
        evidence_rows, trend_rows = [], []
        axes = {
            "publication_count": [(1, 1, "1 publication"), (2, 2, "2 publications"),
                                  (3, 10**6, "3+ publications")],
            "observation_count": [(1, 1, "1 observation"), (2, 2, "2 observations"),
                                  (3, 4, "3-4 observations"), (5, 10**6, "5+ observations")],
        }
        for axis, bands in axes.items():
            values = d[axis].to_numpy(float)
            for model in ("triple", "quad"):
                dec = d[f"d_{model}"].to_numpy(int)
                successes, totals, scores = [], [], []
                for rank, (lo, hi, label) in enumerate(bands):
                    mask = pos_mask & (values >= lo) & (values <= hi)
                    if mask.sum() == 0:
                        continue
                    tp = int(dec[mask].sum())
                    lo_ci, hi_ci = wilson(tp, int(mask.sum()))
                    evidence_rows.append({
                        "axis": axis, "model": model, "stratum": label,
                        "stratum_rank": rank, "n_positive": int(mask.sum()),
                        "recalled": tp, "recall": tp / int(mask.sum()),
                        "recall_ci_low": lo_ci, "recall_ci_high": hi_ci,
                        "mean_probability": float(d[f"p_{model}"].to_numpy()[mask].mean()),
                        "mean_max_tanimoto_to_train": float(d.max_tanimoto.to_numpy()[mask].mean()),
                    })
                    successes.append(tp)
                    totals.append(int(mask.sum()))
                    scores.append(float(rank))
                z, p = cochran_armitage(successes, totals, scores)
                trend_rows.append({"axis": axis, "model": model, "test": "cochran_armitage",
                                   "z": z, "p_value": p, "n_strata": len(successes),
                                   "n_positive_total": int(sum(totals))})
        # logistic adjustment for chemical similarity, the obvious confounder
        adjust_rows = []
        for axis in axes:
            for model in ("triple", "quad"):
                sub = d[pos_mask]
                x_evidence = np.log(sub[axis].to_numpy(float))
                sim = sub.max_tanimoto.to_numpy(float)
                outcome = sub[f"d_{model}"].to_numpy(int)
                if len(set(outcome)) < 2:
                    continue
                unadj = LogisticRegression(penalty=None, max_iter=1000).fit(
                    x_evidence.reshape(-1, 1), outcome)
                adj = LogisticRegression(penalty=None, max_iter=1000).fit(
                    np.column_stack([x_evidence, sim]), outcome)
                adjust_rows.append({
                    "axis": axis, "model": model,
                    "unadjusted_log_odds_per_log_evidence": float(unadj.coef_[0][0]),
                    "similarity_adjusted_log_odds_per_log_evidence": float(adj.coef_[0][0]),
                    "similarity_log_odds": float(adj.coef_[0][1]),
                    "n": int(len(sub))})
        _write_csv(tmp / "A_evidence_strength_recall.csv", pd.DataFrame(evidence_rows))
        _write_csv(tmp / "A_evidence_trend_tests.csv", pd.DataFrame(trend_rows))
        _write_csv(tmp / "A_evidence_similarity_adjustment.csv", pd.DataFrame(adjust_rows))
        print("A: evidence-strength stratification done", flush=True)

        # ================= B & C. structures =================================
        structure_rows, scaffold_rows, drug_rows = [], [], []
        train_scaffolds = set(descriptors_from_smiles(
            [str(x) for x in d1_train_smiles]).scaffold) - {""}
        print(f"training partition contributes {len(train_scaffolds)} distinct scaffolds",
              flush=True)

        for name, frame in cohorts.items():
            desc = descriptors_from_smiles(frame.source_smiles.astype(str).tolist())
            merged = pd.concat([frame.reset_index(drop=True), desc], axis=1)
            merged["cohort"] = name
            structure_rows.append(merged[[
                "cohort", "label", "p_triple", "p_quad", "d_triple", "d_quad",
                "qed", "mol_weight", "clogp", "h_donors", "h_acceptors",
                "lipinski_violations", "scaffold", "parsable"]])
            y = merged.label.to_numpy(int)

            # --- B. drug-likeness strata --------------------------------------
            qed = merged.qed.to_numpy(float)
            finite = np.isfinite(qed)
            bands = [(0.0, 0.3, "QED < 0.3"), (0.3, 0.5, "QED 0.3-0.5"),
                     (0.5, 0.7, "QED 0.5-0.7"), (0.7, 1.01, "QED >= 0.7")]
            for model in ("triple", "quad"):
                dec = merged[f"d_{model}"].to_numpy(int)
                for rank, (lo, hi, label) in enumerate(bands):
                    mask = finite & (qed >= lo) & (qed < hi)
                    if mask.sum() < 5:
                        continue
                    drug_rows.append({"cohort": name, "model": model, "axis": "qed",
                                      "stratum": label, "stratum_rank": rank,
                                      **stratum_metrics(y, dec, mask)})
                viol = merged.lipinski_violations.to_numpy(float)
                for v in (0, 1, 2, 3, 4):
                    mask = np.isfinite(viol) & (viol == v)
                    if mask.sum() < 5:
                        continue
                    drug_rows.append({"cohort": name, "model": model,
                                      "axis": "lipinski_violations",
                                      "stratum": f"{v} violation(s)", "stratum_rank": v,
                                      **stratum_metrics(y, dec, mask)})

            # --- C. scaffolds --------------------------------------------------
            scaffolds = merged.scaffold.to_numpy()
            in_train = np.array([s in train_scaffolds and s != "" for s in scaffolds])
            for model in ("triple", "quad"):
                dec = merged[f"d_{model}"].to_numpy(int)
                for label, mask in (("scaffold seen in training", in_train),
                                    ("scaffold unseen", ~in_train)):
                    if mask.sum() < 5:
                        continue
                    scaffold_rows.append({
                        "cohort": name, "model": model, "analysis": "scaffold_overlap",
                        "stratum": label, "coverage": float(mask.mean()),
                        **stratum_metrics(y, dec, mask)})
            counts = pd.Series(scaffolds[scaffolds != ""]).value_counts()
            scaffold_rows.append({
                "cohort": name, "model": "n/a", "analysis": "scaffold_diversity",
                "stratum": "summary", "n": int(len(merged)),
                "n_distinct_scaffolds": int(len(counts)),
                "largest_family_size": int(counts.iloc[0]) if len(counts) else 0,
                "singleton_scaffold_fraction": float((counts == 1).mean()) if len(counts) else np.nan,
                "fraction_scaffold_seen_in_training": float(in_train.mean())})
            print(f"B,C: {name} done "
                  f"({int(len(counts))} scaffolds, "
                  f"{100*in_train.mean():.0f}% seen in training)", flush=True)

        _write_csv(tmp / "B_drug_likeness_strata.csv", pd.DataFrame(drug_rows))
        _write_csv(tmp / "C_scaffold_analysis.csv", pd.DataFrame(scaffold_rows))
        _write_csv(tmp / "structures_per_compound.csv",
                   pd.concat(structure_rows, ignore_index=True))

        manifest = {
            "schema_version": f"{SCHEMA}.run_manifest.v1", "run_id": run_id,
            "protocol_sha256": protocol_sha256,
            "explained_models": {"triple": TRIPLE, "quad": QUAD},
            "fixed_threshold": THRESHOLD,
            "any_model_is_refitted": False,
            "labels_used_only_as_stratified_outcome": True,
            "evidence_axes": {k: [b[2] for b in v] for k, v in axes.items()},
            "confounder_control": (
                "mean maximum Tanimoto similarity to the training compounds is reported "
                "per evidence stratum, and a logistic model of correctness on log evidence "
                "adjusted for that similarity is fitted alongside the unadjusted model"),
            "drug_likeness_axes": ["qed", "lipinski_violations"],
            "scaffold": "Bemis-Murcko frameworks via RDKit MurckoScaffold",
            "n_training_scaffolds": int(len(train_scaffolds)),
            "paper_split_sha256": split_sha256,
            "runtime": {"python": platform.python_version(), "platform": platform.platform()},
            "sealed_inputs_sha256": {k: sha256_file(root / v["path"])
                                     for k, v in sealed.items()},
            "existing_runs_modified": False,
        }
        atomic_write_json(tmp / "RUN_MANIFEST.json", manifest)
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
