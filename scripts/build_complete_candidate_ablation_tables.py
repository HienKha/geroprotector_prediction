#!/usr/bin/env python3
"""Build the complete six-family fifth-component ablation supplement.

The script is deliberately presentation-only. It verifies and reads immutable
``ablation_grid.csv`` artifacts, then writes a combined CSV, four LaTeX
longtables mirroring the manuscript's core ablation tables, a standalone
supplement wrapper, and a source-hash manifest. It never refits a model and
never edits a source run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


RUNS = {
    "CatBoost": "ablation_catboost_20260823",
    "LightGBM": "ablation_lightgbm_20260823",
    "XGBoost": "ablation_xgboost_20260823",
    "BiSHop": "ablation_bishop_20260823",
    "TabM": "ablation_tabm_20260823",
    "TabNet": "ablation_tabnet_20260828",
}
COMPONENTS = [
    "has_paper_svm",
    "has_tanimoto_svc",
    "has_tabpfn_v2",
    "has_tabfm",
    "has_candidate",
]
METRICS = [
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "Kappa",
    "AUROC",
    "AUPRC",
    "Brier",
    "MCC",
    "MacroF1",
]
COHORTS = ["d1_test", "cv5_train_oof"]
DISPLAY = {
    "Accuracy": "Acc.",
    "Sensitivity": "Sens.",
    "Specificity": "Spec.",
    "Kappa": "$\\kappa$",
    "AUROC": "AUROC",
    "AUPRC": "AUPRC",
    "Brier": "Brier",
    "MCC": "MCC",
    "MacroF1": "M-$F_1$",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_completed(run_dir: Path) -> dict:
    completed_path = run_dir / "COMPLETED.json"
    manifest_path = run_dir / "RUN_MANIFEST.json"
    if not completed_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Unsealed run: {run_dir}")
    completed = json.loads(completed_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if completed.get("status") != "COMPLETE":
        raise RuntimeError(f"Run is not COMPLETE: {run_dir}")
    if completed.get("run_manifest_sha256") != sha256(manifest_path):
        raise RuntimeError(f"Manifest hash mismatch: {run_dir}")
    for relative, expected in completed.get("artifact_hashes", {}).items():
        artifact = run_dir / relative
        if not artifact.is_file() or sha256(artifact) != expected:
            raise RuntimeError(f"Artifact hash mismatch: {artifact}")
    return manifest


def load_all(root: Path) -> tuple[pd.DataFrame, list[dict]]:
    outputs = root / "outputs"
    frames: list[pd.DataFrame] = []
    audit: list[dict] = []
    for family, run_id in RUNS.items():
        run_dir = outputs / run_id
        manifest = verify_completed(run_dir)
        expected_candidate = family.lower()
        if family == "BiSHop":
            expected_candidate = "bishop"
        if manifest.get("candidate_fifth_component") != expected_candidate:
            raise RuntimeError(f"Candidate mismatch in {run_id}")
        grid_path = run_dir / "ablation_grid.csv"
        frame = pd.read_csv(grid_path)
        required = {
            "cohort", "metric", "subset", "k", "value", "ci_low", "ci_high",
            "q_vs_full", *COMPONENTS,
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RuntimeError(f"Missing columns in {grid_path}: {missing}")
        if set(frame["cohort"]) != set(COHORTS):
            raise RuntimeError(f"Unexpected cohorts in {grid_path}")
        if set(frame["metric"]) != set(METRICS):
            raise RuntimeError(f"Unexpected metrics in {grid_path}")
        for cohort in COHORTS:
            part = frame[frame.cohort == cohort]
            if len(part) != 31 * len(METRICS):
                raise RuntimeError(f"Wrong row count for {family} {cohort}")
            if part["subset"].nunique() != 31:
                raise RuntimeError(f"Not all 31 subsets exist for {family} {cohort}")
            key_counts = part.groupby(["subset", "metric"]).size()
            if not (key_counts == 1).all():
                raise RuntimeError(f"Duplicate or missing subset-metric rows: {family} {cohort}")
            component_patterns = (
                part.drop_duplicates("subset")[COMPONENTS].astype(int)
            )
            patterns = {tuple(row) for row in component_patterns.to_numpy()}
            expected = {
                tuple((mask >> bit) & 1 for bit in range(5))
                for mask in range(1, 32)
            }
            if patterns != expected:
                raise RuntimeError(f"Incomplete component combinations: {family} {cohort}")
        frame.insert(0, "candidate_family", family)
        frame.insert(1, "source_run", run_id)
        frames.append(frame)
        audit.append({
            "candidate_family": family,
            "source_run": run_id,
            "candidate": manifest["candidate_fifth_component"],
            "status": "COMPLETE",
            "n_subsets_per_cohort": 31,
            "n_metrics": len(METRICS),
            "n_cohorts": len(COHORTS),
            "grid_rows": len(frame),
            "grid_sha256": sha256(grid_path),
            "manifest_sha256": sha256(run_dir / "RUN_MANIFEST.json"),
            "completed_sha256": sha256(run_dir / "COMPLETED.json"),
        })
    return pd.concat(frames, ignore_index=True), audit


def tick(value: object) -> str:
    return "$\\checkmark$" if int(value) else ""


def fmt(value: float, bold: bool = False, star: bool = False) -> str:
    text = f"{value:.3f}"
    if bold:
        text = f"\\textbf{{{text}}}"
    if star:
        text += "$^{*}$"
    return text


def ordered_subsets(frame: pd.DataFrame) -> list[str]:
    one_metric = frame[frame.metric == "Accuracy"].copy()
    one_metric["source_order"] = range(len(one_metric))
    return one_metric.sort_values(["k", "source_order"])["subset"].tolist()


def point_table(
    all_rows: pd.DataFrame,
    cohort: str,
    table_id: str,
    caption: str,
    supplement_heading: bool = False,
) -> str:
    lines = [
        "\\begin{landscape}",
    ]
    if supplement_heading:
        lines += [
            "\\section*{Additional file 1: Complete alternative-component ablation}",
            "This file reports every nonempty subset for each of the six prespecified "
            "alternative fifth-component families. It supplements the compact marginal-effect "
            "table in the main manuscript and mirrors its four core component-ablation tables.",
            "\\vspace{4pt}",
        ]
    lines += [
        "\\tiny",
        "\\setlength{\\tabcolsep}{1.6pt}",
        "\\begin{longtable}{lccccc" + "r" * len(METRICS) + "}",
        f"\\caption{{{caption}}}\\label{{{table_id}}}\\\\",
        "\\toprule",
        "Family & SVM & Tan. & PFN & FM & Alt. & "
        + " & ".join(DISPLAY[m] for m in METRICS) + " \\\\",
        "\\midrule",
        "\\endfirsthead",
        f"\\multicolumn{{{6 + len(METRICS)}}}{{l}}{{\\tablename\\ \\thetable\\ continued}}\\\\",
        "\\toprule",
        "Family & SVM & Tan. & PFN & FM & Alt. & "
        + " & ".join(DISPLAY[m] for m in METRICS) + " \\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        f"\\multicolumn{{{6 + len(METRICS)}}}{{r}}{{Continued on next page}}\\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    for family in RUNS:
        part = all_rows[(all_rows.candidate_family == family) & (all_rows.cohort == cohort)]
        subsets = ordered_subsets(part)
        metric_best = {}
        for metric in METRICS:
            values = part[part.metric == metric].set_index("subset")["value"]
            metric_best[metric] = values.min() if metric == "Brier" else values.max()
        for row_number, subset in enumerate(subsets):
            rows = part[part.subset == subset].set_index("metric")
            base = rows.iloc[0]
            family_cell = family if row_number == 0 else ""
            cells = [family_cell] + [tick(base[column]) for column in COMPONENTS]
            for metric in METRICS:
                row = rows.loc[metric]
                cells.append(fmt(
                    row.value,
                    bold=abs(row.value - metric_best[metric]) < 5e-13,
                    star=pd.notna(row.q_vs_full) and row.q_vs_full < 0.05,
                ))
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\addlinespace")
    lines += [
        "\\end{longtable}",
        "\\noindent\\begin{minipage}{\\linewidth}\\footnotesize",
        "Each panel contains all 31 nonempty equal-weight subsets of SVM, Tanimoto SVC (Tan.), "
        "TabPFN-v2 (PFN), TabFM (FM), and the named alternative component (Alt.). "
        "The threshold was fixed at 0.5 and no model was refitted during subset enumeration. "
        "Higher is better except for Brier. Bold denotes the best point estimate within a "
        "candidate family. $^{*}q<0.05$ versus that family's complete five-component blend, "
        "using a paired 10,000-resample compound bootstrap and Holm correction across 30 "
        "reduced subsets within metric and registry.",
        "\\end{minipage}",
        "\\end{landscape}",
    ]
    return "\n".join(lines)


def ci_table(all_rows: pd.DataFrame, cohort: str, table_id: str, caption: str) -> str:
    lines = [
        "\\begin{landscape}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{longtable}{lccccc ll}",
        f"\\caption{{{caption}}}\\label{{{table_id}}}\\\\",
        "\\toprule",
        "Family & SVM & Tan. & PFN & FM & Alt. & AUPRC (95\\% CI) & MCC (95\\% CI) \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\multicolumn{8}{l}{\\tablename\\ \\thetable\\ continued}\\\\",
        "\\toprule",
        "Family & SVM & Tan. & PFN & FM & Alt. & AUPRC (95\\% CI) & MCC (95\\% CI) \\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        "\\multicolumn{8}{r}{Continued on next page}\\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    for family in RUNS:
        part = all_rows[(all_rows.candidate_family == family) & (all_rows.cohort == cohort)]
        subsets = ordered_subsets(part)
        for row_number, subset in enumerate(subsets):
            rows = part[part.subset == subset].set_index("metric")
            base = rows.iloc[0]
            family_cell = family if row_number == 0 else ""
            cells = [family_cell] + [tick(base[column]) for column in COMPONENTS]
            for metric in ["AUPRC", "MCC"]:
                row = rows.loc[metric]
                cells.append(f"{row.value:.3f} ({row.ci_low:.3f}--{row.ci_high:.3f})")
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\addlinespace")
    lines += [
        "\\end{longtable}",
        "\\noindent\\begin{minipage}{\\linewidth}\\footnotesize",
        "Point estimates and 95\\% percentile confidence intervals were obtained from 10,000 "
        "compound-level bootstrap resamples. Component ticks and abbreviations are as in "
        "Tables~\\ref{tab:supp-candidate-d1} and~\\ref{tab:supp-candidate-cv}.",
        "\\end{minipage}",
        "\\end{landscape}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manuscript-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    manuscript_dir = args.manuscript_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    all_rows, audit = load_all(root)
    all_rows.to_csv(output_dir / "complete_candidate_ablation_long.csv", index=False)
    pd.DataFrame(audit).to_csv(output_dir / "source_run_audit.csv", index=False)

    tables = "\n\n".join([
        point_table(
            all_rows, "d1_test", "tab:supp-candidate-d1",
            "Complete six-family fifth-component ablation on the held-out "
            "Geroprotectors.org/ChEMBL test partition.", supplement_heading=True),
        point_table(
            all_rows, "cv5_train_oof", "tab:supp-candidate-cv",
            "Complete six-family fifth-component ablation under five-fold "
            "cross-validation on the development partition."),
        ci_table(
            all_rows, "d1_test", "tab:supp-candidate-d1-ci",
            "AUPRC and MCC with 95 percent confidence intervals for every candidate-family "
            "subset on the held-out Geroprotectors.org/ChEMBL test partition."),
        ci_table(
            all_rows, "cv5_train_oof", "tab:supp-candidate-cv-ci",
            "AUPRC and MCC with 95 percent confidence intervals for every candidate-family "
            "subset under five-fold cross-validation."),
    ])
    tables_path = output_dir / "complete_candidate_ablation_tables.tex"
    tables_path.write_text(tables + "\n")

    relative_tables = tables_path.relative_to(manuscript_dir)
    wrapper = "\n".join([
        "\\documentclass[10pt]{article}",
        "\\usepackage[a4paper,margin=10mm]{geometry}",
        "\\usepackage{booktabs,longtable,pdflscape,amssymb}",
        "\\begin{document}",
        "\\renewcommand{\\thetable}{S\\arabic{table}}",
        f"\\input{{{relative_tables.as_posix()}}}",
        "\\end{document}",
        "",
    ])
    wrapper_path = manuscript_dir / "gb4-candidate-ablation-supplement.tex"
    if wrapper_path.exists():
        raise RuntimeError(f"Refusing to overwrite {wrapper_path}")
    wrapper_path.write_text(wrapper)

    generated = [
        output_dir / "complete_candidate_ablation_long.csv",
        output_dir / "source_run_audit.csv",
        tables_path,
        wrapper_path,
    ]
    manifest = {
        "schema_version": "gb4.complete_candidate_ablation_tables.v1",
        "source_runs": audit,
        "candidate_families": list(RUNS),
        "cohorts": COHORTS,
        "metrics": METRICS,
        "subsets_per_family_per_cohort": 31,
        "total_subset_by_registry_evaluations": len(RUNS) * len(COHORTS) * 31,
        "models_refitted": False,
        "source_runs_modified": False,
        "generated_artifact_hashes": {str(p): sha256(p) for p in generated},
        "generator_sha256": sha256(Path(__file__).resolve()),
    }
    (output_dir / "TABLE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": "COMPLETE",
        "output_dir": str(output_dir),
        "combined_rows": len(all_rows),
        "candidate_families": len(RUNS),
        "subset_by_registry_evaluations": len(RUNS) * len(COHORTS) * 31,
        "wrapper": str(wrapper_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
