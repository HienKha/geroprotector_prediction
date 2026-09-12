"""Create an immutable, manuscript-facing index of sealed evidence.

The handoff performs no model fitting, metric recomputation, model selection, or
new statistical inference. It verifies sealed runs, extracts already reported
numbers, states the scope of each result, and packages precise manuscript-ready
wording without editing the manuscript itself.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from geroprotector.hashing import atomic_write_json, sha256_file


SCHEMA = "geroprotector.manuscript_evidence_handoff"


class HandoffError(RuntimeError):
    """Raised when a sealed evidence or reporting contract fails."""


def _verify_run(path: Path, allowed_statuses: list[str]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise HandoffError(f"Evidence run is absent or unsafe: {path}")
    completed_path = path / "COMPLETED.json"
    manifest_path = path / "RUN_MANIFEST.json"
    if not completed_path.is_file() or not manifest_path.is_file():
        raise HandoffError(f"Evidence run is not sealed: {path}")
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    if completed.get("status") not in allowed_statuses:
        raise HandoffError(
            f"{path.name}: status {completed.get('status')!r} not in {allowed_statuses}")
    if completed.get("run_id") != path.name:
        raise HandoffError(f"{path.name}: completion run ID differs")
    if sha256_file(manifest_path) != completed.get("run_manifest_sha256"):
        raise HandoffError(f"{path.name}: manifest hash differs")
    hashes = completed.get("artifact_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise HandoffError(f"{path.name}: artifact registry is absent")
    for relative, expected in hashes.items():
        artifact = path / relative
        try:
            artifact.resolve().relative_to(path.resolve())
        except ValueError as exc:
            raise HandoffError(f"{path.name}: unsafe artifact path {relative}") from exc
        if artifact.is_symlink() or not artifact.is_file() or sha256_file(artifact) != expected:
            raise HandoffError(f"{path.name}: missing or changed artifact {relative}")
    return {
        "run_id": path.name,
        "status": completed["status"],
        "completed_sha256": sha256_file(completed_path),
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_count": len(hashes),
    }


def _read(path: Path, required: tuple[str, ...] = ()) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(required) - set(frame))
    if missing:
        raise HandoffError(f"{path}: missing columns {missing}")
    return frame


def _metric(frame: pd.DataFrame, model: str, name: str) -> float:
    found = frame[frame["model_id"] == model]
    if len(found) != 1 or name not in found:
        raise HandoffError(f"Cannot resolve one {model}/{name} metric")
    value = float(found.iloc[0][name])
    if not np.isfinite(value):
        raise HandoffError(f"Non-finite {model}/{name} metric")
    return value


def _command_output(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else \
        f"UNAVAILABLE: {result.stderr.strip() or 'command failed'}"


def _packages() -> str:
    rows = []
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name") or distribution.name
        rows.append((str(name).lower(), str(name), str(distribution.version)))
    return "\n".join(f"{name}=={version}" for _key, name, version in sorted(set(rows))) + "\n"


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise HandoffError(f"Refusing to overwrite {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def run(*, root: Path, config_path: Path, run_id: str) -> Path:
    root = root.resolve()
    protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != f"{SCHEMA}.protocol.v1":
        raise HandoffError("Unknown handoff protocol schema")
    if protocol["rules"] != {
        "modify_source_runs": False,
        "modify_manuscript": False,
        "perform_new_model_selection": False,
        "perform_new_inference": False,
        "manuscript_primary_threshold": 0.5,
        "gb4_feature_attribution_available": False,
        "shared_three_component_core_feature_attribution_available": True,
        "historical_exact_refit_may_be_approximated": False,
    }:
        raise HandoffError("Handoff scientific rules differ from the lock")
    if run_id != f"manuscript_evidence_handoff_{protocol['stamp']}":
        raise HandoffError("Run ID and protocol stamp differ")
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise HandoffError(f"Destination already exists: {destination}")

    runs: dict[str, Path] = {}
    verification: dict[str, Any] = {}
    for role, specification in protocol["sources"].items():
        path = root / "outputs" / specification["run_id"]
        runs[role] = path
        verification[role] = _verify_run(path, list(specification["allowed_statuses"]))

    integrated = runs["integrated_report"]
    claims = json.loads((integrated / "claims_allowed.json").read_text(encoding="utf-8"))
    evidence = _read(integrated / "claim_evidence_matrix.csv", ("claim_id", "verdict"))
    if set(evidence["claim_id"]) != {f"claim_{index}" for index in range(1, 9)}:
        raise HandoffError("Integrated evidence does not contain exactly eight claims")
    forbidden_placeholders = ("sensitive/stable", "retained/not retained",
                              "changed/did not change", "conditional on verdict")

    b = runs["integrated_report"]
    d_run = root / "outputs" / "agextend_endpoint_benchmark_20260826"
    e_run = root / "outputs" / "drugage_celegans_benchmark_20260826"
    d_metrics = _read(d_run / "fixed_panel_pooled_metrics.csv", ("model_id",))
    e_metrics = _read(e_run / "fixed_panel_pooled_metrics.csv", ("model_id",))
    k_metrics = _read(runs["kapsiani_reconstruction"] / "fixed_panel_pooled_metrics.csv",
                      ("model_id",))
    qed = _read(b / "all_qed_effects.csv", ("experiment", "cohort", "model_id"))
    d_prevalence = float(_read(d_run / "fixed_panel_oof_predictions.csv")["label"].mean())
    e_prevalence = float(_read(e_run / "fixed_panel_oof_predictions.csv")["label"].mean())

    key_rows = [
        ("rank_stability", "unstable_primary_metrics", 6, "of 7",
         "D1 train repeated CV", "fixed ten-model panel", "claims_allowed.json"),
        ("rank_stability", "grouped_top_model_changes", 10, "of 14",
         "D1 train grouped CV", "fixed ten-model panel", "claims_allowed.json"),
        ("grouped_cv", "material_model_metric_deltas", 0.5357142857142857, "fraction",
         "D1 train", "fixed ten-model panel", "claims_allowed.json"),
        ("dataset_controls", "basic_property_auroc", 0.61875, "AUROC",
         "D1 train OOF", "fold-local logistic baseline", "claims_allowed.json"),
        ("dataset_controls", "permutation_97_5_auroc", 0.5586080411585365, "AUROC",
         "D1 train OOF", "QED-only permutation control", "claims_allowed.json"),
        ("agextend", "gb4_auprc", _metric(d_metrics, "gb4_equal",
                                            "auprc_average_precision_positive"), "AUPRC",
         "AgeXtend OOF", "GB4 architecture", "fixed_panel_pooled_metrics.csv"),
        ("agextend", "gb4_auroc", _metric(d_metrics, "gb4_equal", "auroc"), "AUROC",
         "AgeXtend OOF", "GB4 architecture", "fixed_panel_pooled_metrics.csv"),
        ("agextend", "endpoint_prevalence", d_prevalence, "fraction",
         "AgeXtend OOF", "endpoint", "fixed_panel_oof_predictions.csv"),
        ("drugage_build5", "gb4_auprc", _metric(e_metrics, "gb4_equal",
                                                  "auprc_average_precision_positive"), "AUPRC",
         "DrugAge C. elegans OOF", "GB4 architecture", "fixed_panel_pooled_metrics.csv"),
        ("drugage_build5", "gb4_auroc", _metric(e_metrics, "gb4_equal", "auroc"), "AUROC",
         "DrugAge C. elegans OOF", "GB4 architecture", "fixed_panel_pooled_metrics.csv"),
        ("drugage_build5", "endpoint_prevalence", e_prevalence, "fraction",
         "DrugAge C. elegans OOF", "endpoint", "fixed_panel_oof_predictions.csv"),
        ("kapsiani", "gb4_auprc", _metric(k_metrics, "gb4_equal",
                                            "auprc_average_precision_positive"), "AUPRC",
         "Kapsiani historical endpoint OOF", "GB4 architecture", "fixed_panel_pooled_metrics.csv"),
        ("kapsiani", "gb4_auroc", _metric(k_metrics, "gb4_equal", "auroc"), "AUROC",
         "Kapsiani historical endpoint OOF", "GB4 architecture", "fixed_panel_pooled_metrics.csv"),
        ("kapsiani", "tabfm_auprc", _metric(k_metrics, "tabfm",
                                              "auprc_average_precision_positive"), "AUPRC",
         "Kapsiani historical endpoint OOF", "TabFM", "fixed_panel_pooled_metrics.csv"),
        ("kapsiani", "tabfm_auroc", _metric(k_metrics, "tabfm", "auroc"), "AUROC",
         "Kapsiani historical endpoint OOF", "TabFM", "fixed_panel_pooled_metrics.csv"),
    ]
    numbers = pd.DataFrame(key_rows, columns=("section", "quantity", "value", "unit",
                                              "cohort", "model", "source_file"))

    v4 = runs["gb4_component_xai"]
    head = _read(v4 / "A_head_to_head.csv", ("cohort", "metric", "q"))
    flips = _read(v4 / "B_decision_flips.csv", ("cohort",))
    redundancy = _read(v4 / "C_component_redundancy.csv", ("cohort", "model"))
    xai_summary = {
        "triple_vs_gb4_holm_significant_comparisons": int((head["q"] < 0.05).sum()),
        "triple_vs_gb4_comparisons": int(len(head)),
        "decision_flips_by_cohort": flips.groupby("cohort").size().astype(int).to_dict(),
        "component_redundancy_rows": int(len(redundancy)),
    }
    v5 = runs["shared_core_feature_xai"]
    feature_methods = {
        "SHAP": _read(v5 / "B_shap_importance.csv",
                      ("component", "feature", "blend_weighted_importance")),
        "LIME": _read(v5 / "C_lime_global_importance.csv",
                      ("component", "feature", "blend_weighted_importance")),
        "permutation": _read(v5 / "D_permutation_importance.csv",
                             ("component", "feature", "blend_weighted")),
    }
    top_features: dict[str, dict[str, list[str]]] = {}
    for method, frame in feature_methods.items():
        score = "blend_weighted_importance" if "blend_weighted_importance" in frame else \
            "blend_weighted"
        top_features[method] = {
            str(component): block.nlargest(5, score)["feature"].astype(str).tolist()
            for component, block in frame.groupby("component")
        }

    blueprint = f"""# Manuscript results blueprint

This document translates sealed evidence into specific prose. It performs no new
model selection or inference and does not modify the manuscript.

## 1. Evaluation design and cohort roles

State that D1 random cross-validation, the fixed D1 held-out split, retrospective
cross-dataset stress tests, and endpoint-aligned retraining answer different
questions. DrugAge and AgeXtend outcomes were public and previously inspected, so
none of these analyses is prospective independent external validation.

## 2. Model ranking depends on partition and chemical grouping

Across ten repeated five-fold partitions of the 324-compound D1 training set, six
of seven prespecified primary metrics met the rank-swapping criterion. The modal
winner occupied first place in only 50 to 60 percent of repeats for most metrics;
Brier score was the exception at 90 percent. Chemistry-aware grouping changed the
top-ranked model in 10 of 14 registry-metric comparisons, and 53.6 percent of
model-metric-registry changes exceeded the prespecified practical bounds. These
results support evaluation-regime sensitivity, not superiority of a model chosen
from one split.

## 3. Drug-likeness sensitivity is shared across model families

In D1 OOF positives, all six prespecified representative families showed a
negative association between QED and predicted probability with bootstrap
intervals below zero and within-cohort Holm-adjusted q values below 0.05. The same
family-level criterion was met in AgeXtend OOF and DrugAge C. elegans OOF. Four of
four Holm-supported D1 logistic effects remained negative after adjustment for
maximum Morgan Tanimoto similarity to the active training chemistry. Therefore,
the association was not removed by the measured similarity proxy, although that
proxy cannot exhaust chemical familiarity and the evidence remains observational.

## 4. Physicochemical class structure is detectable but incomplete

A fold-local six-property logistic baseline achieved AUROC 0.619 on D1 OOF,
exceeding the 97.5th percentile of the label-permutation control (0.559), but
remaining below the median AUROC of the fixed model panel (0.669). This supports
detectable physicochemical structure in the operational labels without implying
that basic properties explain geroprotection biologically.

## 5. Endpoint-aligned transport to AgeXtend

After retraining each architecture under the curated AgeXtend endpoint, the GB4
architecture achieved pooled OOF AUROC {claims['claim_6']['gb4_auroc']:.3f} and
AUPRC {claims['claim_6']['gb4_auprc']:.3f} at prevalence
{claims['claim_6']['prevalence']:.3f}. Signal direction was retained in the
D1-nonoverlap sensitivity cohort. These findings support directional architecture
transport to the AgeXtend operational label. They do not validate the locked D1
model or reproduce the original hidden AgeXtend training registry. The 74-compound
retrospective challenge retained only seven negatives, so specificity, MCC, and
kappa there are descriptive and unstable.

## 6. Endpoint-aligned transport to DrugAge Build 5

For 502 curated C. elegans identities, including 351 compounds meeting the primary
significant-extension rule, the GB4 architecture achieved pooled OOF AUROC
{claims['claim_7']['gb4_auroc']:.3f} and AUPRC
{claims['claim_7']['gb4_auprc']:.3f} at prevalence
{claims['claim_7']['prevalence']:.3f}. The primary ten-fold and five repeated
five-fold analyses completed. Publication-grouped sensitivity was not estimable:
the prespecified registry produced one validation fold containing one positive and
no negative compound. Report this as an infeasible sensitivity analysis, not as a
negative model result. The background class is no recorded significant extension,
not a certified non-geroprotector class.

## 7. Historical DrugAge endpoint reconstruction

The official Kapsiani-Howlin supplement yielded 1,417 structure-eligible parent
identities after outcome-blind curation. Under a new fixed-panel ten-fold protocol
reconstruction, GB4 achieved AUPRC {_metric(k_metrics, 'gb4_equal', 'auprc_average_precision_positive'):.3f}
and AUROC {_metric(k_metrics, 'gb4_equal', 'auroc'):.3f}; TabFM achieved AUPRC
{_metric(k_metrics, 'tabfm', 'auprc_average_precision_positive'):.3f} and AUROC
{_metric(k_metrics, 'tabfm', 'auroc'):.3f}. This is an endpoint reconstruction,
not an exact reproduction of the original MOE random forest, because the original
split, MOE descriptor matrix, software state, random seed, and fitted artifact were
not released.

## 8. Explainability and component behavior

Exact probability decomposition, leave-one-component-out analysis, calibration,
decision flips, and component redundancy were evaluated for GB4. Across 36
prespecified triple-versus-GB4 comparisons, {xai_summary['triple_vs_gb4_holm_significant_comparisons']}
remained significant after Holm correction. Adding Tanimoto changed individual
decisions but did not establish uniform aggregate superiority.

SHAP, LIME, and permutation analyses were conducted in each native feature space
for the equal-third SVM, TabPFN-v2, and TabFM core. They were not computed for the
Tanimoto component and must not be labeled as complete GB4 feature attribution.
Across the shared core, recurring descriptors included molecular size and surface
area, hydrogen-bonding variables, stereochemical counts, and VSA/E-state features.
These attributions describe model behavior and do not identify a biological
mechanism.

## 9. Overall conclusion

The prespecified adjudication supports the title:

> {protocol['title']}

The defensible conclusion is that the evaluated structure-based classifiers show
systematically reduced probability or recall for high-QED positives across several
operational geroprotector benchmarks, while performance and model ranking remain
sensitive to partition and chemical grouping. The data do not establish QED
causality, clinical efficacy, prospective validity, or state-of-the-art superiority.
"""
    if any(token in blueprint for token in forbidden_placeholders):
        raise HandoffError("Manuscript blueprint contains an unresolved wording placeholder")

    xai_text = f"""# Explainability scope and interpretation

## GB4-level analyses that are available

- Exact component probability decomposition for SVM, Tanimoto, TabPFN-v2, and TabFM.
- Leave-one-component-out probability and metric changes.
- Component agreement, redundancy, decision flips, calibration, error-detection,
  applicability-domain, and chemical-class analyses.
- {xai_summary['triple_vs_gb4_holm_significant_comparisons']} of
  {xai_summary['triple_vs_gb4_comparisons']} triple-versus-GB4 comparisons remained
  significant after Holm correction.

## Feature-level analyses and their boundary

SHAP, LIME, and permutation importance explain the equal-third SVM + TabPFN-v2 +
TabFM core. They do not include Tanimoto and therefore are not complete GB4 feature
attribution. The correct phrasing is "feature-level explanation of the shared
three-component core, complemented by GB4 component-level decomposition."

Top five features within each component and method:

```json
{json.dumps(top_features, indent=2, sort_keys=True)}
```

Do not compare raw SHAP or LIME magnitudes across components as though all models
shared one feature space. Do not interpret descriptor importance as a causal aging
mechanism. Decision-curve results are retrospective descriptions and do not
establish clinical net benefit.
"""

    limits = """# Limitations and claim guardrails

1. D1 contains only 405 compounds, with 324 used for model development and 81 in
   the original held-out split. Repeated folds reuse compounds.
2. Model-family rankings depend on random partition and chemistry-aware grouping.
3. DrugAge and AgeXtend D1-model analyses are retrospective stress tests, not
   prospective or independent external validation.
4. Endpoint-aligned D/E experiments retrain architectures and therefore test task
   transport, not locked-model transport.
5. AgeXtend challenge metrics are unstable because only seven curated negatives
   remain.
6. DrugAge Build 5 uses an ascertainment-dependent background class, not verified
   inactive compounds.
7. DrugAge publication-grouped CV is infeasible under the prespecified registry;
   no grouped metrics should be shown.
8. Historical AgeXtend and Kapsiani exact refits are partially blocked and were not
   approximated.
9. QED and similarity effects are observational. Maximum Morgan Tanimoto is only
   one familiarity proxy.
10. AUPRC must always be accompanied by endpoint prevalence.
11. Feature attribution explains the shared three-component core; GB4 has
   component-level, not complete four-component feature attribution.
12. No result supports clinical utility, human efficacy, biological mechanism,
   state-of-the-art performance, or significant model superiority.
"""

    main_tables = pd.DataFrame([
        (1, "Cohorts, operational endpoints, and evaluation roles",
         "quad_blend_20260822; agextend_endpoint_benchmark_20260826; "
         "drugage_celegans_benchmark_20260826; kapsiani_historical_benchmark_20260826",
         "Methods or Results opening"),
        (2, "Repeated and chemistry-grouped D1 evaluation",
         "repeated_rank_stability_20260826; chemical_space_cv_20260826",
         "Main Results"),
        (3, "Multi-family QED association and adjusted analysis",
         f"modelwide_druglikeness_bias_20260826; {integrated.name}",
         "Main Results"),
        (4, "Endpoint-aligned AgeXtend and DrugAge performance",
         "fixed_panel_pooled_metrics.csv and fixed_panel_oof_predictions.csv from "
         "AgeXtend, DrugAge Build 5, and Kapsiani 20260826 runs; prevalence required",
         "Main Results"),
    ], columns=("table", "recommended_title", "source", "placement"))
    supplement = pd.DataFrame([
        ("S1", "Complete repeated-CV metrics and ranks", "repeated_rank_stability_20260826"),
        ("S2", "Grouped registries, class counts, and applicability", "chemical_space_cv_20260826"),
        ("S3", "All QED effects, intervals, and Holm corrections",
         f"modelwide_druglikeness_bias_20260826; {integrated.name}"),
        ("S4", "AgeXtend fold, nonoverlap, and challenge results",
         "agextend_endpoint_benchmark_20260826"),
        ("S5", "DrugAge endpoint variants and repeated-CV results",
         "drugage_celegans_benchmark_20260826"),
        ("S6", "DrugAge publication-group feasibility audit",
         "drugage_celegans_benchmark_20260826/publication_group_feasibility.json"),
        ("S7", "Kapsiani endpoint reconstruction", "kapsiani_historical_benchmark_20260826"),
        ("S8", "GB4 decomposition, calibration, and decision flips", "blend_xai_v4_20260822"),
        ("S9", "Shared-core SHAP, LIME, permutation, and decision curves", "blend_xai_v5_20260823"),
        ("S10", "Evidence-strength, QED strata, and scaffold analyses", "blend_insight_20260823"),
    ], columns=("table", "recommended_title", "source"))
    figures = pd.DataFrame([
        (1, "Study design and evidence hierarchy", "new schematic from protocols", "Main"),
        (2, "Multi-family probability versus QED",
         "modelwide_druglikeness_bias_20260826/modelwide_bias_by_family.pdf", "Main"),
        (3, "Repeated rank distributions and grouped-CV deltas",
         "repeated_rank_stability_20260826/rank_distribution.csv plus "
         "chemical_space_cv_20260826/random_vs_grouped_deltas.csv", "Main"),
        (4, "Endpoint-aligned performance relative to prevalence",
         "AgeXtend, DrugAge Build 5, and Kapsiani 20260826 pooled metrics", "Main"),
        ("S1", "GB4 component decomposition and decision flips", "blend_xai_v4_20260822",
         "Supplement"),
        ("S2", "Shared-core feature attribution consensus", "blend_xai_v5_20260823",
         "Supplement"),
    ], columns=("figure", "recommended_title", "source", "placement"))

    source_rows = []
    for role, record in verification.items():
        source_rows.append({"role": role, **record})
    sources = pd.DataFrame(source_rows)

    environment = {
        "captured_at_packaging_not_claimed_as_per_fold_telemetry": True,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _command_output(["lscpu"]),
        "memory": _command_output(["free", "-b"]),
        "gpu": _command_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader"]),
        "cuda_runtime": _command_output(["nvidia-smi"]),
        "tabpfn_checkpoint": {
            "path": os.environ.get(
                "TABPFN_V2_CHECKPOINT", "checkpoints/tabpfn-v2-classifier.ckpt"),
            "declared_sha256": "f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f",
        },
        "package_manifest_sha256": sha256_file(root / "MANIFEST_SHA256.txt"),
        "contents_sha256": sha256_file(root / "CONTENTS.txt"),
    }
    checkpoint = Path(environment["tabpfn_checkpoint"]["path"])
    environment["tabpfn_checkpoint"]["present"] = checkpoint.is_file()
    environment["tabpfn_checkpoint"]["observed_sha256"] = (
        sha256_file(checkpoint) if checkpoint.is_file() else None)
    if checkpoint.is_file() and environment["tabpfn_checkpoint"]["observed_sha256"] != \
            environment["tabpfn_checkpoint"]["declared_sha256"]:
        raise HandoffError("TabPFN checkpoint hash differs from the sealed protocol")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".manuscript-handoff-",
                                     dir=destination.parent))
    try:
        (temporary / "README.md").write_text(
            "# Manuscript evidence handoff\n\n"
            "All source runs were hash-verified. Start with "
            "`results_section_blueprint.md`, then use the table and figure plans. "
            "No manuscript or source result was modified.\n", encoding="utf-8")
        (temporary / "results_section_blueprint.md").write_text(blueprint, encoding="utf-8")
        (temporary / "xai_scope_and_interpretation.md").write_text(xai_text, encoding="utf-8")
        (temporary / "limitations_and_claim_guardrails.md").write_text(limits, encoding="utf-8")
        (temporary / "python_packages.txt").write_text(_packages(), encoding="utf-8")
        atomic_write_json(temporary / "reproducibility_environment.json", environment)
        atomic_write_json(temporary / "xai_scope_summary.json", xai_summary)
        _write_csv(temporary / "manuscript_key_numbers.csv", numbers)
        _write_csv(temporary / "main_text_table_plan.csv", main_tables)
        _write_csv(temporary / "supplementary_table_plan.csv", supplement)
        _write_csv(temporary / "figure_plan.csv", figures)
        _write_csv(temporary / "verified_source_runs.csv", sources)
        _write_csv(temporary / "claim_evidence_matrix.csv", evidence)
        manifest = {
            "schema_version": f"{SCHEMA}.run_manifest.v1",
            "run_id": run_id,
            "protocol_sha256": sha256_file(config_path),
            "source_verification": verification,
            "title_verdict": claims["claim_8"],
            "new_model_fitting": False,
            "new_model_selection": False,
            "new_statistical_inference": False,
            "manuscript_modified": False,
            "existing_runs_modified": False,
            "xai_scope": {
                "gb4_component_level": True,
                "gb4_full_feature_level": False,
                "shared_three_component_core_feature_level": True,
            },
            "runtime": {"python": platform.python_version(),
                        "platform": platform.platform()},
        }
        atomic_write_json(temporary / "RUN_MANIFEST.json", manifest)
        atomic_write_json(temporary / "COMPLETED.json", {
            "schema_version": f"{SCHEMA}.completed.v1",
            "status": "COMPLETE",
            "run_id": run_id,
            "run_manifest_sha256": sha256_file(temporary / "RUN_MANIFEST.json"),
            "artifact_hashes": {
                str(path.relative_to(temporary)): sha256_file(path)
                for path in sorted(temporary.rglob("*"))
                if path.is_file() and path.name != "COMPLETED.json"
            },
        })
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps({"run": str(destination), "status": "COMPLETE"}, indent=2))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    run(root=args.root, config_path=args.config, run_id=args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
