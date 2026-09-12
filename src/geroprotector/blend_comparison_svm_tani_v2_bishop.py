"""Seven-way comparison of SVM/Tanimoto/TabPFN-v2/BiSHop blends and thresholds.

This module fits and refits nothing.  It reads already-sealed component
probability streams from three completed runs and produces one comparison table:

  1. equal-thirds SVM/Tanimoto/TabPFNv2 blend @ threshold 0.5          (computed here)
  2. equal-thirds SVM/Tanimoto/TabPFNv2 blend @ threshold OOF-MCC      (computed here)
  3. 0.10/0.60/0.30 SVM/Tanimoto/TabPFNv2 blend @ threshold 0.5        (copied, sealed)
  4. 0.10/0.60/0.30 SVM/Tanimoto/TabPFNv2 blend @ threshold OOF-MCC    (copied, sealed)
  5. equal-thirds SVM/Tanimoto/BiSHop blend @ threshold 0.5            (copied, sealed)
  6. equal-thirds SVM/Tanimoto/BiSHop blend @ threshold OOF-MCC        (copied, sealed)
  7. TabPFNv2 alone @ threshold 0.5 and OOF-MCC                        (copied, sealed)
  8. BiSHop alone @ threshold 0.5 and OOF-MCC                          (copied, sealed)

Only combination (1)-(2) requires new arithmetic: re-weighting the sealed per-row
paper-SVM / Tanimoto-SVC / TabPFN-v2 probability streams into an equal-thirds blend
and selecting its OOF-MCC threshold with the exact same deterministic selector
(`weighted_blend_paper405.select_threshold`) every other run in this repository uses.
No test or external label is used to fit, weight or threshold anything.  Every sealed
input is opened read-only and hash-verified; the run refuses to start if its own
output directory already exists.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.screening_blend_ablation import _safe_metrics
from geroprotector.traditional_paper405 import _read_sources, paper_split_indices
from geroprotector.weighted_blend_paper405 import select_threshold


class BlendComparisonError(RuntimeError):
    """Raised when a sealed input or a leakage contract fails."""


SCHEMA = "geroprotector.blend_comparison_svm_tani_v2_bishop"
EQUAL_WEIGHT = 1.0 / 3.0
D1_ENDPOINT = "paper_binary"
DRUGAGE_ENDPOINT = "significant_positive_retrieval_background_not_certified_negative"
AGEXTEND_ENDPOINT = "published_independent_table6_binary"
METRIC_COLUMNS = [
    "n_test", "auprc_average_precision_positive", "auroc", "brier", "mcc",
    "macro_f1", "recall_sensitivity", "specificity", "accuracy", "balanced_accuracy",
    "precision_positive", "npv", "cohen_kappa", "log_loss", "tn", "fp", "fn", "tp",
]


def _regular_file(path: Path, role: str, expected_sha256: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise BlendComparisonError(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected_sha256 is not None and sha256_file(resolved) != expected_sha256:
        raise BlendComparisonError(f"{role} SHA256 differs from the protocol lock")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = yaml.safe_load(
        _regular_file(path, "comparison protocol").read_text(encoding="utf-8")
    )
    expected_schema = f"{SCHEMA}.protocol.v1"
    if not isinstance(protocol, dict) or protocol.get("schema_version") != expected_schema:
        raise BlendComparisonError("Unknown comparison protocol schema")
    rule = protocol.get("threshold_rule", {})
    if (
        float(rule.get("fixed_0p5", np.nan)) != 0.5
        or rule.get("external_labels_may_select_or_change_threshold") is not False
    ):
        raise BlendComparisonError("Threshold leakage contract differs")
    immutability = protocol.get("immutability", {})
    if immutability.get("existing_run_directories_are_read_only") is not True:
        raise BlendComparisonError("Immutability contract differs")
    for record in protocol.get("sealed_inputs", {}).values():
        _regular_file(root / record["path"], f"sealed input {record['path']}", record["sha256"])
    return protocol, canonical_sha256(protocol)


def _copy_sealed_rows(
    frame: pd.DataFrame,
    *,
    model: str,
    display_model: str,
    weights: str,
    result_source: str,
    op_point_map: dict[str, str],
) -> pd.DataFrame:
    """Reshape sealed rows for one model into the common comparison schema."""

    rows = []
    for operating_point, threshold_rule in op_point_map.items():
        selected = frame[(frame.model == model) & (frame.operating_point == operating_point)]
        if len(selected) == 0:
            raise BlendComparisonError(
                f"Sealed row missing for model={model} operating_point={operating_point}"
            )
        for record in selected.itertuples(index=False):
            rows.append(
                {
                    "cohort": record.cohort,
                    "endpoint": record.endpoint,
                    "model": display_model,
                    "weights": weights,
                    "threshold_rule": threshold_rule,
                    "result_source": result_source,
                    **{column: getattr(record, column) for column in METRIC_COLUMNS},
                    "threshold": record.threshold,
                }
            )
    return pd.DataFrame(rows)


def run(
    *,
    root: Path,
    config_path: Path,
    positive_path: Path,
    negative_path: Path,
    run_id: str,
) -> Path:
    if not re.fullmatch(r"blend_comparison_svm_tani_v2_bishop_[a-z0-9_.-]+", run_id):
        raise BlendComparisonError(
            "RUN_ID must start with blend_comparison_svm_tani_v2_bishop_"
        )
    root = root.resolve()
    protocol, protocol_sha256 = load_protocol(root, config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise BlendComparisonError(f"Run directory already exists: {destination}")

    sealed = protocol["sealed_inputs"]

    # -- 1. equal-thirds SVM/Tanimoto/TabPFNv2: the one new computation ----------
    train_oof = pd.read_csv(root / sealed["weighted_train_oof"]["path"])
    test_components = pd.read_csv(root / sealed["weighted_test_components"]["path"])
    drugage = pd.read_csv(root / sealed["drugage_scored"]["path"])
    agextend = pd.read_csv(root / sealed["agextend_scored"]["path"])
    drugage = drugage.assign(label=drugage.has_significant_positive.astype(int))

    traditional_config = yaml.safe_load(
        (root / "configs" / "traditional_paper405.yaml").read_text(encoding="utf-8")
    )
    if (
        traditional_config["sources"]["positive"]["expected_sha256"]
        != protocol["sources"]["positive_sha256"]
        or traditional_config["sources"]["negative"]["expected_sha256"]
        != protocol["sources"]["negative_sha256"]
        or traditional_config["split"]["expected_assignment_sha256"]
        != protocol["paper_split"]["expected_assignment_sha256"]
    ):
        raise BlendComparisonError("Traditional-paper405 contract differs from the protocol")
    frame, _audit = _read_sources(
        _regular_file(positive_path, "positive source", protocol["sources"]["positive_sha256"]),
        _regular_file(negative_path, "negative source", protocol["sources"]["negative_sha256"]),
        traditional_config,
    )
    if len(frame) != int(protocol["sources"]["expected_rows"]):
        raise BlendComparisonError("D1 row count differs from 405")
    train_indices, _test_indices, split_sha256 = paper_split_indices(traditional_config)
    if split_sha256 != protocol["paper_split"]["expected_assignment_sha256"]:
        raise BlendComparisonError("Paper split differs from the sealed assignment")
    labels_by_row = frame.set_index("paper_row_index")["label"].astype(int)

    train_oof = train_oof.copy()
    train_oof["label"] = labels_by_row.loc[train_oof.paper_row_index].to_numpy()
    observed_rows = np.sort(train_oof.paper_row_index.to_numpy())
    if not np.array_equal(observed_rows, np.sort(train_indices)):
        raise BlendComparisonError("Train-OOF row set differs from the paper train split")

    def _equal_thirds(components: pd.DataFrame) -> np.ndarray:
        return (
            components.probability_paper_svm
            + components.probability_tanimoto_svc
            + components.probability_tabpfn_v2
        ).to_numpy(dtype=float) / 3.0

    train_oof["blend_equal_thirds"] = _equal_thirds(train_oof)
    threshold_oof, oof_mcc = select_threshold(
        train_oof.label.to_numpy(dtype=int), train_oof["blend_equal_thirds"].to_numpy()
    )

    test_components = test_components.copy()
    test_components["blend_equal_thirds"] = _equal_thirds(test_components)
    drugage["blend_equal_thirds"] = _equal_thirds(drugage)
    agextend["blend_equal_thirds"] = _equal_thirds(agextend)

    new_rows = []
    for cohort, endpoint, scored in (
        ("d1_paper_test", D1_ENDPOINT, test_components),
        ("drugage", DRUGAGE_ENDPOINT, drugage),
        ("agextend", AGEXTEND_ENDPOINT, agextend),
    ):
        y = scored.label.to_numpy(dtype=int)
        p = scored["blend_equal_thirds"].to_numpy(dtype=float)
        for threshold_rule, threshold_value in (
            ("fixed_0p5", 0.5),
            ("oof_mcc", threshold_oof),
        ):
            metrics = _safe_metrics(y, p, threshold_value)
            new_rows.append(
                {
                    "cohort": cohort,
                    "endpoint": endpoint,
                    "model": "blend_equal_thirds_svm_tani_tabpfnv2",
                    "weights": "0.3333/0.3333/0.3333",
                    "threshold_rule": threshold_rule,
                    "result_source": "computed_this_run",
                    **{column: metrics[column] for column in METRIC_COLUMNS},
                    "threshold": metrics["threshold"],
                }
            )
    new_frame = pd.DataFrame(new_rows)

    # -- 2. copy the six already-sealed combinations ------------------------------
    ablation_d1 = pd.read_csv(root / sealed["ablation_d1_metrics"]["path"])
    ablation_external = pd.read_csv(root / sealed["ablation_external_metrics"]["path"])
    ablation = pd.concat([ablation_d1, ablation_external], ignore_index=True)
    ablation = ablation[
        ablation.endpoint.isin([D1_ENDPOINT, DRUGAGE_ENDPOINT, AGEXTEND_ENDPOINT])
    ]

    altmodels = pd.read_csv(root / sealed["altmodels_metrics"]["path"])
    altmodels = altmodels[altmodels.result_source == "this_run"]

    copied = pd.concat(
        [
            _copy_sealed_rows(
                ablation,
                model="blend_010_060_030",
                display_model="blend_010_060_030_svm_tani_tabpfnv2",
                weights="0.10/0.60/0.30",
                result_source="sealed_screeningblend_ablation_papersvm_20260818",
                op_point_map={"fixed_0p5": "fixed_0p5", "d1_train_oof_mcc": "oof_mcc"},
            ),
            _copy_sealed_rows(
                ablation,
                model="tabpfn_v2",
                display_model="tabpfn_v2",
                weights="n/a (single model)",
                result_source="sealed_screeningblend_ablation_papersvm_20260818",
                op_point_map={"fixed_0p5": "fixed_0p5", "d1_train_oof_mcc": "oof_mcc"},
            ),
            _copy_sealed_rows(
                altmodels,
                model="blend_equal_thirds_bishop",
                display_model="blend_equal_thirds_svm_tani_bishop",
                weights="0.3333/0.3333/0.3333",
                result_source="sealed_screeningblend_altmodels_20260819",
                op_point_map={"fixed_0p5": "fixed_0p5", "oof_mcc": "oof_mcc"},
            ),
            _copy_sealed_rows(
                altmodels,
                model="bishop",
                display_model="bishop",
                weights="n/a (single model)",
                result_source="sealed_screeningblend_altmodels_20260819",
                op_point_map={"fixed_0p5": "fixed_0p5", "oof_mcc": "oof_mcc"},
            ),
        ],
        ignore_index=True,
    )

    comparison = pd.concat([new_frame, copied], ignore_index=True)
    cohort_order = {"d1_paper_test": 0, "drugage": 1, "agextend": 2}
    model_order = {
        "blend_equal_thirds_svm_tani_tabpfnv2": 0,
        "blend_010_060_030_svm_tani_tabpfnv2": 1,
        "blend_equal_thirds_svm_tani_bishop": 2,
        "tabpfn_v2": 3,
        "bishop": 4,
    }
    threshold_order = {"fixed_0p5": 0, "oof_mcc": 1}
    comparison["_c"] = comparison.cohort.map(cohort_order)
    comparison["_m"] = comparison.model.map(model_order)
    comparison["_t"] = comparison.threshold_rule.map(threshold_order)
    comparison = comparison.sort_values(["_c", "_m", "_t"], kind="stable").drop(
        columns=["_c", "_m", "_t"]
    )
    comparison = comparison[
        [
            "cohort", "endpoint", "n_test", "model", "weights", "threshold_rule",
            "threshold", "result_source", "auprc_average_precision_positive", "auroc",
            "brier", "mcc", "macro_f1", "recall_sensitivity", "specificity", "accuracy",
            "balanced_accuracy", "precision_positive", "npv", "cohen_kappa", "log_loss",
            "tn", "fp", "fn", "tp",
        ]
    ]

    # -- 3. write -------------------------------------------------------------------
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".blendcmp.work-", dir=destination.parent))
    try:
        _write_csv(temporary / "comparison_metrics.csv", comparison)

        columns = [
            ("model", "Model"), ("threshold_rule", "Threshold"), ("n_test", "n"),
            ("auprc_average_precision_positive", "AP"), ("auroc", "AUROC"),
            ("brier", "Brier"), ("mcc", "MCC"), ("macro_f1", "Macro F1"),
            ("recall_sensitivity", "Recall"), ("specificity", "Specificity"),
        ]
        lines = [
            "# Seven-model comparison: SVM/Tanimoto/TabPFNv2 and SVM/Tanimoto/BiSHop",
            "",
            "Rows marked `computed_this_run` are the equal-thirds",
            "SVM/Tanimoto/TabPFNv2 blend, re-weighted here from sealed per-row",
            "component probabilities (no refitting). All other rows are copied",
            "unchanged from already-sealed runs. No test or external label was used",
            "to fit, weight or threshold anything.",
            "",
            f"Equal-thirds SVM/Tanimoto/TabPFNv2 OOF-MCC threshold: {threshold_oof:.6f}",
            f"(selection MCC on the 324-row D1 train OOF: {oof_mcc:.4f}).",
            "",
        ]
        for cohort, title in (
            ("d1_paper_test", "D1 held-out paper test (n=81)"),
            ("drugage", "DrugAge positive-retrieval endpoint (n=446)"),
            ("agextend", "AgeXtend Table 6 endpoint (n=69)"),
        ):
            subset = comparison[comparison.cohort == cohort]
            lines.append(f"## {title}")
            lines.append("")
            lines.append("| " + " | ".join(label for _key, label in columns) + " |")
            lines.append("|" + "---|" * len(columns))
            for record in subset.itertuples(index=False):
                cells = []
                for key, _label in columns:
                    value = getattr(record, key)
                    if isinstance(value, float):
                        cells.append("nan" if not np.isfinite(value) else f"{value:.4f}")
                    else:
                        cells.append(str(value))
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        (temporary / "summary.md").write_text("\n".join(lines), encoding="utf-8")

        manifest = {
            "schema_version": f"{SCHEMA}.run_manifest.v1",
            "run_id": run_id,
            "protocol_sha256": protocol_sha256,
            "paper_split_sha256": split_sha256,
            "equal_thirds_svm_tani_tabpfnv2_oof_threshold": float(threshold_oof),
            "equal_thirds_svm_tani_tabpfnv2_oof_selection_mcc": float(oof_mcc),
            "sealed_inputs_sha256": {
                name: sha256_file(root / record["path"]) for name, record in sealed.items()
            },
            "no_model_was_fit_or_refit": True,
            "test_or_external_labels_used_for_fit_weight_or_threshold": False,
            "existing_runs_modified": False,
        }
        atomic_write_json(temporary / "RUN_MANIFEST.json", manifest)
        atomic_write_json(
            temporary / "COMPLETED.json",
            {
                "schema_version": f"{SCHEMA}.completed.v1",
                "status": "COMPLETE",
                "run_id": run_id,
                "run_manifest_sha256": sha256_file(temporary / "RUN_MANIFEST.json"),
                "artifact_hashes": {
                    str(path.relative_to(temporary)): sha256_file(path)
                    for path in sorted(temporary.rglob("*"))
                    if path.is_file() and path.name != "COMPLETED.json"
                },
            },
        )
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
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args(argv)
    run(
        root=arguments.root,
        config_path=arguments.config,
        positive_path=arguments.positive,
        negative_path=arguments.negative,
        run_id=arguments.run_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
