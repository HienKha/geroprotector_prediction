"""Train-OOF weight search for the SVM / Tanimoto / TabFM blend.

This answers "what is the OOF-optimal weight vector for TabFM in slot 3, and its
OOF-MCC threshold?" using the IDENTICAL machinery that originally produced the
locked 0.10/0.60/0.30 TabPFN blend:

  * the same 171-candidate simplex grid (`weight_grid`, step 0.05, minimum 0.05),
    loaded from the same sealed `configs/weighted_blend_paper405.yaml`;
  * the same primary metric (train-OOF AUPRC) and the same tie-break chain
    (auprc practical tie 0.005 -> mcc -> macro_f1 -> brier -> distance_from_equal
    -> candidate_id), via the same `select_weight_candidate` function;
  * the same deterministic OOF-MCC threshold selector.

so the resulting weights are directly comparable to 0.10/0.60/0.30 rather than
being a differently-defined search.

NOTHING IS REFITTED.  Every component probability already exists:

  * paper_svm and tanimoto_svc OOF/test/external streams come from the sealed
    runs (hash-verified, byte-identical to the published ones);
  * the TabFM OOF/test/external streams come from
    `outputs/screeningblend_tabfm_20260821/`, which cross-fitted TabFM on the
    same 5 folds and proved chemistry parity at 5.55e-17.

The weight search therefore touches ONLY the 324 train-OOF rows.  D1 test,
DrugAge and AgeXtend are scored once afterwards by re-weighting saved per-row
probabilities -- no model sees a test or external label at any point.

SELECTION-MULTIPLICITY CAVEAT (recorded in the manifest, not hidden): this is a
new weight search performed after other models' test/external results had
already been viewed in this project.  The search itself is clean (train-OOF
only), but the resulting weights are a *newly selected* configuration, not a
pre-registered one.  Report it as an exploratory optimum, not as a confirmatory
replacement for the locked 0.10/0.60/0.30 model.
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

from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.screening_blend_ablation import _safe_metrics
from geroprotector.traditional_paper405 import _read_sources, paper_split_indices
from geroprotector.weighted_blend_paper405 import (
    load_protocol as load_weighted_protocol,
)
from geroprotector.weighted_blend_paper405 import (
    select_threshold,
    select_weight_candidate,
)


class WeightedBlendTabFMError(RuntimeError):
    """Raised when a sealed input or a leakage contract fails."""


SCHEMA = "geroprotector.weighted_blend_tabfm"
D1_ENDPOINT = "paper_binary"
DRUGAGE_ENDPOINT = "significant_positive_retrieval_background_not_certified_negative"
AGEXTEND_ENDPOINT = "published_independent_table6_binary"
EQUAL_THIRDS = np.full(3, 1.0 / 3.0)


def _regular_file(path: Path, role: str, expected: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise WeightedBlendTabFMError(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected is not None and sha256_file(resolved) != expected:
        raise WeightedBlendTabFMError(f"{role} SHA256 differs from the protocol lock")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    text = _regular_file(path, "TabFM weight-search protocol").read_text(encoding="utf-8")
    protocol = yaml.safe_load(text)
    expected_schema = f"{SCHEMA}.protocol.v1"
    if not isinstance(protocol, dict) or protocol.get("schema_version") != expected_schema:
        raise WeightedBlendTabFMError("Unknown TabFM weight-search protocol schema")
    contract = protocol.get("search_contract", {})
    for key in (
        "weights_selected_on_train_oof_only",
        "reuses_the_sealed_171_candidate_grid",
        "no_component_is_refitted",
    ):
        if contract.get(key) is not True:
            raise WeightedBlendTabFMError(f"Search contract differs at {key}")
    if contract.get("test_or_external_labels_used_for_weight_or_threshold") is not False:
        raise WeightedBlendTabFMError("Leakage contract differs")
    immutability = protocol.get("immutability", {})
    if immutability.get("existing_run_directories_are_read_only") is not True:
        raise WeightedBlendTabFMError("Immutability contract differs")
    for record in protocol.get("sealed_inputs", {}).values():
        _regular_file(root / record["path"], f"sealed input {record['path']}", record["sha256"])
    return protocol, canonical_sha256(protocol)


def run(
    *, root: Path, config_path: Path, positive_path: Path, negative_path: Path, run_id: str
) -> Path:
    if not re.fullmatch(r"weightedblend_tabfm_[a-z0-9_.-]+", run_id):
        raise WeightedBlendTabFMError("RUN_ID must start with weightedblend_tabfm_")
    root = root.resolve()
    protocol, protocol_sha256 = load_protocol(root, config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise WeightedBlendTabFMError(f"Run directory already exists: {destination}")
    sealed = protocol["sealed_inputs"]

    # -- the ORIGINAL weight-search protocol: identical grid + selection rule ----
    weighted_protocol, _ = load_weighted_protocol(
        _regular_file(
            root / sealed["weighted_protocol"]["path"],
            "sealed weight-search protocol",
            sealed["weighted_protocol"]["sha256"],
        )
    )

    # -- labels + the locked split ----------------------------------------------
    traditional = yaml.safe_load(
        (root / "configs" / "traditional_paper405.yaml").read_text(encoding="utf-8")
    )
    frame, _audit = _read_sources(
        _regular_file(positive_path, "positive source", protocol["sources"]["positive_sha256"]),
        _regular_file(negative_path, "negative source", protocol["sources"]["negative_sha256"]),
        traditional,
    )
    train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    if split_sha256 != protocol["paper_split"]["expected_assignment_sha256"]:
        raise WeightedBlendTabFMError("Paper split differs from the sealed assignment")
    labels = frame["label"].to_numpy(dtype=int)

    # -- component streams: all pre-existing, nothing refitted -------------------
    tabfm_oof = pd.read_csv(root / sealed["tabfm_train_oof"]["path"])
    if not np.array_equal(
        np.sort(tabfm_oof.paper_row_index.to_numpy()), np.sort(train_indices)
    ):
        raise WeightedBlendTabFMError("TabFM OOF rows are not the 324 paper-train rows")
    if not np.array_equal(
        tabfm_oof.label.to_numpy(dtype=int), labels[tabfm_oof.paper_row_index.to_numpy()]
    ):
        raise WeightedBlendTabFMError("TabFM OOF labels differ from D1")

    y_train = tabfm_oof.label.to_numpy(dtype=int)
    # NOTE: the sealed grid helper names slot 3 "tabpfn"; here slot 3 IS TabFM.
    oof_components = {
        "paper_svm": tabfm_oof.probability_paper_svm.to_numpy(float),
        "tanimoto_svc": tabfm_oof.probability_tanimoto_svc.to_numpy(float),
        "tabpfn_v2": tabfm_oof.probability_tabfm.to_numpy(float),
    }

    print("running the sealed 171-candidate weight grid on the 324 train-OOF rows", flush=True)
    table, record = select_weight_candidate(y_train, oof_components, weighted_protocol)
    weights = np.asarray(
        [record["weight_svm"], record["weight_tanimoto"], record["weight_tabpfn"]], dtype=float
    )
    blend_oof = np.column_stack(
        [oof_components[k] for k in ("paper_svm", "tanimoto_svc", "tabpfn_v2")]
    ) @ weights
    oof_threshold, oof_mcc = select_threshold(y_train, blend_oof)
    if not np.isclose(oof_threshold, record["primary_threshold"], rtol=0.0, atol=1e-15):
        raise WeightedBlendTabFMError("Recomputed OOF threshold differs from the search record")
    print(
        f"WINNER weights svm={weights[0]:.2f} tanimoto={weights[1]:.2f} tabfm={weights[2]:.2f} "
        f"| OOF AUPRC {record['selection_auprc']:.4f} | OOF-MCC threshold {oof_threshold!r}",
        flush=True,
    )

    # -- score the three cohorts by re-weighting saved per-row probabilities -----
    def _load(path_key: str, label_from: str | None = None) -> pd.DataFrame:
        d = pd.read_csv(root / sealed[path_key]["path"])
        if label_from is not None:
            d["label"] = d[label_from].astype(int)
        return d

    d1 = _load("tabfm_d1_test")
    if not np.array_equal(d1.paper_row_index.to_numpy(), test_indices):
        raise WeightedBlendTabFMError("TabFM D1-test rows are not the 81 held-out rows")
    drugage = _load("tabfm_drugage")
    agextend = _load("tabfm_agextend")

    cohorts = {
        "d1_paper_test": (D1_ENDPOINT, d1),
        "drugage": (DRUGAGE_ENDPOINT, drugage),
        "agextend": (AGEXTEND_ENDPOINT, agextend),
    }
    for name, (_endpoint, scored) in cohorts.items():
        matrix = np.column_stack([
            scored.probability_paper_svm.to_numpy(float),
            scored.probability_tanimoto_svc.to_numpy(float),
            scored.probability_tabfm.to_numpy(float),
        ])
        scored["blend_oof_weighted_tabfm"] = matrix @ weights
        # Recompute the equal-thirds column too and prove it matches the sealed one,
        # which confirms the streams are aligned row-for-row.
        recomputed_equal = matrix @ EQUAL_THIRDS
        drift = float(np.max(np.abs(recomputed_equal - scored.blend_equal_thirds_tabfm)))
        if drift > 1e-12:
            raise WeightedBlendTabFMError(f"{name}: equal-thirds recomputation drifted")
        print(f"scored {name}: {len(scored)} rows", flush=True)

    # -- metrics: new weighted blend + the equal-thirds TabFM reference ----------
    metric_rows = []
    for cohort, (endpoint, scored) in cohorts.items():
        y = scored.label.to_numpy(dtype=int)
        for model_name, column, threshold in (
            ("blend_oof_weighted_tabfm", "blend_oof_weighted_tabfm", float(oof_threshold)),
            ("blend_equal_thirds_tabfm", "blend_equal_thirds_tabfm",
             float(protocol["reference"]["equal_thirds_tabfm_oof_threshold"])),
        ):
            probability = scored[column].to_numpy(float)
            for point, value in (("oof_mcc", threshold), ("fixed_0p5", 0.5)):
                metric_rows.append({
                    "cohort": cohort, "endpoint": endpoint, "model": model_name,
                    "operating_point": point, "result_source": "this_run",
                    **_safe_metrics(y, probability, value),
                })
    metrics = pd.DataFrame(metric_rows)

    # -- comparison against every other equal-thirds blend + the locked model ----
    comparison_columns = [
        "cohort", "endpoint", "model", "threshold_rule", "threshold", "result_source",
        "auprc_average_precision_positive", "auroc", "brier", "mcc", "macro_f1",
        "recall_sensitivity", "specificity",
    ]
    mine = metrics.rename(columns={"operating_point": "threshold_rule"}).copy()
    mine["result_source"] = "computed_this_run"

    tabfm_metrics = pd.read_csv(root / sealed["tabfm_metrics"]["path"])
    tabfm_metrics = tabfm_metrics[
        tabfm_metrics.model.isin(["tabfm", "paper_svm", "tanimoto_svc"])
    ].rename(columns={"operating_point": "threshold_rule"}).copy()
    tabfm_metrics["result_source"] = "sealed_screeningblend_tabfm_20260821"
    native = tabfm_metrics.threshold_rule == "published_svc_native_predict"
    tabfm_metrics.loc[native, "threshold_rule"] = "oof_mcc"

    eq = pd.read_csv(root / sealed["eq_thirds_comparison"]["path"])
    eq = eq[eq.model.isin([
        "blend_equal_thirds_svm_tani_tabpfnv2", "blend_010_060_030_svm_tani_tabpfnv2",
        "blend_equal_thirds_svm_tani_bishop", "tabpfn_v2", "bishop",
    ])].copy()
    eq["result_source"] = "sealed_blend_comparison_20260819"

    alt = pd.read_csv(root / sealed["altmodels_metrics"]["path"])
    alt = alt[(alt.result_source == "this_run") & (alt.model.isin([
        "blend_equal_thirds_tabm", "blend_equal_thirds_tabnet", "tabm", "tabnet",
    ]))].rename(columns={"operating_point": "threshold_rule"}).copy()
    alt["result_source"] = "sealed_screeningblend_altmodels_20260819"

    comparison = pd.concat(
        [
            mine[comparison_columns], tabfm_metrics[comparison_columns],
            eq[comparison_columns], alt[comparison_columns],
        ],
        ignore_index=True,
    )
    order = {"d1_paper_test": 0, "drugage": 1, "agextend": 2}
    comparison["_c"] = comparison.cohort.map(order)
    comparison = comparison.sort_values(
        ["_c", "threshold_rule", "auprc_average_precision_positive"],
        ascending=[True, True, False], kind="mergesort",
    ).drop(columns=["_c"])

    # -- write --------------------------------------------------------------------
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".wtabfm.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "train_oof_weight_search.csv", table)
        _write_csv(tmp / "d1_train_oof_predictions.csv", pd.DataFrame({
            "paper_row_index": tabfm_oof.paper_row_index.to_numpy(),
            "label": y_train,
            "probability_paper_svm": oof_components["paper_svm"],
            "probability_tanimoto_svc": oof_components["tanimoto_svc"],
            "probability_tabfm": oof_components["tabpfn_v2"],
            "blend_oof_weighted_tabfm": blend_oof,
        }))
        for name, (_endpoint, scored) in cohorts.items():
            _write_csv(tmp / f"scored_{name}.csv", scored)
        _write_csv(tmp / "metrics_all.csv", metrics)
        _write_csv(tmp / "comparison_all.csv", comparison)

        columns = [
            ("model", "Model"), ("n_test", "n"),
            ("auprc_average_precision_positive", "AP"), ("auroc", "AUROC"),
            ("brier", "Brier"), ("mcc", "MCC"), ("macro_f1", "Macro F1"),
            ("recall_sensitivity", "Recall"), ("specificity", "Specificity"),
        ]
        lines = [
            "# OOF-optimal SVM / Tanimoto / TabFM weights",
            "",
            f"Winning weights: **paper_svm {weights[0]:.2f} / tanimoto_svc {weights[1]:.2f} / "
            f"tabfm {weights[2]:.2f}**",
            "",
            "- train-OOF AUPRC at selection: "
            f"{record['selection_auprc']:.4f} "
            f"(best in grid {record['best_grid_auprc']:.4f}, "
            f"{record['eligible_candidates_in_tie_band']} candidates inside "
            "the 0.005 tie band)",
            f"- train-OOF MCC at selection: {record['selection_mcc']:.4f}",
            f"- **OOF-MCC threshold: {oof_threshold!r}**",
            "",
            "Selected on the 324 train-OOF rows only, with the same 171-candidate grid and the",
            "same tie-break chain that produced the locked 0.10/0.60/0.30 TabPFN blend.",
            "Nothing was refitted; the weight search re-weights saved per-row probabilities.",
            "",
            "Exploratory optimum, not a pre-registered configuration -- see the manifest's",
            "selection-multiplicity note.",
            "",
        ]
        primary = metrics[metrics.operating_point != "fixed_0p5"]
        for cohort, title in (
            ("d1_paper_test", "D1 held-out paper test (n=81)"),
            ("drugage", "DrugAge positive-retrieval endpoint (n=446)"),
            ("agextend", "AgeXtend Table 6 endpoint (n=69)"),
        ):
            sub = primary[primary.cohort == cohort]
            lines += [f"## {title}", "",
                      "| " + " | ".join(lbl for _k, lbl in columns) + " |",
                      "|" + "---|" * len(columns)]
            for row in sub.itertuples(index=False):
                cells = []
                for key, _lbl in columns:
                    v = getattr(row, key)
                    cells.append(str(int(v)) if key == "n_test"
                                 else (str(v) if isinstance(v, str)
                                       else ("nan" if not np.isfinite(v) else f"{v:.4f}")))
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        (tmp / "summary.md").write_text("\n".join(lines), encoding="utf-8")

        manifest = {
            "schema_version": f"{SCHEMA}.run_manifest.v1",
            "run_id": run_id, "protocol_sha256": protocol_sha256,
            "paper_split_sha256": split_sha256,
            "slot_3_component": "tabfm (google-research/tabfm v1.0.1)",
            "grid_helper_names_slot_3_tabpfn_but_it_is_tabfm_here": True,
            "winning_weights": {
                "paper_svm": float(weights[0]),
                "tanimoto_svc": float(weights[1]),
                "tabfm": float(weights[2]),
            },
            "weight_search_record": record,
            "oof_mcc_threshold": float(oof_threshold),
            "oof_mcc_at_threshold": float(oof_mcc),
            "equal_thirds_tabfm_oof_threshold": float(
                protocol["reference"]["equal_thirds_tabfm_oof_threshold"]
            ),
            "weights_selected_on_train_oof_only": True,
            "no_component_is_refitted": True,
            "test_or_external_labels_used_for_weight_or_threshold": False,
            "selection_multiplicity_caveat": (
                "new weight search run after other models' test/external results were "
                "already viewed in this project; the search itself is train-OOF only, but "
                "the resulting weights are a newly selected exploratory optimum, not a "
                "pre-registered configuration"
            ),
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
