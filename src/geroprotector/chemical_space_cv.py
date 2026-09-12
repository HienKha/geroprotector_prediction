"""Experiment C: chemistry-aware grouped validation on the D1 training partition.

Random five-fold cross-validation lets a close analogue of a validation compound
sit in the fitting rows.  Two prespecified grouped registries keep related
chemistry inside a single fold:

  C1  Bemis-Murcko frameworks.  Acyclic compounds have an empty framework and are
      therefore clustered by Morgan connected components at Tanimoto 0.40 rather
      than lumped into one giant group or split into singletons.
  C2  Morgan similarity connected components over all 324 compounds at Tanimoto
      0.40, each component indivisible.

Both registries are constructed label-free; only the group-to-fold assignment is
approximately stratified.  C2 carries a feasibility contract declared in the YAML
*before* the components are computed: if a giant component makes five
non-degenerate folds impossible, the registry stops and a formal feasibility
report is written.  The similarity threshold is never changed after seeing
performance.

The comparison baseline is the sealed random seed-42 registry, read from the
existing cross-fitted runs and not recomputed.  The 81 D1 test rows are untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

from geroprotector.study_common import (bind_checkpoint_directory, md_table,
                                        morgan_generator, morgan_matrix)
from geroprotector.d1_training import load_d1_train
from geroprotector.model_panel import (
    BASE_MODELS, BLENDS, PANEL, Panel, add_blends, run_folds)
from geroprotector.fixed_blend_paper405 import _features
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, sha256_file
from geroprotector.modelwide_druglikeness_bias import _train_oof_probabilities
from geroprotector.nine_ml_featuresets import _metrics


class ChemicalSpaceCVError(RuntimeError):
    """Raised when a contract, a feasibility rule or a leakage guard fails."""


SCHEMA = "geroprotector.chemical_space_cv"
PRIMARY_METRICS = ("auprc_average_precision_positive", "auroc", "accuracy", "mcc",
                   "macro_f1", "brier", "cohen_kappa")
LOWER_IS_BETTER = ("brier",)
SIMILARITY_GRID = (0.3, 0.4, 0.5, 0.6)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "geroprotector.chemical_space_cv.protocol.v1":
        raise ChemicalSpaceCVError("Unknown protocol schema")
    if payload["data_boundary"]["d1_test_labels_loaded"] is not False:
        raise ChemicalSpaceCVError("Protocol permits loading D1 test labels")
    return payload, sha256_file(path)


def _similarity_matrix(bits: np.ndarray) -> np.ndarray:
    x = bits.astype(np.float32)
    intersection = x @ x.T
    union = x.sum(1)[:, None] + x.sum(1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection),
                     where=union > 0)


def _components_at(similarity: np.ndarray, threshold: float,
                   subset: np.ndarray | None = None) -> np.ndarray:
    """Connected-component labels for rows in `subset` at the given threshold."""
    block = similarity if subset is None else similarity[np.ix_(subset, subset)]
    adjacency = csr_matrix(block >= threshold)
    _n, labels = connected_components(adjacency, directed=False)
    return labels


def build_registries(smiles: np.ndarray, labels: np.ndarray, protocol: dict
                     ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    settings = protocol["registries"]
    bits = morgan_matrix(list(smiles), morgan_generator(
        radius=int(settings["C1_scaffold"]["fingerprint"]["radius"]),
        bits=int(settings["C1_scaffold"]["fingerprint"]["n_bits"]),
        chirality=bool(settings["C1_scaffold"]["fingerprint"]["use_chirality"])))
    similarity = _similarity_matrix(bits)

    # ---- C1: Bemis-Murcko frameworks, acyclic compounds clustered -----------
    frameworks = []
    for text in smiles:
        molecule = Chem.MolFromSmiles(str(text))
        if molecule is None:
            raise ChemicalSpaceCVError(f"Unparsable structure: {text!r}")
        frameworks.append(MurckoScaffold.MurckoScaffoldSmiles(
            mol=molecule, includeChirality=False))
    frameworks = np.asarray(frameworks, dtype=object)
    acyclic = np.flatnonzero(frameworks == "")
    group_c1 = np.array([f"scaffold::{f}" if f else "" for f in frameworks], dtype=object)
    if len(acyclic):
        clusters = _components_at(
            similarity, float(settings["C1_scaffold"]["acyclic_similarity_threshold"]),
            acyclic)
        for position, row in enumerate(acyclic):
            group_c1[row] = f"acyclic_cluster::{int(clusters[position])}"
    c1 = pd.DataFrame({"row": np.arange(len(smiles)), "group": group_c1,
                       "bemis_murcko_framework": frameworks,
                       "is_acyclic": frameworks == "", "label": labels})

    # ---- C2: similarity connected components over all rows ------------------
    threshold = float(settings["C2_similarity_component"]["similarity_threshold"])
    component = _components_at(similarity, threshold)
    c2 = pd.DataFrame({"row": np.arange(len(smiles)),
                       "group": [f"component::{int(c)}" for c in component],
                       "component_id": component, "label": labels})

    sizes = c2.groupby("group").size()
    feasibility = {
        "threshold": threshold,
        "n_components": int(sizes.size),
        "largest_component_rows": int(sizes.max()),
        "largest_component_fraction": float(sizes.max() / len(smiles)),
        "singleton_components": int((sizes == 1).sum()),
        "max_single_group_fraction_allowed":
            float(settings["C2_similarity_component"]["feasibility"]
                  ["max_single_group_fraction_of_rows"]),
        "c1_n_groups": int(c1.groupby("group").size().size),
        "c1_largest_group_rows": int(c1.groupby("group").size().max()),
        "c1_acyclic_rows": int(len(acyclic)),
    }
    feasibility["feasible"] = bool(
        feasibility["largest_component_fraction"] <=
        feasibility["max_single_group_fraction_allowed"])
    return c1, c2, feasibility


def _assign_folds(registry: pd.DataFrame, settings: dict, feasibility: dict,
                  name: str) -> pd.DataFrame:
    splitter = StratifiedGroupKFold(n_splits=int(settings["n_splits"]),
                                    shuffle=bool(settings["shuffle"]),
                                    random_state=int(settings["random_state"]))
    fold = np.full(len(registry), -1, dtype=int)
    for index, (_fit, validation) in enumerate(splitter.split(
            registry["row"].to_numpy(), registry["label"].to_numpy(),
            groups=registry["group"].to_numpy())):
        fold[validation] = index
    if (fold < 0).any():
        raise ChemicalSpaceCVError(f"{name}: fold assignment is incomplete")
    registry = registry.copy()
    registry["fold"] = fold
    # no group may cross a fold boundary
    crossing = registry.groupby("group")["fold"].nunique()
    if int(crossing.max()) != 1:
        raise ChemicalSpaceCVError(
            f"{name}: {int((crossing > 1).sum())} groups cross a fold boundary")
    counts = registry.groupby("fold")["label"].agg(["size", "sum"])
    feasibility[f"{name}_fold_sizes"] = counts["size"].astype(int).tolist()
    feasibility[f"{name}_fold_positives"] = counts["sum"].astype(int).tolist()
    feasibility[f"{name}_fold_negatives"] = (counts["size"] - counts["sum"]).astype(int).tolist()
    feasibility_settings = settings.get("feasibility", {})
    min_positive = int(feasibility_settings.get("min_positive_per_fold", 1))
    min_negative = int(feasibility_settings.get("min_negative_per_fold", 1))
    if int(counts["sum"].min()) < min_positive or \
       int((counts["size"] - counts["sum"]).min()) < min_negative:
        raise ChemicalSpaceCVError(
            f"{name}: assigned folds violate the prespecified minimum of "
            f"{min_positive} positives and {min_negative} negatives")
    feasibility[f"{name}_fold_size_ratio_max_min"] = float(
        counts["size"].max() / counts["size"].min())
    return registry


def _run_registry(panel: Panel, features, labels, paper_row_indices, registry, name,
                  checkpoint_dir: Path) -> pd.DataFrame:
    checkpoint = checkpoint_dir / f"{name}.csv"
    if checkpoint.is_file():
        print(f"  {name}: resuming from checkpoint", flush=True)
        return pd.read_csv(checkpoint)
    fold = registry["fold"].to_numpy(int)
    order = sorted(np.unique(fold))
    jobs = [(np.flatnonzero(fold != index), np.flatnonzero(fold == index))
            for index in order]
    columns_absolute = run_folds(panel, features, labels, jobs,
                                 [42 + int(index) for index in order],
                                 n_rows=len(labels), tag=f"{name} fold")
    columns = columns_absolute
    blended = add_blends(columns)
    frame = pd.DataFrame({"registry": name,
                          "paper_row_index": np.asarray(paper_row_indices, int),
                          "fold": fold, "label": labels,
                          **{f"p_{k}": v for k, v in blended.items()}})
    frame.to_csv(checkpoint, index=False, lineterminator="\n")
    return frame


def _applicability(bits: np.ndarray, fold: np.ndarray, registry_name: str) -> pd.DataFrame:
    similarity = _similarity_matrix(bits)
    np.fill_diagonal(similarity, -1.0)
    nearest = np.full(len(fold), np.nan)
    for index in np.unique(fold):
        validation = np.flatnonzero(fold == index)
        fit_rows = np.flatnonzero(fold != index)
        nearest[validation] = similarity[np.ix_(validation, fit_rows)].max(axis=1)
    row = {"registry": registry_name, "n": int(len(fold)),
           "mean_max_train_tanimoto": float(nearest.mean()),
           "median_max_train_tanimoto": float(np.median(nearest)),
           "p10": float(np.percentile(nearest, 10)),
           "p90": float(np.percentile(nearest, 90))}
    for cut in SIMILARITY_GRID:
        row[f"fraction_above_{cut}"] = float((nearest > cut).mean())
    return pd.DataFrame([row]), nearest


def run(*, root: Path, config_path: Path, run_id: str) -> Path:
    if not re.fullmatch(r"chemical_space_cv_[a-z0-9_.-]+", run_id):
        raise ChemicalSpaceCVError("RUN_ID must start with chemical_space_cv_")
    root = root.resolve()
    protocol, protocol_sha = _load_protocol(config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise ChemicalSpaceCVError(f"Run directory already exists: {destination}")

    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    frame, audit = load_d1_train(root)
    labels = frame["label"].to_numpy(int)
    paper_row_indices = frame["paper_row_index"].to_numpy(int)
    raw_smiles = frame["smiles"].astype(str).to_numpy(object)
    standardized_smiles = frame["standardized_parent_smiles"].astype(str).to_numpy(object)
    features = _features(frame, raw_smiles.tolist(), fixed_protocol)
    split_sha = audit["paper_split_sha256"]
    y_train = labels

    c1, c2, feasibility = build_registries(standardized_smiles, y_train, protocol)
    c1.insert(1, "paper_row_index", paper_row_indices)
    c2.insert(1, "paper_row_index", paper_row_indices)
    print(f"C1 scaffold groups: {feasibility['c1_n_groups']} "
          f"(largest {feasibility['c1_largest_group_rows']} rows, "
          f"{feasibility['c1_acyclic_rows']} acyclic)", flush=True)
    print(f"C2 similarity components at {feasibility['threshold']}: "
          f"{feasibility['n_components']} (largest {feasibility['largest_component_rows']} "
          f"rows = {feasibility['largest_component_fraction']:.3f} of the partition); "
          f"feasible={feasibility['feasible']}", flush=True)

    c1 = _assign_folds(c1, protocol["registries"]["C1_scaffold"], feasibility, "C1_scaffold")
    registries = {"C1_scaffold": c1}
    if feasibility["feasible"]:
        try:
            c2 = _assign_folds(c2, protocol["registries"]["C2_similarity_component"],
                               feasibility, "C2_similarity_component")
        except ChemicalSpaceCVError as exc:
            feasibility["feasible"] = False
            feasibility["C2_status"] = "BLOCKED_INFEASIBLE_ASSIGNMENT"
            feasibility["C2_blocked_reason"] = str(exc)
        else:
            registries["C2_similarity_component"] = c2
            feasibility["C2_status"] = "RUN"
    else:
        feasibility["C2_status"] = "BLOCKED_INFEASIBLE"
        feasibility["C2_blocked_reason"] = (
            "A single Morgan connected component at Tanimoto "
            f"{feasibility['threshold']} holds "
            f"{feasibility['largest_component_fraction']:.3f} of the 324 training "
            f"compounds, above the prespecified ceiling of "
            f"{feasibility['max_single_group_fraction_allowed']}. Five "
            "non-degenerate indivisible folds are therefore not achievable. The "
            "similarity threshold was not changed after this was observed.")
        print("  C2 registry BLOCKED: " + feasibility["C2_blocked_reason"], flush=True)

    checkpoint_dir = root / "outputs" / f".{run_id}.work"
    checkpoint_contract = {
        "run_id": run_id, "protocol_sha256": protocol_sha,
        "paper_split_sha256": split_sha,
        "train_boundary_inputs": {
            "split_registry_sha256": audit["split_registry"]["sha256"],
            "train_oof_sha256": audit["train_oof_label_source"]["sha256"]},
        "source_sha256": {
            name: sha256_file(root / "src" / "geroprotector" / name)
            for name in ("chemical_space_cv.py", "model_panel.py", "study_common.py",
                         "d1_training.py")},
    }
    checkpoint_binding_sha = bind_checkpoint_directory(
        checkpoint_dir, checkpoint_contract, error_cls=ChemicalSpaceCVError)
    panel = Panel(root)
    predictions = [_run_registry(panel, features, labels, paper_row_indices, registry,
                                 name, checkpoint_dir)
                   for name, registry in registries.items()]

    # ---- reference: sealed random seed-42 registry --------------------------
    reference = _train_oof_probabilities(root)
    reference = reference.sort_values("paper_row_index").reset_index(drop=True)
    reference_columns = {m: reference[f"p_{m}"].to_numpy(float) for m in BASE_MODELS}
    reference_block = pd.DataFrame({
        "registry": "random_seed42", "paper_row_index": reference["paper_row_index"],
        "fold": reference["fold"], "label": reference["label"],
        **{f"p_{k}": v for k, v in add_blends(reference_columns).items()}})
    predictions.append(reference_block)
    all_predictions = pd.concat(predictions, ignore_index=True)

    pooled, per_fold = [], []
    for registry_name, block in all_predictions.groupby("registry"):
        for model in PANEL:
            probability = block[f"p_{model}"].to_numpy(float)
            y = block["label"].to_numpy(int)
            pooled.append({"registry": registry_name, "model_id": model,
                           **_metrics(y, probability, probability)})
            for index in sorted(block["fold"].unique()):
                mask = (block["fold"] == index).to_numpy()
                per_fold.append({"registry": registry_name, "fold": int(index),
                                 "model_id": model,
                                 **_metrics(y[mask], probability[mask], probability[mask])})
    pooled = pd.DataFrame(pooled)
    per_fold = pd.DataFrame(per_fold)

    delta_rows = []
    baseline = pooled[pooled.registry == "random_seed42"].set_index("model_id")
    for registry_name in registries:
        piece = pooled[pooled.registry == registry_name].set_index("model_id")
        for metric in PRIMARY_METRICS:
            better = ("lower" if metric in LOWER_IS_BETTER else "higher")
            for model in PANEL:
                grouped, random_value = float(piece.loc[model, metric]), float(baseline.loc[model, metric])
                delta_rows.append({
                    "registry": registry_name, "model_id": model, "metric": metric,
                    "random_seed42": random_value, "grouped": grouped,
                    "delta_grouped_minus_random": grouped - random_value,
                    "direction_better": better,
                    "grouped_is_worse": bool((grouped < random_value) if better == "higher"
                                             else (grouped > random_value))})
        # rank changes
        for metric in PRIMARY_METRICS:
            ascending = metric in LOWER_IS_BETTER
            random_rank = baseline[metric].rank(ascending=ascending, method="min")
            grouped_rank = piece[metric].rank(ascending=ascending, method="min")
            for model in PANEL:
                delta_rows.append({
                    "registry": registry_name, "model_id": model,
                    "metric": f"rank::{metric}",
                    "random_seed42": float(random_rank[model]),
                    "grouped": float(grouped_rank[model]),
                    "delta_grouped_minus_random": float(grouped_rank[model] - random_rank[model]),
                    "direction_better": "lower", "grouped_is_worse":
                        bool(grouped_rank[model] > random_rank[model])})
    deltas = pd.DataFrame(delta_rows)

    bits = morgan_matrix(list(standardized_smiles), morgan_generator())
    applicability = []
    nearest_by_registry = {}
    for registry_name, registry in registries.items():
        piece, nearest = _applicability(bits, registry["fold"].to_numpy(int), registry_name)
        applicability.append(piece)
        nearest_by_registry[registry_name] = nearest
    reference_fold = reference.set_index("paper_row_index").loc[paper_row_indices, "fold"].to_numpy(int)
    piece, nearest = _applicability(bits, reference_fold, "random_seed42")
    applicability.append(piece)
    nearest_by_registry["random_seed42"] = nearest
    applicability = pd.concat(applicability, ignore_index=True)
    for registry_name, nearest in nearest_by_registry.items():
        mapping = dict(zip(paper_row_indices, nearest))
        mask = all_predictions["registry"] == registry_name
        all_predictions.loc[mask, "maximum_tanimoto_to_active_fit_fold"] = \
            all_predictions.loc[mask, "paper_row_index"].map(mapping)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".chemcv.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "scaffold_group_registry.csv", c1)
        _write_csv(tmp / "similarity_component_registry.csv", c2)
        atomic_write_json(tmp / "group_feasibility_audit.json", feasibility)
        _write_csv(tmp / "per_row_grouped_oof_predictions.csv", all_predictions)
        _write_csv(tmp / "per_fold_grouped_metrics.csv", per_fold)
        _write_csv(tmp / "pooled_grouped_metrics.csv", pooled)
        _write_csv(tmp / "random_vs_grouped_deltas.csv", deltas)
        _write_csv(tmp / "applicability_by_registry.csv", applicability)

        lines = ["# Experiment C -- chemistry-aware grouped validation", "",
                 f"run_id: `{run_id}`  |  D1 train only  |  threshold fixed at 0.5", "",
                 "## Group feasibility audit", "",
                 "```json", json.dumps(feasibility, indent=2), "```", "",
                 "## Applicability by registry", "",
                 md_table(applicability, "{:.4f}"), ""]
        for registry_name in list(registries) + ["random_seed42"]:
            piece = pooled[pooled.registry == registry_name][
                ["model_id"] + list(PRIMARY_METRICS)]
            lines += [f"## Pooled OOF metrics -- {registry_name}", "",
                      md_table(piece.reset_index(drop=True), "{:.4f}"), ""]
        for registry_name in registries:
            piece = deltas[(deltas.registry == registry_name) &
                           (~deltas.metric.str.startswith("rank::"))]
            wide = piece.pivot(index="model_id", columns="metric",
                               values="delta_grouped_minus_random").reset_index()
            lines += [f"## Change from the random seed-42 registry -- {registry_name}",
                      "", md_table(wide, "{:+.4f}"), ""]
        (tmp / "chemical_space_cv_summary.md").write_text("\n".join(lines) + "\n")

        atomic_write_json(tmp / "RUN_MANIFEST.json", {
            "schema_version": f"{SCHEMA}.run_manifest.v1", "run_id": run_id,
            "protocol_sha256": protocol_sha, "paper_split_sha256": split_sha,
            "checkpoint_binding_sha256": checkpoint_binding_sha,
            "registries_run": list(registries),
            "feasibility_audit": feasibility,
            "panel": list(PANEL),
            "blend_definitions": {k: list(v) for k, v in BLENDS.items()},
            "resolved_model_settings": panel.resolved_settings,
            "fixed_threshold": 0.5, "threshold_is_tuned": False,
            "groups_constructed_label_free": True,
            "similarity_threshold_changed_after_seeing_results": False,
            "rows": "D1 train only (324)",
            "d1_test_labels_loaded": False, "external_outcomes_loaded": False,
            "test_or_external_labels_used_in_fit_selection_or_threshold": False,
            "data_audit": audit,
            "runtime": {"python": platform.python_version(),
                        "platform": platform.platform()},
            "existing_runs_modified": False})
        atomic_write_json(tmp / "COMPLETED.json", {
            "schema_version": f"{SCHEMA}.completed.v1",
            "status": "COMPLETE" if feasibility["feasible"] else "COMPLETE_C2_BLOCKED",
            "run_id": run_id,
            "run_manifest_sha256": sha256_file(tmp / "RUN_MANIFEST.json"),
            "artifact_hashes": {str(p.relative_to(tmp)): sha256_file(p)
                                for p in sorted(tmp.rglob("*"))
                                if p.is_file() and p.name != "COMPLETED.json"}})
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
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
