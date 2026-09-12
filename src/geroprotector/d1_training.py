"""Load the 324-row D1 development partition without loading held-out labels.

The original source files encode class membership by file, so reading them in a
CV module necessarily materializes the 81 held-out labels.  This loader instead
joins two sealed artifacts: the train-only OOF stream supplies row IDs/labels,
and the published split registry supplies chemistry while its label column is
never read.  The seven paper descriptors are recomputed with the pinned
DataWarrior/OpenChemLib implementation and are later parity-gated through the
seed-42 paper-SVM stream.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geroprotector.chemistry.standardize import STANDARDIZATION_CONTRACT, standardize_smiles
from geroprotector.hashing import canonical_sha256, sha256_file
from geroprotector.screening_blend_external import (
    PAPER_COLUMNS,
    _datawarrior_descriptors,
    load_protocol as load_external_protocol,
)


class D1TrainBoundaryError(RuntimeError):
    """Raised when a sealed input, row alignment, or chemistry contract differs."""


SPLIT_RUN = "weightedblend405_20260817"
OOF_RUN = "quad_blend_20260822"


def _verified_artifact(root: Path, run_id: str, relative: str) -> tuple[Path, str]:
    run = root / "outputs" / run_id
    completed_path = run / "COMPLETED.json"
    if not completed_path.is_file():
        raise D1TrainBoundaryError(f"Missing completion lock: {completed_path}")
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    manifest_name = "run_manifest.json" if (run / "run_manifest.json").is_file() else "RUN_MANIFEST.json"
    manifest_path = run / manifest_name
    if not manifest_path.is_file():
        raise D1TrainBoundaryError(f"Missing run manifest: {manifest_path}")
    expected_manifest = completed.get("run_manifest_sha256")
    if expected_manifest and sha256_file(manifest_path) != expected_manifest:
        raise D1TrainBoundaryError(f"Manifest hash mismatch: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = (completed.get("artifact_hashes") or {}).get(relative)
    if expected is None:
        expected = (manifest.get("artifact_hashes") or {}).get(relative)
    target = run / relative
    if not target.is_file() or expected is None or sha256_file(target) != expected:
        raise D1TrainBoundaryError(
            f"Artifact is absent or not hash-verified by its sealed run: {target}")
    return target, expected


def load_d1_train(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return train-only chemistry, labels, descriptors, and provenance."""
    root = Path(root).resolve()
    split_path, split_hash = _verified_artifact(root, SPLIT_RUN, "split_registry.csv")
    oof_path, oof_hash = _verified_artifact(
        root, OOF_RUN, "cv5_train_oof_predictions.csv")
    split_manifest = json.loads(
        (root / "outputs" / SPLIT_RUN / "run_manifest.json").read_text(encoding="utf-8"))

    # Intentionally exclude the split registry's label column. Held-out chemistry
    # is present in this provenance table, but held-out outcomes never enter memory.
    chemistry = pd.read_csv(
        split_path, usecols=["paper_row_index", "compound_name", "smiles", "role"])
    chemistry = chemistry[chemistry["role"] == "paper_train"].drop(columns="role")
    oof = pd.read_csv(oof_path, usecols=["paper_row_index", "label"])
    if len(chemistry) != 324 or len(oof) != 324:
        raise D1TrainBoundaryError(
            f"D1 train boundary must contain 324 rows, got {len(chemistry)} and {len(oof)}")
    if chemistry["paper_row_index"].duplicated().any() or oof["paper_row_index"].duplicated().any():
        raise D1TrainBoundaryError("D1 train row IDs are duplicated")
    frame = oof.merge(chemistry, on="paper_row_index", how="inner", validate="one_to_one")
    if len(frame) != 324 or set(frame["label"].astype(int).unique()) != {0, 1}:
        raise D1TrainBoundaryError("D1 train identity join or label vector is invalid")

    standardized = [standardize_smiles(value) for value in frame["smiles"].astype(str)]
    frame["standardized_parent_smiles"] = [item.standardized_parent_smiles for item in standardized]
    frame["connectivity_inchikey"] = [item.connectivity_inchikey for item in standardized]
    frame["input_component_count"] = [item.component_count for item in standardized]

    external_protocol, external_protocol_sha = load_external_protocol(
        root, root / "configs" / "screening_blend_external_protocol.yaml")
    request = pd.DataFrame({
        "stable_id": frame["paper_row_index"].astype(str),
        "compound_name": frame["compound_name"].astype(str),
        "source_smiles": frame["smiles"].astype(str),
    })
    descriptors, descriptor_audit = _datawarrior_descriptors(
        root, external_protocol, request, d1_parity=True)
    if descriptors["stable_id"].astype(str).tolist() != request["stable_id"].tolist():
        raise D1TrainBoundaryError("DataWarrior descriptor rows were reordered")
    for column in PAPER_COLUMNS:
        frame[column] = pd.to_numeric(descriptors[column], errors="raise").to_numpy(float)
    if not np.isfinite(frame[list(PAPER_COLUMNS)].to_numpy(float)).all():
        raise D1TrainBoundaryError("DataWarrior returned a non-finite paper descriptor")

    audit = {
        "rows": 324,
        "positive": int(frame["label"].sum()),
        "negative": int((frame["label"] == 0).sum()),
        "paper_row_index_sha256": canonical_sha256(
            frame["paper_row_index"].astype(int).tolist()),
        "paper_split_sha256": split_manifest["paper_split_sha256"],
        "split_registry": {"path": str(split_path), "sha256": split_hash,
                           "label_column_loaded": False},
        "train_oof_label_source": {"path": str(oof_path), "sha256": oof_hash},
        "d1_test_labels_loaded": False,
        "datawarrior": descriptor_audit,
        "external_protocol_sha256": external_protocol_sha,
        "standardization_contract": STANDARDIZATION_CONTRACT,
        "multicomponent_input_rows": int((frame["input_component_count"] > 1).sum()),
    }
    return frame, audit
