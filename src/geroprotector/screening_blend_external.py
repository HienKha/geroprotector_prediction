"""Retrospective DrugAge and AgeXtend challenges for the locked screening blend."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.special import expit

from geroprotector.fixed_blend_paper405 import _tabpfn_probability, _tanimoto
from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.screening_blend_paper405 import (
    LOCKED_WEIGHTS,
    load_locked_bundle,
)
from geroprotector.traditional_paper405 import evaluation_metrics


class ScreeningBlendExternalError(RuntimeError):
    """Raised when a retrospective external contract is violated."""


PAPER_COLUMNS = (
    "Total Molweight",
    "cLogP",
    "H-Acceptors",
    "H-Donors",
    "Total Surface Area",
    "Relative PSA",
    "Rotatable Bonds",
)
_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_XLSX_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _regular_file(path: Path, role: str, expected_sha256: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ScreeningBlendExternalError(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected_sha256 is not None and sha256_file(resolved) != expected_sha256:
        raise ScreeningBlendExternalError(f"{role} SHA256 differs from the protocol")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite immutable CSV: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(_regular_file(path, "JSON artifact").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ScreeningBlendExternalError(f"JSON artifact is not an object: {path}")
    return value


def _safe_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    return re.sub(r"[^a-z0-9]+", "", text.encode("ascii", "ignore").decode().lower())


def _parent_identity(smiles: object) -> dict[str, Any]:
    text = str(smiles).strip()
    raw = Chem.MolFromSmiles(text)
    if raw is None:
        raise ScreeningBlendExternalError(f"RDKit cannot parse SMILES: {text!r}")
    component_count = len(Chem.GetMolFrags(raw))
    excluded_elements = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}
    metals = sorted(
        {atom.GetAtomicNum() for atom in raw.GetAtoms()} - excluded_elements
    )
    try:
        parent = rdMolStandardize.FragmentParent(rdMolStandardize.Cleanup(raw))
        parent = rdMolStandardize.Uncharger().uncharge(parent)
        Chem.SanitizeMol(parent)
    except Exception as exc:
        raise ScreeningBlendExternalError(
            f"RDKit cannot standardize SMILES: {text!r}"
        ) from exc
    full_key = Chem.MolToInchiKey(parent)
    if not full_key:
        raise ScreeningBlendExternalError("RDKit could not derive an InChIKey")
    return {
        "raw_smiles": text,
        "standardized_parent_smiles": Chem.MolToSmiles(
            parent, canonical=True, isomericSmiles=True
        ),
        "full_inchikey": full_key,
        "connectivity_inchikey": full_key.split("-", maxsplit=1)[0],
        "component_count": component_count,
        "metal_atomic_numbers": metals,
    }


def _xlsx_column_index(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference)
    if letters is None:
        raise ScreeningBlendExternalError("Invalid XLSX cell reference")
    value = 0
    for character in letters.group():
        value = value * 26 + ord(character) - 64
    return value - 1


def _xlsx_sheet(path: Path, sheet_name: str, requested: tuple[str, ...]) -> pd.DataFrame:
    """Read selected XLSX columns without adding an Excel dependency to the lock."""

    with ZipFile(path) as archive:
        names = set(archive.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in item.iter(_XLSX_NS + "t"))
                for item in shared_root.findall(_XLSX_NS + "si")
            ]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in relationships}
        sheets = workbook.find(_XLSX_NS + "sheets")
        if sheets is None:
            raise ScreeningBlendExternalError("XLSX has no sheets")
        selected = next(
            (item for item in sheets if item.attrib.get("name") == sheet_name), None
        )
        if selected is None:
            raise ScreeningBlendExternalError(f"XLSX sheet is absent: {sheet_name}")
        target = targets[selected.attrib[_XLSX_REL + "id"]]
        target = target.lstrip("/") if target.startswith("/") else "xl/" + target
        root = ET.fromstring(archive.read(target))
        raw_rows: list[dict[int, str]] = []
        for row in root.iter(_XLSX_NS + "row"):
            values: dict[int, str] = {}
            for cell in row.findall(_XLSX_NS + "c"):
                value_node = cell.find(_XLSX_NS + "v")
                value = "" if value_node is None else value_node.text or ""
                if cell.attrib.get("t") == "s" and value:
                    value = shared[int(value)]
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.iter(_XLSX_NS + "t")
                    )
                values[_xlsx_column_index(cell.attrib["r"])] = value
            raw_rows.append(values)
    if not raw_rows:
        raise ScreeningBlendExternalError("XLSX sheet is empty")
    header = raw_rows[0]
    by_name = {str(value).strip(): index for index, value in header.items()}
    missing = set(requested) - set(by_name)
    if missing:
        raise ScreeningBlendExternalError(f"XLSX columns missing: {sorted(missing)}")
    records = [
        {name: row.get(by_name[name], "") for name in requested}
        for row in raw_rows[1:]
        if any(row.get(by_name[name], "") != "" for name in requested)
    ]
    return pd.DataFrame(records, columns=requested)


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    value = yaml.safe_load(_regular_file(path, "external protocol").read_text())
    if not isinstance(value, dict) or value.get("schema_version") != (
        "geroprotector.screening_blend_external.protocol.v2"
    ):
        raise ScreeningBlendExternalError("Unknown external protocol schema")
    if value.get("model_weights") != [0.1, 0.6, 0.3] or value.get(
        "decision_thresholds"
    ) != [0.5299579802368826, 0.5]:
        raise ScreeningBlendExternalError("Weights or thresholds differ from the lock")
    if value.get("threshold_reoptimization_on_external") != "forbidden":
        raise ScreeningBlendExternalError("External thresholds may not be optimized")
    for source in value["sources"].values():
        _regular_file(Path(source["path"]), source["role"], source["sha256"])
    _regular_file(root / value["datawarrior"]["java_source"], "DataWarrior CLI source")
    return value, canonical_sha256(value)


def _d1_frame(protocol: dict[str, Any]) -> pd.DataFrame:
    positive = protocol["sources"]["d1_positive"]
    negative = protocol["sources"]["d1_negative"]
    first = pd.read_csv(
        positive["path"], sep="\t", encoding="latin1"
    ).rename(columns={"Compound Name": "compound_name", "Smiles": "smiles"})
    second = pd.read_csv(negative["path"], encoding="latin1").rename(
        columns={"canonical_smiles": "smiles"}
    )
    frame = pd.concat(
        [first.assign(label=1), second.assign(label=0)], ignore_index=True
    )
    if len(frame) != 405 or tuple(frame[list(PAPER_COLUMNS)].columns) != PAPER_COLUMNS:
        raise ScreeningBlendExternalError("D1 source reconstruction differs")
    frame.insert(0, "paper_row_index", np.arange(len(frame), dtype=int))
    return frame


def _d1_firewall(frame: pd.DataFrame) -> tuple[set[str], set[str]]:
    identities: set[str] = set()
    for value in frame.smiles:
        identities.add(_parent_identity(value)["connectivity_inchikey"])
    return identities, {_safe_name(value) for value in frame.compound_name}


def _prepare_records(
    *,
    cohort: str,
    mapping: pd.DataFrame,
    d1_identities: set[str],
    d1_names: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for row in mapping.itertuples(index=False):
        base = {
            "provider_record_id": str(row.provider_record_id),
            "compound_name": str(row.compound_name).strip(),
            "mapping_source": str(row.mapping_source),
            "provider_identifier": str(row.provider_identifier),
            "source_smiles": str(row.source_smiles).strip(),
        }
        try:
            identity = _parent_identity(base["source_smiles"])
        except ScreeningBlendExternalError:
            records.append({**base, "included": False, "exclusion_reason": "parse_failed"})
            continue
        reason = ""
        if identity["connectivity_inchikey"] in d1_identities:
            reason = "overlap_d1_parent_connectivity"
        elif _safe_name(base["compound_name"]) in d1_names:
            reason = "overlap_d1_normalized_name"
        elif identity["component_count"] > 1:
            reason = "multi_component_or_salt_requires_adjudication"
        elif identity["metal_atomic_numbers"]:
            reason = "metal_or_coordination_requires_adjudication"
        records.append(
            {
                **base,
                **identity,
                "included": not reason,
                "exclusion_reason": reason,
            }
        )
    ledger = pd.DataFrame(records).sort_values(
        ["included", "exclusion_reason", "provider_record_id"],
        ascending=[False, True, True],
        kind="stable",
    )
    included = ledger[ledger.included].copy()
    grouped: list[dict[str, Any]] = []
    for connectivity, group in included.groupby("connectivity_inchikey", sort=True):
        group = group.sort_values(
            ["source_smiles", "compound_name", "provider_record_id"], kind="stable"
        )
        representative = group.iloc[0]
        grouped.append(
            {
                "external_id": f"{cohort}::{connectivity}",
                "cohort": cohort,
                "connectivity_inchikey": connectivity,
                "full_inchikey": representative.full_inchikey,
                "compound_name": representative.compound_name,
                "source_smiles": representative.source_smiles,
                "standardized_parent_smiles": representative.standardized_parent_smiles,
                "source_names_json": json.dumps(
                    sorted(set(group.compound_name.astype(str))), ensure_ascii=False
                ),
                "provider_record_ids_json": json.dumps(
                    sorted(set(group.provider_record_id.astype(str)))
                ),
                "provider_identifiers_json": json.dumps(
                    sorted(set(group.provider_identifier.astype(str)))
                ),
                "collapsed_mapping_rows": len(group),
            }
        )
    identities = pd.DataFrame(grouped).sort_values("external_id", kind="stable")
    if identities.external_id.duplicated().any() or identities.empty:
        raise ScreeningBlendExternalError("Prepared identities are empty or duplicated")
    return identities, ledger


def _drugage_mapping(protocol: dict[str, Any]) -> pd.DataFrame:
    source = protocol["sources"]
    names = set(
        pd.read_csv(source["drugage_raw"]["path"], usecols=["compound_name"])[
            "compound_name"
        ].astype(str)
    )
    mapping = pd.read_csv(source["drugage_mapping"]["path"])
    required = {
        "compound_name",
        "cid",
        "canonical_smiles",
        "isomeric_smiles",
        "inchikey",
        "pubchem_status",
    }
    if set(mapping) != required or set(mapping.pubchem_status.astype(str)) != {"ok"}:
        raise ScreeningBlendExternalError("DrugAge mapping schema/status differs")
    mapping = mapping[mapping.compound_name.astype(str).isin(names)].copy()
    mapping["source_smiles"] = mapping.isomeric_smiles.fillna("").astype(str)
    blank = mapping.source_smiles.str.strip().eq("")
    mapping.loc[blank, "source_smiles"] = mapping.loc[blank, "canonical_smiles"]
    mapping["provider_record_id"] = "drugage-name::" + mapping.compound_name.astype(str)
    mapping["mapping_source"] = "pinned_pubchem_name_mapping"
    mapping["provider_identifier"] = "PubChemCID:" + mapping.cid.astype(str)
    return mapping[
        [
            "provider_record_id",
            "compound_name",
            "mapping_source",
            "provider_identifier",
            "source_smiles",
        ]
    ]


def _agextend_table(protocol: dict[str, Any], outcomes: bool) -> pd.DataFrame:
    columns = (
        "S.No.",
        "Compound Name",
        "Isomeric_SMILES",
        "Label",
        "Source",
    ) if outcomes else ("S.No.", "Compound Name", "Isomeric_SMILES")
    return _xlsx_sheet(
        Path(protocol["sources"]["agextend_external_table"]["path"]),
        "Supplementary Table 6",
        columns,
    )


def _agextend_mapping(protocol: dict[str, Any]) -> pd.DataFrame:
    table = _agextend_table(protocol, outcomes=False)
    return pd.DataFrame(
        {
            "provider_record_id": "agextend-table6::" + table["S.No."].astype(str),
            "compound_name": table["Compound Name"].astype(str),
            "mapping_source": "provider_isomeric_smiles",
            "provider_identifier": "SupplementaryTable6:" + table["S.No."].astype(str),
            "source_smiles": table["Isomeric_SMILES"].astype(str),
        }
    )


def prepare_cohort(
    root: Path, protocol: dict[str, Any], protocol_sha256: str, run: Path, cohort: str
) -> Path:
    destination = run / cohort / "prepared"
    lock_path = destination / "PREPARED.json"
    if lock_path.is_file() and not lock_path.is_symlink():
        lock = _load_json(lock_path)
        for name, expected in lock["artifact_hashes"].items():
            if sha256_file(_regular_file(destination / name, name)) != expected:
                raise ScreeningBlendExternalError(f"Prepared artifact drift: {name}")
        return destination
    if destination.exists() or destination.is_symlink():
        raise ScreeningBlendExternalError(f"Unsealed prepared directory exists: {destination}")
    paper = _d1_frame(protocol)
    d1_identities, d1_names = _d1_firewall(paper)
    mapping = _drugage_mapping(protocol) if cohort == "drugage" else _agextend_mapping(protocol)
    identities, ledger = _prepare_records(
        cohort=cohort,
        mapping=mapping,
        d1_identities=d1_identities,
        d1_names=d1_names,
    )
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".prepared.work-", dir=parent))
    try:
        _write_csv(temporary / "identity_only.csv", identities)
        _write_csv(temporary / "mapping_and_exclusion_ledger.csv", ledger)
        artifacts = {
            name: sha256_file(temporary / name)
            for name in ("identity_only.csv", "mapping_and_exclusion_ledger.csv")
        }
        atomic_write_json(
            temporary / "PREPARED.json",
            {
                "schema_version": "geroprotector.screening_blend_external.prepared.v1",
                "cohort": cohort,
                "protocol_sha256": protocol_sha256,
                "d1_rows_audited": len(paper),
                "d1_unique_parent_identities": len(d1_identities),
                "included_parent_identities": len(identities),
                "mapping_rows": len(ledger),
                "exclusion_counts": {
                    str(key): int(value)
                    for key, value in ledger.exclusion_reason.replace("", "included")
                    .value_counts()
                    .items()
                },
                "outcomes_used_for_identity_or_exclusion": False,
                "public_outcomes_co_located_in_provider_file": True,
                "artifact_hashes": artifacts,
            },
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _datawarrior_descriptors(
    root: Path,
    protocol: dict[str, Any],
    rows: pd.DataFrame,
    *,
    d1_parity: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    settings = protocol["datawarrior"]
    java_root = Path(settings["jdk_root"])
    java = _regular_file(java_root / "bin" / "java", "Java runtime")
    javac = _regular_file(java_root / "bin" / "javac", "Java compiler")
    jar = _regular_file(
        Path(settings["openchemlib_jar"]), "OpenChemLib", settings["jar_sha256"]
    )
    source = _regular_file(root / settings["java_source"], "DataWarrior CLI source")
    repository = Path(settings["datawarrior_repository"])
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if commit != settings["datawarrior_commit"]:
        raise ScreeningBlendExternalError("DataWarrior checkout differs from tag v5.5.0")
    with tempfile.TemporaryDirectory(prefix="datawarrior-descriptors-") as temporary_name:
        classes = Path(temporary_name)
        subprocess.run(
            [str(javac), "-cp", str(jar), "-d", str(classes), str(source)], check=True
        )
        lines: list[str] = []
        for row in rows.itertuples(index=False):
            smiles = str(row.source_smiles).strip()
            if d1_parity and str(row.compound_name) == "Ozone":
                smiles = "O=O=O"
            lines.append(f"{row.stable_id}\t{smiles}\n")
        result = subprocess.run(
            [
                str(java),
                "-cp",
                f"{jar}:{classes}",
                "DataWarriorDescriptors",
            ],
            input="".join(lines),
            text=True,
            capture_output=True,
            check=True,
        )
    output = pd.read_csv(io.StringIO(result.stdout), sep="\t")
    if output.stable_id.astype(str).tolist() != rows.stable_id.astype(str).tolist():
        raise ScreeningBlendExternalError("DataWarrior output order/IDs differ")
    return output, {
        "datawarrior_commit": commit,
        "openchemlib_sha256": sha256_file(jar),
        "java_source_sha256": sha256_file(source),
        "java_version": subprocess.run(
            [str(java), "-version"], text=True, capture_output=True, check=True
        ).stderr.splitlines()[0],
        "stderr": result.stderr.strip(),
    }


def _datawarrior_parity(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    paper = _d1_frame(protocol).rename(columns={"smiles": "source_smiles"})
    paper["stable_id"] = paper.paper_row_index.astype(str)
    calculated, audit = _datawarrior_descriptors(
        root,
        protocol,
        paper[["stable_id", "compound_name", "source_smiles"]],
        d1_parity=True,
    )
    expected = paper[list(PAPER_COLUMNS)].to_numpy(dtype=float)
    observed = calculated[list(PAPER_COLUMNS)].to_numpy(dtype=float)
    if not np.array_equal(expected, observed):
        difference = float(np.nanmax(np.abs(expected - observed)))
        raise ScreeningBlendExternalError(
            f"DataWarrior parity failed on D1 (maximum absolute difference {difference})"
        )
    return {**audit, "paper_rows": 405, "all_seven_values_exact": True}


def _rdkit_features(
    identities: pd.DataFrame, bundle: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    names = [name for name, _function in Descriptors._descList]
    if names != bundle["descriptor_names"]:
        raise ScreeningBlendExternalError("RDKit descriptor registry differs from bundle")
    contract = bundle["portable_contract"]
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(contract["morgan_radius"]),
        fpSize=int(contract["morgan_bits"]),
        includeChirality=bool(contract["morgan_include_chirality"]),
    )
    bits = np.zeros((len(identities), int(contract["morgan_bits"])), dtype=np.uint8)
    descriptors = np.full((len(identities), len(names)), np.nan, dtype=np.float64)
    for row_index, smiles in enumerate(identities.source_smiles.astype(str)):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ScreeningBlendExternalError("Prepared structure became unparsable")
        bits[row_index] = np.asarray(generator.GetFingerprint(molecule), dtype=np.uint8)
        for column_index, (_name, function) in enumerate(Descriptors._descList):
            try:
                value = float(function(molecule))
            except Exception:
                value = np.nan
            # Compare in Python/float64 space.  Comparing a very large RDKit
            # descriptor directly with the numpy float32 scalar can itself
            # trigger a noisy overflow warning before the value is rejected.
            float32_max = float(np.finfo(np.float32).max)
            descriptors[row_index, column_index] = (
                value if np.isfinite(value) and abs(value) <= float32_max else np.nan
            )
    return bits, descriptors


def _frozen_descriptor_transform(
    raw: np.ndarray, context: dict[str, np.ndarray]
) -> np.ndarray:
    retained = raw[:, np.asarray(context["finite_any_mask"], dtype=bool)]
    medians = np.asarray(context["medians_after_finite_any"], dtype=float)
    retained = np.where(np.isfinite(retained), retained, medians)
    output = retained[:, np.asarray(context["varying_after_imputation_mask"], dtype=bool)]
    output = output.astype(np.float32)
    if not np.isfinite(output).all() or output.shape[1] != context["context_features"].shape[1]:
        raise ScreeningBlendExternalError("Frozen TabPFN descriptor transform failed")
    return output


def _verified_bundles(
    root: Path, protocol: dict[str, Any]
) -> tuple[dict[str, Any], float, float]:
    lock_path = _regular_file(root / protocol["model_lock"], "model lock")
    lock = _load_json(lock_path)
    if lock.get("weights") != [0.1, 0.6, 0.3] or lock.get(
        "selection_used_outer_test"
    ) is not False:
        raise ScreeningBlendExternalError("Model lock semantics differ")
    loaded = []
    for record in lock["bundles"]:
        bundle_path = lock_path.parent / record["path"]
        bundle = load_locked_bundle(
            bundle_path, expected_artifact_sha256=record["sha256"]
        )
        loaded.append((bundle, float(record["decision_threshold"])))
    if len(loaded) != 2 or loaded[0][0]["component_state_sha256"] != loaded[1][0][
        "component_state_sha256"
    ]:
        raise ScreeningBlendExternalError("Decision bundles do not share one model")
    thresholds = {value for _bundle, value in loaded}
    if thresholds != {0.5, 0.5299579802368826}:
        raise ScreeningBlendExternalError("Decision thresholds differ")
    return loaded[0][0], 0.5299579802368826, 0.5


def predict_cohort(
    root: Path, protocol: dict[str, Any], protocol_sha256: str, run: Path, cohort: str
) -> Path:
    prepared = prepare_cohort(root, protocol, protocol_sha256, run, cohort)
    destination = run / cohort / "predicted"
    lock_path = destination / "PREDICTION_LOCK.json"
    if lock_path.is_file() and not lock_path.is_symlink():
        lock = _load_json(lock_path)
        if sha256_file(destination / "predictions.csv") != lock["predictions_sha256"]:
            raise ScreeningBlendExternalError("Sealed predictions drifted")
        return destination
    if destination.exists() or destination.is_symlink():
        raise ScreeningBlendExternalError(
            f"Unsealed prediction directory exists: {destination}"
        )
    bundle, threshold_oof, threshold_fixed = _verified_bundles(root, protocol)
    identities = pd.read_csv(prepared / "identity_only.csv")
    dw_parity = _datawarrior_parity(root, protocol)
    dw_input = identities[["external_id", "compound_name", "source_smiles"]].rename(
        columns={"external_id": "stable_id"}
    )
    paper_features, dw_audit = _datawarrior_descriptors(
        root, protocol, dw_input, d1_parity=False
    )
    valid = paper_features[list(PAPER_COLUMNS)].notna().all(axis=1).to_numpy()
    feature_exclusions = identities.loc[~valid, ["external_id", "compound_name"]].copy()
    feature_exclusions["exclusion_reason"] = "datawarrior_descriptor_failure"
    identities = identities.loc[valid].reset_index(drop=True)
    paper_matrix = paper_features.loc[valid, list(PAPER_COLUMNS)].to_numpy(dtype=float)
    if list(PAPER_COLUMNS) != bundle["paper_svm_feature_names"]:
        raise ScreeningBlendExternalError("Paper-SVM feature order differs")
    bits, raw_descriptors = _rdkit_features(identities, bundle)
    tabpfn_query = _frozen_descriptor_transform(
        raw_descriptors, bundle["tabpfn_context"]
    )
    probability_svm = np.asarray(
        bundle["paper_svm"].predict_proba(paper_matrix)[:, 1], dtype=float
    )
    similarity = _tanimoto(bits, np.asarray(bundle["tanimoto_train_bits"], dtype=np.uint8))
    probability_tanimoto = expit(bundle["tanimoto_svc"].decision_function(similarity))
    probability_tabpfn, tabpfn_audit = _tabpfn_probability(
        np.asarray(bundle["tabpfn_context"]["context_features"], dtype=np.float32),
        np.asarray(bundle["fit_labels"], dtype=int),
        tabpfn_query,
        bundle["tabpfn_checkpoint"],
        42,
    )
    components = np.column_stack(
        [probability_svm, probability_tanimoto, probability_tabpfn]
    )
    probability = components @ LOCKED_WEIGHTS
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ScreeningBlendExternalError("External blend probabilities are invalid")
    predictions = identities.copy()
    predictions["probability_paper_svm"] = probability_svm
    predictions["probability_tanimoto_svc"] = probability_tanimoto
    predictions["probability_tabpfn_v2"] = probability_tabpfn
    predictions["blend_probability"] = probability
    predictions["rank_descending"] = pd.Series(probability).rank(
        method="min", ascending=False
    ).astype(int)
    predictions["decision_oof_mcc_0p5299579802368826"] = probability >= threshold_oof
    predictions["decision_fixed_0p5"] = probability >= threshold_fixed
    predictions["maximum_tanimoto_to_fitted_train"] = similarity.max(axis=1)
    predictions["within_similarity_0p40"] = similarity.max(axis=1) >= 0.40
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".predicted.work-", dir=destination.parent))
    try:
        _write_csv(temporary / "predictions.csv", predictions)
        _write_csv(temporary / "feature_exclusions.csv", feature_exclusions)
        feature_audit = {
            "schema_version": "geroprotector.screening_blend_external.features.v1",
            "cohort": cohort,
            "input_identities": len(valid),
            "predicted_identities": len(predictions),
            "datawarrior_failures": int((~valid).sum()),
            "datawarrior_d1_parity": dw_parity,
            "datawarrior_external_run": dw_audit,
            "rdkit_descriptor_count": raw_descriptors.shape[1],
            "tabpfn_transformed_columns": tabpfn_query.shape[1],
            "tabpfn_audit": tabpfn_audit,
            "model_component_state_sha256": bundle["component_state_sha256"],
            "weights": LOCKED_WEIGHTS.tolist(),
            "outcome_columns_passed_to_prediction": False,
            "external_refit_calibration_or_threshold_selection": False,
        }
        atomic_write_json(temporary / "feature_audit.json", feature_audit)
        atomic_write_json(
            temporary / "PREDICTION_LOCK.json",
            {
                "schema_version": "geroprotector.screening_blend_external.prediction_lock.v1",
                "cohort": cohort,
                "protocol_sha256": protocol_sha256,
                "prepared_lock_sha256": sha256_file(prepared / "PREPARED.json"),
                "model_lock_sha256": sha256_file(root / protocol["model_lock"]),
                "predictions_sha256": sha256_file(temporary / "predictions.csv"),
                "feature_exclusions_sha256": sha256_file(
                    temporary / "feature_exclusions.csv"
                ),
                "feature_audit_sha256": sha256_file(temporary / "feature_audit.json"),
                "n_predictions": len(predictions),
                "thresholds": [threshold_oof, threshold_fixed],
                "outcome_endpoint_aggregation_or_metrics_run_before_lock": False,
                "public_provider_file_contains_outcome_columns": True,
            },
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _drugage_outcomes(protocol: dict[str, Any], identities: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        name: row.external_id
        for row in identities.itertuples(index=False)
        for name in json.loads(row.source_names_json)
    }
    raw = pd.read_csv(protocol["sources"]["drugage_raw"]["path"])
    raw["external_id"] = raw.compound_name.astype(str).map(aliases)
    raw = raw[raw.external_id.notna()].copy()
    raw["effect"] = pd.to_numeric(raw.avg_lifespan_change_percent, errors="coerce")
    raw["significance"] = raw.avg_lifespan_significance.fillna("").astype(str).str.strip()
    rows = []
    for external_id, group in raw.groupby("external_id", sort=True):
        finite = group.effect.notna()
        positive = (group.significance == "S") & (group.effect > 0)
        decrease = (group.significance == "S") & (group.effect < 0)
        publications = sorted(
            {
                str(int(value)) if float(value).is_integer() else str(value)
                for value in pd.to_numeric(group.pubmed_id, errors="coerce").dropna()
            }
        )
        all_nonincrease = bool(finite.all() and (group.effect <= 0).all())
        if positive.any() and not decrease.any():
            status = "positive_clean"
        elif (
            not positive.any()
            and not decrease.any()
            and all_nonincrease
            and len(publications) >= 2
        ):
            status = "negative_clean"
        else:
            status = "uncertain"
        rows.append(
            {
                "external_id": external_id,
                "observation_count": len(group),
                "publication_count": len(publications),
                "publication_ids_json": json.dumps(publications),
                "has_significant_positive": bool(positive.any()),
                "has_significant_decrease": bool(decrease.any()),
                "all_observed_avg_effects_nonincrease": all_nonincrease,
                "strict_status": status,
            }
        )
    return pd.DataFrame(rows)


def _agextend_outcomes(protocol: dict[str, Any], identities: pd.DataFrame) -> pd.DataFrame:
    provider_to_external = {
        provider_id: row.external_id
        for row in identities.itertuples(index=False)
        for provider_id in json.loads(row.provider_record_ids_json)
    }
    table = _agextend_table(protocol, outcomes=True)
    table["provider_record_id"] = "agextend-table6::" + table["S.No."].astype(str)
    table["external_id"] = table.provider_record_id.map(provider_to_external)
    table = table[table.external_id.notna()].copy()
    table["label"] = pd.to_numeric(table.Label, errors="raise").astype(int)
    if not set(table.label).issubset({0, 1}):
        raise ScreeningBlendExternalError("AgeXtend labels are not binary")
    rows = []
    for external_id, group in table.groupby("external_id", sort=True):
        labels = set(group.label)
        if len(labels) != 1:
            status = "conflict"
            label: int | None = None
        else:
            label = labels.pop()
            status = "positive_clean" if label == 1 else "negative_clean"
        rows.append(
            {
                "external_id": external_id,
                "label": label,
                "strict_status": status,
                "observation_count": len(group),
                "publication_count": group.Source.astype(str).nunique(),
                "publication_ids_json": json.dumps(sorted(set(group.Source.astype(str)))),
            }
        )
    return pd.DataFrame(rows)


def _metric_rows(frame: pd.DataFrame, endpoint: str) -> list[dict[str, Any]]:
    labels = frame.label.to_numpy(dtype=int)
    probability = frame.blend_probability.to_numpy(dtype=float)
    rows = []
    for name, threshold in (
        ("oof_mcc", 0.5299579802368826),
        ("fixed_0p5", 0.5),
    ):
        metrics = evaluation_metrics(
            labels, probability, probability, threshold=threshold
        )
        rows.append({"endpoint": endpoint, "operating_point": name, **metrics})
    return rows


def _top_k(frame: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["blend_probability", "external_id"], ascending=[False, True], kind="stable"
    )
    prevalence = float(ordered.label.mean())
    rows = []
    for k in (5, 10, 20, 50, 100):
        if k > len(ordered):
            continue
        selected = ordered.head(k)
        hits = int(selected.label.sum())
        rows.append(
            {
                "endpoint": endpoint,
                "k": k,
                "positive_hits": hits,
                "precision_at_k": hits / k,
                "recall_at_k": hits / int(ordered.label.sum()),
                "enrichment_over_prevalence": (hits / k) / prevalence,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_metrics(
    frame: pd.DataFrame, endpoint: str, resamples: int, seed: int
) -> pd.DataFrame:
    y = frame.label.to_numpy(dtype=int)
    probability = frame.blend_probability.to_numpy(dtype=float)
    groups = [np.flatnonzero(y == value) for value in (0, 1)]
    rng = np.random.default_rng(seed)
    names = (
        "auprc_average_precision_positive",
        "auroc",
        "brier",
        "mcc",
        "macro_f1",
        "recall_sensitivity",
        "specificity",
    )
    output = []
    for operating_point, threshold in (
        ("oof_mcc", 0.5299579802368826),
        ("fixed_0p5", 0.5),
    ):
        point = evaluation_metrics(y, probability, probability, threshold=threshold)
        draws = {name: [] for name in names}
        for _ in range(resamples):
            index = np.concatenate(
                [rng.choice(group, size=len(group), replace=True) for group in groups]
            )
            metric = evaluation_metrics(
                y[index], probability[index], probability[index], threshold=threshold
            )
            for name in names:
                draws[name].append(metric[name])
        for name in names:
            output.append(
                {
                    "endpoint": endpoint,
                    "operating_point": operating_point,
                    "metric": name,
                    "estimate": point[name],
                    "ci_lower": float(np.quantile(draws[name], 0.025)),
                    "ci_upper": float(np.quantile(draws[name], 0.975)),
                    "resamples": resamples,
                    "sampling_unit": "stratified_parent_connectivity_compound",
                }
            )
    return pd.DataFrame(output)


def _publication_geometry(frame: pd.DataFrame) -> dict[str, Any]:
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(first: str, second: str) -> None:
        a, b = find(first), find(second)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for row in frame.itertuples(index=False):
        compound = "compound:" + row.external_id
        find(compound)
        for publication in json.loads(row.publication_ids_json):
            union(compound, "publication:" + publication)
    compound_roots = [find("compound:" + value) for value in frame.external_id]
    counts = pd.Series(compound_roots).value_counts()
    return {
        "publication_compound_union_units": len(counts),
        "largest_union_compounds": int(counts.max()),
        "largest_union_fraction": float(counts.max() / len(frame)),
        "ordinary_cluster_inference_not_dominated": bool(counts.max() / len(frame) <= 0.5),
    }


def score_cohort(
    root: Path, protocol: dict[str, Any], protocol_sha256: str, run: Path, cohort: str
) -> Path:
    predicted = predict_cohort(root, protocol, protocol_sha256, run, cohort)
    destination = run / cohort / "scored"
    completed = destination / "SCORED.json"
    if completed.is_file() and not completed.is_symlink():
        lock = _load_json(completed)
        for name, expected in lock["artifact_hashes"].items():
            if sha256_file(destination / name) != expected:
                raise ScreeningBlendExternalError(f"Scored artifact drift: {name}")
        return destination
    if destination.exists() or destination.is_symlink():
        raise ScreeningBlendExternalError(f"Unsealed scored directory exists: {destination}")
    predictions = pd.read_csv(predicted / "predictions.csv")
    prediction_lock = _load_json(predicted / "PREDICTION_LOCK.json")
    if sha256_file(predicted / "predictions.csv") != prediction_lock["predictions_sha256"]:
        raise ScreeningBlendExternalError("Prediction bytes differ before scoring")
    if cohort == "drugage":
        outcomes = _drugage_outcomes(protocol, predictions)
        merged = predictions.merge(outcomes, on="external_id", validate="one_to_one")
        retrieval = merged.assign(label=merged.has_significant_positive.astype(int))
        strict = merged[merged.strict_status.isin(["positive_clean", "negative_clean"])].copy()
        strict["label"] = (strict.strict_status == "positive_clean").astype(int)
        endpoint_frames = {
            "significant_positive_retrieval_background_not_certified_negative": retrieval,
            "strict_binary_sensitivity": strict,
        }
    else:
        outcomes = _agextend_outcomes(protocol, predictions)
        merged = predictions.merge(outcomes, on="external_id", validate="one_to_one")
        strict = merged[merged.strict_status != "conflict"].copy()
        endpoint_frames = {"published_independent_table6_binary": strict}
    metrics: list[dict[str, Any]] = []
    top_k: list[pd.DataFrame] = []
    intervals: list[pd.DataFrame] = []
    geometries: dict[str, Any] = {}
    for endpoint, frame in endpoint_frames.items():
        if set(frame.label.astype(int)) != {0, 1}:
            continue
        metrics.extend(_metric_rows(frame, endpoint))
        top_k.append(_top_k(frame, endpoint))
        intervals.append(
            _bootstrap_metrics(
                frame,
                endpoint,
                int(protocol["evaluation"]["bootstrap_resamples"]),
                int(protocol["evaluation"]["bootstrap_seed"]),
            )
        )
        geometries[endpoint] = _publication_geometry(frame)
    metrics_frame = pd.DataFrame(metrics)
    top_frame = pd.concat(top_k, ignore_index=True) if top_k else pd.DataFrame()
    interval_frame = (
        pd.concat(intervals, ignore_index=True) if intervals else pd.DataFrame()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".scored.work-", dir=destination.parent))
    try:
        _write_csv(temporary / "outcome_ledger.csv", outcomes)
        _write_csv(temporary / "scored_predictions.csv", merged)
        _write_csv(temporary / "metrics.csv", metrics_frame)
        _write_csv(temporary / "top_k.csv", top_frame)
        _write_csv(temporary / "bootstrap_intervals.csv", interval_frame)
        atomic_write_json(
            temporary / "evaluation_audit.json",
            {
                "schema_version": "geroprotector.screening_blend_external.evaluation.v1",
                "cohort": cohort,
                "role": "published_retrospective_post_hoc_stress_test",
                "fresh_or_temporally_blinded": False,
                "prediction_lock_sha256": sha256_file(predicted / "PREDICTION_LOCK.json"),
                "prediction_file_sha256": sha256_file(predicted / "predictions.csv"),
                "external_model_refit_calibration_weight_or_threshold_selection": False,
                "thresholds": [0.5299579802368826, 0.5],
                "endpoint_counts": {
                    endpoint: {
                        "n": len(frame),
                        "positive": int(frame.label.sum()),
                        "negative_or_background": int((1 - frame.label).sum()),
                    }
                    for endpoint, frame in endpoint_frames.items()
                },
                "publication_cluster_geometry": geometries,
                "drugage_background_is_a_certified_negative_class": False,
                "agextend_negative_count_is_adequate_for_strong_inference": False,
            },
        )
        artifacts = {
            name: sha256_file(temporary / name)
            for name in (
                "outcome_ledger.csv",
                "scored_predictions.csv",
                "metrics.csv",
                "top_k.csv",
                "bootstrap_intervals.csv",
                "evaluation_audit.json",
            )
        }
        atomic_write_json(
            temporary / "SCORED.json",
            {
                "schema_version": "geroprotector.screening_blend_external.scored.v1",
                "cohort": cohort,
                "protocol_sha256": protocol_sha256,
                "artifact_hashes": artifacts,
            },
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _summary(run: Path) -> str:
    lines = [
        "# Locked screening blend — retrospective external stress tests",
        "",
        "The locked 0.10/0.60/0.30 blend was applied without refitting, calibration,",
        "threshold optimization or model choice. Both prespecified thresholds are shown.",
        "These public outcomes were already available and are not fresh confirmatory data.",
        "",
    ]
    for cohort in ("drugage", "agextend"):
        metrics = pd.read_csv(run / cohort / "scored" / "metrics.csv")
        audit = _load_json(run / cohort / "scored" / "evaluation_audit.json")
        lines.extend([f"## {cohort}", ""])
        for endpoint, group in metrics.groupby("endpoint", sort=False):
            count = audit["endpoint_counts"][endpoint]
            lines.append(
                f"### {endpoint} (n={count['n']}, positive={count['positive']}, "
                f"negative/background={count['negative_or_background']})"
            )
            lines.extend(
                [
                    "",
                    "| Threshold | AP | AUROC | MCC | Macro F1 | Recall | Specificity |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in group.itertuples(index=False):
                lines.append(
                    f"| {row.threshold:.15g} | {row.auprc_average_precision_positive:.4f} | "
                    f"{row.auroc:.4f} | {row.mcc:.4f} | {row.macro_f1:.4f} | "
                    f"{row.recall_sensitivity:.4f} | {row.specificity:.4f} |"
                )
            lines.append("")
        if cohort == "drugage":
            lines.append(
                "DrugAge non-positive/background compounds are not certified experimental "
                "negatives; strict binary results are a conservative sensitivity only."
            )
        else:
            lines.append(
                "After D1/formulation exclusions, very few AgeXtend negatives remain; binary "
                "metrics and their intervals are therefore unstable."
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def run_all(root: Path, config_path: Path, run_id: str) -> Path:
    if not re.fullmatch(r"screeningblend_external_[a-z0-9_.-]+", run_id):
        raise ScreeningBlendExternalError("RUN_ID must start with screeningblend_external_")
    root = root.resolve()
    protocol, protocol_sha256 = load_protocol(root, config_path)
    output_root = root / "outputs"
    output_root.mkdir(exist_ok=True)
    run = output_root / run_id
    run.mkdir(exist_ok=True)
    completed = run / "COMPLETED.json"
    if completed.is_file() and not completed.is_symlink():
        value = _load_json(completed)
        if sha256_file(run / "summary.md") != value["summary_sha256"]:
            raise ScreeningBlendExternalError("Completed external summary drifted")
        return run
    for cohort in ("drugage", "agextend"):
        score_cohort(root, protocol, protocol_sha256, run, cohort)
    summary_path = run / "summary.md"
    if summary_path.exists():
        raise ScreeningBlendExternalError("Unsealed summary already exists")
    summary_path.write_text(_summary(run), encoding="utf-8")
    atomic_write_json(
        completed,
        {
            "schema_version": "geroprotector.screening_blend_external.completed.v1",
            "status": "COMPLETE",
            "run_id": run_id,
            "protocol_sha256": protocol_sha256,
            "summary_sha256": sha256_file(summary_path),
            "drugage_scored_sha256": sha256_file(run / "drugage" / "scored" / "SCORED.json"),
            "agextend_scored_sha256": sha256_file(run / "agextend" / "scored" / "SCORED.json"),
            "model_or_threshold_selected_from_external_results": False,
            "scientific_role": "published_retrospective_post_hoc_stress_test",
        },
    )
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    destination = run_all(Path(args.root), Path(args.config).resolve(), args.run_id)
    print(f"Complete: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
