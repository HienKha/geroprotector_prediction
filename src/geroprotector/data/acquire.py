"""Pinned upstream acquisition with no implicit network access elsewhere."""

from __future__ import annotations

import urllib.request
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from ..hashing import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_sha256,
    sha256_bytes,
    sha256_file,
)
from ..logging import utc_now

UPSTREAM_COMMIT = "c8f458925f5ea2beeba87c7c2dda62eefacf618c"
RAW_ROOT = (
    "https://raw.githubusercontent.com/BioAgeLab/Geroprotectors-Project-INGER/"
    f"{UPSTREAM_COMMIT}/"
)


@dataclass(frozen=True)
class SourceSpec:
    source_role: str
    relative_path: str
    cache_name: str
    sha256: str
    rows: int
    unique_smiles: int
    separator: str
    encoding: str
    name_column: str
    smiles_column: str
    supervised_label: int | None

    @property
    def url(self) -> str:
        return RAW_ROOT + self.relative_path.replace(" ", "%20")


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        source_role="prior_ml_unlabeled",
        relative_path="5.Chemical space/Geroprotectors by ML.csv",
        cache_name="geroprotectors_by_ml.csv",
        sha256="9bce3bdfc296bf777901232349f690ca5453a36f904235ae8fb0e7d5db58df4d",
        rows=1488,
        unique_smiles=1488,
        separator=",",
        encoding="latin-1",
        name_column="Name",
        smiles_column="smiles",
        supervised_label=None,
    ),
    SourceSpec(
        source_role="reported_positive",
        relative_path="0.Data/Geroprotectors_Clean_Descriptors_2024.csv",
        cache_name="geroprotectors_reported.tsv",
        sha256="41cc8451d80b9aec09574842e4c7007132d68cd613436460b2a784154f609e74",
        rows=206,
        unique_smiles=205,
        separator="\t",
        encoding="latin-1",
        name_column="Compound Name",
        smiles_column="Smiles",
        supervised_label=1,
    ),
    SourceSpec(
        source_role="weak_chembl_reference",
        relative_path="0.Data/No_geroprotectors_and_Toxicos.csv",
        cache_name="no_geroprotectors_and_toxicos.csv",
        sha256="396f78e7546d76529c808abb01d02da617da92c3d4aed83551901cc3ffd09c14",
        rows=199,
        unique_smiles=198,
        separator=",",
        encoding="latin-1",
        name_column="compound_name",
        smiles_column="canonical_smiles",
        supervised_label=0,
    ),
)


class AcquisitionError(RuntimeError):
    pass


def acquire_sources(
    output_dir: str | Path,
    *,
    allow_network: bool,
    source_config: Mapping[str, Mapping[str, Any]],
    expected_upstream_commit: str,
    local_sources: Mapping[str, str | Path] | None = None,
) -> dict:
    """Verify supplied/cache files or explicitly download the pinned byte streams."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if str(expected_upstream_commit) != UPSTREAM_COMMIT:
        raise AcquisitionError("Configured upstream commit differs from the code contract")
    expected_roles = {spec.source_role for spec in SOURCE_SPECS}
    if set(source_config) != expected_roles:
        raise AcquisitionError("Configured source roles differ from the code contract")
    records: list[dict] = []
    for spec in SOURCE_SPECS:
        configured = source_config[spec.source_role]
        expected_config = {
            "cache_name": spec.cache_name,
            "sha256": spec.sha256,
            "rows": spec.rows,
            "unique_smiles": spec.unique_smiles,
            "delimiter": "tab" if spec.separator == "\t" else "comma",
            "encoding": spec.encoding,
            "name_column": spec.name_column,
            "smiles_column": spec.smiles_column,
        }
        observed_config = {key: configured.get(key) for key in expected_config}
        if observed_config != expected_config:
            raise AcquisitionError(
                f"Configured source contract differs for {spec.source_role}: "
                f"{observed_config} != {expected_config}"
            )
        supplied = None if local_sources is None else local_sources.get(spec.source_role)
        target = destination / spec.cache_name
        if target.is_symlink():
            raise AcquisitionError(f"Pinned cache target cannot be a symlink: {target}")
        if supplied is not None:
            source = Path(supplied)
            if source.is_symlink() or not source.is_file():
                raise AcquisitionError(f"Local source must be regular/non-symlink: {source}")
            payload = source.read_bytes()
        elif target.is_file() and not target.is_symlink():
            payload = target.read_bytes()
        else:
            if not allow_network:
                raise AcquisitionError(
                    f"Missing {spec.source_role}; network is disabled. Expected {target}"
                )
            try:
                with urllib.request.urlopen(spec.url, timeout=60) as response:
                    payload = response.read()
            except Exception as exc:
                raise AcquisitionError(f"Could not download pinned source: {spec.url}") from exc
        actual = sha256_bytes(payload)
        if actual != spec.sha256:
            raise AcquisitionError(
                f"SHA256 mismatch for {spec.source_role}: {actual} != {spec.sha256}"
            )
        if not target.exists():
            atomic_write_bytes(target, payload)
        elif sha256_file(target) != spec.sha256:
            raise AcquisitionError(f"Existing cache changed: {target}")
        parsed = pd.read_csv(
            BytesIO(payload),
            sep=spec.separator,
            encoding=spec.encoding,
        )
        if len(parsed) != spec.rows:
            raise AcquisitionError(
                f"Observed rows changed for {spec.source_role}: {len(parsed)} != {spec.rows}"
            )
        required_columns = {spec.name_column, spec.smiles_column}
        if not required_columns.issubset(parsed.columns):
            raise AcquisitionError(
                f"Observed columns changed for {spec.source_role}: "
                f"missing {sorted(required_columns - set(parsed.columns))}"
            )
        unique_smiles = parsed[spec.smiles_column].astype(str).str.strip().nunique()
        if unique_smiles != spec.unique_smiles:
            raise AcquisitionError(
                f"Observed unique SMILES changed for {spec.source_role}: "
                f"{unique_smiles} != {spec.unique_smiles}"
            )
        records.append(
            {
                **asdict(spec),
                "url": spec.url,
                "cached_path": target.as_posix(),
                "verified_sha256": actual,
                "observed_rows": len(parsed),
                "observed_columns": list(map(str, parsed.columns)),
                "observed_dtypes": {
                    str(column): str(dtype) for column, dtype in parsed.dtypes.items()
                },
                "parser": {
                    "pandas": pd.__version__,
                    "separator": spec.separator,
                    "encoding": spec.encoding,
                },
            }
        )
    manifest = {
        "schema_version": "geroprotector.acquisition.v1",
        "upstream_commit": UPSTREAM_COMMIT,
        "created_utc": utc_now(),
        "sources": records,
    }
    manifest["canonical_sha256"] = canonical_sha256(manifest)
    manifest_path = destination / "acquisition_manifest.json"
    if manifest_path.exists():
        import json

        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise AcquisitionError("Existing acquisition manifest is not a regular file")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        claimed = existing.pop("canonical_sha256", None)
        if claimed != canonical_sha256(existing):
            raise AcquisitionError("Existing acquisition manifest integrity failed")
        existing["canonical_sha256"] = claimed
        comparable_existing = {
            key: value for key, value in existing.items() if key != "created_utc"
        }
        comparable_current = {
            key: value for key, value in manifest.items() if key != "created_utc"
        }
        # The canonical digest itself includes created_utc, so compare semantic payloads
        # without either self-hash after revalidating every source byte above.
        comparable_existing.pop("canonical_sha256", None)
        comparable_current.pop("canonical_sha256", None)
        if comparable_existing != comparable_current:
            raise AcquisitionError("Existing acquisition manifest differs from verified data")
        return existing
    atomic_write_json(manifest_path, manifest)
    return manifest
