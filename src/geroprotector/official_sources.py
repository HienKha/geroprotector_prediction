"""Small, dependency-free readers and guards for pinned official sources.

The insight-suite environments intentionally do not add ``openpyxl``.  These
helpers read the cell values needed from an XLSX archive while preserving the
source file byte-for-byte.  They are deliberately strict: a source with a
different digest or an ambiguous header is rejected before any experiment is
started.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from geroprotector.hashing import sha256_file


_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


class OfficialSourceError(RuntimeError):
    """Raised when an official source does not match its pinned contract."""


def require_pinned_file(path: str | Path, sha256: str, *, role: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise OfficialSourceError(f"{role} is not a regular pinned file: {candidate}")
    actual = sha256_file(candidate)
    if actual != str(sha256):
        raise OfficialSourceError(
            f"{role} SHA256 differs: expected {sha256}, observed {actual}")
    return candidate


def xlsx_rows(path: str | Path, sheet_name: str) -> list[dict[str, str]]:
    """Return one mapping of Excel column letter to text for every sheet row."""
    path = Path(path)
    with ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.get("Id"): item.get("Target") for item in relationships}
        relation_id = None
        for sheet in workbook.iter(f"{_NS}sheet"):
            if sheet.get("name") == sheet_name:
                relation_id = sheet.get(f"{_REL}id")
                break
        if relation_id is None:
            raise OfficialSourceError(f"XLSX sheet is absent: {sheet_name}")
        location = str(targets[relation_id])
        location = location.lstrip("/") if location.startswith("/") \
            else "xl/" + location.lstrip("/")

        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.text or "" for node in item.iter(f"{_NS}t"))
                      for item in root.iter(f"{_NS}si")]

        rows: list[dict[str, str]] = []
        for row in ET.fromstring(archive.read(location)).iter(f"{_NS}row"):
            cells: dict[str, str] = {}
            for cell in row.iter(f"{_NS}c"):
                match = re.match(r"([A-Z]+)", str(cell.get("r")))
                if match is None:
                    raise OfficialSourceError("Malformed XLSX cell address")
                column = match.group(1)
                value_node = cell.find(f"{_NS}v")
                value = "" if value_node is None else value_node.text or ""
                if cell.get("t") == "s" and value:
                    value = shared[int(value)]
                elif cell.get("t") == "inlineStr":
                    value = "".join(node.text or "" for node in cell.iter(f"{_NS}t"))
                cells[column] = str(value)
            rows.append(cells)
    return rows


def xlsx_table(path: str | Path, sheet_name: str, *,
               required_columns: tuple[str, ...]) -> pd.DataFrame:
    """Read a table whose header may follow title or explanatory rows.

    The first row containing every required column exactly once is the header.
    Requiring the full declared set avoids silently selecting a prose row.
    """
    rows = xlsx_rows(path, sheet_name)
    header_index = None
    column_by_name: dict[str, str] = {}
    for index, row in enumerate(rows):
        inverse = {str(value).strip(): column for column, value in row.items()}
        if set(required_columns).issubset(inverse):
            header_index = index
            column_by_name = {name: inverse[name] for name in required_columns}
            break
    if header_index is None:
        raise OfficialSourceError(
            f"Could not locate required header in {sheet_name}: {required_columns}")
    records = []
    for row in rows[header_index + 1:]:
        record = {name: row.get(column_by_name[name], "") for name in required_columns}
        if any(str(value).strip() for value in record.values()):
            records.append(record)
    return pd.DataFrame(records, columns=required_columns)
