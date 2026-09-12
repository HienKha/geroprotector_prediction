"""Stable row and identity identifiers."""

from __future__ import annotations

import hashlib
import json


def raw_row_id(source_role: str, row_index: int, name: str, smiles: str) -> str:
    payload = json.dumps(
        {
            "source_role": source_role,
            "row_index": int(row_index),
            "name": str(name),
            "smiles": str(smiles),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"raw:{source_role}:{row_index:06d}:{digest[:16]}"


def compound_id(connectivity_inchikey: str) -> str:
    return f"cmp::{str(connectivity_inchikey).strip()}"
