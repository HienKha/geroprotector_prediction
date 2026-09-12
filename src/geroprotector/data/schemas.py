"""JSON-schema loading and row validation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


def load_schema(root: str | Path, name: str) -> dict[str, Any]:
    path = Path(root) / "schemas" / name
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Schema must be a regular non-symlink file: {path}")
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return schema


def validate_records(records: Iterable[dict[str, Any]], schema: dict[str, Any]) -> None:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for index, record in enumerate(records):
        errors = sorted(validator.iter_errors(record), key=lambda item: list(item.path))
        if errors:
            summary = "; ".join(error.message for error in errors[:5])
            raise ValueError(f"Schema validation failed at record {index}: {summary}")
