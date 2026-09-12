#!/usr/bin/env python3
"""Deterministically regenerate or verify the source-package inventory/checksums."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

EXCLUDED_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "__pycache__",
    "artifacts",
    "checkpoints",
    "external_data",
    "external_sources",
    "licenses",
    "outputs",
    "resource_telemetry",
    "third_party",
    "tools",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
MANIFEST_NAME = "MANIFEST_SHA256.txt"
CONTENTS_NAME = "CONTENTS.txt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(root: Path, *, include_contents: bool) -> tuple[Path, ...]:
    selected: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRECTORIES for part in relative.parts):
            continue
        if path.is_symlink():
            raise RuntimeError(f"Package source cannot contain symlinks: {relative}")
        if not path.is_file() or path.suffix in EXCLUDED_SUFFIXES:
            continue
        if relative.as_posix() == MANIFEST_NAME:
            continue
        if not include_contents and relative.as_posix() == CONTENTS_NAME:
            continue
        selected.append(path)
    return tuple(sorted(selected, key=lambda item: item.relative_to(root).as_posix()))


def _contents_bytes(root: Path) -> bytes:
    rows = [
        "Geroprotector prediction implementation package contents",
        "",
        "Generated deterministically; excludes caches, runtime checkpoints/licenses, "
        "artifacts, outputs and this file.",
        "",
    ]
    rows.extend(
        f"{path.relative_to(root).as_posix()}\t{path.stat().st_size} bytes"
        for path in _source_files(root, include_contents=False)
    )
    return ("\n".join(rows) + "\n").encode("utf-8")


def _manifest_bytes(root: Path) -> bytes:
    rows = [
        f"{_sha256(path)}  ./{path.relative_to(root).as_posix()}"
        for path in _source_files(root, include_contents=True)
    ]
    return ("\n".join(rows) + "\n").encode("utf-8")


def _atomic_replace(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    contents = root / CONTENTS_NAME
    manifest = root / MANIFEST_NAME
    if args.check:
        if contents.read_bytes() != _contents_bytes(root):
            raise SystemExit("CONTENTS.txt is stale")
        if manifest.read_bytes() != _manifest_bytes(root):
            raise SystemExit("MANIFEST_SHA256.txt is stale")
        return 0
    _atomic_replace(contents, _contents_bytes(root))
    _atomic_replace(manifest, _manifest_bytes(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
