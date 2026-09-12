#!/usr/bin/env python3
"""Verify a completion lock and every artifact hash before a run is skipped."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geroprotector.hashing import sha256_file


def verify(run: Path) -> None:
    run = run.resolve()
    completed_path = run / "COMPLETED.json"
    if not run.is_dir() or not completed_path.is_file():
        raise SystemExit(f"not a completed run directory: {run}")
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    recorded = completed.get("artifact_hashes")
    if not isinstance(recorded, dict) or not recorded:
        raise SystemExit(f"completion lock has no artifact_hashes mapping: {completed_path}")
    for relative, expected in sorted(recorded.items()):
        target = run / relative
        if not target.is_file() or target.is_symlink():
            raise SystemExit(f"recorded artifact is absent or not a regular file: {target}")
        actual = sha256_file(target)
        if actual != expected:
            raise SystemExit(f"artifact hash mismatch: {target}\nexpected {expected}\nactual   {actual}")
    expected_manifest = completed.get("run_manifest_sha256")
    if expected_manifest:
        manifest = run / "RUN_MANIFEST.json"
        if not manifest.is_file() or sha256_file(manifest) != expected_manifest:
            raise SystemExit(f"RUN_MANIFEST hash mismatch: {manifest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    verify(args.run_dir)
    print(f"verified completed run: {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
