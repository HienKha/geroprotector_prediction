from pathlib import Path

import json
import pytest

from geroprotector.manuscript_evidence_handoff import HandoffError, _verify_run
from geroprotector.hashing import sha256_file


def test_handoff_verifier_rejects_changed_artifact(tmp_path: Path) -> None:
    run = tmp_path / "sealed"
    run.mkdir()
    artifact = run / "result.txt"
    artifact.write_text("original", encoding="utf-8")
    manifest = run / "RUN_MANIFEST.json"
    manifest.write_text('{}\n', encoding="utf-8")
    completed = {
        "run_id": "sealed",
        "status": "COMPLETE",
        "run_manifest_sha256": sha256_file(manifest),
        "artifact_hashes": {"result.txt": sha256_file(artifact)},
    }
    (run / "COMPLETED.json").write_text(json.dumps(completed), encoding="utf-8")
    assert _verify_run(run, ["COMPLETE"])["artifact_count"] == 1
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(HandoffError, match="changed"):
        _verify_run(run, ["COMPLETE"])


def test_handoff_source_has_no_stale_integrated_report_pointer() -> None:
    source = Path(__file__).parents[1] / "src/geroprotector/manuscript_evidence_handoff.py"
    text = source.read_text(encoding="utf-8")
    assert 'modelwide_druglikeness_bias_20260826; insight_suite_report_20260826"' not in text
    assert 'f"modelwide_druglikeness_bias_20260826; {integrated.name}"' in text
