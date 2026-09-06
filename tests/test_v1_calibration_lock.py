import json
from pathlib import Path

import pytest

from deepmm.calibration.v1_calibration_lock import (
    V1_CALIBRATION_EVIDENCE_SHA256,
    V1_COMMITTED_CALIBRATION_JSON_SHA256,
    assert_v1_calibration_lock,
    load_v1_calibration_lock,
)


def test_committed_v1_calibration_is_complete_and_locked():
    payload = load_v1_calibration_lock()
    assert payload["calibration_evidence_sha256"] == V1_CALIBRATION_EVIDENCE_SHA256
    assert len(payload["conditions"]) == 15
    assert all(
        row["status"] == "fitted"
        for row in payload["conditions"]["clean"]["calibrators"].values()
    )


def test_v1_calibration_lock_rejects_modified_file(tmp_path: Path):
    source = Path("artifacts/locked/v1_calibration.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["calibration_C"] = 2.0
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact changed"):
        assert_v1_calibration_lock(changed)


def test_calibration_file_digest_is_the_published_digest():
    import hashlib

    raw = Path("artifacts/locked/v1_calibration.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == V1_COMMITTED_CALIBRATION_JSON_SHA256
