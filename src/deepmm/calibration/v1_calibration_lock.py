"""Immutable provenance lock for the held-out V1 calibration evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from deepmm.evaluation.v1_final_config import (
    V1_CALIBRATION_C,
    V1_FINAL_POLICY_SHA256,
    V1_STRESS_PLAN_SHA256,
)
from deepmm.training.v1_selection_lock import (
    V1_DATASET_MANIFEST_SHA256,
    V1_QUALITY_MODEL_HASH,
    V1_SELECTED_MODELS,
)


V1_CALIBRATION_RUN_ID = 34020134495
V1_CALIBRATION_HEAD_SHA = "f029736f46b5c100e8c222871b474d19b703bb53"
V1_CALIBRATION_ARTIFACT_ID = 9985217324
V1_CALIBRATION_ARTIFACT_ZIP_SHA256 = (
    "f44ac02e6501e79056995a4f96f0b981cb14309c78ade0e6616b83120823ba67"
)
V1_COMMITTED_CALIBRATION_JSON_SHA256 = (
    "7418827a89c17afb8bda2c5e24408e71585399f9f2aaeddcf7a799293d97e795"
)
V1_CALIBRATION_EVIDENCE_SHA256 = (
    "047a4a786e5d88364d44227c03a4ae471ba6ff0481e011a72a7dda3295c56eaf"
)
V1_CALIBRATION_TRIAL_SHA256 = (
    "f6104160a138bccebc1f4b03fd5012be1712027d41c44ad5f916a8c34639ca88"
)
V1_CALIBRATION_PATH = Path("artifacts/locked/v1_calibration.json")
V1_SYSTEM_IDS = ("U-FP", "U-FV", "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3S")


def _default_path() -> Path:
    return Path(__file__).resolve().parents[3] / V1_CALIBRATION_PATH


def load_v1_calibration_lock(path: Path | None = None) -> dict[str, Any]:
    """Load and fully verify the exact calibration artifact used by final scoring."""
    source = _default_path() if path is None else Path(path)
    raw = source.read_bytes()
    actual_file_hash = hashlib.sha256(raw).hexdigest()
    if actual_file_hash != V1_COMMITTED_CALIBRATION_JSON_SHA256:
        raise RuntimeError(
            "V1 calibration artifact changed: "
            f"expected {V1_COMMITTED_CALIBRATION_JSON_SHA256}, got {actual_file_hash}"
        )
    payload = json.loads(raw)
    evidence_hash = payload.pop("calibration_evidence_sha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    actual_evidence_hash = hashlib.sha256(canonical).hexdigest()
    payload["calibration_evidence_sha256"] = evidence_hash

    expected_checkpoints = {
        family: V1_SELECTED_MODELS[family]["expected_checkpoint_hash"]
        for family in ("D1", "D2", "D3S")
    }
    checks = {
        "scope": payload.get("scope") == "V1 held-out calibration only; final role untouched",
        "final firewall": payload.get("final_images_read") is False,
        "checkpoint firewall": payload.get("development_checkpoints_verified_before_calibration") is True,
        "dataset": payload.get("dataset_manifest_sha256") == V1_DATASET_MANIFEST_SHA256,
        "calibration trials": payload.get("calibration_trial_manifest_sha256") == V1_CALIBRATION_TRIAL_SHA256,
        "final policy": payload.get("training_final_policy_sha256") == V1_FINAL_POLICY_SHA256,
        "stress plan": payload.get("stress_plan_sha256") == V1_STRESS_PLAN_SHA256,
        "quality model": payload.get("quality_model_hash") == V1_QUALITY_MODEL_HASH,
        "checkpoints": payload.get("selected_checkpoint_hashes") == expected_checkpoints,
        "calibration C": payload.get("calibration_C") == V1_CALIBRATION_C,
        "evidence hash": evidence_hash == V1_CALIBRATION_EVIDENCE_SHA256 == actual_evidence_hash,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"V1 calibration lock failed: {', '.join(failed)}")

    conditions = payload.get("conditions", {})
    if len(conditions) != 15 or "clean" not in conditions:
        raise RuntimeError("V1 calibration lock must contain exactly the 15 frozen conditions")
    for condition_id, record in conditions.items():
        calibrators = record.get("calibrators", {})
        if set(calibrators) != set(V1_SYSTEM_IDS):
            raise RuntimeError(f"{condition_id}: calibration system order/set changed")
        for system_id, calibrator in calibrators.items():
            status = calibrator.get("status")
            expected_unavailable = (
                (condition_id == "missing-fingerprint" and system_id == "U-FP")
                or (condition_id == "missing-finger-vein" and system_id == "U-FV")
            )
            if expected_unavailable:
                if status != "unavailable":
                    raise RuntimeError(f"{condition_id}/{system_id}: expected unavailable")
            elif status != "fitted" or float(calibrator.get("slope", 0.0)) <= 0.0:
                raise RuntimeError(f"{condition_id}/{system_id}: calibrator is not valid")
    return payload


def assert_v1_calibration_lock(path: Path | None = None) -> None:
    load_v1_calibration_lock(path)
