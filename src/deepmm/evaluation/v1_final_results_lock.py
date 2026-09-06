"""Immutable provenance and integrity lock for the completed V1 Q1--Q3 campaign."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from deepmm.calibration.v1_calibration_lock import V1_CALIBRATION_EVIDENCE_SHA256
from deepmm.evaluation.v1_final_config import (
    V1_FINAL_POLICY_SHA256,
    V1_FINAL_TRIAL_MANIFEST_SHA256,
)
from deepmm.training.v1_selection_lock import (
    V1_DATASET_MANIFEST_SHA256,
    V1_SELECTION_LOCK_SHA256,
)
from deepmm.validation import run_manifest_hash, validate_run_manifest


V1_FINAL_RUN_ID = 34021350579
V1_FINAL_RUN_ATTEMPT = 2
V1_FINAL_HEAD_SHA = "480c1f4e67757e4789b270b5ea12ecd0e9eac16b"
V1_FINAL_ARTIFACT_ID = 9985665528
V1_FINAL_ARTIFACT_ZIP_SHA256 = (
    "743ae2d30d7cf5747dce5b3cad66e789d8cc3c9923fe91f35510d0bd473180c5"
)
V1_FINAL_ARTIFACT_FILES = {
    "v1_final_results.json": "092399fb06857be57d55bcceda97ccf25b9821f9e1158d370573f7c9cfb4984e",
    "v1_final_run_manifests.json": "3db9b9a830619004dd80c3b6606490442fbb8afd3affa48a2574ab095fc652f7",
    "v1_final_scores.npz": "b77fce5f86e1ea82978ec7f78c72534348078b9c8da434254a1226c477247d0a",
    "v1_final_trials.json": "d1ac5c1f778dbb8bce1123c7911ee905b13c4ec2f395483c4950966d5178767f",
}
V1_FINAL_TRIAL_GZIP_SHA256 = (
    "1583cf6222ff898aa17721c16acb9d1ba9f54231a09064af019bfc02cad81fa1"
)
V1_FINAL_COMMITTED_PARTS = {
    "v1_final_scores.npz": (
        "acbbe50aef934f8bc4eb9f04fb991df9107e798d4c739a27f136b5a37b501a40",
        "22438eec10f74c3af97356d56b76d8f84b26b7d02296a0c9fdceca9f6f9e91c1",
        "3f6665af602f7c977f9d84fbe6def7a3865fd46bb538daf665431db56e011e9f",
        "cc2b67921df94e23073e486b4c8d8dc7f81445704c9ccf44266dfaec5f7465d3",
        "5057648bc87d006e8bc33964807a522ac2791a7f7cf732b1696d3d36ae2a9e28",
        "378dffd7260fe4b2f3bfb86a483e8ee38a5e95494c23df200c413c69429a7239",
        "b73aa09020d165d01955b84d6d5532ddeda2233df14593c601eddcd4b307cb88",
        "f0887c4fb1653978504f258365817dbbac8e616451751c9de589c1154bb35647",
        "43d34edda55c98964805378699f5332d614ad47e5047e5a42158e431678fe2da",
        "bfa680ddddc453b1e76016f22bfa491f3755fb902dbce8a80bf41311b7817e68",
        "7d734d95dfd37c8b286c28657552d99fc21352ef8b270e204d9ea0533285cdb1",
        "9197adc928880aa44457c09723302face1eac08fce0b72622d1e873685ece772",
        "af38642d42254df2f8a0566c6bb8f18089aaa935bb69ba8d01b492d373619e8a",
        "93e948e133c092f5147d6c891d0312f85bd6dc94c3ea6fffed63da0e31cc4fd9",
        "855710a4985d7f066a9173227c215f809a70722b93816a129ab53fae39be2125",
        "a5629aeb9b3073a619a0d2cdf8a04d51ac7015673e78906e2471069deca00237",
        "cfce04d3057a1ffd36c68e7bdfac946647704306e58efd3b1401bcd6b29ee09e",
        "ef1d8543c8409c7dac836acff102ada10298f006c1e519441d4e74eb8feef8f1",
        "b8fd9879e205a7ffc902d6913b56c5825032ea47b64d6a412e6f7c493142942b",
    ),
    "v1_final_trials.json.gz": (
        "ee70ffb64cf6b24a37078469b3a2261c88f17a559b8b47179227878cbc2ffe74",
        "5a844abe4e777ac671ae498955e0d6ff4c4dd0a9f8c411e307fba47649f3f696",
    ),
}
V1_FINAL_RESULTS_PATH = Path("artifacts/locked/v1_final")


def v1_final_results_lock_payload() -> dict[str, Any]:
    return {
        "run_id": V1_FINAL_RUN_ID,
        "run_attempt": V1_FINAL_RUN_ATTEMPT,
        "head_sha": V1_FINAL_HEAD_SHA,
        "artifact_id": V1_FINAL_ARTIFACT_ID,
        "artifact_zip_sha256": V1_FINAL_ARTIFACT_ZIP_SHA256,
        "artifact_files": dict(V1_FINAL_ARTIFACT_FILES),
        "committed_trial_gzip_sha256": V1_FINAL_TRIAL_GZIP_SHA256,
        "committed_parts": {
            name: list(hashes) for name, hashes in V1_FINAL_COMMITTED_PARTS.items()
        },
        "dataset_manifest_sha256": V1_DATASET_MANIFEST_SHA256,
        "selection_lock_sha256": V1_SELECTION_LOCK_SHA256,
        "calibration_evidence_sha256": V1_CALIBRATION_EVIDENCE_SHA256,
        "final_policy_sha256": V1_FINAL_POLICY_SHA256,
        "final_trial_manifest_sha256": V1_FINAL_TRIAL_MANIFEST_SHA256,
    }


def v1_final_results_lock_hash() -> str:
    canonical = json.dumps(
        v1_final_results_lock_payload(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


V1_FINAL_RESULTS_LOCK_SHA256 = "fab7227ad87dd8da0f1f9065ab377052e18b5f429fc8a9c2d1dcc79ccd036468"


def _root(path: Path | None) -> Path:
    if path is not None:
        return Path(path)
    return Path(__file__).resolve().parents[3] / V1_FINAL_RESULTS_PATH


def read_v1_final_file(name: str, path: Path | None = None) -> bytes:
    """Recompose and verify one transport-split binary from committed parts."""
    root = _root(path)
    expected_parts = V1_FINAL_COMMITTED_PARTS[name]
    chunks = []
    for index, expected_hash in enumerate(expected_parts):
        part = root / f"{name}.part-{index:03d}"
        data = part.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise RuntimeError(f"V1 final evidence part changed: {part.name}")
        chunks.append(data)
    combined = b"".join(chunks)
    expected = (
        V1_FINAL_TRIAL_GZIP_SHA256
        if name == "v1_final_trials.json.gz"
        else V1_FINAL_ARTIFACT_FILES[name]
    )
    if hashlib.sha256(combined).hexdigest() != expected:
        raise RuntimeError(f"recomposed V1 final evidence changed: {name}")
    return combined


def load_v1_final_results_lock(path: Path | None = None) -> dict[str, Any]:
    """Verify committed evidence and return the locked result summary."""
    if v1_final_results_lock_hash() != V1_FINAL_RESULTS_LOCK_SHA256:
        raise RuntimeError("V1 final-results lock constants changed")
    root = _root(path)
    for name, expected in V1_FINAL_ARTIFACT_FILES.items():
        if name in {"v1_final_trials.json", "v1_final_scores.npz"}:
            continue
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"V1 final artifact {name} changed: expected {expected}, got {actual}")

    compressed = read_v1_final_file("v1_final_trials.json.gz", root)
    trial_bytes = gzip.decompress(compressed)
    if hashlib.sha256(trial_bytes).hexdigest() != V1_FINAL_ARTIFACT_FILES["v1_final_trials.json"]:
        raise RuntimeError("decompressed final trials differ from the workflow artifact")

    result = json.loads((root / "v1_final_results.json").read_text(encoding="utf-8"))
    expected_fields = {
        "status": "complete",
        "final_images_read": True,
        "code_commit": V1_FINAL_HEAD_SHA,
        "dataset_manifest_sha256": V1_DATASET_MANIFEST_SHA256,
        "selection_lock_sha256": V1_SELECTION_LOCK_SHA256,
        "calibration_evidence_sha256": V1_CALIBRATION_EVIDENCE_SHA256,
        "final_policy_sha256": V1_FINAL_POLICY_SHA256,
        "final_clean_trial_manifest_sha256": V1_FINAL_TRIAL_MANIFEST_SHA256,
    }
    failed = [key for key, value in expected_fields.items() if result.get(key) != value]
    if failed:
        raise RuntimeError(f"V1 final result summary failed fields: {', '.join(failed)}")
    if len(result.get("conditions", {})) != 15:
        raise RuntimeError("V1 final result summary must contain exactly 15 conditions")
    clean = result["conditions"]["clean"]
    if (clean["n_trials"], clean["n_genuine"], clean["n_impostor"]) != (4000, 200, 3800):
        raise RuntimeError("V1 final clean trial counts changed")

    manifests = json.loads((root / "v1_final_run_manifests.json").read_text(encoding="utf-8"))
    if len(manifests) != 148:
        raise RuntimeError("V1 final evidence must contain 148 available system-condition manifests")
    pairs: set[tuple[str, str]] = set()
    for row in manifests:
        stored = row.get("run_manifest_hash")
        core = {key: value for key, value in row.items() if key != "run_manifest_hash"}
        validated = validate_run_manifest(core)
        if stored != run_manifest_hash(validated):
            raise RuntimeError(f"invalid run-manifest hash for {row.get('run_id')}")
        pairs.add((validated["condition_id"], validated["method_id"]))
    if len(pairs) != 148:
        raise RuntimeError("duplicate V1 final system-condition manifest")

    score_bytes = read_v1_final_file("v1_final_scores.npz", root)
    with np.load(io.BytesIO(score_bytes), allow_pickle=False) as arrays:
        if len(arrays.files) != 444:
            raise RuntimeError("V1 final score bundle must contain 444 arrays")
        for name in arrays.files:
            values = arrays[name]
            if values.shape != (4000,) or not np.all(np.isfinite(values)):
                raise RuntimeError(f"invalid score array {name}")
    return result


def assert_v1_final_results_lock(path: Path | None = None) -> None:
    load_v1_final_results_lock(path)
