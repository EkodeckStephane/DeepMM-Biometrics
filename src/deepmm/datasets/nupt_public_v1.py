"""Deterministic V1 protocol for the official public NUPT-FPV subset.

V1 uses the public directory identifier as a *biometric instance identity*. It does
not infer that the 20 public identifiers are 20 independent human volunteers.
Every multimodal evidence unit is one (instance, session, capture) tuple with both
fingerprint and finger-vein samples.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from itertools import product
from typing import Any

from deepmm.validation import trial_manifest_hash, validate_dataset_records, validate_trial_records


V1_IDENTITY_SCOPE = "public_biometric_instance"
V1_ROLE_CAPTURES = {
    "fit": {"session": "1", "enroll": ("01", "02"), "probe": ("03", "04", "05")},
    "selection": {"session": "1", "enroll": ("06",), "probe": ("07",)},
    "calibration": {"session": "1", "enroll": ("08",), "probe": ("09",)},
    "final": {"enroll_session": "1", "enroll": ("10",), "probe_session": "2", "probe": tuple(f"{i:02d}" for i in range(1, 11))},
}
_REQUIRED_MODALITIES = ("fingerprint", "finger_vein")


def _evidence_id(instance_id: str, session_id: str, capture_id: str) -> str:
    return f"nupt-v1-{session_id}-{instance_id}-{capture_id}"


def build_v1_evidence_units(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Build complete two-modality evidence units from a NUPT public manifest."""
    rows = validate_dataset_records(records)
    grouped: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        modality = row["modality"]
        if modality not in _REQUIRED_MODALITIES:
            continue
        key = (row["instance_id"], row["session_id"], row["capture_id"])
        if modality in grouped[key]:
            raise ValueError(f"duplicate {modality} sample for evidence unit {key}")
        grouped[key][modality] = row

    if not grouped:
        raise ValueError("no NUPT V1 evidence units found")

    units: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, by_modality in sorted(grouped.items()):
        missing = [m for m in _REQUIRED_MODALITIES if m not in by_modality]
        if missing:
            raise ValueError(f"evidence unit {key} is missing modalities: {missing}")
        instance_id, session_id, capture_id = key
        units[key] = {
            "evidence_id": _evidence_id(instance_id, session_id, capture_id),
            "instance_id": instance_id,
            "session_id": session_id,
            "capture_id": capture_id,
            "fingerprint_sample_id": by_modality["fingerprint"]["sample_id"],
            "finger_vein_sample_id": by_modality["finger_vein"]["sample_id"],
            "fingerprint_path": by_modality["fingerprint"]["relative_path"],
            "finger_vein_path": by_modality["finger_vein"]["relative_path"],
        }
    return units


def _required_unit(
    units: Mapping[tuple[str, str, str], Mapping[str, Any]],
    instance_id: str,
    session_id: str,
    capture_id: str,
) -> Mapping[str, Any]:
    key = (instance_id, session_id, capture_id)
    try:
        return units[key]
    except KeyError as exc:
        raise ValueError(f"missing required V1 evidence unit {key}") from exc


def generate_v1_trials(
    records: Iterable[Mapping[str, Any]],
    role: str,
    *,
    condition_id: str = "clean",
) -> list[dict[str, Any]]:
    """Generate the frozen all-vs-all verification trials for one V1 role.

    Roles are sample-disjoint by construction. `final` uses session-1 capture 10 as
    enrollment and all ten session-2 captures as probes. Other roles use disjoint
    capture sets from session 1. All 20 public instance IDs remain in every role;
    consequently V1 measures enrolled-instance verification, not unseen-person
    generalization.
    """
    role = str(role).strip().lower()
    if role not in V1_ROLE_CAPTURES:
        raise ValueError(f"unknown V1 role {role!r}")
    condition_id = str(condition_id).strip()
    if not condition_id:
        raise ValueError("condition_id must be non-empty")

    units = build_v1_evidence_units(records)
    instances = sorted({key[0] for key in units})
    if len(instances) < 2:
        raise ValueError("V1 verification requires at least two public instance IDs")

    cfg = V1_ROLE_CAPTURES[role]
    if role == "final":
        enroll_session = str(cfg["enroll_session"])
        probe_session = str(cfg["probe_session"])
    else:
        enroll_session = probe_session = str(cfg["session"])
    enroll_captures = tuple(cfg["enroll"])
    probe_captures = tuple(cfg["probe"])

    trials: list[dict[str, Any]] = []
    for enroll_instance in instances:
        for probe_instance in instances:
            label = int(enroll_instance == probe_instance)
            for enroll_capture, probe_capture in product(enroll_captures, probe_captures):
                enrollment = _required_unit(units, enroll_instance, enroll_session, enroll_capture)
                probe = _required_unit(units, probe_instance, probe_session, probe_capture)
                trial_id = (
                    f"nupt-v1-{role}-{condition_id}-"
                    f"e{enroll_instance}s{enroll_session}c{enroll_capture}-"
                    f"p{probe_instance}s{probe_session}c{probe_capture}"
                )
                trials.append(
                    {
                        "trial_id": trial_id,
                        "label": label,
                        "anchor_subject_id": enroll_instance,
                        "enrollment_subject_id": enroll_instance,
                        "probe_subject_id": probe_instance,
                        "enrollment_sample_id": enrollment["evidence_id"],
                        "probe_sample_id": probe["evidence_id"],
                        "condition_id": condition_id,
                        "identity_scope": V1_IDENTITY_SCOPE,
                        "role": role,
                        "enrollment_session_id": enroll_session,
                        "probe_session_id": probe_session,
                        "enrollment_capture_id": enroll_capture,
                        "probe_capture_id": probe_capture,
                        "enrollment_fingerprint_sample_id": enrollment["fingerprint_sample_id"],
                        "enrollment_finger_vein_sample_id": enrollment["finger_vein_sample_id"],
                        "probe_fingerprint_sample_id": probe["fingerprint_sample_id"],
                        "probe_finger_vein_sample_id": probe["finger_vein_sample_id"],
                    }
                )
    return validate_trial_records(trials)


def v1_trial_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return counts and frozen hashes for all V1 clean trial roles."""
    rows = list(records)
    result: dict[str, Any] = {"identity_scope": V1_IDENTITY_SCOPE, "roles": {}}
    for role in ("fit", "selection", "calibration", "final"):
        trials = generate_v1_trials(rows, role)
        result["roles"][role] = {
            "n_trials": len(trials),
            "n_genuine": sum(t["label"] == 1 for t in trials),
            "n_impostor": sum(t["label"] == 0 for t in trials),
            "trial_manifest_hash": trial_manifest_hash(trials),
        }
    return result
