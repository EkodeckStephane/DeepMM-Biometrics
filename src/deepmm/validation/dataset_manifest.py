"""Dataset-manifest validation for real multimodal archive audits.

The manifest deliberately models the *human* subject separately from a nested
biometric instance (for example, a finger). This prevents a multi-finger database
from silently inflating the biological sample size or leaking one person's
instances across outer train/development/calibration/test partitions.

No raw biometric pixels are handled here. A validated manifest contains only
identity/session/capture metadata and dataset-local file references/hashes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import PurePosixPath
import re
from typing import Any

from .hashing import hash_ordered_records


REQUIRED_DATASET_FIELDS = (
    "sample_id",
    "person_id",
    "instance_id",
    "modality",
    "session_id",
    "capture_id",
    "relative_path",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _text(record: Mapping[str, Any], field: str) -> str:
    if field not in record:
        raise ValueError(f"missing required dataset field {field!r}")
    value = str(record[field]).strip()
    if not value:
        raise ValueError(f"dataset field {field!r} must not be empty")
    return value


def _relative_dataset_path(value: str) -> str:
    raw = value.replace("\\", "/").strip()
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise ValueError("relative_path must be a normalized dataset-local relative path")
    normalized = str(path)
    if normalized in {".", ""}:
        raise ValueError("relative_path must identify a file")
    return normalized


def validate_dataset_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalize an ordered dataset archive manifest.

    Optional fields:
    - ``file_sha256``: lowercase SHA-256 of the raw file when local policy permits
      hashing it;
    - ``file_size_bytes``: non-negative integer;
    - arbitrary non-null metadata fields, retained for provenance.

    ``instance_id`` is nested inside ``person_id``. For NUPT-FPV it can represent
    a finger designation such as ``left_index``; the same designation may therefore
    occur for many people.
    """
    rows: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    paths: set[str] = set()

    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"dataset record {index} must be a mapping")
        row = dict(record)
        normalized = {
            "sample_id": _text(row, "sample_id"),
            "person_id": _text(row, "person_id"),
            "instance_id": _text(row, "instance_id"),
            "modality": _text(row, "modality").lower(),
            "session_id": _text(row, "session_id"),
            "capture_id": _text(row, "capture_id"),
            "relative_path": _relative_dataset_path(_text(row, "relative_path")),
        }

        if normalized["sample_id"] in sample_ids:
            raise ValueError(f"duplicate sample_id {normalized['sample_id']!r}")
        if normalized["relative_path"] in paths:
            raise ValueError(f"duplicate relative_path {normalized['relative_path']!r}")
        sample_ids.add(normalized["sample_id"])
        paths.add(normalized["relative_path"])

        if "file_sha256" in row and row["file_sha256"] is not None:
            digest = str(row["file_sha256"]).strip().lower()
            if not _SHA256_RE.fullmatch(digest):
                raise ValueError("file_sha256 must be a 64-character lowercase SHA-256 digest")
            normalized["file_sha256"] = digest

        if "file_size_bytes" in row and row["file_size_bytes"] is not None:
            size = row["file_size_bytes"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError("file_size_bytes must be a non-negative integer")
            normalized["file_size_bytes"] = int(size)

        for key, value in row.items():
            if key not in normalized and key not in {"file_sha256", "file_size_bytes"}:
                if value is None:
                    raise ValueError(f"optional dataset field {key!r} must not be null when present")
                normalized[key] = value

        rows.append(normalized)

    if not rows:
        raise ValueError("dataset manifest must not be empty")
    return rows


def dataset_manifest_hash(records: Iterable[Mapping[str, Any]]) -> str:
    """Return an order-independent archive-manifest SHA-256.

    File-system traversal order is not scientific evidence, so validated rows are
    sorted by ``sample_id`` before hashing. Any metadata/path/hash change still
    changes the digest.
    """
    rows = validate_dataset_records(records)
    rows.sort(key=lambda row: row["sample_id"])
    return hash_ordered_records(rows)


def assert_person_partition_disjointness(
    records: Iterable[Mapping[str, Any]],
    sample_to_partition: Mapping[str, str],
) -> dict[str, tuple[str, ...]]:
    """Ensure every human person appears in exactly one outer partition.

    The mapping is sample-level on purpose: this function detects accidental
    placement of different fingers/captures from one person into different splits.
    """
    rows = validate_dataset_records(records)
    partitions_by_person: dict[str, set[str]] = defaultdict(set)
    missing: list[str] = []
    for row in rows:
        sample_id = row["sample_id"]
        partition = sample_to_partition.get(sample_id)
        if partition is None or not str(partition).strip():
            missing.append(sample_id)
            continue
        partitions_by_person[row["person_id"]].add(str(partition).strip())

    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"partition mapping is missing {len(missing)} samples; first: {preview}")

    leaking = {
        person: sorted(parts)
        for person, parts in partitions_by_person.items()
        if len(parts) != 1
    }
    if leaking:
        first_person = sorted(leaking)[0]
        raise ValueError(
            f"person-level partition leakage for {first_person!r}: {leaking[first_person]}"
        )
    return {person: tuple(sorted(parts)) for person, parts in sorted(partitions_by_person.items())}


def audit_multimodal_topology(
    records: Iterable[Mapping[str, Any]],
    *,
    required_modalities: Sequence[str],
    min_samples_per_modality: int = 1,
    min_sessions_per_modality: int = 1,
    require_capture_alignment: bool = False,
) -> dict[str, Any]:
    """Audit completeness of person/instance-level multimodal evidence.

    ``require_capture_alignment`` is intentionally optional. Some multimodal
    datasets collect modalities in the same visit without one-to-one simultaneous
    capture indices; imposing alignment without documentation would discard valid
    data. When enabled, every required modality must expose the same set of
    ``(session_id, capture_id)`` keys for each person/instance.
    """
    rows = validate_dataset_records(records)
    modalities = tuple(str(value).strip().lower() for value in required_modalities)
    if len(modalities) < 2 or any(not value for value in modalities):
        raise ValueError("required_modalities must contain at least two non-empty modalities")
    if len(set(modalities)) != len(modalities):
        raise ValueError("required_modalities must be unique")
    if min_samples_per_modality < 1 or min_sessions_per_modality < 1:
        raise ValueError("minimum sample/session requirements must be >= 1")

    allowed = set(modalities)
    by_identity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["modality"] in allowed:
            by_identity[(row["person_id"], row["instance_id"])].append(row)
    if not by_identity:
        raise ValueError("no manifest records match required_modalities")

    incomplete: list[dict[str, Any]] = []
    aligned_count = 0
    per_modality_total = Counter()
    persons: set[str] = set()

    for (person_id, instance_id), identity_rows in sorted(by_identity.items()):
        persons.add(person_id)
        rows_by_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in identity_rows:
            rows_by_modality[row["modality"]].append(row)
            per_modality_total[row["modality"]] += 1

        reasons: list[str] = []
        capture_keys: dict[str, set[tuple[str, str]]] = {}
        for modality in modalities:
            current = rows_by_modality.get(modality, [])
            if len(current) < min_samples_per_modality:
                reasons.append(
                    f"{modality}: samples={len(current)} < {min_samples_per_modality}"
                )
            sessions = {row["session_id"] for row in current}
            if len(sessions) < min_sessions_per_modality:
                reasons.append(
                    f"{modality}: sessions={len(sessions)} < {min_sessions_per_modality}"
                )
            capture_keys[modality] = {
                (row["session_id"], row["capture_id"]) for row in current
            }

        if require_capture_alignment:
            first_keys = capture_keys[modalities[0]]
            if any(capture_keys[modality] != first_keys for modality in modalities[1:]):
                reasons.append("capture keys are not aligned across required modalities")

        if reasons:
            incomplete.append(
                {
                    "person_id": person_id,
                    "instance_id": instance_id,
                    "reasons": tuple(reasons),
                }
            )
        else:
            aligned_count += 1

    return {
        "n_records": len(rows),
        "n_people": len(persons),
        "n_instances": len(by_identity),
        "n_complete_instances": aligned_count,
        "n_incomplete_instances": len(incomplete),
        "required_modalities": modalities,
        "samples_by_modality": dict(sorted(per_modality_total.items())),
        "incomplete_instances": tuple(incomplete),
        "complete": len(incomplete) == 0,
    }
