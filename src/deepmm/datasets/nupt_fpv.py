"""NUPT-FPV archive adapter.

The public NUPT-FPV repository exposes a small fingerprint/finger-vein subset for
technical inspection. Its folder identifiers are biometric-instance identifiers;
the public README does not establish how those public instance IDs map to the 140
human volunteers in the complete database. The scanner therefore refuses to invent
that biological mapping.

A real scientific archive may provide a separately verified ``instance_to_person``
mapping. Only then are records marked as person-resolved and eligible for outer
person-disjoint splitting.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

from deepmm.validation import validate_dataset_records


NUPT_PUBLIC_UNRESOLVED_PERSON = "UNRESOLVED_PUBLIC_NUPT_PERSON"

_MODALITY_DIRS = {
    "Fingerprint": "fingerprint",
    "FingerVein": "finger_vein",
}
_SESSION_RE = re.compile(r"^Session(?P<session>\d+)$")
_FILE_RE = re.compile(r"^(?P<instance>\d{3})_(?P<capture>\d{2})\.bmp$", re.IGNORECASE)
_INSTANCE_RE = re.compile(r"^\d{3}$")


def _normalize_mapping(instance_to_person: Mapping[str, str] | None) -> dict[str, str] | None:
    if instance_to_person is None:
        return None
    normalized: dict[str, str] = {}
    for instance, person in instance_to_person.items():
        instance_id = str(instance).strip()
        person_id = str(person).strip()
        if not _INSTANCE_RE.fullmatch(instance_id):
            raise ValueError(f"invalid NUPT instance id {instance_id!r}")
        if not person_id:
            raise ValueError(f"empty person id for NUPT instance {instance_id!r}")
        normalized[instance_id] = person_id
    if not normalized:
        raise ValueError("instance_to_person must not be empty when provided")
    return normalized


def scan_nupt_fpv(
    root: str | Path,
    *,
    instance_to_person: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Scan a local NUPT-FPV tree into the DeepMM metadata-manifest schema.

    Expected source layout::

        image/Session1/Fingerprint/001/001_01.bmp
        image/Session1/FingerVein/001/001_01.bmp
        image/Session2/...

    When ``instance_to_person`` is omitted, all rows deliberately carry the common
    sentinel ``UNRESOLVED_PUBLIC_NUPT_PERSON`` and
    ``person_mapping_status='unresolved'``. This permits structural smoke testing
    while making biological-subject inference impossible to claim accidentally.
    """
    dataset_root = Path(root)
    image_root = dataset_root / "image"
    if not image_root.is_dir():
        raise ValueError(f"NUPT-FPV image directory not found: {image_root}")

    mapping = _normalize_mapping(instance_to_person)
    rows: list[dict[str, Any]] = []
    encountered_instances: set[str] = set()

    session_dirs = sorted(path for path in image_root.iterdir() if path.is_dir())
    if not session_dirs:
        raise ValueError("NUPT-FPV image directory contains no session directories")

    for session_dir in session_dirs:
        match = _SESSION_RE.fullmatch(session_dir.name)
        if match is None:
            raise ValueError(f"unexpected NUPT-FPV session directory {session_dir.name!r}")
        session_id = match.group("session")

        for source_modality, canonical_modality in _MODALITY_DIRS.items():
            modality_root = session_dir / source_modality
            if not modality_root.is_dir():
                raise ValueError(
                    f"missing NUPT-FPV modality directory {modality_root.relative_to(dataset_root)}"
                )

            instance_dirs = sorted(path for path in modality_root.iterdir() if path.is_dir())
            if not instance_dirs:
                raise ValueError(f"no biometric instances found under {modality_root}")

            for instance_dir in instance_dirs:
                instance_id = instance_dir.name
                if not _INSTANCE_RE.fullmatch(instance_id):
                    raise ValueError(f"invalid NUPT-FPV instance directory {instance_id!r}")
                encountered_instances.add(instance_id)

                files = sorted(path for path in instance_dir.iterdir() if path.is_file())
                if not files:
                    raise ValueError(f"no captures found under {instance_dir}")

                for file_path in files:
                    file_match = _FILE_RE.fullmatch(file_path.name)
                    if file_match is None:
                        raise ValueError(
                            f"unexpected NUPT-FPV capture filename {file_path.name!r}"
                        )
                    filename_instance = file_match.group("instance")
                    if filename_instance != instance_id:
                        raise ValueError(
                            f"filename instance {filename_instance!r} does not match "
                            f"directory {instance_id!r}"
                        )
                    capture_id = file_match.group("capture")

                    if mapping is None:
                        person_id = NUPT_PUBLIC_UNRESOLVED_PERSON
                        mapping_status = "unresolved"
                    else:
                        if instance_id not in mapping:
                            raise ValueError(
                                f"verified person mapping is missing NUPT instance {instance_id!r}"
                            )
                        person_id = mapping[instance_id]
                        mapping_status = "resolved"

                    relative_path = file_path.relative_to(dataset_root).as_posix()
                    rows.append(
                        {
                            "sample_id": (
                                f"nupt-{session_id}-{canonical_modality}-"
                                f"{instance_id}-{capture_id}"
                            ),
                            "person_id": person_id,
                            "instance_id": instance_id,
                            "modality": canonical_modality,
                            "session_id": session_id,
                            "capture_id": capture_id,
                            "relative_path": relative_path,
                            "file_size_bytes": file_path.stat().st_size,
                            "dataset_id": "NUPT-FPV",
                            "person_mapping_status": mapping_status,
                        }
                    )

    if mapping is not None:
        extra = sorted(set(mapping) - encountered_instances)
        if extra:
            raise ValueError(
                "verified person mapping contains instance ids absent from archive: "
                + ", ".join(extra[:10])
            )

    return validate_dataset_records(rows)


def assert_nupt_person_mapping_resolved(records: list[Mapping[str, Any]]) -> None:
    """Reject NUPT manifests whose human-person mapping is not verified."""
    if not records:
        raise ValueError("NUPT-FPV records must not be empty")
    unresolved = [
        str(row.get("instance_id", "?"))
        for row in records
        if row.get("person_mapping_status") != "resolved"
        or row.get("person_id") == NUPT_PUBLIC_UNRESOLVED_PERSON
    ]
    if unresolved:
        preview = ", ".join(sorted(set(unresolved))[:10])
        raise ValueError(
            "NUPT-FPV human-person mapping is unresolved; technical-only evidence "
            f"for instance ids: {preview}"
        )
