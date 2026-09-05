#!/usr/bin/env python3
"""Audit the official NUPT-FPV public subset for the bounded V1 study.

The script validates archive structure, cross-modality/session naming, actual BMP
metadata, and the deterministic V1 fit/selection/calibration/final trial manifests.
It deliberately does not infer a mapping from public instance IDs to human
volunteers.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import struct
import sys

from deepmm.datasets import (
    assert_nupt_person_mapping_resolved,
    scan_nupt_fpv,
    v1_trial_summary,
)
from deepmm.validation import audit_multimodal_topology, dataset_manifest_hash


EXPECTED_PUBLIC_INSTANCES = 20
EXPECTED_CAPTURES_PER_SESSION = 10
EXPECTED_SESSIONS = 2
EXPECTED_MODALITIES = ("fingerprint", "finger_vein")
EXPECTED_RECORDS = (
    EXPECTED_PUBLIC_INSTANCES
    * EXPECTED_CAPTURES_PER_SESSION
    * EXPECTED_SESSIONS
    * len(EXPECTED_MODALITIES)
)


def _bmp_metadata(path: Path) -> tuple[int, int, int]:
    header = path.read_bytes()[:54]
    if len(header) < 30 or header[:2] != b"BM":
        raise ValueError(f"not a readable BMP file: {path}")
    width, height = struct.unpack_from("<ii", header, 18)
    bits_per_pixel = struct.unpack_from("<H", header, 28)[0]
    if width <= 0 or height == 0 or bits_per_pixel <= 0:
        raise ValueError(f"invalid BMP dimensions/bit depth: {path}")
    return width, abs(height), bits_per_pixel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the official NUPT-FPV GitHub public subset for V1."
    )
    parser.add_argument("root", type=Path, help="Root of a local NUPT-FPV checkout")
    args = parser.parse_args(argv)

    try:
        rows = scan_nupt_fpv(args.root)
        topology = audit_multimodal_topology(
            rows,
            required_modalities=EXPECTED_MODALITIES,
            min_samples_per_modality=EXPECTED_CAPTURES_PER_SESSION * EXPECTED_SESSIONS,
            min_sessions_per_modality=EXPECTED_SESSIONS,
            require_capture_alignment=True,
        )
        trial_summary = v1_trial_summary(rows)

        dimensions: dict[str, set[tuple[int, int, int]]] = defaultdict(set)
        for row in rows:
            dimensions[row["modality"]].add(
                _bmp_metadata(args.root / row["relative_path"])
            )
    except (OSError, TypeError, ValueError, struct.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    instance_ids = sorted({row["instance_id"] for row in rows})
    sessions = sorted({row["session_id"] for row in rows})
    modality_counts = {
        modality: sum(row["modality"] == modality for row in rows)
        for modality in EXPECTED_MODALITIES
    }

    structural_ok = (
        topology["complete"]
        and len(rows) == EXPECTED_RECORDS
        and len(instance_ids) == EXPECTED_PUBLIC_INSTANCES
        and sessions == ["1", "2"]
        and modality_counts == {"fingerprint": 400, "finger_vein": 400}
        and all(len(dimensions[modality]) == 1 for modality in EXPECTED_MODALITIES)
    )

    try:
        assert_nupt_person_mapping_resolved(rows)
    except ValueError:
        scientific_identity_resolved = False
    else:
        scientific_identity_resolved = True

    serial_dimensions = {
        modality: [
            {"width": width, "height": height, "bits_per_pixel": bit_depth}
            for width, height, bit_depth in sorted(values)
        ]
        for modality, values in sorted(dimensions.items())
    }

    result = {
        "dataset": "NUPT-FPV",
        "evidence_scope": "v1_public_subset_structural_and_protocol",
        "n_records": len(rows),
        "n_public_instance_ids": len(instance_ids),
        "sessions": sessions,
        "modality_counts": modality_counts,
        "bmp_metadata_by_modality": serial_dimensions,
        "capture_alignment_by_name": topology["complete"],
        "structural_expectations_pass": structural_ok,
        "scientific_human_identity_mapping_resolved": scientific_identity_resolved,
        "dataset_manifest_hash": dataset_manifest_hash(rows),
        "v1_clean_trials": trial_summary,
        "warning": (
            "Public instance IDs 001-020 are biometric-instance identities only. "
            "V1 must not describe them as 20 independent human volunteers; "
            "person-level inference is deferred to the complete-dataset V2."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    return 0 if structural_ok and not scientific_identity_resolved else 1


if __name__ == "__main__":
    raise SystemExit(main())
