#!/usr/bin/env python3
"""Audit the official NUPT-FPV public subset as technical evidence only.

This script validates archive structure and cross-modality/session naming. It does
not convert the public 001-020 instance identifiers into human-person identifiers.
Accordingly, a successful run is infrastructure evidence, never a scientific
performance/dataset-lock result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from deepmm.datasets import assert_nupt_person_mapping_resolved, scan_nupt_fpv
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the official NUPT-FPV GitHub public subset (technical-only)."
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
    except (OSError, TypeError, ValueError) as exc:
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
    )

    try:
        assert_nupt_person_mapping_resolved(rows)
    except ValueError:
        scientific_identity_resolved = False
    else:
        scientific_identity_resolved = True

    result = {
        "dataset": "NUPT-FPV",
        "evidence_scope": "technical_public_subset_only",
        "n_records": len(rows),
        "n_public_instance_ids": len(instance_ids),
        "sessions": sessions,
        "modality_counts": modality_counts,
        "capture_alignment_by_name": topology["complete"],
        "structural_expectations_pass": structural_ok,
        "scientific_human_identity_mapping_resolved": scientific_identity_resolved,
        "dataset_manifest_hash": dataset_manifest_hash(rows),
        "warning": (
            "Public instance IDs 001-020 are not treated as human-volunteer IDs. "
            "This output cannot lock the scientific dataset or support performance claims."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    # The expected correct public-subset state is structurally complete while still
    # biologically unresolved. If person identity appears resolved without an
    # explicit verified mapping, the technical pilot should fail loudly.
    return 0 if structural_ok and not scientific_identity_resolved else 1


if __name__ == "__main__":
    raise SystemExit(main())
