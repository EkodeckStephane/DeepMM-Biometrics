#!/usr/bin/env python3
"""Audit a DeepMM dataset manifest without reading raw biometric content."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from deepmm.validation import audit_multimodal_topology, dataset_manifest_hash


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a metadata-only multimodal dataset archive manifest."
    )
    parser.add_argument("manifest", type=Path, help="CSV file with DeepMM dataset-manifest fields")
    parser.add_argument(
        "--modalities",
        required=True,
        help="Comma-separated required modalities, e.g. fingerprint,finger_vein",
    )
    parser.add_argument("--min-samples", type=_positive_int, default=1)
    parser.add_argument("--min-sessions", type=_positive_int, default=1)
    parser.add_argument(
        "--require-capture-alignment",
        action="store_true",
        help="Require identical (session,capture) keys across selected modalities",
    )
    args = parser.parse_args(argv)

    modalities = tuple(part.strip() for part in args.modalities.split(",") if part.strip())
    if len(modalities) < 2:
        parser.error("--modalities must contain at least two modalities")
    if not args.manifest.is_file():
        parser.error(f"manifest does not exist: {args.manifest}")

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    try:
        summary = audit_multimodal_topology(
            rows,
            required_modalities=modalities,
            min_samples_per_modality=args.min_samples,
            min_sessions_per_modality=args.min_sessions,
            require_capture_alignment=args.require_capture_alignment,
        )
        digest = dataset_manifest_hash(rows)
    except (TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    serializable = dict(summary)
    serializable["dataset_manifest_hash"] = digest
    print(json.dumps(serializable, indent=2, sort_keys=True))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
