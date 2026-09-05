"""Validation and hashing of reproducible DeepMM experiment-run manifests."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .hashing import hash_ordered_records

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

REQUIRED_RUN_FIELDS = (
    "run_id",
    "method_id",
    "family",
    "seed",
    "condition_id",
    "code_commit",
    "split_hash",
    "trial_manifest_hash",
    "config_hash",
    "score_manifest_hash",
)

OPTIONAL_SHA256_FIELDS = (
    "dataset_manifest_hash",
    "checkpoint_hash",
    "environment_hash",
)


def _nonempty_text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if value is None:
        raise ValueError(f"missing required run field {field!r}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"run field {field!r} must not be empty")
    return text


def _sha256(record: Mapping[str, Any], field: str, *, required: bool) -> str | None:
    value = record.get(field)
    if value is None:
        if required:
            raise ValueError(f"missing required run field {field!r}")
        return None
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"run field {field!r} must be a 64-character lowercase SHA-256 digest")
    return text


def validate_run_manifest(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one immutable experiment-run provenance record.

    The manifest binds a scientific result to the exact code commit, subject split,
    ordered verification trials, configuration, random seed, experimental condition
    and ordered score output. Optional dataset/checkpoint/environment hashes may be
    added when those artifacts exist.

    A full 40-character Git commit SHA is required. Short SHAs are intentionally
    rejected because final evidence records must be globally unambiguous within the
    repository history.
    """
    if not isinstance(record, Mapping):
        raise TypeError("run manifest must be a mapping")
    row = dict(record)

    run_id = _nonempty_text(row, "run_id")
    method_id = _nonempty_text(row, "method_id")
    family = _nonempty_text(row, "family")
    condition_id = _nonempty_text(row, "condition_id")

    raw_seed = row.get("seed")
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, int) or raw_seed < 0:
        raise ValueError("run field 'seed' must be a non-negative integer")
    seed = int(raw_seed)

    code_commit = _nonempty_text(row, "code_commit").lower()
    if not _GIT_SHA_RE.fullmatch(code_commit):
        raise ValueError("run field 'code_commit' must be a full 40-character lowercase Git SHA")

    required_hashes = {
        field: _sha256(row, field, required=True)
        for field in ("split_hash", "trial_manifest_hash", "config_hash", "score_manifest_hash")
    }
    optional_hashes = {
        field: _sha256(row, field, required=False)
        for field in OPTIONAL_SHA256_FIELDS
    }

    row.update(
        run_id=run_id,
        method_id=method_id,
        family=family,
        seed=seed,
        condition_id=condition_id,
        code_commit=code_commit,
        **required_hashes,
    )
    for field, value in optional_hashes.items():
        if value is not None:
            row[field] = value
        elif field in row:
            # Do not let explicit null values silently participate in supposedly
            # immutable manifests: omit the field when the artifact does not exist.
            row.pop(field)

    return row


def run_manifest_hash(record: Mapping[str, Any]) -> str:
    """Return SHA-256 of the validated canonical run record."""
    return hash_ordered_records([validate_run_manifest(record)])
