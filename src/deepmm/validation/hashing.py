"""Deterministic SHA-256 helpers for experiment locks.

Hashes are provenance identifiers, not security claims. They make accidental
changes to subject splits and ordered trial lists detectable across experiments.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_text(text: str) -> str:
    """Return lowercase SHA-256 hex digest of UTF-8 text."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_split_manifest(split_subjects: Mapping[str, Iterable[str]]) -> str:
    """Hash a split->subject mapping independent of input ordering.

    Split names and subject identifiers are string-normalized and sorted. Duplicate
    subject IDs within a split are rejected because silently collapsing them could
    hide a malformed manifest.
    """
    canonical: dict[str, list[str]] = {}
    if not split_subjects:
        raise ValueError("split_subjects must not be empty")
    for split, values in split_subjects.items():
        split_name = str(split)
        subjects = [str(v) for v in values]
        if not subjects:
            raise ValueError(f"split {split_name!r} must not be empty")
        if len(set(subjects)) != len(subjects):
            raise ValueError(f"duplicate subject ID within split {split_name!r}")
        canonical[split_name] = sorted(subjects)
    return sha256_text(_canonical_json(canonical))


def hash_ordered_records(records: Iterable[Mapping[str, Any]]) -> str:
    """Hash an ordered sequence of record dictionaries.

    Dictionary key order does not matter, but **record order does**. This is the
    intended behavior for a frozen verification trial list: reordering, inserting,
    deleting, or modifying any trial changes the digest.
    """
    lines: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} must be a mapping")
        lines.append(_canonical_json(dict(record)))
    if not lines:
        raise ValueError("records must not be empty")
    return sha256_text("\n".join(lines) + "\n")
