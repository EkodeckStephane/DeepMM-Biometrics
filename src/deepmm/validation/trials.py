"""Validation rules for frozen biometric verification trial manifests.

A verification result is scientifically meaningful only if every system is scored
on the same, immutable trial list. This module validates the identity semantics,
order, and score coverage of that list before any metric is computed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .hashing import hash_ordered_records

REQUIRED_TRIAL_FIELDS = (
    "trial_id",
    "label",
    "anchor_subject_id",
    "enrollment_subject_id",
    "probe_subject_id",
    "enrollment_sample_id",
    "probe_sample_id",
    "condition_id",
)


def _text(record: Mapping[str, Any], field: str, index: int) -> str:
    value = record.get(field)
    if value is None:
        raise ValueError(f"trial {index}: missing field {field!r}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"trial {index}: empty field {field!r}")
    return text


def _label(record: Mapping[str, Any], index: int) -> int:
    """Accept literal integer/string 0 or 1; reject booleans and float-like labels."""
    raw = record.get("label")
    if isinstance(raw, bool):
        raise ValueError(f"trial {index}: label must be exactly 0 or 1, not boolean")
    text = str(raw).strip()
    if text not in {"0", "1"}:
        raise ValueError(f"trial {index}: label must be exactly 0 or 1")
    return int(text)


def validate_trial_records(
    records: Iterable[Mapping[str, Any]],
    *,
    require_anchor_member: bool = True,
    reject_self_match: bool = True,
) -> list[dict[str, Any]]:
    """Validate and canonicalize an ordered verification trial list.

    Label convention: ``1`` = genuine and ``0`` = impostor.

    Scientific consistency rules:
    - genuine trials require equal enrollment/probe identity;
    - impostor trials require different identities;
    - a self-comparison of exactly the same sample is rejected by default;
    - trial IDs are unique;
    - enrollment/probe sample IDs may reappear across different trials, as expected
      in biometric verification, but their identity assignment must be globally
      consistent within the manifest;
    - the subject-centric ``anchor_subject_id`` must be one of the two trial
      identities when ``require_anchor_member=True``.

    The returned records preserve input order and use canonical string IDs / int
    labels. Additional fields (sessions, sensors, corruption severity, modality
    availability, etc.) are retained unchanged and therefore participate in
    downstream ordered hashing.
    """
    canonical: list[dict[str, Any]] = []
    seen_trials: set[str] = set()
    sample_owner: dict[str, str] = {}

    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise TypeError(f"trial {index} must be a mapping")
        row = dict(raw)
        trial_id = _text(row, "trial_id", index)
        if trial_id in seen_trials:
            raise ValueError(f"trial {index}: duplicate trial_id {trial_id!r}")
        seen_trials.add(trial_id)

        label = _label(row, index)
        anchor = _text(row, "anchor_subject_id", index)
        enroll_subject = _text(row, "enrollment_subject_id", index)
        probe_subject = _text(row, "probe_subject_id", index)
        enroll_sample = _text(row, "enrollment_sample_id", index)
        probe_sample = _text(row, "probe_sample_id", index)
        condition = _text(row, "condition_id", index)

        if label == 1 and enroll_subject != probe_subject:
            raise ValueError(f"trial {trial_id!r}: genuine label requires equal subject IDs")
        if label == 0 and enroll_subject == probe_subject:
            raise ValueError(f"trial {trial_id!r}: impostor label requires different subject IDs")
        if reject_self_match and enroll_sample == probe_sample:
            raise ValueError(f"trial {trial_id!r}: self-match of the same sample is not allowed")
        if require_anchor_member and anchor not in {enroll_subject, probe_subject}:
            raise ValueError(f"trial {trial_id!r}: anchor_subject_id must belong to the trial")

        for sample_id, subject_id in (
            (enroll_sample, enroll_subject),
            (probe_sample, probe_subject),
        ):
            previous = sample_owner.get(sample_id)
            if previous is not None and previous != subject_id:
                raise ValueError(
                    f"sample {sample_id!r} is assigned to multiple subjects: {previous!r}, {subject_id!r}"
                )
            sample_owner[sample_id] = subject_id

        row.update(
            trial_id=trial_id,
            label=label,
            anchor_subject_id=anchor,
            enrollment_subject_id=enroll_subject,
            probe_subject_id=probe_subject,
            enrollment_sample_id=enroll_sample,
            probe_sample_id=probe_sample,
            condition_id=condition,
        )
        canonical.append(row)

    if not canonical:
        raise ValueError("trial manifest must not be empty")
    labels = {row["label"] for row in canonical}
    if labels != {0, 1}:
        raise ValueError("trial manifest must contain both genuine and impostor trials")
    return canonical


def validate_score_records(
    trial_records: Iterable[Mapping[str, Any]],
    score_records: Iterable[Mapping[str, Any]],
    *,
    score_field: str = "score",
) -> list[dict[str, Any]]:
    """Require one finite score for every frozen trial in the exact same order.

    Missing, extra, and reordered trial scores are rejected. If a model cannot
    produce a score for a planned trial, that is a run failure or an explicitly
    modelled failure-to-acquire outcome; the row is not silently dropped.
    """
    import math

    if not isinstance(score_field, str) or not score_field.strip():
        raise ValueError("score_field must be a non-empty string")

    trials = validate_trial_records(trial_records)
    scores: list[dict[str, Any]] = []
    for index, raw in enumerate(score_records):
        if not isinstance(raw, Mapping):
            raise TypeError(f"score row {index} must be a mapping")
        scores.append(dict(raw))
    if len(scores) != len(trials):
        raise ValueError("score record count must exactly match trial count")

    canonical_scores: list[dict[str, Any]] = []
    for index, (trial, row) in enumerate(zip(trials, scores)):
        trial_id = str(row.get("trial_id", "")).strip()
        if trial_id != trial["trial_id"]:
            raise ValueError(
                f"score row {index}: trial_id/order mismatch; expected {trial['trial_id']!r}, got {trial_id!r}"
            )
        try:
            score = float(row[score_field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"score row {index}: invalid {score_field!r}") from exc
        if not math.isfinite(score):
            raise ValueError(f"score row {index}: {score_field!r} must be finite")
        row["trial_id"] = trial_id
        row[score_field] = score
        canonical_scores.append(row)
    return canonical_scores


def trial_manifest_hash(records: Iterable[Mapping[str, Any]]) -> str:
    """Validate and hash the complete ordered trial manifest with SHA-256."""
    return hash_ordered_records(validate_trial_records(records))


def score_manifest_hash(
    trial_records: Iterable[Mapping[str, Any]],
    score_records: Iterable[Mapping[str, Any]],
    *,
    score_field: str = "score",
) -> str:
    """Validate score coverage/order and hash the complete ordered score records."""
    canonical_scores = validate_score_records(
        trial_records, score_records, score_field=score_field
    )
    return hash_ordered_records(canonical_scores)
