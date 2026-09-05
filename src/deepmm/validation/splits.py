"""Hard validation rules for biometric dataset manifests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def _as_set(values: Iterable[str], name: str) -> set[str]:
    result = {str(v) for v in values}
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def assert_disjoint_subject_splits(
    train_subjects: Iterable[str],
    val_subjects: Iterable[str],
    test_subjects: Iterable[str],
) -> None:
    """Raise ValueError if any biometric identity appears in multiple splits."""
    splits = {
        "train": _as_set(train_subjects, "train_subjects"),
        "val": _as_set(val_subjects, "val_subjects"),
        "test": _as_set(test_subjects, "test_subjects"),
    }
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    violations: list[str] = []
    for a, b in pairs:
        overlap = splits[a] & splits[b]
        if overlap:
            preview = ", ".join(sorted(overlap)[:10])
            violations.append(f"{a}/{b}: {len(overlap)} overlap(s) [{preview}]")
    if violations:
        raise ValueError("subject leakage detected; " + "; ".join(violations))


def assert_unique_sample_ids(split_samples: Mapping[str, Iterable[str]]) -> None:
    """Raise ValueError if a sample identifier is duplicated within/across splits.

    A dataset adapter should use a content hash rather than only a file path when
    duplicate-file detection is required. This function deliberately accepts any
    stable identifier so callers can validate both path IDs and cryptographic hashes.
    """
    owner: dict[str, str] = {}
    duplicates: list[str] = []
    for split, sample_ids in split_samples.items():
        for raw_id in sample_ids:
            sample_id = str(raw_id)
            previous = owner.get(sample_id)
            if previous is not None:
                duplicates.append(f"{sample_id} ({previous}, {split})")
            else:
                owner[sample_id] = split
    if duplicates:
        preview = "; ".join(duplicates[:10])
        raise ValueError(f"duplicate sample identifier(s) detected: {preview}")
