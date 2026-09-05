"""Dataset-agnostic missing-modality utilities.

Missingness is a second experimental axis, not a hidden property of raw arrays.
These helpers operate on explicit availability masks and preserve canonical zero
placeholders required by :mod:`deepmm.fusion.contracts`.
"""

from __future__ import annotations

import numpy as np

from .contracts import EmbeddingEvidence, ScoreEvidence


def deterministic_modality_dropout_mask(
    n_samples: int,
    n_modalities: int,
    *,
    drop_probability: float,
    seed: int,
) -> np.ndarray:
    """Generate a deterministic availability mask for modality-dropout training.

    Each modality is independently dropped with the requested probability. Rows in
    which every modality would be dropped are repaired deterministically using the
    same RNG by retaining one uniformly selected modality. The returned mask is
    boolean and always contains at least one available modality per sample.

    This helper is for training/development policy construction. Final-test missing
    conditions must be explicit preregistered patterns, not random draws.
    """
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if not isinstance(n_modalities, int) or n_modalities < 2:
        raise ValueError("n_modalities must be an integer >= 2")
    p = float(drop_probability)
    if not 0.0 <= p < 1.0:
        raise ValueError("drop_probability must lie in [0, 1)")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(seed)
    available = rng.random((n_samples, n_modalities)) >= p
    empty_rows = np.flatnonzero(np.sum(available, axis=1) == 0)
    if empty_rows.size:
        retained = rng.integers(0, n_modalities, size=empty_rows.size)
        available[empty_rows, retained] = True
    return available


def fixed_subset_mask(
    n_samples: int,
    modality_names,
    available_modalities,
) -> np.ndarray:
    """Return a repeated explicit availability pattern for a named modality subset."""
    names = tuple(str(x).strip() for x in modality_names)
    if len(names) < 2 or len(set(names)) != len(names) or any(not x for x in names):
        raise ValueError("modality_names must contain at least two unique non-empty names")
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    selected = {str(x).strip() for x in available_modalities}
    unknown = selected - set(names)
    if unknown:
        raise ValueError(f"unknown available modalities: {sorted(unknown)}")
    if not selected:
        raise ValueError("at least one modality must remain available")
    row = np.array([name in selected for name in names], dtype=bool)
    return np.tile(row, (n_samples, 1))


def apply_score_availability(
    evidence: ScoreEvidence,
    availability,
) -> ScoreEvidence:
    """Return score evidence under a new explicit availability pattern.

    A modality can only be removed from the existing evidence; this function cannot
    synthesize evidence that was originally unavailable. Scores and quality values
    for newly unavailable modalities are replaced by canonical zero placeholders.
    """
    target = np.asarray(availability)
    if target.shape != evidence.availability.shape:
        raise ValueError("availability shape must match the evidence")
    if target.dtype != np.bool_:
        if not np.issubdtype(target.dtype, np.integer) or not np.all(np.isin(target, [0, 1])):
            raise ValueError("availability must be boolean or exact integer 0/1")
        target = target.astype(bool)
    if np.any(target & ~evidence.availability):
        raise ValueError("cannot make originally unavailable score evidence available")
    if np.any(np.sum(target, axis=1) == 0):
        raise ValueError("each trial must retain at least one available modality")

    scores = np.where(target, evidence.scores, 0.0)
    quality = None if evidence.quality is None else np.where(target, evidence.quality, 0.0)
    return ScoreEvidence(scores, evidence.modality_names, target, quality)


def apply_embedding_availability(
    evidence: EmbeddingEvidence,
    *,
    enrollment_availability=None,
    probe_availability=None,
) -> EmbeddingEvidence:
    """Return embedding evidence with additional explicit modality removals.

    Enrollment and probe masks can be manipulated independently, supporting both
    symmetric missing-modality tests and flexible asymmetric enrollment/query tests.
    Omitted masks keep the corresponding original availability.
    """

    def _target(value, original, side: str):
        if value is None:
            return np.array(original, copy=True)
        arr = np.asarray(value)
        if arr.shape != original.shape:
            raise ValueError(f"{side}_availability shape must match the evidence")
        if arr.dtype != np.bool_:
            if not np.issubdtype(arr.dtype, np.integer) or not np.all(np.isin(arr, [0, 1])):
                raise ValueError(f"{side}_availability must be boolean or exact integer 0/1")
            arr = arr.astype(bool)
        if np.any(arr & ~original):
            raise ValueError(f"cannot make originally unavailable {side} evidence available")
        if np.any(np.sum(arr, axis=1) == 0):
            raise ValueError(f"each trial must retain at least one available {side} modality")
        return arr.astype(bool, copy=False)

    enroll_av = _target(enrollment_availability, evidence.enrollment_availability, "enrollment")
    probe_av = _target(probe_availability, evidence.probe_availability, "probe")

    enrollment = tuple(
        np.where(enroll_av[:, m, None], array, 0.0)
        for m, array in enumerate(evidence.enrollment)
    )
    probe = tuple(
        np.where(probe_av[:, m, None], array, 0.0)
        for m, array in enumerate(evidence.probe)
    )
    quality = None if evidence.quality is None else np.where(probe_av, evidence.quality, 0.0)
    return EmbeddingEvidence(
        enrollment,
        probe,
        evidence.modality_names,
        enroll_av,
        probe_av,
        quality,
    )


def masked_weighted_score_sum(
    evidence: ScoreEvidence,
    weights,
) -> np.ndarray:
    """Fuse scores by renormalizing fixed non-negative weights over available modalities.

    This is an explicit deterministic M0-style fallback utility. It does not learn
    weights and therefore accepts no labels. Inputs should already have the score
    normalization required by the relevant experimental protocol.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] != evidence.n_modalities:
        raise ValueError("weights must be a 1-D vector matching the modality count")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0) or np.sum(w) <= 0.0:
        raise ValueError("weights must be finite, non-negative, and not all zero")

    effective = evidence.availability * w[None, :]
    denom = np.sum(effective, axis=1)
    if np.any(denom <= 0.0):
        raise ValueError("a trial has no positive-weight available modality")
    normalized = effective / denom[:, None]
    return np.sum(normalized * evidence.scores, axis=1)


def modality_subset_id(modality_names, availability_row) -> str:
    """Canonical human-readable identifier for one availability subset."""
    names = tuple(str(x).strip() for x in modality_names)
    row = np.asarray(availability_row)
    if row.ndim != 1 or row.shape[0] != len(names):
        raise ValueError("availability_row must match modality_names")
    if row.dtype != np.bool_:
        if not np.issubdtype(row.dtype, np.integer) or not np.all(np.isin(row, [0, 1])):
            raise ValueError("availability_row must be boolean or exact integer 0/1")
        row = row.astype(bool)
    selected = [name for name, keep in zip(names, row) if keep]
    if not selected:
        raise ValueError("empty modality subset is invalid")
    return "+".join(selected)
