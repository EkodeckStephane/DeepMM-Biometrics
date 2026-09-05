"""Evidence-tier contracts for fair multimodal fusion comparisons.

The primary DeepMM benchmark studies the *fusion mechanism* while keeping upstream
unimodal evidence controlled. These lightweight containers make the information
available to each method explicit and prevent hidden missingness through NaN values.

They deliberately contain no labels. Labels belong to development/evaluation
routines, never to a transform-time evidence object.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import numpy as np


class EvidenceTier(str, Enum):
    """Information strata used by the confirmatory benchmark."""

    SCORE = "score"
    EMBEDDING = "embedding"
    TOKEN = "token"


def _names(names: Iterable[str]) -> tuple[str, ...]:
    out = tuple(str(name).strip() for name in names)
    if len(out) < 2:
        raise ValueError("multimodal evidence requires at least two modalities")
    if any(not name for name in out):
        raise ValueError("modality names must be non-empty")
    if len(set(out)) != len(out):
        raise ValueError("modality names must be unique")
    return out


def _matrix(value, *, name: str, dtype=np.float64) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D array")
    return arr


def _availability(value, shape: tuple[int, int], *, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if arr.dtype != np.bool_:
        # Integer 0/1 is accepted only after an exact-value check. Floats are
        # rejected to avoid treating arbitrary quality-like values as masks.
        if not np.issubdtype(arr.dtype, np.integer) or not np.all(np.isin(arr, [0, 1])):
            raise ValueError(f"{name} must be boolean or exact integer 0/1")
        arr = arr.astype(bool)
    return arr.astype(bool, copy=False)


def _quality(value, shape: tuple[int, int], availability: np.ndarray) -> np.ndarray | None:
    if value is None:
        return None
    q = np.asarray(value, dtype=np.float64)
    if q.shape != shape:
        raise ValueError(f"quality must have shape {shape}")
    if not np.all(np.isfinite(q)) or np.any(q < 0.0):
        raise ValueError("quality values must be finite and non-negative")
    if np.any(q[~availability] != 0.0):
        raise ValueError("quality for unavailable modalities must use the canonical zero placeholder")
    return q


def _require_row_evidence(availability: np.ndarray, *, name: str) -> None:
    if np.any(np.sum(availability, axis=1) == 0):
        raise ValueError(f"each trial must retain at least one available modality in {name}")


@dataclass(frozen=True)
class ScoreEvidence:
    """Canonical trial-level score evidence.

    ``scores`` has shape ``(n_trials, n_modalities)``. Missingness is represented
    only by ``availability``; unavailable score slots must be the exact canonical
    value ``0.0`` rather than NaN/inf/sentinel values. This prevents a method from
    receiving extra missingness information through its raw score magnitude.

    ``quality`` is optional. If present, it uses the same trial x modality shape;
    unavailable slots must be zero.
    """

    scores: np.ndarray
    modality_names: tuple[str, ...]
    availability: np.ndarray
    quality: np.ndarray | None = None

    def __post_init__(self) -> None:
        names = _names(self.modality_names)
        scores = _matrix(self.scores, name="scores")
        if scores.shape[1] != len(names):
            raise ValueError("scores modality dimension must match modality_names")
        availability = _availability(self.availability, scores.shape, name="availability")
        _require_row_evidence(availability, name="availability")
        if not np.all(np.isfinite(scores)):
            raise ValueError("scores must be finite; encode missingness only through availability")
        if np.any(scores[~availability] != 0.0):
            raise ValueError("unavailable score slots must use the canonical zero placeholder")
        quality = _quality(self.quality, scores.shape, availability)

        object.__setattr__(self, "scores", np.array(scores, dtype=np.float64, copy=True))
        object.__setattr__(self, "modality_names", names)
        object.__setattr__(self, "availability", np.array(availability, dtype=bool, copy=True))
        object.__setattr__(self, "quality", None if quality is None else np.array(quality, copy=True))

    @property
    def tier(self) -> EvidenceTier:
        return EvidenceTier.SCORE

    @property
    def n_trials(self) -> int:
        return int(self.scores.shape[0])

    @property
    def n_modalities(self) -> int:
        return int(self.scores.shape[1])

    def complete_case_mask(self) -> np.ndarray:
        return np.all(self.availability, axis=1)


@dataclass(frozen=True)
class EmbeddingEvidence:
    """Canonical paired enrollment/probe embeddings for verification trials.

    Each modality may have a different embedding dimension, so embeddings are a
    tuple of arrays with shape ``(n_trials, d_m)``. Availability is explicit and
    separate for enrollment and probe. Unavailable embeddings must be all-zero;
    hidden NaN/sentinel missingness is forbidden.

    Trial-level quality is associated with the probe by default because Q3 stress
    manipulations are typically applied to observed/probe evidence. If a study
    needs side-specific quality, the protocol must introduce and freeze that
    extension before final testing rather than smuggling it into an embedding.
    """

    enrollment: tuple[np.ndarray, ...]
    probe: tuple[np.ndarray, ...]
    modality_names: tuple[str, ...]
    enrollment_availability: np.ndarray
    probe_availability: np.ndarray
    quality: np.ndarray | None = None

    def __post_init__(self) -> None:
        names = _names(self.modality_names)
        enrollment = tuple(np.asarray(x, dtype=np.float64) for x in self.enrollment)
        probe = tuple(np.asarray(x, dtype=np.float64) for x in self.probe)
        if len(enrollment) != len(names) or len(probe) != len(names):
            raise ValueError("one enrollment/probe embedding array is required per modality")
        if not enrollment:
            raise ValueError("embedding evidence must not be empty")

        n_trials: int | None = None
        for index, (enroll, query) in enumerate(zip(enrollment, probe)):
            if enroll.ndim != 2 or query.ndim != 2 or enroll.shape != query.shape:
                raise ValueError(
                    f"modality {names[index]!r}: enrollment and probe must be matching 2-D arrays"
                )
            if enroll.shape[0] == 0 or enroll.shape[1] == 0:
                raise ValueError(f"modality {names[index]!r}: embeddings must be non-empty")
            if n_trials is None:
                n_trials = int(enroll.shape[0])
            elif enroll.shape[0] != n_trials:
                raise ValueError("all modality embedding arrays must share the same trial count")
            if not np.all(np.isfinite(enroll)) or not np.all(np.isfinite(query)):
                raise ValueError("embeddings must be finite; encode missingness only through availability")

        assert n_trials is not None
        shape = (n_trials, len(names))
        enroll_av = _availability(
            self.enrollment_availability, shape, name="enrollment_availability"
        )
        probe_av = _availability(self.probe_availability, shape, name="probe_availability")
        _require_row_evidence(enroll_av, name="enrollment_availability")
        _require_row_evidence(probe_av, name="probe_availability")

        for m, (enroll, query) in enumerate(zip(enrollment, probe)):
            if np.any(enroll[~enroll_av[:, m]] != 0.0):
                raise ValueError(
                    f"modality {names[m]!r}: unavailable enrollment embeddings must be all-zero"
                )
            if np.any(query[~probe_av[:, m]] != 0.0):
                raise ValueError(
                    f"modality {names[m]!r}: unavailable probe embeddings must be all-zero"
                )

        quality = _quality(self.quality, shape, probe_av)
        object.__setattr__(
            self, "enrollment", tuple(np.array(x, dtype=np.float64, copy=True) for x in enrollment)
        )
        object.__setattr__(
            self, "probe", tuple(np.array(x, dtype=np.float64, copy=True) for x in probe)
        )
        object.__setattr__(self, "modality_names", names)
        object.__setattr__(self, "enrollment_availability", np.array(enroll_av, copy=True))
        object.__setattr__(self, "probe_availability", np.array(probe_av, copy=True))
        object.__setattr__(self, "quality", None if quality is None else np.array(quality, copy=True))

    @property
    def tier(self) -> EvidenceTier:
        return EvidenceTier.EMBEDDING

    @property
    def n_trials(self) -> int:
        return int(self.enrollment[0].shape[0])

    @property
    def n_modalities(self) -> int:
        return len(self.modality_names)

    @property
    def embedding_dims(self) -> tuple[int, ...]:
        return tuple(int(x.shape[1]) for x in self.enrollment)

    def complete_case_mask(self) -> np.ndarray:
        return np.all(self.enrollment_availability & self.probe_availability, axis=1)


@dataclass(frozen=True)
class FusionMethodSpec:
    """Predeclared information-access contract for one fusion family."""

    method_id: str
    name: str
    evidence_tier: EvidenceTier
    uses_quality: bool = False
    uses_availability: bool = False
    confirmatory: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        method_id = str(self.method_id).strip().upper()
        name = str(self.name).strip()
        if not method_id or not name:
            raise ValueError("method_id and name must be non-empty")
        object.__setattr__(self, "method_id", method_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "evidence_tier", EvidenceTier(self.evidence_tier))

    def validate_evidence(self, evidence: ScoreEvidence | EmbeddingEvidence) -> None:
        """Reject comparisons in which a method receives the wrong information tier."""
        if evidence.tier != self.evidence_tier:
            raise ValueError(
                f"{self.method_id} requires {self.evidence_tier.value} evidence, "
                f"received {evidence.tier.value}"
            )
        if self.uses_quality and evidence.quality is None:
            raise ValueError(f"{self.method_id} requires predeclared quality variables")
        if not self.uses_quality and evidence.quality is not None:
            raise ValueError(
                f"{self.method_id} is not permitted to consume quality variables in this contract"
            )
        if not self.uses_availability:
            complete = evidence.complete_case_mask()
            if not np.all(complete):
                raise ValueError(
                    f"{self.method_id} has no missingness access; evaluate it only on complete evidence"
                )


def canonical_confirmatory_method_specs() -> tuple[FusionMethodSpec, ...]:
    """Return the Gate-4-locked method-information strata.

    Architecture details remain data-dependent, but these IDs and information
    tiers define what can be called a within-stratum fusion comparison.
    """
    return (
        FusionMethodSpec("C1", "Equal normalized score fusion", EvidenceTier.SCORE),
        FusionMethodSpec("C2", "Validation-weighted score fusion", EvidenceTier.SCORE),
        FusionMethodSpec("C3", "Regularized logistic score fusion", EvidenceTier.SCORE),
        FusionMethodSpec(
            "C5", "Classical quality-weighted score fusion", EvidenceTier.SCORE, uses_quality=True
        ),
        FusionMethodSpec("D1", "Compact nonlinear score fusion", EvidenceTier.SCORE),
        FusionMethodSpec(
            "D3S",
            "Learned score quality/availability gate",
            EvidenceTier.SCORE,
            uses_quality=True,
            uses_availability=True,
        ),
        FusionMethodSpec("C4", "Controlled feature concatenation", EvidenceTier.EMBEDDING),
        FusionMethodSpec("D2", "Compact nonlinear feature fusion", EvidenceTier.EMBEDDING),
        FusionMethodSpec(
            "D3F",
            "Learned feature quality/availability gate",
            EvidenceTier.EMBEDDING,
            uses_quality=True,
            uses_availability=True,
        ),
    )


def method_spec(method_id: str) -> FusionMethodSpec:
    """Lookup a canonical method specification by ID."""
    key = str(method_id).strip().upper()
    matches = [spec for spec in canonical_confirmatory_method_specs() if spec.method_id == key]
    if not matches:
        raise KeyError(f"unknown confirmatory method_id {method_id!r}")
    return matches[0]
