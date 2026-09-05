"""Architecture-agnostic contracts for DeepMM neural fusion heads.

This module intentionally avoids importing a deep-learning framework. It freezes
what a neural family is allowed to consume and how much tuning it is allowed to
receive before a concrete PyTorch/JAX backend is selected. Final hidden widths may
remain data-dimensionality dependent, but the search space and information access
must be declared before final-test evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from math import prod
from typing import Iterable

from .contracts import EvidenceTier, FusionMethodSpec, method_spec


class NeuralHeadKind(str, Enum):
    SCORE_MLP = "score_mlp"
    FEATURE_MLP = "feature_mlp"
    SCORE_GATE = "score_gate"
    FEATURE_GATE = "feature_gate"


_ALLOWED_ACTIVATIONS = {"relu", "gelu", "silu", "tanh"}
_ALLOWED_OBJECTIVES = {"eer", "auc", "cllr", "composite_development_only"}


@dataclass(frozen=True)
class NeuralHeadConfig:
    """One candidate neural fusion-head configuration.

    ``input_dim`` is the actual numeric dimension received by the head after the
    upstream evidence contract has been materialized. It is recorded rather than
    inferred silently so parameter counts remain auditable.
    """

    method_id: str
    kind: NeuralHeadKind
    input_dim: int
    hidden_dims: tuple[int, ...]
    output_dim: int = 1
    activation: str = "relu"
    dropout: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        method_id = str(self.method_id).strip().upper()
        kind = NeuralHeadKind(self.kind)
        activation = str(self.activation).strip().lower()
        hidden_dims = tuple(int(value) for value in self.hidden_dims)
        if not method_id:
            raise ValueError("method_id must be non-empty")
        if not isinstance(self.input_dim, int) or self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if not hidden_dims or any(value <= 0 for value in hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")
        if not isinstance(self.output_dim, int) or self.output_dim <= 0:
            raise ValueError("output_dim must be a positive integer")
        if activation not in _ALLOWED_ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ALLOWED_ACTIVATIONS)}")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")

        spec = method_spec(method_id)
        expected = {
            NeuralHeadKind.SCORE_MLP: ("D1", EvidenceTier.SCORE),
            NeuralHeadKind.FEATURE_MLP: ("D2", EvidenceTier.EMBEDDING),
            NeuralHeadKind.SCORE_GATE: ("D3S", EvidenceTier.SCORE),
            NeuralHeadKind.FEATURE_GATE: ("D3F", EvidenceTier.EMBEDDING),
        }[kind]
        if method_id != expected[0]:
            raise ValueError(f"{kind.value} must use method_id {expected[0]}")
        if spec.evidence_tier != expected[1]:
            raise RuntimeError("canonical method registry is inconsistent with neural head kind")

        object.__setattr__(self, "method_id", method_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "hidden_dims", hidden_dims)
        object.__setattr__(self, "activation", activation)
        object.__setattr__(self, "dropout", float(self.dropout))

    @property
    def evidence_spec(self) -> FusionMethodSpec:
        return method_spec(self.method_id)

    @property
    def dense_parameter_count(self) -> int:
        """Count affine-layer parameters for the declared MLP skeleton.

        This is framework-independent and counts each dense layer as
        ``in_features * out_features + out_features``. Additional normalization,
        projection, attention, or quality-estimator parameters must be accounted
        for separately by the concrete backend rather than hidden here.
        """
        dims = (self.input_dim,) + self.hidden_dims + (self.output_dim,)
        return int(sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1)))

    def as_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "kind": self.kind.value,
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "output_dim": self.output_dim,
            "activation": self.activation,
            "dropout": self.dropout,
            "seed": self.seed,
            "dense_parameter_count": self.dense_parameter_count,
        }


@dataclass(frozen=True)
class TrainingBudget:
    """Predeclared resource/search budget shared across comparable neural methods."""

    max_epochs: int
    early_stopping_patience: int
    max_candidate_configs: int
    seeds: tuple[int, ...]
    tuning_objective: str = "eer"
    max_training_runs: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.max_epochs, int) or self.max_epochs <= 0:
            raise ValueError("max_epochs must be a positive integer")
        if not isinstance(self.early_stopping_patience, int) or self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be a non-negative integer")
        if self.early_stopping_patience >= self.max_epochs:
            raise ValueError("early_stopping_patience must be smaller than max_epochs")
        if not isinstance(self.max_candidate_configs, int) or self.max_candidate_configs <= 0:
            raise ValueError("max_candidate_configs must be a positive integer")
        seeds = tuple(int(seed) for seed in self.seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be a non-empty tuple of unique integers")
        objective = str(self.tuning_objective).strip().lower()
        if objective not in _ALLOWED_OBJECTIVES:
            raise ValueError(f"tuning_objective must be one of {sorted(_ALLOWED_OBJECTIVES)}")
        implied_runs = self.max_candidate_configs * len(seeds)
        max_runs = implied_runs if self.max_training_runs is None else self.max_training_runs
        if not isinstance(max_runs, int) or max_runs <= 0:
            raise ValueError("max_training_runs must be a positive integer")
        if implied_runs > max_runs:
            raise ValueError(
                "max_candidate_configs * number_of_seeds exceeds max_training_runs"
            )
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "tuning_objective", objective)
        object.__setattr__(self, "max_training_runs", int(max_runs))

    def as_dict(self) -> dict[str, object]:
        return {
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "max_candidate_configs": self.max_candidate_configs,
            "seeds": list(self.seeds),
            "tuning_objective": self.tuning_objective,
            "max_training_runs": self.max_training_runs,
        }


@dataclass(frozen=True)
class NeuralSearchSpace:
    """Finite preregisterable candidate set for one neural method family."""

    method_id: str
    candidates: tuple[NeuralHeadConfig, ...]
    budget: TrainingBudget

    def __post_init__(self) -> None:
        method_id = str(self.method_id).strip().upper()
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("candidates must not be empty")
        if len(candidates) > self.budget.max_candidate_configs:
            raise ValueError("candidate count exceeds the declared training budget")
        if any(candidate.method_id != method_id for candidate in candidates):
            raise ValueError("every candidate must belong to method_id")
        canonical = [json.dumps(candidate.as_dict(), sort_keys=True) for candidate in candidates]
        if len(canonical) != len(set(canonical)):
            raise ValueError("candidate configurations must be unique")
        object.__setattr__(self, "method_id", method_id)
        object.__setattr__(self, "candidates", candidates)

    @property
    def planned_training_runs(self) -> int:
        return len(self.candidates) * len(self.budget.seeds)

    @property
    def search_hash(self) -> str:
        payload = {
            "method_id": self.method_id,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "budget": self.budget.as_dict(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def assert_matched_training_budgets(search_spaces: Iterable[NeuralSearchSpace]) -> tuple[NeuralSearchSpace, ...]:
    """Require matched training/search budgets for a confirmatory neural comparison.

    Candidate architectures may differ by family, but epochs, patience, number of
    candidates, seed set, tuning objective, and run cap must be identical. A
    deliberate exception must be documented as a non-matched/system-level analysis.
    """
    spaces = tuple(search_spaces)
    if len(spaces) < 2:
        raise ValueError("at least two neural search spaces are required")
    first = spaces[0].budget
    for space in spaces[1:]:
        if space.budget != first:
            raise ValueError("neural methods do not have matched training/tuning budgets")
    ids = [space.method_id for space in spaces]
    if len(ids) != len(set(ids)):
        raise ValueError("search spaces must have unique method IDs")
    return spaces
