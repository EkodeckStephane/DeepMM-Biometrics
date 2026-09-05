"""Frozen stress-condition specifications for Q3 robustness experiments.

This module defines *what* a stress condition is without choosing image-specific
operators or severity values prematurely. Exact corruptions are data/modality
specific and are frozen later in configuration files before final-test access.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Iterable


class StressKind(str, Enum):
    CLEAN = "clean"
    CORRUPTION = "corruption"
    MISSING = "missing"


Scalar = str | int | float | bool


def _targets(values) -> tuple[str, ...]:
    out = tuple(str(x).strip() for x in values)
    if any(not x for x in out) or len(set(out)) != len(out):
        raise ValueError("target_modalities must be unique non-empty names")
    return out


def _parameters(values) -> tuple[tuple[str, Scalar], ...]:
    out: list[tuple[str, Scalar]] = []
    seen: set[str] = set()
    for raw_key, value in values:
        key = str(raw_key).strip()
        if not key or key in seen:
            raise ValueError("parameter keys must be unique non-empty strings")
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError("stress parameter values must be JSON scalar values")
        if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
            raise ValueError("stress parameter values must be finite")
        seen.add(key)
        out.append((key, value))
    return tuple(sorted(out))


@dataclass(frozen=True)
class StressCondition:
    """One preregisterable clean/corruption/missingness condition."""

    condition_id: str
    kind: StressKind
    target_modalities: tuple[str, ...] = ()
    operator: str = "none"
    severity_rank: int = 0
    parameters: tuple[tuple[str, Scalar], ...] = ()

    def __post_init__(self) -> None:
        condition_id = str(self.condition_id).strip()
        operator = str(self.operator).strip().lower()
        if not condition_id:
            raise ValueError("condition_id must be non-empty")
        if not operator:
            raise ValueError("operator must be non-empty")
        kind = StressKind(self.kind)
        targets = _targets(self.target_modalities)
        parameters = _parameters(self.parameters)
        if not isinstance(self.severity_rank, int) or self.severity_rank < 0:
            raise ValueError("severity_rank must be a non-negative integer")

        if kind is StressKind.CLEAN:
            if targets or operator != "none" or self.severity_rank != 0 or parameters:
                raise ValueError("clean condition must have no targets/operator parameters and severity 0")
        elif kind is StressKind.CORRUPTION:
            if not targets:
                raise ValueError("corruption condition requires target modalities")
            if operator in {"none", "missing"}:
                raise ValueError("corruption condition requires a concrete corruption operator")
            if self.severity_rank < 1:
                raise ValueError("corruption severity_rank must be >= 1")
        else:
            if not targets:
                raise ValueError("missing condition requires target modalities")
            if operator != "missing":
                raise ValueError("missing condition operator must be 'missing'")
            if self.severity_rank != 1:
                raise ValueError("categorical missingness uses severity_rank=1")
            if parameters:
                raise ValueError("missing condition does not accept corruption parameters")

        object.__setattr__(self, "condition_id", condition_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "target_modalities", targets)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "parameters", parameters)

    def as_dict(self) -> dict[str, object]:
        return {
            "condition_id": self.condition_id,
            "kind": self.kind.value,
            "target_modalities": list(self.target_modalities),
            "operator": self.operator,
            "severity_rank": self.severity_rank,
            "parameters": {key: value for key, value in self.parameters},
        }


def clean_condition() -> StressCondition:
    return StressCondition("clean", StressKind.CLEAN)


def validate_stress_plan(
    conditions: Iterable[StressCondition],
    modality_names,
) -> tuple[StressCondition, ...]:
    """Validate a complete ordered Q3 plan without selecting final severities.

    The plan requires exactly one clean condition and unique IDs. Corruption
    severities for the same operator/target tuple cannot reuse a rank. A missing
    condition cannot remove every modality because verification would have no
    biometric evidence left.
    """
    plan = tuple(conditions)
    if not plan:
        raise ValueError("stress plan must not be empty")
    modalities = tuple(str(x).strip() for x in modality_names)
    if len(modalities) < 2 or any(not x for x in modalities) or len(set(modalities)) != len(modalities):
        raise ValueError("modality_names must contain at least two unique non-empty names")

    ids = [condition.condition_id for condition in plan]
    if len(ids) != len(set(ids)):
        raise ValueError("stress condition IDs must be unique")
    if sum(condition.kind is StressKind.CLEAN for condition in plan) != 1:
        raise ValueError("stress plan must contain exactly one clean condition")

    seen_severity: set[tuple[str, tuple[str, ...], int]] = set()
    modality_set = set(modalities)
    for condition in plan:
        unknown = set(condition.target_modalities) - modality_set
        if unknown:
            raise ValueError(
                f"condition {condition.condition_id!r} targets unknown modalities: {sorted(unknown)}"
            )
        if condition.kind is StressKind.MISSING and set(condition.target_modalities) == modality_set:
            raise ValueError("a missing condition cannot remove every modality")
        if condition.kind is StressKind.CORRUPTION:
            key = (condition.operator, tuple(sorted(condition.target_modalities)), condition.severity_rank)
            if key in seen_severity:
                raise ValueError("duplicate corruption severity for the same operator/target set")
            seen_severity.add(key)
    return plan


def stress_plan_hash(conditions: Iterable[StressCondition], modality_names) -> str:
    """SHA-256 hash of the validated ordered stress plan."""
    plan = validate_stress_plan(conditions, modality_names)
    payload = {
        "modality_names": [str(x).strip() for x in modality_names],
        "conditions": [condition.as_dict() for condition in plan],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()
