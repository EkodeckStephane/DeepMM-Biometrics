"""Framework-independent contracts for neural model development.

These objects describe *where* fitting, model selection and calibration may occur.
They do not choose final partition sizes or hyperparameters before a real dataset is
locked.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


def _name(value: str, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


@dataclass(frozen=True)
class FinalTestFirewall:
    """Declare mutually exclusive development roles and the untouched final test.

    `fit_partition` provides gradient/parameter fitting data.
    `selection_partition` is used for early stopping/hyperparameter selection.
    `calibration_partition` is reserved for post-hoc score calibration when used.
    `final_test_partition` must never appear in any development role.

    The contract is role-level. Actual sample/person disjointness is checked by the
    dataset/split validators after the raw archive is available.
    """

    fit_partition: str
    selection_partition: str
    calibration_partition: str
    final_test_partition: str

    def __post_init__(self) -> None:
        normalized = {
            "fit_partition": _name(self.fit_partition, "fit_partition"),
            "selection_partition": _name(self.selection_partition, "selection_partition"),
            "calibration_partition": _name(self.calibration_partition, "calibration_partition"),
            "final_test_partition": _name(self.final_test_partition, "final_test_partition"),
        }
        if len(set(normalized.values())) != 4:
            raise ValueError(
                "fit, selection, calibration and final-test partitions must be distinct"
            )
        for key, value in normalized.items():
            object.__setattr__(self, key, value)

    def assert_development_partition(self, partition: str) -> str:
        """Return a normalized development partition or reject final-test access."""
        value = _name(partition, "partition")
        if value == self.final_test_partition:
            raise ValueError("final-test partition is forbidden during model development")
        if value not in {
            self.fit_partition,
            self.selection_partition,
            self.calibration_partition,
        }:
            raise ValueError(f"unknown development partition {value!r}")
        return value

    def as_dict(self) -> dict[str, str]:
        return {
            "fit_partition": self.fit_partition,
            "selection_partition": self.selection_partition,
            "calibration_partition": self.calibration_partition,
            "final_test_partition": self.final_test_partition,
        }

    @property
    def protocol_hash(self) -> str:
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
