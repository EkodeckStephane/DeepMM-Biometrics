"""Comparable computational-cost records for the Q2 benchmark.

A latency number is scientifically comparable only when the measurement context is
explicit and matched. These utilities record that context and provide a small
hardware-agnostic timing helper. GPU callers may supply a synchronization callback.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class MeasurementContext:
    """Frozen context required for a headline cost comparison."""

    hardware_id: str
    device: str
    batch_size: int
    precision: str
    scope: str = "fusion_only"
    num_threads: int | None = None

    def __post_init__(self) -> None:
        hardware_id = str(self.hardware_id).strip()
        device = str(self.device).strip()
        precision = str(self.precision).strip().lower()
        scope = str(self.scope).strip().lower()
        if not hardware_id or not device or not precision:
            raise ValueError("hardware_id, device and precision must be non-empty")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if scope not in {"fusion_only", "end_to_end"}:
            raise ValueError("scope must be 'fusion_only' or 'end_to_end'")
        if self.num_threads is not None and (
            not isinstance(self.num_threads, int) or self.num_threads <= 0
        ):
            raise ValueError("num_threads must be a positive integer or None")
        object.__setattr__(self, "hardware_id", hardware_id)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "scope", scope)


@dataclass(frozen=True)
class LatencySummary:
    n: int
    median_ms: float
    q1_ms: float
    q3_ms: float
    mean_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or self.n < 2:
            raise ValueError("latency summary requires at least two repetitions")
        values = np.array([self.median_ms, self.q1_ms, self.q3_ms, self.mean_ms], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("latency summary values must be finite and non-negative")
        if self.q1_ms > self.median_ms or self.median_ms > self.q3_ms:
            raise ValueError("latency quartiles must satisfy q1 <= median <= q3")

    @property
    def iqr_ms(self) -> float:
        return float(self.q3_ms - self.q1_ms)


@dataclass(frozen=True)
class CostRecord:
    """One method's cost evidence under an explicit measurement context."""

    method_id: str
    context: MeasurementContext
    latency: LatencySummary
    trainable_params: int
    total_params: int
    macs: float | None = None
    peak_memory_mb: float | None = None

    def __post_init__(self) -> None:
        method_id = str(self.method_id).strip().upper()
        if not method_id:
            raise ValueError("method_id must be non-empty")
        for name, value in (
            ("trainable_params", self.trainable_params),
            ("total_params", self.total_params),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.trainable_params > self.total_params:
            raise ValueError("trainable_params cannot exceed total_params")
        for name, value in (("macs", self.macs), ("peak_memory_mb", self.peak_memory_mb)):
            if value is not None and (not np.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be finite and non-negative or None")
        object.__setattr__(self, "method_id", method_id)


def latency_summary(samples_ms) -> LatencySummary:
    """Summarize raw latency repetitions without discarding outliers silently."""
    x = np.asarray(samples_ms, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("samples_ms must be a finite non-negative 1-D array with >=2 values")
    q1, median, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    return LatencySummary(
        n=int(x.size),
        median_ms=float(median),
        q1_ms=float(q1),
        q3_ms=float(q3),
        mean_ms=float(np.mean(x)),
    )


def measure_latency(
    fn: Callable[[], object],
    *,
    warmup: int = 10,
    repeats: int = 100,
    synchronize: Callable[[], object] | None = None,
) -> tuple[LatencySummary, np.ndarray]:
    """Measure callable latency and return summary plus all raw samples.

    ``synchronize`` should be supplied for asynchronous accelerators (for example a
    framework-specific GPU synchronize function). Raw samples are returned because
    the final article must preserve cost evidence rather than only a single mean.
    """
    if not callable(fn):
        raise TypeError("fn must be callable")
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if not isinstance(repeats, int) or repeats < 2:
        raise ValueError("repeats must be an integer >= 2")
    if synchronize is not None and not callable(synchronize):
        raise TypeError("synchronize must be callable or None")

    for _ in range(warmup):
        fn()
    if synchronize is not None:
        synchronize()

    samples = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        if synchronize is not None:
            synchronize()
        start = perf_counter_ns()
        fn()
        if synchronize is not None:
            synchronize()
        end = perf_counter_ns()
        samples[i] = (end - start) / 1_000_000.0
    return latency_summary(samples), samples


def assert_comparable_cost_context(records: Iterable[CostRecord]) -> tuple[CostRecord, ...]:
    """Reject headline cost comparisons measured under different conditions."""
    values = tuple(records)
    if len(values) < 2:
        raise ValueError("at least two cost records are required for comparison")
    reference = values[0].context
    for record in values[1:]:
        if record.context != reference:
            raise ValueError(
                "cost records are not directly comparable: hardware/device/batch/precision/"
                "scope/thread context differs"
            )
    ids = [record.method_id for record in values]
    if len(ids) != len(set(ids)):
        raise ValueError("cost comparison must contain unique method IDs")
    return values
