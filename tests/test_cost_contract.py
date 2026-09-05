import numpy as np
import pytest

from deepmm.evaluation import (
    CostRecord,
    MeasurementContext,
    assert_comparable_cost_context,
    latency_summary,
    measure_latency,
)


def _context(**kwargs):
    base = dict(
        hardware_id="ci-cpu",
        device="cpu",
        batch_size=32,
        precision="fp32",
        scope="fusion_only",
        num_threads=1,
    )
    base.update(kwargs)
    return MeasurementContext(**base)


def test_latency_summary_preserves_count_and_quartile_order():
    summary = latency_summary([1.0, 2.0, 3.0, 4.0])
    assert summary.n == 4
    assert summary.q1_ms <= summary.median_ms <= summary.q3_ms
    assert summary.iqr_ms >= 0.0


def test_cost_record_rejects_impossible_parameter_counts():
    summary = latency_summary([1.0, 1.1])
    with pytest.raises(ValueError, match="cannot exceed"):
        CostRecord("D1", _context(), summary, trainable_params=20, total_params=10)


def test_cost_context_must_match_for_headline_comparison():
    summary = latency_summary([1.0, 1.1, 0.9])
    a = CostRecord("C3", _context(), summary, 10, 10)
    b = CostRecord("D1", _context(), summary, 100, 100)
    assert len(assert_comparable_cost_context([a, b])) == 2

    different_batch = CostRecord("D2", _context(batch_size=1), summary, 100, 100)
    with pytest.raises(ValueError, match="not directly comparable"):
        assert_comparable_cost_context([a, different_batch])


def test_duplicate_method_cost_records_are_rejected():
    summary = latency_summary([1.0, 1.1])
    a = CostRecord("D1", _context(), summary, 100, 100)
    with pytest.raises(ValueError, match="unique method IDs"):
        assert_comparable_cost_context([a, a])


def test_measure_latency_returns_all_raw_repetitions():
    counter = {"n": 0}

    def fn():
        counter["n"] += 1
        return counter["n"]

    summary, raw = measure_latency(fn, warmup=2, repeats=5)
    assert summary.n == 5
    assert raw.shape == (5,)
    assert np.all(raw >= 0.0)
    assert counter["n"] == 7


def test_measurement_scope_distinguishes_fusion_only_from_end_to_end():
    assert _context(scope="fusion_only") != _context(scope="end_to_end")
    with pytest.raises(ValueError):
        MeasurementContext("h", "cpu", 1, "fp32", scope="unknown")
