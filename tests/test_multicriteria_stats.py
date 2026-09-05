import numpy as np
import pytest

from deepmm.stats.multicriteria import (
    bootstrap_dominance_probability,
    kendall_tau_b,
    non_dominated_mask,
    non_dominated_probability,
    pairwise_rank_reversals,
)


def test_pareto_front_respects_mixed_directions():
    # criterion 0: EER (minimize); criterion 1: TAR (maximize); criterion 2: latency (minimize)
    values = np.array(
        [
            [0.10, 0.90, 10.0],  # A dominates B
            [0.15, 0.85, 15.0],  # B
            [0.08, 0.92, 30.0],  # C trades accuracy for cost versus A
        ]
    )
    mask = non_dominated_mask(values, minimize=[True, False, True])
    assert mask.tolist() == [True, False, True]


def test_bootstrap_dominance_probability_and_front_probability():
    base = np.array(
        [
            [0.10, 0.90, 10.0],
            [0.15, 0.85, 15.0],
            [0.08, 0.92, 30.0],
        ]
    )
    samples = np.repeat(base[None, :, :], repeats=20, axis=0)
    p = bootstrap_dominance_probability(samples, minimize=[True, False, True])
    assert p[0, 1] == pytest.approx(1.0)
    assert p[1, 0] == pytest.approx(0.0)
    front = non_dominated_probability(samples, minimize=[True, False, True])
    assert front.tolist() == pytest.approx([1.0, 0.0, 1.0])


def test_kendall_tau_b_same_and_reverse_rankings():
    clean = np.array([1.0, 2.0, 3.0, 4.0])
    same = np.array([10.0, 20.0, 30.0, 40.0])
    reverse = same[::-1]
    assert kendall_tau_b(clean, same) == pytest.approx(1.0)
    assert kendall_tau_b(clean, reverse) == pytest.approx(-1.0)


def test_kendall_tau_b_handles_ties_and_degenerate_case():
    a = np.array([1.0, 1.0, 2.0])
    b = np.array([1.0, 2.0, 3.0])
    value = kendall_tau_b(a, b)
    assert 0.0 < value < 1.0
    assert np.isnan(kendall_tau_b([1.0, 1.0], [2.0, 2.0]))


def test_pairwise_rank_reversal_excludes_ties():
    clean = np.array([1.0, 2.0, 3.0])
    stress = np.array([3.0, 2.0, 1.0])
    reversals = pairwise_rank_reversals(clean, stress)
    assert reversals == [(0, 2)]  # pairs involving method 1 become ties, not reversals

    strict_stress = np.array([3.0, 2.5, 1.0])
    assert pairwise_rank_reversals(clean, strict_stress) == [(0, 1), (0, 2), (1, 2)]
