import numpy as np
import pytest

from deepmm.stats.inference import holm_adjust, holm_reject, paired_cluster_permutation_test


def _mean_score(labels, scores):
    del labels
    return float(np.mean(scores))


def _toy():
    clusters = np.repeat(np.array(["s1", "s2", "s3", "s4"]), 2)
    labels = np.tile(np.array([0, 1]), 4)
    a = np.array([0.1, 0.9, 0.2, 1.0, 0.15, 0.95, 0.25, 1.05])
    b = a - 0.2
    return labels, a, b, clusters


def test_exact_cluster_permutation_is_deterministic_and_detects_strong_direction():
    y, a, b, c = _toy()
    observed, p_value, mode = paired_cluster_permutation_test(
        y, a, b, c, _mean_score, alternative="greater", exact_max_clusters=8
    )
    assert observed == pytest.approx(0.2)
    assert mode == "exact"
    # With four positive subject blocks, only the all-unswapped assignment is as
    # extreme in the one-sided positive direction: 1 / 16.
    assert p_value == pytest.approx(1.0 / 16.0)


def test_identical_systems_have_unit_two_sided_p_value():
    y, a, _, c = _toy()
    observed, p_value, mode = paired_cluster_permutation_test(
        y, a, a.copy(), c, _mean_score, exact_max_clusters=8
    )
    assert observed == pytest.approx(0.0)
    assert p_value == pytest.approx(1.0)
    assert mode == "exact"


def test_monte_carlo_mode_is_seed_deterministic():
    clusters = np.repeat(np.array([f"s{i}" for i in range(20)]), 2)
    labels = np.tile(np.array([0, 1]), 20)
    a = np.linspace(0.0, 1.0, 40)
    b = a - 0.05
    r1 = paired_cluster_permutation_test(
        labels,
        a,
        b,
        clusters,
        _mean_score,
        alternative="greater",
        n_permutations=200,
        seed=77,
        exact_max_clusters=8,
    )
    r2 = paired_cluster_permutation_test(
        labels,
        a,
        b,
        clusters,
        _mean_score,
        alternative="greater",
        n_permutations=200,
        seed=77,
        exact_max_clusters=8,
    )
    assert r1 == pytest.approx(r2[:2]) + (r2[2],) if False else r1 == r2
    assert r1[2] == "monte-carlo"


def test_holm_adjust_known_example_and_rejection():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    adjusted = holm_adjust(p)
    # sorted p: .01,.03,.04,.20 -> max(.04,.09,.08,.20) step-down
    assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.20])
    assert holm_reject(p, alpha=0.05).tolist() == [True, False, False, False]


def test_invalid_p_values_are_rejected():
    with pytest.raises(ValueError):
        holm_adjust([0.1, 1.2])
