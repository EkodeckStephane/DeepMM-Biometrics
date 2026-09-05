import numpy as np
import pytest

from deepmm.stats.bootstrap import (
    cluster_bootstrap_metric,
    paired_cluster_bootstrap_difference,
    percentile_interval,
)


def _separation(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    return float(np.mean(scores[labels == 1]) - np.mean(scores[labels == 0]))


def _clustered_toy():
    # Each subject-centric cluster contains both a genuine and an impostor trial,
    # so every cluster-bootstrap replicate remains evaluable.
    clusters = np.repeat(np.array(["s1", "s2", "s3", "s4"]), 2)
    labels = np.tile(np.array([0, 1]), 4)
    scores = np.array([0.1, 0.8, 0.2, 0.9, 0.3, 0.75, 0.15, 0.85])
    return labels, scores, clusters


def test_cluster_bootstrap_is_seed_deterministic():
    y, s, c = _clustered_toy()
    a = cluster_bootstrap_metric(y, s, c, _separation, n_boot=50, seed=17)
    b = cluster_bootstrap_metric(y, s, c, _separation, n_boot=50, seed=17)
    assert np.array_equal(a, b)
    assert np.all(np.isfinite(a))


def test_paired_bootstrap_identical_systems_has_zero_difference():
    y, s, c = _clustered_toy()
    delta = paired_cluster_bootstrap_difference(y, s, s.copy(), c, _separation, n_boot=40, seed=3)
    assert np.all(delta == pytest.approx(0.0))


def test_within_cluster_resampling_runs_and_is_deterministic():
    y, s, c = _clustered_toy()
    a = cluster_bootstrap_metric(
        y, s, c, _separation, n_boot=30, seed=9, resample_within_cluster=True
    )
    b = cluster_bootstrap_metric(
        y, s, c, _separation, n_boot=30, seed=9, resample_within_cluster=True
    )
    assert np.array_equal(a, b)


def test_percentile_interval_and_validation():
    lo, hi = percentile_interval(np.arange(101, dtype=float), confidence=0.90)
    assert lo == pytest.approx(5.0)
    assert hi == pytest.approx(95.0)
    with pytest.raises(ValueError):
        percentile_interval([1.0])


def test_bootstrap_requires_multiple_clusters():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match="at least two clusters"):
        cluster_bootstrap_metric(y, s, ["s1"] * 4, _separation, n_boot=10)
