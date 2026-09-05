"""Cluster-aware bootstrap utilities for biometric verification.

Verification trials are usually dependent because many trials share the same
biometric subject. These helpers therefore resample a predeclared *cluster ID*
rather than individual trials.

The cluster definition is part of the experimental protocol. For a subject-centric
verification trial table it can be the probe/claimed identity. For symmetric
all-vs-all pair construction, a one-way cluster bootstrap can be insufficient
because each impostor trial depends on two identities; such protocols require a
subsets/multiway bootstrap or a deterministic subject-level reconstruction rule
before these helpers are used for headline inference.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

ScalarMetric = Callable[[np.ndarray, np.ndarray], float]


def _arrays(labels, scores, cluster_ids):
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    c = np.asarray(cluster_ids)
    if y.ndim != 1 or s.ndim != 1 or c.ndim != 1:
        raise ValueError("labels, scores and cluster_ids must be 1-D arrays")
    if not (y.size == s.size == c.size) or y.size == 0:
        raise ValueError("labels, scores and cluster_ids must have equal non-zero length")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    if np.unique(c).size < 2:
        raise ValueError("at least two clusters are required for clustered bootstrap")
    return y, s, c


def _bootstrap_indices(
    cluster_ids: np.ndarray,
    rng: np.random.Generator,
    *,
    resample_within_cluster: bool,
) -> np.ndarray:
    """Draw one cluster bootstrap replicate and return row indices.

    Every unique cluster is sampled with replacement. A sampled cluster contributes
    its complete trial block by default. If ``resample_within_cluster`` is true,
    rows inside each selected block are also sampled with replacement, preserving
    the original block size.
    """
    clusters = np.unique(cluster_ids)
    sampled = rng.choice(clusters, size=clusters.size, replace=True)
    blocks: list[np.ndarray] = []
    for cluster in sampled:
        idx = np.flatnonzero(cluster_ids == cluster)
        if resample_within_cluster:
            idx = rng.choice(idx, size=idx.size, replace=True)
        blocks.append(np.asarray(idx, dtype=np.int64))
    return np.concatenate(blocks)


def percentile_interval(
    replicates,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a two-sided percentile interval for finite bootstrap replicates."""
    x = np.asarray(replicates, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("replicates must be a finite 1-D array with at least two values")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    alpha = 1.0 - confidence
    lo, hi = np.quantile(x, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def cluster_bootstrap_metric(
    labels,
    scores,
    cluster_ids,
    metric: ScalarMetric,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    resample_within_cluster: bool = False,
) -> np.ndarray:
    """Return cluster-bootstrap replicates of a scalar verification metric.

    ``metric`` must accept ``(labels, scores)`` and return one finite scalar. Use a
    small wrapper for tuple-valued repository metrics, e.g. ``lambda y, s: eer(y,s)[0]``.

    The function does not choose the cluster definition. That decision must be
    frozen with the trial protocol before final evaluation.
    """
    if not isinstance(n_boot, int) or n_boot < 2:
        raise ValueError("n_boot must be an integer >= 2")
    y, s, c = _arrays(labels, scores, cluster_ids)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = _bootstrap_indices(c, rng, resample_within_cluster=resample_within_cluster)
        value = float(metric(y[idx], s[idx]))
        if not np.isfinite(value):
            raise ValueError(f"metric returned a non-finite value at bootstrap replicate {b}")
        out[b] = value
    return out


def paired_cluster_bootstrap_difference(
    labels,
    scores_a,
    scores_b,
    cluster_ids,
    metric: ScalarMetric,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    resample_within_cluster: bool = False,
) -> np.ndarray:
    """Bootstrap the paired metric difference ``metric(A) - metric(B)``.

    Both systems are evaluated on the exact same resampled rows in every replicate.
    This preserves the paired experimental design and is preferred to comparing two
    independently bootstrapped confidence intervals.
    """
    y, a, c = _arrays(labels, scores_a, cluster_ids)
    b = np.asarray(scores_b, dtype=np.float64)
    if b.ndim != 1 or b.shape != a.shape or not np.all(np.isfinite(b)):
        raise ValueError("scores_b must be a finite 1-D array matching scores_a")
    if not isinstance(n_boot, int) or n_boot < 2:
        raise ValueError("n_boot must be an integer >= 2")

    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=np.float64)
    for r in range(n_boot):
        idx = _bootstrap_indices(c, rng, resample_within_cluster=resample_within_cluster)
        va = float(metric(y[idx], a[idx]))
        vb = float(metric(y[idx], b[idx]))
        delta = va - vb
        if not np.isfinite(delta):
            raise ValueError(f"metric difference is non-finite at bootstrap replicate {r}")
        out[r] = delta
    return out
