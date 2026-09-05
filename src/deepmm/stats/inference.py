"""Paired cluster-level randomization tests and multiplicity control.

These routines are intended for preregistered subject-centric comparisons. They
operate on complete cluster blocks, never on individual verification trials as if
those trials were independent subjects.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product

import numpy as np

ScalarMetric = Callable[[np.ndarray, np.ndarray], float]


def _paired_arrays(labels, scores_a, scores_b, cluster_ids):
    y = np.asarray(labels)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    c = np.asarray(cluster_ids)
    if y.ndim != 1 or a.ndim != 1 or b.ndim != 1 or c.ndim != 1:
        raise ValueError("labels, scores_a, scores_b and cluster_ids must be 1-D arrays")
    if not (y.size == a.size == b.size == c.size) or y.size == 0:
        raise ValueError("all paired arrays must have equal non-zero length")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("scores must be finite")
    clusters = np.unique(c)
    if clusters.size < 2:
        raise ValueError("at least two clusters are required")
    return y, a, b, c, clusters


def paired_cluster_permutation_test(
    labels,
    scores_a,
    scores_b,
    cluster_ids,
    metric: ScalarMetric,
    *,
    alternative: str = "two-sided",
    n_permutations: int = 10000,
    seed: int = 0,
    exact_max_clusters: int = 16,
) -> tuple[float, float, str]:
    """Test a paired system difference by swapping complete cluster blocks.

    The observed statistic is ``metric(A) - metric(B)``. Under the paired null,
    system labels A/B are exchangeable **within each predeclared cluster**, so an
    entire cluster's score block is either retained or swapped between systems.

    If the number of clusters is at most ``exact_max_clusters``, all ``2^K`` swap
    assignments are enumerated. Otherwise a seeded Monte-Carlo randomization test
    is used, with the standard +1 correction.

    Returns ``(observed_difference, p_value, mode)`` where mode is ``"exact"`` or
    ``"monte-carlo"``.

    This one-way cluster test inherits the same limitation as the one-way cluster
    bootstrap: dense symmetric all-vs-all impostor protocols may require a
    subsets/multiway dependence treatment instead.
    """
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    if not isinstance(n_permutations, int) or n_permutations < 1:
        raise ValueError("n_permutations must be an integer >= 1")
    if not isinstance(exact_max_clusters, int) or exact_max_clusters < 1:
        raise ValueError("exact_max_clusters must be an integer >= 1")

    y, a, b, c, clusters = _paired_arrays(labels, scores_a, scores_b, cluster_ids)
    observed = float(metric(y, a) - metric(y, b))
    if not np.isfinite(observed):
        raise ValueError("observed metric difference must be finite")

    cluster_indices = [np.flatnonzero(c == cluster) for cluster in clusters]

    def permuted_delta(swap_bits) -> float:
        pa = a.copy()
        pb = b.copy()
        for do_swap, idx in zip(swap_bits, cluster_indices):
            if do_swap:
                pa[idx], pb[idx] = b[idx], a[idx]
        value = float(metric(y, pa) - metric(y, pb))
        if not np.isfinite(value):
            raise ValueError("permuted metric difference must be finite")
        return value

    def extreme(value: float) -> bool:
        eps = np.finfo(float).eps * 16.0
        if alternative == "two-sided":
            return abs(value) >= abs(observed) - eps
        if alternative == "greater":
            return value >= observed - eps
        return value <= observed + eps

    if clusters.size <= exact_max_clusters:
        total = 0
        count = 0
        for bits in product((False, True), repeat=int(clusters.size)):
            value = permuted_delta(bits)
            total += 1
            count += int(extreme(value))
        return observed, float(count / total), "exact"

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        bits = rng.integers(0, 2, size=clusters.size, dtype=np.int8).astype(bool)
        count += int(extreme(permuted_delta(bits)))
    p_value = float((count + 1) / (n_permutations + 1))
    return observed, p_value, "monte-carlo"


def holm_adjust(p_values) -> np.ndarray:
    """Return Holm step-down family-wise-error adjusted p-values.

    Input p-values must be finite and in ``[0,1]``. The output preserves the
    original order. This function adjusts a *predeclared inferential family*; it
    does not decide which comparisons belong in that family.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("p_values must be a non-empty 1-D array")
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_values must be finite and lie in [0,1]")

    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    m = p.size
    adjusted_sorted = np.empty(m, dtype=np.float64)
    running = 0.0
    for i, value in enumerate(sorted_p):
        candidate = min(1.0, float((m - i) * value))
        running = max(running, candidate)
        adjusted_sorted[i] = running
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def holm_reject(p_values, alpha: float = 0.05) -> np.ndarray:
    """Return rejection mask using Holm-adjusted p-values at family-wise alpha."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    return holm_adjust(p_values) <= alpha
