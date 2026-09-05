"""Pareto and rank-stability utilities for Q2/Q3.

Rows represent model/fusion families; columns represent locked evaluation
criteria. Criterion direction is explicit through a Boolean ``minimize`` vector.
"""

from __future__ import annotations

import numpy as np


def _matrix(values) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("values must be a 2-D matrix with >=2 methods and >=1 criterion")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must be finite")
    return x


def _directions(minimize, n_criteria: int) -> np.ndarray:
    d = np.asarray(minimize, dtype=bool)
    if d.ndim != 1 or d.size != n_criteria:
        raise ValueError("minimize must be a Boolean vector matching the criteria")
    return d


def _to_minimization(values: np.ndarray, minimize: np.ndarray) -> np.ndarray:
    signs = np.where(minimize, 1.0, -1.0)
    return values * signs


def non_dominated_mask(values, minimize, *, atol: float = 0.0) -> np.ndarray:
    """Return Boolean mask of Pareto non-dominated methods.

    ``atol`` is a predeclared numerical tolerance, not a practical-equivalence
    margin. A scientific equivalence margin, if used, must be defined separately
    before final testing.
    """
    if atol < 0.0:
        raise ValueError("atol must be >= 0")
    x = _matrix(values)
    d = _directions(minimize, x.shape[1])
    z = _to_minimization(x, d)
    keep = np.ones(z.shape[0], dtype=bool)
    for i in range(z.shape[0]):
        for j in range(z.shape[0]):
            if i == j:
                continue
            no_worse = np.all(z[j] <= z[i] + atol)
            strictly_better = np.any(z[j] < z[i] - atol)
            if no_worse and strictly_better:
                keep[i] = False
                break
    return keep


def bootstrap_dominance_probability(samples, minimize, *, atol: float = 0.0) -> np.ndarray:
    """Return pairwise probability that row method i Pareto-dominates method j.

    ``samples`` has shape ``(bootstrap_replicates, methods, criteria)`` and must be
    based on matched bootstrap replicates across all methods/criteria.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] < 2 or x.shape[1] < 2 or x.shape[2] < 1:
        raise ValueError("samples must have shape (replicates>=2, methods>=2, criteria>=1)")
    if not np.all(np.isfinite(x)):
        raise ValueError("samples must be finite")
    d = _directions(minimize, x.shape[2])
    z = _to_minimization(x, d)
    m = z.shape[1]
    result = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            no_worse = np.all(z[:, i, :] <= z[:, j, :] + atol, axis=1)
            strictly_better = np.any(z[:, i, :] < z[:, j, :] - atol, axis=1)
            result[i, j] = float(np.mean(no_worse & strictly_better))
    return result


def non_dominated_probability(samples, minimize, *, atol: float = 0.0) -> np.ndarray:
    """Return probability each method lies on the bootstrap Pareto frontier."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] < 2:
        raise ValueError("samples must have shape (replicates>=2, methods, criteria)")
    counts = np.zeros(x.shape[1], dtype=np.float64)
    for b in range(x.shape[0]):
        counts += non_dominated_mask(x[b], minimize, atol=atol)
    return counts / float(x.shape[0])


def kendall_tau_b(values_a, values_b, *, atol: float = 0.0) -> float:
    """Return Kendall tau-b between two model rankings, handling ties.

    The arrays contain comparable criterion values for the same methods. Direction
    does not need to be specified because reversing the direction of *both* arrays
    leaves pair concordance unchanged.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size < 2:
        raise ValueError("values_a and values_b must be same-length 1-D arrays with >=2 methods")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("ranking values must be finite")
    if atol < 0.0:
        raise ValueError("atol must be >= 0")

    concordant = discordant = ties_a = ties_b = 0
    for i in range(a.size - 1):
        for j in range(i + 1, a.size):
            da = a[i] - a[j]
            db = b[i] - b[j]
            sa = 0 if abs(da) <= atol else (1 if da > 0 else -1)
            sb = 0 if abs(db) <= atol else (1 if db > 0 else -1)
            if sa == 0 and sb == 0:
                continue
            if sa == 0:
                ties_a += 1
            elif sb == 0:
                ties_b += 1
            elif sa == sb:
                concordant += 1
            else:
                discordant += 1

    numerator = float(concordant - discordant)
    denom = np.sqrt(
        float(concordant + discordant + ties_a)
        * float(concordant + discordant + ties_b)
    )
    if denom == 0.0:
        return float("nan")
    return numerator / denom


def pairwise_rank_reversals(clean_values, stress_values, *, atol: float = 0.0) -> list[tuple[int, int]]:
    """Return method-index pairs whose ordering reverses from clean to stress.

    Ties in either condition are not called reversals. Direction may be lower- or
    higher-is-better as long as it is the same in both conditions.
    """
    clean = np.asarray(clean_values, dtype=np.float64)
    stress = np.asarray(stress_values, dtype=np.float64)
    if clean.ndim != 1 or stress.ndim != 1 or clean.shape != stress.shape or clean.size < 2:
        raise ValueError("clean_values and stress_values must be same-length 1-D arrays")
    if not (np.all(np.isfinite(clean)) and np.all(np.isfinite(stress))):
        raise ValueError("ranking values must be finite")
    if atol < 0.0:
        raise ValueError("atol must be >= 0")

    reversals: list[tuple[int, int]] = []
    for i in range(clean.size - 1):
        for j in range(i + 1, clean.size):
            dc = clean[i] - clean[j]
            ds = stress[i] - stress[j]
            if abs(dc) <= atol or abs(ds) <= atol:
                continue
            if dc * ds < 0.0:
                reversals.append((i, j))
    return reversals
