"""Calibration metrics for held-out binary verification trials.

These functions evaluate probabilities already produced by a calibration stage.
They do not fit a calibrator and therefore cannot accidentally consume test labels
for model fitting.
"""

from __future__ import annotations

import numpy as np


def _validated(labels, probabilities) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or y.shape[0] != p.shape[0]:
        raise ValueError("labels and probabilities must be 1-D arrays of equal length")
    if y.size == 0:
        raise ValueError("empty probability vector")
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("labels must be binary")
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    return y, p


def brier_score(labels, probabilities) -> float:
    """Mean squared probability error (lower is better)."""
    y, p = _validated(labels, probabilities)
    return float(np.mean((p - y) ** 2))


def negative_log_likelihood(labels, probabilities, eps: float = 1e-15) -> float:
    """Binary log loss / negative log-likelihood in nats (lower is better)."""
    y, p = _validated(labels, probabilities)
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must lie in (0, 0.5)")
    p = np.clip(p, eps, 1.0 - eps)
    loss = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    return float(np.mean(loss))


def expected_calibration_error(labels, probabilities, n_bins: int = 15) -> float:
    """Equal-width Expected Calibration Error.

    The binning rule and number of bins must be frozen before final test analysis.
    Empty bins contribute zero weight.
    """
    y, p = _validated(labels, probabilities)
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Probabilities exactly equal to 1 belong to the last bin.
    bins = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
    ece = 0.0
    n = float(y.size)
    for b in range(n_bins):
        mask = bins == b
        if not np.any(mask):
            continue
        confidence = float(np.mean(p[mask]))
        frequency = float(np.mean(y[mask]))
        ece += (float(np.sum(mask)) / n) * abs(confidence - frequency)
    return float(ece)
