"""Calibration metrics for held-out binary verification trials.

Probability metrics evaluate probabilities already produced by a calibration stage.
The likelihood-ratio metrics follow the BOSARIS convention: larger log-likelihood
ratios support the genuine/target hypothesis.

No function in this module fits a *parametric* calibrator on test data. ``min_cllr``
is an evaluation statistic: it uses an isotonic/PAV-equivalent monotonic mapping on
the evaluated scores to estimate the best calibration achievable without changing
their ranking. It must not be confused with a deployable held-out calibrator.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


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


def _score_vector(values, name: str, *, allow_inf: bool = False) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if np.any(np.isnan(x)) or (not allow_inf and not np.all(np.isfinite(x))):
        requirement = "must not contain NaN" if allow_inf else "must be finite"
        raise ValueError(f"{name} {requirement}")
    return x


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


def cllr(non_target_llrs, target_llrs) -> float:
    """Return BOSARIS cost of log likelihood ratio ``C_llr``.

    Inputs must already be **natural-log likelihood ratios**. Positive values
    support the target/genuine hypothesis; negative values support non-target.
    The result is expressed in bits because the BOSARIS definition uses log base 2.

    ``+/-inf`` are accepted so that a mathematically perfect monotonic calibration
    can have zero cost; NaN is always rejected.
    """
    non = _score_vector(non_target_llrs, "non_target_llrs", allow_inf=True)
    tar = _score_vector(target_llrs, "target_llrs", allow_inf=True)
    ln2 = np.log(2.0)
    target_cost = np.mean(np.logaddexp(0.0, -tar) / ln2)
    non_target_cost = np.mean(np.logaddexp(0.0, non) / ln2)
    return float(0.5 * (target_cost + non_target_cost))


def min_cllr(non_target_scores, target_scores) -> float:
    """Return minimum ``C_llr`` obtainable by a monotonic score mapping.

    This is the discrimination component used in BOSARIS-style calibration
    analysis. Scores only need to obey the convention "higher = more target-like";
    they do **not** need to be calibrated LLRs.

    The implementation uses isotonic regression, which is the pool-adjacent-
    violators solution for the monotonic posterior mapping. Tied raw scores are
    handled jointly by the isotonic fit rather than being artificially ordered by
    class label.
    """
    non = _score_vector(non_target_scores, "non_target_scores")
    tar = _score_vector(target_scores, "target_scores")
    scores = np.concatenate([non, tar])
    labels = np.concatenate([
        np.zeros(non.size, dtype=np.float64),
        np.ones(tar.size, dtype=np.float64),
    ])

    isotonic = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
        y_min=0.0,
        y_max=1.0,
    )
    posterior = np.asarray(isotonic.fit_transform(scores, labels), dtype=np.float64)

    prior = float(tar.size) / float(tar.size + non.size)
    log_prior_odds = np.log(prior) - np.log1p(-prior)
    with np.errstate(divide="ignore", invalid="ignore"):
        posterior_log_odds = np.log(posterior) - np.log1p(-posterior)
    llrs = posterior_log_odds - log_prior_odds

    return cllr(llrs[: non.size], llrs[non.size :])


def cllr_calibration_loss(non_target_llrs, target_llrs) -> float:
    """Return ``C_llr_cal = C_llr - C_llr_min``.

    ``C_llr`` assesses the supplied calibrated LLRs; ``C_llr_min`` removes
    monotonic calibration error while preserving ranking. Small negative values
    caused only by floating-point roundoff are clipped to zero.
    """
    non = _score_vector(non_target_llrs, "non_target_llrs", allow_inf=True)
    tar = _score_vector(target_llrs, "target_llrs", allow_inf=True)
    actual = cllr(non, tar)
    # min_cllr requires finite raw scores. Actual deployed LLRs should be finite;
    # perfect +/-inf toy cases have zero calibration loss by construction.
    if not (np.all(np.isfinite(non)) and np.all(np.isfinite(tar))):
        if actual == 0.0:
            return 0.0
        raise ValueError("calibration loss requires finite LLRs unless C_llr is exactly zero")
    minimum = min_cllr(non, tar)
    loss = actual - minimum
    if loss < 0.0 and abs(loss) <= 1e-12:
        loss = 0.0
    return float(loss)
