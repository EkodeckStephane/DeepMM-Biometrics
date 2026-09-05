"""Held-out affine logistic calibration for biometric verification scores.

The calibrator is fitted on development/calibration labels only and transforms raw
verification scores into approximate natural-log likelihood ratios (LLRs). It uses
class-balanced sample weights so the fitted logistic decision function corresponds
to an effective target prior of 0.5; under that prior posterior log-odds equal LLR.

The regularization strength is intentionally a required constructor argument. It
must be frozen/tuned without test labels rather than hidden behind an arbitrary
benchmark default.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _scores(values) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("scores must be a non-empty 1-D array")
    if not np.all(np.isfinite(x)):
        raise ValueError("scores must be finite")
    return x


def _labels(values, n: int) -> np.ndarray:
    y = np.asarray(values)
    if y.ndim != 1 or y.size != n:
        raise ValueError("labels must be a 1-D array matching scores")
    if np.issubdtype(y.dtype, np.bool_):
        y = y.astype(np.int8)
    try:
        y = y.astype(np.int8)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must contain only 0 and 1") from exc
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("labels must contain both 0 (impostor) and 1 (genuine)")
    return y


def posterior_probability_from_llr(llrs, *, target_prior: float) -> np.ndarray:
    """Convert natural-log LLRs to target posterior probability at a fixed prior."""
    if not 0.0 < target_prior < 1.0:
        raise ValueError("target_prior must lie strictly between 0 and 1")
    x = _scores(llrs)
    log_prior_odds = np.log(target_prior) - np.log1p(-target_prior)
    z = x + log_prior_odds
    # Stable sigmoid without scipy dependency.
    out = np.empty_like(z)
    positive = z >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    out[~positive] = ez / (1.0 + ez)
    return out


class LogisticLLRCalibrator:
    """Affine regularized logistic score-to-LLR calibration.

    Parameters
    ----------
    C:
        Inverse L2 regularization strength passed to scikit-learn logistic
        regression. The benchmark configuration must lock this value before final
        test evaluation.

    Notes
    -----
    Repository score convention is ``higher = more genuine``. A fitted non-positive
    slope is rejected because silently flipping a system's ranking at calibration
    time would mix discrimination repair with calibration assessment.
    """

    def __init__(self, *, C: float):
        if not np.isfinite(C) or C <= 0.0:
            raise ValueError("C must be a finite positive number")
        self.C = float(C)

    def fit(self, scores, labels) -> "LogisticLLRCalibrator":
        x = _scores(scores)
        y = _labels(labels, x.size)

        n = y.size
        n_non = int(np.sum(y == 0))
        n_tar = int(np.sum(y == 1))
        # Each class contributes total weight n/2. This keeps the objective scale
        # comparable to an ordinary n-sample fit while imposing effective prior .5.
        sample_weight = np.where(
            y == 1,
            n / (2.0 * n_tar),
            n / (2.0 * n_non),
        )

        model = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=5000,
            random_state=0,
        )
        model.fit(x.reshape(-1, 1), y, sample_weight=sample_weight)
        slope = float(model.coef_[0, 0])
        intercept = float(model.intercept_[0])
        if not np.isfinite(slope) or not np.isfinite(intercept):
            raise RuntimeError("calibration fit produced non-finite parameters")
        if slope <= 0.0:
            raise ValueError(
                "calibration slope is non-positive; fix score orientation/discrimination before calibration"
            )

        self.model_ = model
        self.slope_ = slope
        self.intercept_ = intercept
        return self

    def transform(self, scores) -> np.ndarray:
        """Transform scores to natural-log LLRs without consuming labels."""
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before transform")
        x = _scores(scores)
        return np.asarray(self.model_.decision_function(x.reshape(-1, 1)), dtype=np.float64)

    def predict_probability(self, scores, *, target_prior: float) -> np.ndarray:
        """Return posterior target probabilities for a predeclared target prior."""
        return posterior_probability_from_llr(
            self.transform(scores), target_prior=target_prior
        )
