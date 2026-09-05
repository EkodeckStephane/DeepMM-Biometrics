"""Classical score-fusion baselines for controlled multimodal verification.

All fitting methods are intentionally explicit: callers must supply development /
validation scores. Test labels must never be passed to ``fit`` in the benchmark.
Higher scores are assumed to indicate stronger evidence for a genuine match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from deepmm.metrics.verification import eer, roc_auc


def _matrix(scores) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("scores must be a 2-D array: trials x modalities")
    if x.shape[0] == 0 or x.shape[1] < 2:
        raise ValueError("scores must contain at least one trial and two modalities")
    if not np.all(np.isfinite(x)):
        raise ValueError("scores must be finite; missing modalities require an explicit policy")
    return x


def _labels(labels, n: int) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int8)
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError("labels must be a 1-D array matching the number of trials")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("labels must contain both 0 (impostor) and 1 (genuine)")
    return y


def zscore_fit(scores) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-modality mean/std on development scores only."""
    x = _matrix(scores)
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    if np.any(scale <= np.finfo(np.float64).eps):
        raise ValueError("each modality must have non-zero score variance")
    return mean, scale


def zscore_transform(scores, mean, scale) -> np.ndarray:
    """Apply frozen per-modality z-score normalization."""
    x = _matrix(scores)
    mu = np.asarray(mean, dtype=np.float64)
    sd = np.asarray(scale, dtype=np.float64)
    if mu.shape != (x.shape[1],) or sd.shape != (x.shape[1],):
        raise ValueError("normalization vectors must match the modality dimension")
    if np.any(sd <= 0) or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sd)):
        raise ValueError("invalid normalization parameters")
    return (x - mu) / sd


@dataclass
class WeightedScoreFusion:
    """Deterministic non-negative weighted score fusion.

    ``fit`` performs a simplex grid search on development data. The grid and
    objective must be frozen before final test evaluation. For two modalities,
    ``grid_step=0.05`` yields weights [0, .05, ..., 1]. For more modalities the
    class deliberately raises ``NotImplementedError`` rather than silently use an
    uncontrolled optimizer; the primary DeepMM benchmark currently targets two
    modalities.
    """

    grid_step: float = 0.05
    objective: str = "eer"
    normalize: bool = True

    def fit(self, scores, labels) -> "WeightedScoreFusion":
        x = _matrix(scores)
        y = _labels(labels, x.shape[0])
        if x.shape[1] != 2:
            raise NotImplementedError("current controlled grid search supports exactly two modalities")
        if not (0.0 < self.grid_step <= 1.0):
            raise ValueError("grid_step must be in (0, 1]")
        if self.objective not in {"eer", "auc"}:
            raise ValueError("objective must be 'eer' or 'auc'")

        if self.normalize:
            self.mean_, self.scale_ = zscore_fit(x)
            z = zscore_transform(x, self.mean_, self.scale_)
        else:
            self.mean_ = None
            self.scale_ = None
            z = x

        n_steps = int(round(1.0 / self.grid_step))
        weights = np.linspace(0.0, 1.0, n_steps + 1)
        best_key = None
        best_weight = None
        for w in weights:
            fused = w * z[:, 0] + (1.0 - w) * z[:, 1]
            if self.objective == "eer":
                value, _ = eer(y, fused)
                key = (value, abs(w - 0.5), w)
            else:
                value = roc_auc(y, fused)
                key = (-value, abs(w - 0.5), w)
            if best_key is None or key < best_key:
                best_key = key
                best_weight = float(w)

        self.weight_ = best_weight
        self.weights_ = np.array([best_weight, 1.0 - best_weight], dtype=np.float64)
        return self

    def transform(self, scores) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("fit must be called before transform")
        x = _matrix(scores)
        if x.shape[1] != 2:
            raise ValueError("modality dimension differs from fitted model")
        z = zscore_transform(x, self.mean_, self.scale_) if self.normalize else x
        return z @ self.weights_

    def fit_transform(self, scores, labels) -> np.ndarray:
        return self.fit(scores, labels).transform(scores)


class LogisticScoreFusion:
    """Regularized logistic-regression score fusion.

    This is a **classical learned fusion baseline**, not a deep-learning method.
    The returned decision score is the logistic log-odds, preserving a natural
    higher-is-more-genuine convention for verification metrics.
    """

    def __init__(self, *, C: float = 1.0, normalize: bool = True, max_iter: int = 2000):
        if C <= 0:
            raise ValueError("C must be positive")
        self.C = float(C)
        self.normalize = bool(normalize)
        self.max_iter = int(max_iter)

    def fit(self, scores, labels) -> "LogisticScoreFusion":
        x = _matrix(scores)
        y = _labels(labels, x.shape[0])
        if self.normalize:
            self.mean_, self.scale_ = zscore_fit(x)
            z = zscore_transform(x, self.mean_, self.scale_)
        else:
            self.mean_ = None
            self.scale_ = None
            z = x
        self.model_ = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=0,
        )
        self.model_.fit(z, y)
        return self

    def transform(self, scores) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before transform")
        x = _matrix(scores)
        z = zscore_transform(x, self.mean_, self.scale_) if self.normalize else x
        return np.asarray(self.model_.decision_function(z), dtype=np.float64)

    def predict_proba(self, scores) -> np.ndarray:
        """Return genuine-class probability from the fitted development model."""
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before predict_proba")
        x = _matrix(scores)
        z = zscore_transform(x, self.mean_, self.scale_) if self.normalize else x
        return np.asarray(self.model_.predict_proba(z)[:, 1], dtype=np.float64)
