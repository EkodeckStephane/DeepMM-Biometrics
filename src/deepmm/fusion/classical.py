"""Classical score-fusion baselines for controlled multimodal verification.

All fitting methods are intentionally explicit: callers must supply development /
validation scores. Test labels must never be passed to a trainable ``fit`` method
in the benchmark. Higher scores are assumed to indicate stronger evidence for a
genuine match.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

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


class EqualScoreFusion:
    """Equal-weight fusion after development-only score normalization.

    This is the minimum-complexity multimodal reference. ``fit`` uses no class
    labels; it only freezes normalization statistics from development scores.
    """

    def __init__(self, *, normalize: bool = True):
        self.normalize = bool(normalize)

    def fit(self, scores) -> "EqualScoreFusion":
        x = _matrix(scores)
        self.n_modalities_ = int(x.shape[1])
        if self.normalize:
            self.mean_, self.scale_ = zscore_fit(x)
        else:
            self.mean_ = None
            self.scale_ = None
        return self

    def transform(self, scores) -> np.ndarray:
        if not hasattr(self, "n_modalities_"):
            raise RuntimeError("fit must be called before transform")
        x = _matrix(scores)
        if x.shape[1] != self.n_modalities_:
            raise ValueError("modality dimension differs from fitted model")
        z = zscore_transform(x, self.mean_, self.scale_) if self.normalize else x
        return np.mean(z, axis=1)

    def fit_transform(self, scores) -> np.ndarray:
        return self.fit(scores).transform(scores)


def _integer_compositions(total: int, parts: int):
    """Yield deterministic non-negative integer compositions of total."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _integer_compositions(total - first, parts - 1):
            yield (first,) + rest


@dataclass
class WeightedScoreFusion:
    """Deterministic non-negative simplex-weighted score fusion.

    ``fit`` performs an exhaustive simplex grid search on development data. The
    grid and objective must be frozen before final test evaluation. ``grid_step``
    must exactly divide 1 (within floating-point tolerance). A candidate-count cap
    prevents an accidental combinatorial search when many modalities are used.
    """

    grid_step: float = 0.05
    objective: str = "eer"
    normalize: bool = True
    max_candidates: int = 10000

    def fit(self, scores, labels) -> "WeightedScoreFusion":
        x = _matrix(scores)
        y = _labels(labels, x.shape[0])
        if not (0.0 < self.grid_step <= 1.0):
            raise ValueError("grid_step must be in (0, 1]")
        if self.objective not in {"eer", "auc"}:
            raise ValueError("objective must be 'eer' or 'auc'")
        if not isinstance(self.max_candidates, int) or self.max_candidates < 1:
            raise ValueError("max_candidates must be an integer >= 1")

        n_steps = int(round(1.0 / self.grid_step))
        if n_steps < 1 or not np.isclose(n_steps * self.grid_step, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("grid_step must divide 1 exactly, e.g. 0.1, 0.05, 0.02")
        n_candidates = comb(n_steps + x.shape[1] - 1, x.shape[1] - 1)
        if n_candidates > self.max_candidates:
            raise ValueError(
                f"simplex grid would contain {n_candidates} candidates; "
                "increase grid_step or max_candidates deliberately"
            )

        if self.normalize:
            self.mean_, self.scale_ = zscore_fit(x)
            z = zscore_transform(x, self.mean_, self.scale_)
        else:
            self.mean_ = None
            self.scale_ = None
            z = x

        equal = np.full(x.shape[1], 1.0 / x.shape[1], dtype=np.float64)
        best_key = None
        best_weights = None
        for composition in _integer_compositions(n_steps, x.shape[1]):
            weights = np.asarray(composition, dtype=np.float64) / float(n_steps)
            fused = z @ weights
            if self.objective == "eer":
                value, _ = eer(y, fused)
                primary = value
            else:
                value = roc_auc(y, fused)
                primary = -value
            # Prefer the simpler/less extreme solution on exact metric ties, then
            # a deterministic lexicographic ordering for reproducibility.
            key = (primary, float(np.sum(np.abs(weights - equal))), tuple(weights.tolist()))
            if best_key is None or key < best_key:
                best_key = key
                best_weights = weights

        self.weights_ = np.asarray(best_weights, dtype=np.float64)
        self.n_modalities_ = int(x.shape[1])
        # Compatibility convenience for the common two-modality case.
        self.weight_ = float(self.weights_[0]) if self.n_modalities_ == 2 else None
        self.n_candidates_ = int(n_candidates)
        return self

    def transform(self, scores) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("fit must be called before transform")
        x = _matrix(scores)
        if x.shape[1] != self.n_modalities_:
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
        self.n_modalities_ = int(x.shape[1])
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
        if x.shape[1] != self.n_modalities_:
            raise ValueError("modality dimension differs from fitted model")
        z = zscore_transform(x, self.mean_, self.scale_) if self.normalize else x
        return np.asarray(self.model_.decision_function(z), dtype=np.float64)

    def predict_proba(self, scores) -> np.ndarray:
        """Return genuine-class probability from the fitted development model."""
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before predict_proba")
        x = _matrix(scores)
        if x.shape[1] != self.n_modalities_:
            raise ValueError("modality dimension differs from fitted model")
        z = zscore_transform(x, self.mean_, self.scale_) if self.normalize else x
        return np.asarray(self.model_.predict_proba(z)[:, 1], dtype=np.float64)
