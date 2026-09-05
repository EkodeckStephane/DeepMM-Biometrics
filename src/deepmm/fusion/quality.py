"""Classical quality-aware score-fusion baseline.

The baseline deliberately separates *quality-aware fusion* from deep gating. It
uses externally supplied, predeclared quality values and tunes only a scalar
quality exponent on development data. Test labels are never accepted by
``transform``.
"""

from __future__ import annotations

import numpy as np

from deepmm.fusion.classical import _labels, _matrix, zscore_fit, zscore_transform
from deepmm.metrics.verification import eer, roc_auc


def _quality_matrix(quality, shape: tuple[int, int]) -> np.ndarray:
    q = np.asarray(quality, dtype=np.float64)
    if q.shape != shape:
        raise ValueError("quality must match the trials x modalities score matrix")
    if not np.all(np.isfinite(q)) or np.any(q < 0.0):
        raise ValueError("quality values must be finite and non-negative")
    if np.any(np.sum(q, axis=1) <= 0.0):
        raise ValueError("each trial must have at least one positive-quality modality")
    return q


class QualityWeightedScoreFusion:
    """Classical dynamic score fusion using per-trial modality quality.

    For normalized modality scores ``z_m`` and supplied qualities ``q_m``, the
    fused score is

    ``sum_m w_m z_m`` with ``w_m proportional to q_m ** gamma``.

    ``gamma`` is selected from a frozen finite grid on development data. This is a
    deliberately low-capacity comparator for a learned quality-aware/gating model.
    It should receive the same quality variables as any deep gate used in a
    headline comparison.

    Missing modalities are **not** encoded implicitly through NaN or absent score
    values. The missing-modality benchmark uses a separate explicit policy.
    """

    def __init__(
        self,
        *,
        gamma_grid=(0.0, 0.5, 1.0, 2.0, 4.0),
        objective: str = "eer",
        normalize: bool = True,
    ):
        grid = np.asarray(gamma_grid, dtype=np.float64)
        if grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid)):
            raise ValueError("gamma_grid must be a non-empty finite 1-D sequence")
        if np.any(grid < 0.0):
            raise ValueError("gamma values must be non-negative")
        if self._has_duplicates(grid):
            raise ValueError("gamma_grid must not contain duplicates")
        if objective not in {"eer", "auc"}:
            raise ValueError("objective must be 'eer' or 'auc'")
        self.gamma_grid = tuple(float(v) for v in grid)
        self.objective = objective
        self.normalize = bool(normalize)

    @staticmethod
    def _has_duplicates(values: np.ndarray) -> bool:
        return np.unique(values).size != values.size

    @staticmethod
    def _weights(quality: np.ndarray, gamma: float) -> np.ndarray:
        if gamma == 0.0:
            powered = np.ones_like(quality, dtype=np.float64)
        else:
            powered = np.power(quality, gamma)
        denom = np.sum(powered, axis=1, keepdims=True)
        if np.any(denom <= 0.0):
            raise ValueError("quality weighting produced an all-zero trial")
        return powered / denom

    def fit(self, scores, quality, labels) -> "QualityWeightedScoreFusion":
        x = _matrix(scores)
        q = _quality_matrix(quality, x.shape)
        y = _labels(labels, x.shape[0])

        if self.normalize:
            self.mean_, self.scale_ = zscore_fit(x)
            z = zscore_transform(x, self.mean_, self.scale_)
        else:
            self.mean_ = None
            self.scale_ = None
            z = x

        best_key = None
        best_gamma = None
        for gamma in self.gamma_grid:
            weights = self._weights(q, gamma)
            fused = np.sum(weights * z, axis=1)
            if self.objective == "eer":
                value, _ = eer(y, fused)
                key = (value, gamma)
            else:
                value = roc_auc(y, fused)
                key = (-value, gamma)
            if best_key is None or key < best_key:
                best_key = key
                best_gamma = gamma

        self.gamma_ = float(best_gamma)
        self.n_modalities_ = int(x.shape[1])
        return self

    def transform(self, scores, quality) -> np.ndarray:
        if not hasattr(self, "gamma_"):
            raise RuntimeError("fit must be called before transform")
        x = _matrix(scores)
        if x.shape[1] != self.n_modalities_:
            raise ValueError("modality dimension differs from fitted model")
        q = _quality_matrix(quality, x.shape)
        z = zscore_transform(x, self.mean_, self.scale_) if self.normalize else x
        weights = self._weights(q, self.gamma_)
        return np.sum(weights * z, axis=1)

    def quality_weights(self, quality) -> np.ndarray:
        """Return frozen per-trial quality weights for audit/interpretability."""
        if not hasattr(self, "gamma_"):
            raise RuntimeError("fit must be called before quality_weights")
        q = np.asarray(quality, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != self.n_modalities_:
            raise ValueError("quality must be a trials x fitted-modalities matrix")
        q = _quality_matrix(q, q.shape)
        return self._weights(q, self.gamma_)
