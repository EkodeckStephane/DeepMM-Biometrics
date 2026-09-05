"""Verification metrics with explicit score-direction conventions.

All functions assume larger scores indicate stronger evidence for a genuine match.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _validated(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or y.shape[0] != s.shape[0]:
        raise ValueError("labels and scores must be 1-D arrays of equal length")
    if y.size == 0:
        raise ValueError("empty score vector")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    values = set(np.unique(y).tolist())
    if values != {0, 1}:
        raise ValueError("labels must contain both 0 (impostor) and 1 (genuine)")
    return y, s


def roc_auc(labels, scores) -> float:
    """Return ROC-AUC for genuine=1, impostor=0 trials."""
    y, s = _validated(labels, scores)
    return float(roc_auc_score(y, s))


def eer(labels, scores) -> tuple[float, float]:
    """Return (EER, threshold) using linear interpolation at FAR=FRR.

    The threshold is linearly interpolated between adjacent ROC operating points.
    If an exact crossing is present, that operating point is returned directly.
    """
    y, s = _validated(labels, scores)
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1, drop_intermediate=False)
    fnr = 1.0 - tpr
    diff = fpr - fnr

    exact = np.flatnonzero(np.isclose(diff, 0.0, rtol=0.0, atol=1e-15))
    if exact.size:
        i = int(exact[0])
        return float(fpr[i]), float(thresholds[i])

    crossing = np.flatnonzero(diff[:-1] * diff[1:] < 0.0)
    if crossing.size:
        i = int(crossing[0])
        d0, d1 = diff[i], diff[i + 1]
        alpha = -d0 / (d1 - d0)
        value = fpr[i] + alpha * (fpr[i + 1] - fpr[i])
        threshold = thresholds[i] + alpha * (thresholds[i + 1] - thresholds[i])
        return float(value), float(threshold)

    # Discrete ROC curves can occasionally jump over the crossing numerically.
    i = int(np.argmin(np.abs(diff)))
    value = 0.5 * (fpr[i] + fnr[i])
    return float(value), float(thresholds[i])


def tar_at_far(labels, scores, target_far: float) -> tuple[float, float, float]:
    """Return conservative (TAR, achieved_FAR, threshold) at target FAR.

    The selected ROC point is the one with the largest TAR among observed points
    satisfying FAR <= target_far. No interpolation to an unattained FAR is used.
    This avoids optimistic reporting at very low FAR when the finite impostor
    trial count cannot resolve the requested operating point.
    """
    if not 0.0 <= target_far <= 1.0:
        raise ValueError("target_far must be in [0, 1]")
    y, s = _validated(labels, scores)
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1, drop_intermediate=False)
    eligible = np.flatnonzero(fpr <= target_far + np.finfo(float).eps)
    if eligible.size == 0:
        raise RuntimeError("no ROC operating point satisfies the requested FAR")
    best_tar = np.max(tpr[eligible])
    candidates = eligible[np.flatnonzero(np.isclose(tpr[eligible], best_tar))]
    # Prefer the highest achieved FAR among equal-TAR candidates: it is the least
    # unnecessarily conservative threshold while remaining inside the FAR budget.
    i = int(candidates[np.argmax(fpr[candidates])])
    return float(tpr[i]), float(fpr[i]), float(thresholds[i])
