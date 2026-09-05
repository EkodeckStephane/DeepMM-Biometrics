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
    """Return empirical linearly interpolated ``(EER, threshold)``.

    This function intersects the ordinary empirical ROC polyline with FAR=FRR.
    ``eer_rocch`` is provided separately for the BOSARIS-style ROC-convex-hull
    interpretation used by parts of the biometric/speaker-verification literature.

    When the interpolated crossing has no single finite deterministic threshold
    (for example when all scores are tied), the threshold is returned as ``NaN``
    rather than fabricating an attainable operating point.
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
        if np.isfinite(thresholds[i]) and np.isfinite(thresholds[i + 1]):
            threshold = thresholds[i] + alpha * (thresholds[i + 1] - thresholds[i])
        else:
            threshold = np.nan
        return float(value), float(threshold)

    i = int(np.argmin(np.abs(diff)))
    value = 0.5 * (fpr[i] + fnr[i])
    return float(value), float(thresholds[i])


def _upper_roc_convex_hull(fpr: np.ndarray, tpr: np.ndarray) -> np.ndarray:
    """Return the upper-left ROC convex hull as increasing-FPR points.

    For equal FPR values only the maximum TPR is retained. Hull segment slopes
    are constrained to be non-increasing, which yields the concave upper ROC
    envelope relevant to randomized optimal operating points.
    """
    unique_x = np.unique(fpr)
    points = [(float(x), float(np.max(tpr[fpr == x]))) for x in unique_x]
    hull: list[tuple[float, float]] = []

    def slope(a: tuple[float, float], b: tuple[float, float]) -> float:
        dx = b[0] - a[0]
        if dx == 0.0:
            return np.inf
        return (b[1] - a[1]) / dx

    for point in points:
        while len(hull) >= 2:
            left_slope = slope(hull[-2], hull[-1])
            right_slope = slope(hull[-1], point)
            if right_slope > left_slope + 1e-15:
                hull.pop()
            else:
                break
        hull.append(point)
    return np.asarray(hull, dtype=np.float64)


def eer_rocch(labels, scores) -> float:
    """Return EER on the ROC convex hull (ROCCH).

    This follows the BOSARIS evaluation interpretation: the EER is the
    intersection of the upper ROC convex hull with ``TPR = 1 - FPR``. Hull
    segments can represent randomized mixtures of adjacent deterministic
    thresholds, so this function returns the EER value only.
    """
    y, s = _validated(labels, scores)
    fpr, tpr, _ = roc_curve(y, s, pos_label=1, drop_intermediate=False)
    hull = _upper_roc_convex_hull(fpr, tpr)

    for i in range(hull.shape[0] - 1):
        x0, y0 = hull[i]
        x1, y1 = hull[i + 1]
        d0 = x0 + y0 - 1.0
        d1 = x1 + y1 - 1.0
        if abs(d0) <= 1e-15:
            return float(x0)
        if d0 <= 0.0 <= d1:
            denom = d1 - d0
            if abs(denom) <= 1e-15:
                return float(0.5 * (x0 + (1.0 - y0)))
            alpha = -d0 / denom
            crossing_fpr = x0 + alpha * (x1 - x0)
            crossing_tpr = y0 + alpha * (y1 - y0)
            return float(0.5 * (crossing_fpr + (1.0 - crossing_tpr)))

    x, yv = hull[-1]
    if abs(x + yv - 1.0) <= 1e-15:
        return float(x)
    raise RuntimeError("ROC convex hull did not cross the equal-error line")


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
    i = int(candidates[np.argmax(fpr[candidates])])
    return float(tpr[i]), float(fpr[i]), float(thresholds[i])
