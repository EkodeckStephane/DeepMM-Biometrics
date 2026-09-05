import numpy as np
import pytest

from deepmm.metrics.verification import eer, roc_auc, tar_at_far


def test_perfect_separation_has_zero_eer_and_unit_auc():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    value, threshold = eer(labels, scores)
    assert value == pytest.approx(0.0)
    assert threshold >= 0.8
    assert roc_auc(labels, scores) == pytest.approx(1.0)


def test_reversed_separation_has_unit_eer_and_zero_auc():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    value, _ = eer(labels, scores)
    assert value == pytest.approx(1.0)
    assert roc_auc(labels, scores) == pytest.approx(0.0)


def test_tar_at_far_is_conservative():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.4, 0.6, 0.8, 0.9])
    tar, achieved_far, threshold = tar_at_far(labels, scores, target_far=0.0)
    assert achieved_far <= 0.0
    assert 0.0 <= tar <= 1.0
    assert np.isfinite(threshold)


def test_invalid_labels_are_rejected():
    with pytest.raises(ValueError):
        eer([1, 1], [0.1, 0.2])
