import numpy as np
import pytest

from deepmm.fusion.quality import QualityWeightedScoreFusion
from deepmm.metrics.verification import roc_auc


def test_quality_weighting_beats_equal_quality_when_one_modality_is_reversed():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    good = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 1.0])
    bad = good[::-1]
    scores = np.column_stack([good, bad])
    quality = np.column_stack([np.ones(labels.size), np.full(labels.size, 0.1)])

    model = QualityWeightedScoreFusion(gamma_grid=(0.0, 1.0, 2.0), objective="auc")
    model.fit(scores, quality, labels)
    fused = model.transform(scores, quality)

    assert model.gamma_ == pytest.approx(1.0)
    assert roc_auc(labels, fused) == pytest.approx(1.0)
    weights = model.quality_weights(quality)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.all(weights[:, 0] > weights[:, 1])


def test_gamma_zero_is_equal_weighting_independent_of_quality_magnitude():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([[0.1, 0.2], [0.2, 0.3], [0.8, 0.7], [0.9, 0.8]])
    quality = np.array([[1.0, 0.01], [0.5, 2.0], [0.2, 0.8], [10.0, 1.0]])
    model = QualityWeightedScoreFusion(gamma_grid=(0.0,), objective="eer").fit(scores, quality, labels)
    weights = model.quality_weights(quality)
    assert np.allclose(weights, 0.5)


def test_quality_validation_rejects_negative_and_all_zero_rows():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([[0.1, 0.2], [0.2, 0.3], [0.8, 0.7], [0.9, 0.8]])

    with pytest.raises(ValueError, match="non-negative"):
        QualityWeightedScoreFusion().fit(scores, [[1, 1], [1, -1], [1, 1], [1, 1]], labels)

    with pytest.raises(ValueError, match="positive-quality"):
        QualityWeightedScoreFusion().fit(scores, [[1, 1], [0, 0], [1, 1], [1, 1]], labels)
