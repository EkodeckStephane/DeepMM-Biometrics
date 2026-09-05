import numpy as np
import pytest

from deepmm.fusion.classical import (
    EqualScoreFusion,
    LogisticScoreFusion,
    WeightedScoreFusion,
    zscore_fit,
    zscore_transform,
)
from deepmm.metrics.verification import roc_auc


def _toy_scores():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # modality 0 is strongly discriminative; modality 1 is mostly noise/reversed.
    scores = np.array(
        [
            [0.10, 0.80],
            [0.20, 0.70],
            [0.15, 0.60],
            [0.25, 0.55],
            [0.75, 0.45],
            [0.80, 0.35],
            [0.90, 0.30],
            [0.85, 0.20],
        ]
    )
    return labels, scores


def test_zscore_uses_frozen_parameters():
    _, scores = _toy_scores()
    mean, scale = zscore_fit(scores)
    z = zscore_transform(scores, mean, scale)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-12)


def test_equal_fusion_is_arithmetic_mean_after_frozen_normalization():
    _, scores = _toy_scores()
    fusion = EqualScoreFusion().fit(scores)
    expected = zscore_transform(scores, fusion.mean_, fusion.scale_).mean(axis=1)
    assert fusion.transform(scores) == pytest.approx(expected)


def test_weighted_fusion_prefers_stronger_modality():
    labels, scores = _toy_scores()
    fusion = WeightedScoreFusion(grid_step=0.1, objective="auc").fit(scores, labels)
    assert fusion.weights_[0] > fusion.weights_[1]
    assert fusion.n_candidates_ == 11
    fused = fusion.transform(scores)
    assert roc_auc(labels, fused) == pytest.approx(1.0)


def test_weighted_fusion_supports_three_modality_simplex():
    labels, scores = _toy_scores()
    third = 0.5 * scores[:, 0] + 0.5 * np.linspace(0.0, 0.1, labels.size)
    x = np.column_stack([scores, third])
    fusion = WeightedScoreFusion(grid_step=0.5, objective="auc").fit(x, labels)
    assert fusion.weights_.shape == (3,)
    assert fusion.weights_.sum() == pytest.approx(1.0)
    # Integer compositions of 2 units into 3 parts: C(4,2)=6.
    assert fusion.n_candidates_ == 6
    assert roc_auc(labels, fusion.transform(x)) == pytest.approx(1.0)


def test_logistic_score_fusion_returns_ordered_scores_and_probabilities():
    labels, scores = _toy_scores()
    fusion = LogisticScoreFusion().fit(scores, labels)
    decision = fusion.transform(scores)
    probability = fusion.predict_proba(scores)
    assert decision.shape == labels.shape
    assert probability.shape == labels.shape
    assert np.all((probability > 0.0) & (probability < 1.0))
    assert roc_auc(labels, decision) == pytest.approx(1.0)


def test_missing_modalities_require_explicit_policy():
    labels, scores = _toy_scores()
    scores = scores.copy()
    scores[0, 1] = np.nan
    with pytest.raises(ValueError, match="explicit policy"):
        WeightedScoreFusion().fit(scores, labels)


def test_invalid_simplex_grid_is_rejected():
    labels, scores = _toy_scores()
    with pytest.raises(ValueError, match="divide 1 exactly"):
        WeightedScoreFusion(grid_step=0.3).fit(scores, labels)


def test_transform_before_fit_is_rejected():
    _, scores = _toy_scores()
    with pytest.raises(RuntimeError):
        EqualScoreFusion().transform(scores)
    with pytest.raises(RuntimeError):
        WeightedScoreFusion().transform(scores)
    with pytest.raises(RuntimeError):
        LogisticScoreFusion().transform(scores)
