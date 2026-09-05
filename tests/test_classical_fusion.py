import numpy as np
import pytest

from deepmm.fusion.classical import LogisticScoreFusion, WeightedScoreFusion, zscore_fit, zscore_transform
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


def test_weighted_fusion_prefers_stronger_modality():
    labels, scores = _toy_scores()
    fusion = WeightedScoreFusion(grid_step=0.1, objective="auc").fit(scores, labels)
    assert fusion.weights_[0] > fusion.weights_[1]
    fused = fusion.transform(scores)
    assert roc_auc(labels, fused) == pytest.approx(1.0)


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


def test_transform_before_fit_is_rejected():
    _, scores = _toy_scores()
    with pytest.raises(RuntimeError):
        WeightedScoreFusion().transform(scores)
    with pytest.raises(RuntimeError):
        LogisticScoreFusion().transform(scores)
