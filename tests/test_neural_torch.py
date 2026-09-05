import pytest


torch = pytest.importorskip("torch")

from deepmm.fusion.neural_torch import (
    FeatureFusionMLP,
    FeatureQualityGate,
    ScoreMLPFusion,
    ScoreQualityGate,
    parameter_count,
)


def test_score_mlp_shape_and_parameter_count():
    model = ScoreMLPFusion(2, (8, 4), activation="relu")
    x = torch.tensor([[0.2, 0.4], [0.9, -0.1], [-0.5, 0.2]], dtype=torch.float32)
    out = model(x)
    assert out.shape == (3,)
    assert parameter_count(model) > 0


def test_feature_fusion_uses_shared_encoder_and_returns_cosine_scores():
    torch.manual_seed(1)
    model = FeatureFusionMLP((3, 5), (7,), fused_dim=4)
    enrollment = [torch.randn(6, 3), torch.randn(6, 5)]
    probe = [block.clone() for block in enrollment]
    scores = model(enrollment, probe)
    assert scores.shape == (6,)
    assert torch.allclose(scores, torch.ones_like(scores), atol=1e-5)


def test_score_quality_gate_renormalizes_over_available_modalities():
    torch.manual_seed(2)
    model = ScoreQualityGate(2, (4,))
    scores = torch.tensor([[0.8, 0.0], [0.2, 0.6]], dtype=torch.float32)
    quality = torch.tensor([[0.9, 0.0], [0.4, 0.8]], dtype=torch.float32)
    availability = torch.tensor([[True, False], [True, True]])
    weights = model.weights(quality, availability)
    assert weights.shape == (2, 2)
    assert weights[0, 0] == pytest.approx(1.0)
    assert weights[0, 1] == pytest.approx(0.0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)
    out = model(scores, quality, availability)
    assert out.shape == (2,)


def test_score_gate_rejects_hidden_missingness():
    model = ScoreQualityGate(2, (4,))
    scores = torch.tensor([[0.8, 99.0]], dtype=torch.float32)
    quality = torch.tensor([[0.9, 0.0]], dtype=torch.float32)
    availability = torch.tensor([[True, False]])
    with pytest.raises(ValueError, match="canonical zero placeholder"):
        model(scores, quality, availability)


def test_feature_quality_gate_supports_heterogeneous_dimensions_and_missingness():
    torch.manual_seed(3)
    model = FeatureQualityGate((3, 5), projection_dim=4, gate_hidden_dims=(6,))
    enrollment = [torch.randn(4, 3), torch.randn(4, 5)]
    probe = [torch.randn(4, 3), torch.randn(4, 5)]
    quality = torch.tensor(
        [[0.8, 0.7], [0.9, 0.0], [0.0, 0.6], [0.5, 0.4]], dtype=torch.float32
    )
    enroll_availability = torch.tensor(
        [[True, True], [True, True], [False, True], [True, True]]
    )
    probe_availability = torch.tensor(
        [[True, True], [True, False], [False, True], [True, True]]
    )
    # Canonical all-zero unavailable embeddings.
    enrollment[0][2] = 0.0
    probe[0][2] = 0.0
    probe[1][1] = 0.0

    scores = model(
        enrollment,
        probe,
        quality,
        enroll_availability,
        probe_availability,
    )
    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()
    assert torch.all(scores <= 1.00001)
    assert torch.all(scores >= -1.00001)


def test_feature_quality_gate_rejects_trials_without_joint_modality():
    model = FeatureQualityGate((2, 2), projection_dim=3, gate_hidden_dims=(4,))
    enrollment = [torch.zeros(1, 2), torch.randn(1, 2)]
    probe = [torch.randn(1, 2), torch.zeros(1, 2)]
    quality = torch.tensor([[0.7, 0.0]], dtype=torch.float32)
    enroll_availability = torch.tensor([[False, True]])
    probe_availability = torch.tensor([[True, False]])
    with pytest.raises(ValueError, match="jointly available"):
        model(enrollment, probe, quality, enroll_availability, probe_availability)
