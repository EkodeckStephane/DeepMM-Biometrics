import pytest

from deepmm.fusion import (
    EvidenceTier,
    NeuralHeadConfig,
    NeuralHeadKind,
    NeuralSearchSpace,
    TrainingBudget,
    assert_matched_training_budgets,
)


def _budget():
    return TrainingBudget(
        max_epochs=50,
        early_stopping_patience=7,
        max_candidate_configs=3,
        seeds=(11, 23, 37),
        tuning_objective="eer",
        max_training_runs=9,
    )


def test_neural_head_kind_is_locked_to_canonical_method_and_tier():
    d1 = NeuralHeadConfig("D1", NeuralHeadKind.SCORE_MLP, 2, (16, 8))
    d2 = NeuralHeadConfig("D2", NeuralHeadKind.FEATURE_MLP, 256, (128, 64))
    d3s = NeuralHeadConfig("D3S", NeuralHeadKind.SCORE_GATE, 6, (16,))
    d3f = NeuralHeadConfig("D3F", NeuralHeadKind.FEATURE_GATE, 260, (64,))
    assert d1.evidence_spec.evidence_tier == EvidenceTier.SCORE
    assert d2.evidence_spec.evidence_tier == EvidenceTier.EMBEDDING
    assert d3s.evidence_spec.uses_quality is True
    assert d3f.evidence_spec.uses_availability is True

    with pytest.raises(ValueError, match="must use method_id D1"):
        NeuralHeadConfig("D2", NeuralHeadKind.SCORE_MLP, 2, (8,))


def test_dense_parameter_count_is_auditable_closed_form():
    config = NeuralHeadConfig("D1", NeuralHeadKind.SCORE_MLP, 2, (4,), output_dim=1)
    # 2*4+4 + 4*1+1 = 17
    assert config.dense_parameter_count == 17
    assert config.as_dict()["dense_parameter_count"] == 17


def test_training_budget_caps_candidates_times_seeds():
    with pytest.raises(ValueError, match="exceeds max_training_runs"):
        TrainingBudget(
            max_epochs=50,
            early_stopping_patience=5,
            max_candidate_configs=4,
            seeds=(1, 2, 3),
            max_training_runs=10,
        )


def test_search_space_hash_is_deterministic_and_sensitive_to_config():
    budget = _budget()
    a = NeuralSearchSpace(
        "D1",
        (
            NeuralHeadConfig("D1", NeuralHeadKind.SCORE_MLP, 2, (8,), seed=0),
            NeuralHeadConfig("D1", NeuralHeadKind.SCORE_MLP, 2, (16,), seed=0),
        ),
        budget,
    )
    b = NeuralSearchSpace(
        "D1",
        (
            NeuralHeadConfig("D1", NeuralHeadKind.SCORE_MLP, 2, (8,), seed=0),
            NeuralHeadConfig("D1", NeuralHeadKind.SCORE_MLP, 2, (16,), seed=0),
        ),
        budget,
    )
    c = NeuralSearchSpace(
        "D1",
        (
            NeuralHeadConfig("D1", NeuralHeadKind.SCORE_MLP, 2, (8,), seed=0),
            NeuralHeadConfig("D1", NeuralHeadKind.SCORE_MLP, 2, (32,), seed=0),
        ),
        budget,
    )
    assert a.search_hash == b.search_hash
    assert a.search_hash != c.search_hash
    assert a.planned_training_runs == 6


def test_search_space_rejects_mixed_method_candidates_and_duplicates():
    budget = _budget()
    with pytest.raises(ValueError, match="every candidate"):
        NeuralSearchSpace(
            "D1",
            (
                NeuralHeadConfig("D1", NeuralHeadKind.SCORE_MLP, 2, (8,)),
                NeuralHeadConfig("D2", NeuralHeadKind.FEATURE_MLP, 10, (8,)),
            ),
            budget,
        )

    duplicate = NeuralHeadConfig("D1", NeuralHeadKind.SCORE_MLP, 2, (8,))
    with pytest.raises(ValueError, match="unique"):
        NeuralSearchSpace("D1", (duplicate, duplicate), budget)


def test_confirmatory_neural_comparison_requires_matched_budgets():
    budget = _budget()
    d1 = NeuralSearchSpace(
        "D1", (NeuralHeadConfig("D1", NeuralHeadKind.SCORE_MLP, 2, (8,)),), budget
    )
    d2 = NeuralSearchSpace(
        "D2", (NeuralHeadConfig("D2", NeuralHeadKind.FEATURE_MLP, 64, (32,)),), budget
    )
    assert len(assert_matched_training_budgets((d1, d2))) == 2

    other_budget = TrainingBudget(
        max_epochs=60,
        early_stopping_patience=7,
        max_candidate_configs=3,
        seeds=(11, 23, 37),
        tuning_objective="eer",
        max_training_runs=9,
    )
    d3 = NeuralSearchSpace(
        "D3S", (NeuralHeadConfig("D3S", NeuralHeadKind.SCORE_GATE, 6, (8,)),), other_budget
    )
    with pytest.raises(ValueError, match="matched training/tuning budgets"):
        assert_matched_training_budgets((d1, d3))


def test_budget_rejects_posthoc_style_unbounded_search():
    with pytest.raises(ValueError):
        TrainingBudget(50, 5, 0, (1, 2, 3))
    with pytest.raises(ValueError):
        TrainingBudget(50, 50, 3, (1, 2, 3))
    with pytest.raises(ValueError):
        TrainingBudget(50, 5, 3, (1, 1, 2))
