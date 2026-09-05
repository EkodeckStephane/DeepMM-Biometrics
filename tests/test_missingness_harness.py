import numpy as np
import pytest

from deepmm.fusion import (
    EmbeddingEvidence,
    ScoreEvidence,
    apply_embedding_availability,
    apply_score_availability,
    deterministic_modality_dropout_mask,
    fixed_subset_mask,
    masked_weighted_score_sum,
    modality_subset_id,
)


def test_dropout_mask_is_deterministic_and_never_empty():
    a = deterministic_modality_dropout_mask(100, 3, drop_probability=0.8, seed=17)
    b = deterministic_modality_dropout_mask(100, 3, drop_probability=0.8, seed=17)
    assert np.array_equal(a, b)
    assert a.dtype == np.bool_
    assert np.all(np.sum(a, axis=1) >= 1)


def test_fixed_subset_mask_uses_names_not_positions_from_call_site():
    mask = fixed_subset_mask(3, ("face", "finger", "iris"), ("finger", "iris"))
    assert mask.tolist() == [[False, True, True]] * 3
    assert modality_subset_id(("face", "finger", "iris"), mask[0]) == "finger+iris"


def test_score_availability_can_only_remove_existing_evidence():
    original = ScoreEvidence(
        np.array([[0.9, 0.7], [0.4, 0.3]]),
        ("face", "finger"),
        np.ones((2, 2), dtype=bool),
        quality=np.array([[0.8, 0.7], [0.6, 0.5]]),
    )
    target = np.array([[1, 0], [1, 1]], dtype=bool)
    masked = apply_score_availability(original, target)
    assert masked.scores[0, 1] == 0.0
    assert masked.quality[0, 1] == 0.0
    assert original.scores[0, 1] == pytest.approx(0.7)

    already_missing = apply_score_availability(original, target)
    with pytest.raises(ValueError, match="cannot make originally unavailable"):
        apply_score_availability(already_missing, np.ones((2, 2), dtype=bool))


def test_masked_weighted_score_sum_renormalizes_over_available_modalities():
    evidence = ScoreEvidence(
        np.array([[0.8, 0.0], [0.2, 0.6]]),
        ("face", "finger"),
        np.array([[1, 0], [1, 1]], dtype=bool),
    )
    fused = masked_weighted_score_sum(evidence, [0.25, 0.75])
    assert fused[0] == pytest.approx(0.8)
    assert fused[1] == pytest.approx(0.25 * 0.2 + 0.75 * 0.6)


def test_embedding_availability_supports_probe_only_missingness():
    enrollment = (np.ones((2, 3)), np.ones((2, 2)))
    probe = (np.ones((2, 3)), np.ones((2, 2)))
    availability = np.ones((2, 2), dtype=bool)
    evidence = EmbeddingEvidence(
        enrollment,
        probe,
        ("face", "finger"),
        availability,
        availability,
        quality=np.ones((2, 2)),
    )
    probe_mask = np.array([[1, 0], [1, 1]], dtype=bool)
    stressed = apply_embedding_availability(evidence, probe_availability=probe_mask)
    assert np.all(stressed.enrollment[1][0] == 1.0)
    assert np.all(stressed.probe[1][0] == 0.0)
    assert stressed.quality[0, 1] == 0.0
    assert stressed.complete_case_mask().tolist() == [False, True]


def test_final_subset_cannot_be_empty():
    with pytest.raises(ValueError, match="at least one modality"):
        fixed_subset_mask(2, ("face", "finger"), ())

    evidence = ScoreEvidence(
        np.ones((2, 2)),
        ("face", "finger"),
        np.ones((2, 2), dtype=bool),
    )
    with pytest.raises(ValueError, match="at least one available modality"):
        apply_score_availability(evidence, np.zeros((2, 2), dtype=bool))
