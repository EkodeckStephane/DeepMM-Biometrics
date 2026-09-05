import numpy as np
import pytest

from deepmm.fusion import (
    EmbeddingEvidence,
    EvidenceTier,
    ScoreEvidence,
    canonical_confirmatory_method_specs,
    method_spec,
)


def test_score_evidence_requires_explicit_zero_missing_placeholder():
    scores = np.array([[0.7, 0.0], [0.2, 0.6]])
    availability = np.array([[1, 0], [1, 1]])
    evidence = ScoreEvidence(scores, ("face", "finger"), availability)
    assert evidence.tier == EvidenceTier.SCORE
    assert evidence.complete_case_mask().tolist() == [False, True]

    with pytest.raises(ValueError, match="canonical zero placeholder"):
        ScoreEvidence(
            np.array([[0.7, 999.0], [0.2, 0.6]]),
            ("face", "finger"),
            availability,
        )


def test_nan_cannot_encode_score_missingness():
    with pytest.raises(ValueError, match="scores must be finite"):
        ScoreEvidence(
            np.array([[0.7, np.nan], [0.2, 0.6]]),
            ("face", "finger"),
            np.array([[1, 0], [1, 1]]),
        )


def test_quality_is_zero_for_unavailable_score_modalities():
    scores = np.array([[0.7, 0.0], [0.2, 0.6]])
    availability = np.array([[1, 0], [1, 1]])
    with pytest.raises(ValueError, match="quality for unavailable modalities"):
        ScoreEvidence(
            scores,
            ("face", "finger"),
            availability,
            quality=np.array([[0.8, 0.2], [0.7, 0.9]]),
        )


def test_embedding_evidence_allows_modality_specific_dimensions():
    enrollment = (
        np.ones((3, 4)),
        np.ones((3, 7)),
    )
    probe = (
        np.ones((3, 4)),
        np.ones((3, 7)),
    )
    availability = np.ones((3, 2), dtype=bool)
    evidence = EmbeddingEvidence(
        enrollment,
        probe,
        ("face", "finger"),
        availability,
        availability,
    )
    assert evidence.tier == EvidenceTier.EMBEDDING
    assert evidence.embedding_dims == (4, 7)
    assert evidence.complete_case_mask().tolist() == [True, True, True]


def test_embedding_missingness_must_be_zero_and_explicit():
    enrollment = (np.ones((2, 3)), np.ones((2, 2)))
    probe_a = np.ones((2, 3))
    probe_b = np.array([[0.0, 0.0], [1.0, 1.0]])
    probe_availability = np.array([[1, 0], [1, 1]], dtype=bool)
    enrollment_availability = np.ones((2, 2), dtype=bool)

    evidence = EmbeddingEvidence(
        enrollment,
        (probe_a, probe_b),
        ("face", "finger"),
        enrollment_availability,
        probe_availability,
    )
    assert evidence.complete_case_mask().tolist() == [False, True]

    with pytest.raises(ValueError, match="unavailable probe embeddings"):
        EmbeddingEvidence(
            enrollment,
            (probe_a, np.ones((2, 2))),
            ("face", "finger"),
            enrollment_availability,
            probe_availability,
        )


def test_non_missingness_method_rejects_incomplete_evidence():
    evidence = ScoreEvidence(
        np.array([[0.7, 0.0], [0.2, 0.6]]),
        ("face", "finger"),
        np.array([[1, 0], [1, 1]]),
    )
    with pytest.raises(ValueError, match="has no missingness access"):
        method_spec("C1").validate_evidence(evidence)


def test_quality_access_is_symmetric_for_quality_aware_methods():
    evidence = ScoreEvidence(
        np.array([[0.7, 0.5], [0.2, 0.6]]),
        ("face", "finger"),
        np.ones((2, 2), dtype=bool),
        quality=np.array([[0.8, 0.5], [0.7, 0.9]]),
    )
    method_spec("C5").validate_evidence(evidence)
    method_spec("D3S").validate_evidence(evidence)

    with pytest.raises(ValueError, match="not permitted to consume quality"):
        method_spec("C1").validate_evidence(evidence)


def test_wrong_information_tier_is_rejected():
    evidence = ScoreEvidence(
        np.array([[0.7, 0.5], [0.2, 0.6]]),
        ("face", "finger"),
        np.ones((2, 2), dtype=bool),
    )
    with pytest.raises(ValueError, match="requires embedding evidence"):
        method_spec("D2").validate_evidence(evidence)


def test_confirmatory_registry_has_unique_ids_and_expected_strata():
    specs = canonical_confirmatory_method_specs()
    ids = [spec.method_id for spec in specs]
    assert len(ids) == len(set(ids))
    assert {"C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3S", "D3F"} == set(ids)
    assert method_spec("c1").evidence_tier == EvidenceTier.SCORE
    assert method_spec("D2").evidence_tier == EvidenceTier.EMBEDDING


def test_labels_are_not_part_of_evidence_contracts():
    score_fields = set(ScoreEvidence.__dataclass_fields__)
    embedding_fields = set(EmbeddingEvidence.__dataclass_fields__)
    assert "labels" not in score_fields
    assert "labels" not in embedding_fields
