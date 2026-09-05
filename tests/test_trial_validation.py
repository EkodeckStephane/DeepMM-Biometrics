import numpy as np
import pytest

from deepmm.validation.trials import (
    score_manifest_hash,
    trial_manifest_hash,
    validate_score_records,
    validate_trial_records,
)


def _trials():
    return [
        {
            "trial_id": "g1",
            "label": 1,
            "anchor_subject_id": "s1",
            "enrollment_subject_id": "s1",
            "probe_subject_id": "s1",
            "enrollment_sample_id": "s1-e1",
            "probe_sample_id": "s1-p1",
            "condition_id": "clean",
            "severity": 0,
        },
        {
            "trial_id": "i1",
            "label": 0,
            "anchor_subject_id": "s1",
            "enrollment_subject_id": "s1",
            "probe_subject_id": "s2",
            "enrollment_sample_id": "s1-e1",
            "probe_sample_id": "s2-p1",
            "condition_id": "clean",
            "severity": 0,
        },
    ]


def test_valid_trial_manifest_is_canonical_and_preserves_extra_fields():
    rows = _trials()
    rows[0]["label"] = "1"
    out = validate_trial_records(rows)
    assert out[0]["label"] == 1
    assert out[0]["severity"] == 0
    assert [row["trial_id"] for row in out] == ["g1", "i1"]


def test_boolean_and_nonliteral_labels_are_rejected():
    rows = _trials()
    rows[0]["label"] = True
    with pytest.raises(ValueError, match="not boolean"):
        validate_trial_records(rows)
    rows = _trials()
    rows[0]["label"] = 1.0
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        validate_trial_records(rows)


def test_duplicate_trial_id_is_rejected():
    rows = _trials()
    rows[1]["trial_id"] = "g1"
    with pytest.raises(ValueError, match="duplicate trial_id"):
        validate_trial_records(rows)


def test_genuine_and_impostor_identity_semantics_are_enforced():
    rows = _trials()
    rows[0]["probe_subject_id"] = "s2"
    with pytest.raises(ValueError, match="genuine label"):
        validate_trial_records(rows)

    rows = _trials()
    rows[1]["probe_subject_id"] = "s1"
    rows[1]["probe_sample_id"] = "s1-p2"
    with pytest.raises(ValueError, match="impostor label"):
        validate_trial_records(rows)


def test_self_match_and_external_anchor_are_rejected():
    rows = _trials()
    rows[0]["probe_sample_id"] = rows[0]["enrollment_sample_id"]
    with pytest.raises(ValueError, match="self-match"):
        validate_trial_records(rows)

    rows = _trials()
    rows[1]["anchor_subject_id"] = "s9"
    with pytest.raises(ValueError, match="anchor_subject_id"):
        validate_trial_records(rows)


def test_sample_cannot_change_identity_inside_manifest():
    rows = _trials()
    rows[1]["probe_sample_id"] = "s1-p1"  # first row assigns this sample to s1
    with pytest.raises(ValueError, match="multiple subjects"):
        validate_trial_records(rows)


def test_manifest_must_contain_both_trial_classes():
    with pytest.raises(ValueError, match="both genuine and impostor"):
        validate_trial_records([_trials()[0]])


def test_score_records_must_cover_exact_frozen_order():
    trials = _trials()
    scores = [{"trial_id": "g1", "score": 0.8}, {"trial_id": "i1", "score": 0.2}]
    out = validate_score_records(trials, scores)
    assert out[0]["score"] == pytest.approx(0.8)

    with pytest.raises(ValueError, match="count"):
        validate_score_records(trials, scores[:1])

    with pytest.raises(ValueError, match="order mismatch"):
        validate_score_records(trials, list(reversed(scores)))


def test_nonfinite_score_and_invalid_score_row_are_rejected():
    trials = _trials()
    scores = [{"trial_id": "g1", "score": np.nan}, {"trial_id": "i1", "score": 0.2}]
    with pytest.raises(ValueError, match="must be finite"):
        validate_score_records(trials, scores)

    with pytest.raises(TypeError, match="score row 0"):
        validate_score_records(trials, [None, {"trial_id": "i1", "score": 0.2}])


def test_trial_hash_is_order_sensitive_and_extra_metadata_sensitive():
    a = _trials()
    b = list(reversed(_trials()))
    assert trial_manifest_hash(a) != trial_manifest_hash(b)

    c = _trials()
    c[0]["severity"] = 1
    assert trial_manifest_hash(a) != trial_manifest_hash(c)


def test_score_hash_changes_when_any_score_changes():
    trials = _trials()
    a = [{"trial_id": "g1", "score": 0.8}, {"trial_id": "i1", "score": 0.2}]
    b = [{"trial_id": "g1", "score": 0.81}, {"trial_id": "i1", "score": 0.2}]
    assert score_manifest_hash(trials, a) != score_manifest_hash(trials, b)
