import pytest

from deepmm.validation.hashing import hash_ordered_records, hash_split_manifest, sha256_text


def test_sha256_text_is_deterministic():
    assert sha256_text("deepmm") == sha256_text("deepmm")
    assert sha256_text("deepmm") != sha256_text("DeepMM")


def test_split_hash_is_independent_of_subject_and_mapping_order():
    a = {
        "train": ["s2", "s1"],
        "val": ["s4", "s3"],
        "test": ["s6", "s5"],
    }
    b = {
        "test": ["s5", "s6"],
        "train": ["s1", "s2"],
        "val": ["s3", "s4"],
    }
    assert hash_split_manifest(a) == hash_split_manifest(b)


def test_split_hash_changes_when_membership_changes():
    a = {"train": ["s1"], "val": ["s2"], "test": ["s3"]}
    b = {"train": ["s1"], "val": ["s3"], "test": ["s2"]}
    assert hash_split_manifest(a) != hash_split_manifest(b)


def test_duplicate_subject_within_split_is_rejected():
    with pytest.raises(ValueError, match="duplicate subject"):
        hash_split_manifest({"train": ["s1", "s1"]})


def test_trial_hash_ignores_dictionary_key_order_but_not_trial_order():
    trials_a = [
        {"left": "a1", "right": "a2", "label": 1},
        {"left": "a1", "right": "b1", "label": 0},
    ]
    trials_same = [
        {"label": 1, "right": "a2", "left": "a1"},
        {"right": "b1", "label": 0, "left": "a1"},
    ]
    trials_reversed = list(reversed(trials_a))
    assert hash_ordered_records(trials_a) == hash_ordered_records(trials_same)
    assert hash_ordered_records(trials_a) != hash_ordered_records(trials_reversed)
