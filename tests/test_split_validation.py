import pytest

from deepmm.validation.splits import assert_disjoint_subject_splits, assert_unique_sample_ids


def test_disjoint_subject_splits_pass():
    assert_disjoint_subject_splits(["s1", "s2"], ["s3"], ["s4", "s5"])


def test_subject_overlap_fails():
    with pytest.raises(ValueError, match="subject leakage"):
        assert_disjoint_subject_splits(["s1", "s2"], ["s2", "s3"], ["s4"])


def test_duplicate_sample_id_fails():
    with pytest.raises(ValueError, match="duplicate sample"):
        assert_unique_sample_ids({"train": ["a", "b"], "val": ["c"], "test": ["b", "d"]})
