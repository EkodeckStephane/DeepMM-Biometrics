import pytest

from deepmm.validation.run_manifest import run_manifest_hash, validate_run_manifest

H0 = "0" * 64
H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64
GIT = "a" * 40


def _run():
    return {
        "run_id": "clean-c2-seed0",
        "method_id": "classical-logistic",
        "family": "classical-score",
        "seed": 0,
        "condition_id": "clean",
        "code_commit": GIT,
        "split_hash": H0,
        "trial_manifest_hash": H1,
        "config_hash": H2,
        "score_manifest_hash": H3,
        "environment_hash": "4" * 64,
    }


def test_valid_run_manifest_is_canonicalized():
    row = _run()
    row["code_commit"] = GIT.upper()
    out = validate_run_manifest(row)
    assert out["code_commit"] == GIT
    assert out["seed"] == 0
    assert out["condition_id"] == "clean"


def test_short_git_sha_is_rejected():
    row = _run()
    row["code_commit"] = "a" * 12
    with pytest.raises(ValueError, match="full 40-character"):
        validate_run_manifest(row)


def test_invalid_sha256_is_rejected():
    row = _run()
    row["split_hash"] = "f" * 63
    with pytest.raises(ValueError, match="SHA-256"):
        validate_run_manifest(row)


def test_seed_must_be_nonnegative_integer_not_boolean():
    row = _run()
    row["seed"] = -1
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_run_manifest(row)

    row = _run()
    row["seed"] = True
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_run_manifest(row)


def test_optional_null_hash_is_omitted_but_invalid_value_rejected():
    row = _run()
    row["checkpoint_hash"] = None
    out = validate_run_manifest(row)
    assert "checkpoint_hash" not in out

    row = _run()
    row["checkpoint_hash"] = "not-a-hash"
    with pytest.raises(ValueError, match="checkpoint_hash"):
        validate_run_manifest(row)


def test_manifest_hash_is_key_order_independent_but_evidence_sensitive():
    a = _run()
    b = dict(reversed(list(a.items())))
    assert run_manifest_hash(a) == run_manifest_hash(b)

    c = _run()
    c["score_manifest_hash"] = "5" * 64
    assert run_manifest_hash(a) != run_manifest_hash(c)


def test_extra_provenance_metadata_participates_in_hash():
    a = _run()
    b = _run()
    b["hardware_id"] = "gpu-a"
    assert run_manifest_hash(a) != run_manifest_hash(b)
