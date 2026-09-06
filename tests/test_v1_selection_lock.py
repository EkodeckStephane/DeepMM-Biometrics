from deepmm.training.v1_selection_lock import (
    V1_DEVELOPMENT_RUN_ID,
    V1_REPORTING_SEED,
    V1_SELECTED_MODELS,
    V1_SELECTION_LOCK_SHA256,
    assert_v1_selection_lock,
    v1_selection_lock_hash,
)


def test_v1_selection_lock_is_exact():
    assert_v1_selection_lock()
    assert V1_DEVELOPMENT_RUN_ID == 34019887091
    assert V1_REPORTING_SEED == 1701
    assert v1_selection_lock_hash() == V1_SELECTION_LOCK_SHA256


def test_v1_selected_candidate_ids_and_reporting_hashes_are_frozen():
    assert V1_SELECTED_MODELS["D1"]["candidate_id"] == "d1-h16"
    assert V1_SELECTED_MODELS["D2"]["candidate_id"] == "d2-h128-z64"
    assert V1_SELECTED_MODELS["D3S"]["candidate_id"] == "d3s-h16"
    for family in ("D1", "D2", "D3S"):
        checkpoint = V1_SELECTED_MODELS[family]["expected_checkpoint_hash"]
        assert isinstance(checkpoint, str)
        assert len(checkpoint) == 64
