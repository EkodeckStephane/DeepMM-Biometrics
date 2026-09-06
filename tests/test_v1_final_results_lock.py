from deepmm.evaluation.v1_final_results_lock import (
    V1_FINAL_ARTIFACT_FILES,
    V1_FINAL_ARTIFACT_ZIP_SHA256,
    V1_FINAL_RESULTS_LOCK_SHA256,
    assert_v1_final_results_lock,
    load_v1_final_results_lock,
    v1_final_results_lock_hash,
)


def test_v1_final_results_are_complete_and_locked():
    result = load_v1_final_results_lock()
    assert result["status"] == "complete"
    assert result["conditions"]["clean"]["n_trials"] == 4000
    assert len(result["conditions"]) == 15


def test_v1_final_results_provenance_is_frozen():
    assert_v1_final_results_lock()
    assert v1_final_results_lock_hash() == V1_FINAL_RESULTS_LOCK_SHA256
    assert V1_FINAL_ARTIFACT_ZIP_SHA256 == "743ae2d30d7cf5747dce5b3cad66e789d8cc3c9923fe91f35510d0bd473180c5"
    assert V1_FINAL_ARTIFACT_FILES["v1_final_scores.npz"] == "b77fce5f86e1ea82978ec7f78c72534348078b9c8da434254a1226c477247d0a"
