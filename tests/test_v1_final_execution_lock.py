from deepmm.evaluation.v1_final_execution_lock import assert_v1_final_script_lock


def test_v1_final_runner_is_frozen():
    assert_v1_final_script_lock()
