from deepmm.evaluation.v1_final_config import (
    V1_CALIBRATION_C,
    V1_ECE_BINS,
    V1_FAR_GRID,
    V1_FINAL_POLICY_SHA256,
    V1_FINAL_TRIAL_MANIFEST_SHA256,
    V1_MISSINGNESS_SCOPE,
    V1_STRESS_PLAN_SHA256,
    assert_v1_final_policy_lock,
    v1_final_policy_hash,
)


def test_v1_final_policy_is_exact_and_self_consistent():
    assert_v1_final_policy_lock()
    assert V1_CALIBRATION_C == 1.0
    assert V1_ECE_BINS == 15
    assert V1_MISSINGNESS_SCOPE == "probe_only"
    assert V1_FAR_GRID == (0.1, 0.01, 0.001)
    assert V1_STRESS_PLAN_SHA256 == "6ba45461396f61dda720e7d289cdade98cac750cf9172b7502518428e022bbd3"
    assert V1_FINAL_TRIAL_MANIFEST_SHA256 == "3b60ce30d0d496c35aefe0bf0b8c48f868cb3befba0c1fdfb52986645293f324"
    assert v1_final_policy_hash() == V1_FINAL_POLICY_SHA256
