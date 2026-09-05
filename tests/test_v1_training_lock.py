from deepmm.training.v1_public_config import (
    V1_NEURAL_BUDGET,
    V1_NEURAL_CANDIDATES,
    V1_PRIMARY_ENCODER,
    V1_REPORTING_SEED,
    V1_TRAINING_LOCK_SHA256,
    assert_v1_training_lock,
    v1_training_lock_hash,
)


def test_v1_training_lock_is_exact_and_self_consistent():
    assert_v1_training_lock()
    assert V1_PRIMARY_ENCODER == "resnet18_imagenet1k_v1"
    assert V1_REPORTING_SEED == 1701
    assert V1_NEURAL_BUDGET.seeds == (1701, 2903, 4307)
    assert V1_NEURAL_BUDGET.max_epochs == 40
    assert V1_NEURAL_BUDGET.early_stopping_patience == 6
    assert V1_NEURAL_BUDGET.max_candidate_configs == 2
    assert V1_NEURAL_BUDGET.max_training_runs == 6
    assert V1_NEURAL_BUDGET.tuning_objective == "eer"
    assert set(V1_NEURAL_CANDIDATES) == {"D1", "D2", "D3S"}
    assert all(len(items) == 2 for items in V1_NEURAL_CANDIDATES.values())
    assert v1_training_lock_hash() == V1_TRAINING_LOCK_SHA256


def test_v1_candidate_ids_are_unique():
    ids = [
        candidate["candidate_id"]
        for candidates in V1_NEURAL_CANDIDATES.values()
        for candidate in candidates
    ]
    assert len(ids) == len(set(ids))
