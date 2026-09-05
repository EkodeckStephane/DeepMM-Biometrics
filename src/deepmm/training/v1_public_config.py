"""Frozen V1 neural-development configuration for the public NUPT-FPV subset.

This module is the preregistered development lock used before any calibration or
final-test image is opened. The primary representation is the already frozen
ResNet18 ImageNet encoder. D3 in the V1 headline comparison is instantiated as
D3S, a quality-aware score gate, so it can be compared directly with the classical
C5 quality-weighted score baseline using exactly the same quality cues.
"""

from __future__ import annotations

import hashlib
import json

from deepmm.fusion.neural_contracts import TrainingBudget


V1_PRIMARY_ENCODER = "resnet18_imagenet1k_v1"
V1_REPORTING_SEED = 1701
V1_BATCH_SIZE = 256
V1_FIT_SAMPLING_RULE = (
    "deterministic class-balanced replication: every impostor trial once and every "
    "genuine trial 19 times; selection remains unmodified"
)

V1_NEURAL_BUDGET = TrainingBudget(
    max_epochs=40,
    early_stopping_patience=6,
    max_candidate_configs=2,
    seeds=(1701, 2903, 4307),
    tuning_objective="eer",
    max_training_runs=6,
)

# Kept as a plain serializable mapping so the scientific lock remains importable
# in the lightweight base test environment where PyTorch is intentionally absent.
# The execution script materializes TorchOptimizerConfig from this exact mapping.
V1_OPTIMIZER = {
    "optimizer": "adamw",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "gradient_clip_norm": 5.0,
    "deterministic_algorithms": True,
}

V1_NEURAL_CANDIDATES = {
    "D1": (
        {
            "candidate_id": "d1-h8",
            "hidden_dims": (8,),
            "activation": "relu",
            "dropout": 0.0,
        },
        {
            "candidate_id": "d1-h16",
            "hidden_dims": (16,),
            "activation": "relu",
            "dropout": 0.0,
        },
    ),
    "D2": (
        {
            "candidate_id": "d2-h128-z64",
            "hidden_dims": (128,),
            "fused_dim": 64,
            "activation": "relu",
            "dropout": 0.0,
        },
        {
            "candidate_id": "d2-h256-z64",
            "hidden_dims": (256,),
            "fused_dim": 64,
            "activation": "relu",
            "dropout": 0.0,
        },
    ),
    "D3S": (
        {
            "candidate_id": "d3s-h8",
            "hidden_dims": (8,),
            "activation": "relu",
            "dropout": 0.0,
        },
        {
            "candidate_id": "d3s-h16",
            "hidden_dims": (16,),
            "activation": "relu",
            "dropout": 0.0,
        },
    ),
}

V1_SELECTION_RULE = (
    "min median selection EER across seeds; then min mean EER; then min trainable "
    "parameters; then candidate_id"
)
V1_FINAL_REPORTING_RULE = (
    "reporting_seed=1701 primary technical realization; all other locked seeds retained "
    "as stochastic sensitivity"
)


def v1_training_lock_payload() -> dict[str, object]:
    return {
        "encoder_id": V1_PRIMARY_ENCODER,
        "neural_budget": V1_NEURAL_BUDGET.as_dict(),
        "optimizer": dict(V1_OPTIMIZER),
        "reporting_seed": V1_REPORTING_SEED,
        "batch_size": V1_BATCH_SIZE,
        "fit_sampling_rule": V1_FIT_SAMPLING_RULE,
        "families": {
            family: [
                {
                    key: (list(value) if isinstance(value, tuple) else value)
                    for key, value in candidate.items()
                }
                for candidate in candidates
            ]
            for family, candidates in V1_NEURAL_CANDIDATES.items()
        },
        "selection_rule": V1_SELECTION_RULE,
        "final_reporting_rule": V1_FINAL_REPORTING_RULE,
    }


def v1_training_lock_hash() -> str:
    encoded = json.dumps(
        v1_training_lock_payload(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


V1_TRAINING_LOCK_SHA256 = "d7a118af2bd02cdb0625602713cf3254f65a8acc06459672a72ff3a48ec22f45"


def assert_v1_training_lock() -> None:
    if V1_REPORTING_SEED not in V1_NEURAL_BUDGET.seeds:
        raise RuntimeError("reporting seed is outside the locked seed set")
    if set(V1_NEURAL_CANDIDATES) != {"D1", "D2", "D3S"}:
        raise RuntimeError("V1 neural family set changed")
    if any(
        len(candidates) != V1_NEURAL_BUDGET.max_candidate_configs
        for candidates in V1_NEURAL_CANDIDATES.values()
    ):
        raise RuntimeError("V1 candidate counts no longer match the locked budget")
    actual = v1_training_lock_hash()
    if actual != V1_TRAINING_LOCK_SHA256:
        raise RuntimeError(
            f"V1 training lock changed: expected {V1_TRAINING_LOCK_SHA256}, got {actual}"
        )
