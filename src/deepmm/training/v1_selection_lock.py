"""Immutable V1 model-selection outcome for the public-subset campaign.

These values were frozen from the successful development-only workflow before any
calibration-role or final-role image was opened by the neural campaign. Subsequent
workflows must reproduce the reporting-seed checkpoint hashes exactly before using
the selected neural systems.
"""

from __future__ import annotations

import hashlib
import json


V1_DEVELOPMENT_RUN_ID = 33986109443
V1_DEVELOPMENT_HEAD_SHA = "e0e41d125eff126fa3a0a583623dfa069a0b3c83"
V1_DEVELOPMENT_ARTIFACT_ID = 9975210497
V1_DEVELOPMENT_ARTIFACT_ZIP_SHA256 = (
    "1070570c085061da6656b70fe9034232e4973f82dba6efb87a113c01ec9fb7e7"
)
V1_COMMITTED_DEVELOPMENT_JSON_SHA256 = (
    "23b9babe0e8503273e27ab06e41168b5978c576691d82f25a22c092f623c4c79"
)
V1_TRAINING_LOCK_SHA256 = (
    "d7a118af2bd02cdb0625602713cf3254f65a8acc06459672a72ff3a48ec22f45"
)
V1_DATASET_MANIFEST_SHA256 = (
    "be7d83e353476e50a6193d77e47d7f176ab0a9cb805f81cb8b8a87f79368238c"
)
V1_FIT_TRIAL_MANIFEST_SHA256 = (
    "8125265b7407bdfdef2507b3ee6592625d6fc65e41657ad6ea8dd04989514eb7"
)
V1_SELECTION_TRIAL_MANIFEST_SHA256 = (
    "482086a4d3d7a57a68cdd9362363399a8b56174d3218f0b3eed138a7fe6d4fb4"
)
V1_ENCODER_WEIGHT_STATE_HASH = (
    "00eb5b1bfe8ea60f64c93ce1aeba14523e5197adf8ba562afb56ee36e91e7e24"
)
V1_QUALITY_MODEL_HASH = (
    "219874362a61c6b181248820c17d24056644564512ec16ccafe5117133ad0f0d"
)
V1_REPORTING_SEED = 1701

V1_SELECTED_MODELS = {
    "D1": {
        "candidate_id": "d1-h16",
        "expected_checkpoint_hash": (
            "19c603fcf1cbfb4e2ae78a7f1621cc34ee594b335689b124f67653c659357e00"
        ),
        "best_epoch": 3,
        "trainable_parameters": 65,
        "selection_eer_median": 0.1763157894736842,
        "selection_eer_mean": 0.17456140350877192,
    },
    "D2": {
        "candidate_id": "d2-h128-z64",
        "expected_checkpoint_hash": (
            "ac046ce1b3ff2c45c5ccef81b4479f20f81e5113c590d50c2850fccfa705f201"
        ),
        "best_epoch": 30,
        "trainable_parameters": 139456,
        "selection_eer_median": 0.05263157894736842,
        "selection_eer_mean": 0.06754385964912281,
    },
    "D3S": {
        "candidate_id": "d3s-h16",
        "expected_checkpoint_hash": (
            "e9a407f6f8d2c81479b06d03d240810cdba06e8b3f731395a3d1d37ab68bcab1"
        ),
        "best_epoch": 1,
        "trainable_parameters": 116,
        "selection_eer_median": 0.15,
        "selection_eer_mean": 0.15,
    },
}


def v1_selection_lock_payload() -> dict[str, object]:
    return {
        "development_run_id": V1_DEVELOPMENT_RUN_ID,
        "development_head_sha": V1_DEVELOPMENT_HEAD_SHA,
        "workflow_artifact_id": V1_DEVELOPMENT_ARTIFACT_ID,
        "workflow_artifact_zip_sha256": V1_DEVELOPMENT_ARTIFACT_ZIP_SHA256,
        "committed_development_json_sha256": V1_COMMITTED_DEVELOPMENT_JSON_SHA256,
        "training_lock_sha256": V1_TRAINING_LOCK_SHA256,
        "dataset_manifest_sha256": V1_DATASET_MANIFEST_SHA256,
        "fit_trial_manifest_sha256": V1_FIT_TRIAL_MANIFEST_SHA256,
        "selection_trial_manifest_sha256": V1_SELECTION_TRIAL_MANIFEST_SHA256,
        "encoder_weight_state_hash": V1_ENCODER_WEIGHT_STATE_HASH,
        "quality_model_hash": V1_QUALITY_MODEL_HASH,
        "reporting_seed": V1_REPORTING_SEED,
        "selected": V1_SELECTED_MODELS,
    }


def v1_selection_lock_hash() -> str:
    encoded = json.dumps(
        v1_selection_lock_payload(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


V1_SELECTION_LOCK_SHA256 = (
    "729ca6603d2e6989a6d8c696dd6e777a001492a7dcc292a031db09690813f029"
)


def assert_v1_selection_lock() -> None:
    actual = v1_selection_lock_hash()
    if actual != V1_SELECTION_LOCK_SHA256:
        raise RuntimeError(
            f"V1 selection lock changed: expected {V1_SELECTION_LOCK_SHA256}, got {actual}"
        )
