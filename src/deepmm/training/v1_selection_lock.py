"""Immutable V1 model-selection outcome for the public-subset campaign.

These values were frozen from the successful development-only workflow before any
calibration-role or final-role image was opened by the neural campaign. Subsequent
workflows must reproduce the reporting-seed checkpoint hashes exactly before using
the selected neural systems.
"""

from __future__ import annotations

import hashlib
import json


V1_DEVELOPMENT_RUN_ID = 34019887091
V1_DEVELOPMENT_HEAD_SHA = "8055c03363ee98e8ca8038ccdecfb045cfe9cdb6"
V1_DEVELOPMENT_ARTIFACT_ID = 9985137496
V1_DEVELOPMENT_ARTIFACT_ZIP_SHA256 = (
    "49b0756ee49b026eaa06f7ac625e4f47010b21c97ac772e87202025f0243c7db"
)
V1_COMMITTED_DEVELOPMENT_JSON_SHA256 = (
    "2818c4f84b083b6d9a0efb4e32151a3e2f00c6d646197f1230f0a9e03a206ecf"
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
            "6bc9eff9e6ac58a5b533ba23dc8f6af77a003f6382bd1ab918e9e09edf1aa997"
        ),
        "best_epoch": 2,
        "trainable_parameters": 65,
        "selection_eer_median": 0.1763157894736842,
        "selection_eer_mean": 0.17456140350877192,
    },
    "D2": {
        "candidate_id": "d2-h128-z64",
        "expected_checkpoint_hash": (
            "055c07b041f655b15e89a672420ff01c6a8f2b2df48f04cd33d2e12793873b48"
        ),
        "best_epoch": 10,
        "trainable_parameters": 139456,
        "selection_eer_median": 0.05263157894736842,
        "selection_eer_mean": 0.06754385964912281,
    },
    "D3S": {
        "candidate_id": "d3s-h16",
        "expected_checkpoint_hash": (
            "fc1987af6ff80d2f19b907857aa300a09fefa21e9b2787590f66462987d65e34"
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
    "83ecd9e4f357babc7e7e70652dc3f7f95c2cf65dacea2c558f4b6c4d656ada14"
)


def assert_v1_selection_lock() -> None:
    actual = v1_selection_lock_hash()
    if actual != V1_SELECTION_LOCK_SHA256:
        raise RuntimeError(
            f"V1 selection lock changed: expected {V1_SELECTION_LOCK_SHA256}, got {actual}"
        )
