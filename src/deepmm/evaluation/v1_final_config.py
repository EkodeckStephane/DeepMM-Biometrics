"""Frozen calibration, robustness, missingness and final-evaluation policy for V1.

This module is intentionally torch-free so the scientific lock is auditable in the
base CI. It freezes the remaining protocol choices before calibration-role or final-
role outcomes are inspected by the selected V1 systems.
"""

from __future__ import annotations

import hashlib
import json


V1_CALIBRATION_C = 1.0
V1_CALIBRATION_EFFECTIVE_TARGET_PRIOR = 0.5
V1_PROBABILITY_REFERENCE_PRIOR = 0.5
V1_ECE_BINS = 15

V1_STRESS_PLAN_SHA256 = "6ba45461396f61dda720e7d289cdade98cac750cf9172b7502518428e022bbd3"
V1_FINAL_TRIAL_MANIFEST_SHA256 = "3b60ce30d0d496c35aefe0bf0b8c48f868cb3befba0c1fdfb52986645293f324"

V1_CALIBRATION_PRIMARY_POLICY = (
    "clean calibrator transferred unchanged to all final conditions"
)
V1_CALIBRATION_SECONDARY_POLICY = (
    "condition-specific held-out recalibration on the matching calibration condition"
)

V1_MISSINGNESS_SCOPE = "probe_only"
V1_MISSINGNESS_FALLBACK_POLICY = (
    "M0 single-available-modality fallback for U/C1/C2/C3/C4/D1/D2; C5 and D3S "
    "use explicit availability-aware renormalization/gating; enrollment remains clean"
)
V1_MISSINGNESS_PLACEHOLDER = "canonical_zero"

V1_COST_POLICY = {
    "primary_scope": "fusion_only",
    "device": "cpu",
    "precision": "float32",
    "batch_size": 256,
    "warmup": 20,
    "repeats": 200,
    "num_threads": 2,
    "end_to_end_scope": "descriptive_secondary",
}

V1_DISCRIMINATION_METRICS = ("rocch_eer", "empirical_eer", "auc", "tar_at_far")
V1_FAR_GRID = (0.1, 0.01, 0.001)
V1_CALIBRATION_METRICS = ("cllr", "min_cllr", "cllr_cal", "brier", "nll", "ece")
V1_Q2_RULE = "pareto_no_posthoc_weighted_composite"
V1_Q3_ANALYSES = ("conditionwise_metrics", "kendall_tau_b", "pairwise_rank_reversals")


def v1_final_policy_payload() -> dict[str, object]:
    return {
        "calibration": {
            "C": V1_CALIBRATION_C,
            "effective_target_prior": V1_CALIBRATION_EFFECTIVE_TARGET_PRIOR,
            "probability_reference_prior": V1_PROBABILITY_REFERENCE_PRIOR,
            "ece_bins": V1_ECE_BINS,
            "primary_condition_policy": V1_CALIBRATION_PRIMARY_POLICY,
            "secondary_condition_policy": V1_CALIBRATION_SECONDARY_POLICY,
        },
        "stress_plan_sha256": V1_STRESS_PLAN_SHA256,
        "missingness": {
            "scope": V1_MISSINGNESS_SCOPE,
            "fallback_policy": V1_MISSINGNESS_FALLBACK_POLICY,
            "placeholder": V1_MISSINGNESS_PLACEHOLDER,
            "no_nan": True,
        },
        "cost": dict(V1_COST_POLICY),
        "final": {
            "trial_manifest_sha256": V1_FINAL_TRIAL_MANIFEST_SHA256,
            "conditions": 15,
            "clean_primary_discrimination": True,
        },
        "metrics": {
            "discrimination": list(V1_DISCRIMINATION_METRICS),
            "far_grid": list(V1_FAR_GRID),
            "calibration": list(V1_CALIBRATION_METRICS),
            "q2": V1_Q2_RULE,
            "q3": list(V1_Q3_ANALYSES),
        },
    }


def v1_final_policy_hash() -> str:
    encoded = json.dumps(
        v1_final_policy_payload(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


V1_FINAL_POLICY_SHA256 = "e9701015b541e9c7e4debccd01fd1f32affecc97abaf2996b8cf6c5811adbfb5"


def assert_v1_final_policy_lock() -> None:
    actual = v1_final_policy_hash()
    if actual != V1_FINAL_POLICY_SHA256:
        raise RuntimeError(
            f"V1 final policy changed: expected {V1_FINAL_POLICY_SHA256}, got {actual}"
        )
    if min(V1_FAR_GRID) < 1.0 / 3800.0:
        raise RuntimeError("requested FAR is below the final impostor-count resolution")
