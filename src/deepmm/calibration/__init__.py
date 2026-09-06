"""Held-out calibration models for DeepMM biometric verification."""

from .logistic import LogisticLLRCalibrator, posterior_probability_from_llr
from .v1_calibration_lock import (
    V1_CALIBRATION_EVIDENCE_SHA256,
    V1_COMMITTED_CALIBRATION_JSON_SHA256,
    assert_v1_calibration_lock,
    load_v1_calibration_lock,
)

__all__ = [
    "LogisticLLRCalibrator",
    "posterior_probability_from_llr",
    "V1_CALIBRATION_EVIDENCE_SHA256",
    "V1_COMMITTED_CALIBRATION_JSON_SHA256",
    "assert_v1_calibration_lock",
    "load_v1_calibration_lock",
]
