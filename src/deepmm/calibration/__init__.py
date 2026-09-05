"""Held-out calibration models for DeepMM biometric verification."""

from .logistic import LogisticLLRCalibrator, posterior_probability_from_llr

__all__ = ["LogisticLLRCalibrator", "posterior_probability_from_llr"]
