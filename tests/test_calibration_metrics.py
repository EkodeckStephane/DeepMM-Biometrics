import numpy as np
import pytest

from deepmm.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)


def test_perfect_probabilities_have_zero_brier_and_ece():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])
    assert brier_score(labels, probabilities) == pytest.approx(0.0)
    assert expected_calibration_error(labels, probabilities, n_bins=10) == pytest.approx(0.0)
    assert negative_log_likelihood(labels, probabilities) < 1e-12


def test_constant_empirical_prevalence_is_calibrated_in_one_occupied_bin():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.full(4, 0.5)
    assert brier_score(labels, probabilities) == pytest.approx(0.25)
    assert expected_calibration_error(labels, probabilities, n_bins=10) == pytest.approx(0.0)
    assert negative_log_likelihood(labels, probabilities) == pytest.approx(np.log(2.0))


def test_probability_range_is_validated():
    with pytest.raises(ValueError):
        brier_score([0, 1], [-0.1, 1.1])
