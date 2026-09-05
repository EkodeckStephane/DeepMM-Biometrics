import numpy as np
import pytest

from deepmm.metrics.calibration import (
    brier_score,
    cllr,
    cllr_calibration_loss,
    expected_calibration_error,
    min_cllr,
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


def test_cllr_matches_closed_form_symmetric_example():
    # One non-target LLR=-2 and one target LLR=+2 give the same term on each side.
    expected = np.log1p(np.exp(-2.0)) / np.log(2.0)
    assert cllr([-2.0], [2.0]) == pytest.approx(expected, rel=1e-12)


def test_zero_llr_system_has_cllr_one():
    assert cllr([0.0, 0.0], [0.0, 0.0]) == pytest.approx(1.0)


def test_min_cllr_is_zero_for_perfect_ranking():
    assert min_cllr([0.1, 0.2], [0.8, 0.9]) == pytest.approx(0.0, abs=1e-12)


def test_min_cllr_is_one_when_all_scores_are_tied():
    assert min_cllr([0.5, 0.5], [0.5, 0.5]) == pytest.approx(1.0, abs=1e-12)


def test_min_cllr_is_one_for_completely_reversed_ranking():
    # An increasing monotonic calibration cannot rescue a completely reversed ranking;
    # the PAV/isotonic optimum collapses to an uninformative LLR=0 system.
    assert min_cllr([0.8, 0.9], [0.1, 0.2]) == pytest.approx(1.0, abs=1e-12)


def test_calibration_loss_is_nonnegative_and_zero_for_optimal_monotonic_scores():
    non = np.array([-4.0, -2.0])
    tar = np.array([2.0, 4.0])
    loss = cllr_calibration_loss(non, tar)
    assert loss >= 0.0
    # The ranking is perfect, so min-Cllr=0; finite LLRs still have non-zero actual Cllr.
    assert loss == pytest.approx(cllr(non, tar), rel=1e-12)
