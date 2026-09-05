import numpy as np
import pytest

from deepmm.calibration import LogisticLLRCalibrator, posterior_probability_from_llr
from deepmm.metrics.calibration import cllr


def _development_scores():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([-2.0, -1.2, -0.8, -0.2, 0.3, 0.9, 1.4, 2.1])
    return labels, scores


def test_calibrator_fits_positive_affine_mapping_and_transforms_without_labels():
    y, s = _development_scores()
    cal = LogisticLLRCalibrator(C=10.0).fit(s, y)
    assert cal.slope_ > 0.0
    llr = cal.transform(np.array([-1.0, 0.0, 1.0]))
    assert llr.shape == (3,)
    assert np.all(np.diff(llr) > 0.0)


def test_probability_conversion_respects_reference_prior():
    llr = np.array([0.0, np.log(3.0), -np.log(3.0)])
    p = posterior_probability_from_llr(llr, target_prior=0.5)
    assert p == pytest.approx([0.5, 0.75, 0.25])

    p_prior = posterior_probability_from_llr([0.0], target_prior=0.2)
    assert p_prior[0] == pytest.approx(0.2)


def test_calibrated_llrs_have_finite_cllr_on_development_toy():
    y, s = _development_scores()
    cal = LogisticLLRCalibrator(C=10.0).fit(s, y)
    llrs = cal.transform(s)
    value = cllr(llrs[y == 0], llrs[y == 1])
    assert np.isfinite(value)
    assert value < 1.0


def test_reversed_score_orientation_is_rejected_not_silently_repaired():
    y, s = _development_scores()
    with pytest.raises(ValueError, match="score orientation"):
        LogisticLLRCalibrator(C=10.0).fit(-s, y)


def test_transform_before_fit_and_invalid_inputs_are_rejected():
    with pytest.raises(RuntimeError):
        LogisticLLRCalibrator(C=1.0).transform([0.1, 0.2])
    with pytest.raises(ValueError):
        LogisticLLRCalibrator(C=0.0)
    with pytest.raises(ValueError):
        posterior_probability_from_llr([0.0], target_prior=1.0)


def test_balancing_makes_zero_llr_map_to_requested_prior_not_empirical_trial_prior():
    # Imbalanced calibration set: the explicit class weighting prevents the raw
    # empirical target fraction from becoming the reference prior by accident.
    non = np.linspace(-2.0, -0.2, 12)
    tar = np.linspace(0.4, 1.8, 3)
    scores = np.concatenate([non, tar])
    labels = np.concatenate([np.zeros(non.size, dtype=int), np.ones(tar.size, dtype=int)])
    cal = LogisticLLRCalibrator(C=10.0).fit(scores, labels)
    probabilities = cal.predict_probability([0.0], target_prior=0.5)
    assert 0.0 < probabilities[0] < 1.0
    # The exact posterior depends on the score; what is forbidden is hard-coding
    # the empirical 3/15 prior into probability conversion.
    assert probabilities[0] != pytest.approx(3.0 / 15.0, abs=1e-3)
