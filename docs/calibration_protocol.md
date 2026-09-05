# Score Calibration Protocol v0.1

**Study:** *Deep Learning Approaches for Multimodal Biometrics*

Calibration is evaluated separately from discrimination. Good EER or AUC does not by itself demonstrate good calibration, and final-test labels are not used to fit the deployable calibration mapping.

## Score convention

All methods expose a scalar score where larger values indicate stronger genuine evidence. A fitted affine calibrator with a non-positive slope is rejected rather than silently reversing the score ranking.

## Held-out logistic LLR calibration

`LogisticLLRCalibrator` is fitted only on the designated development/calibration partition. Class-balanced fitting weights impose an effective target prior of 0.5, so the logistic decision output is used as an approximate natural-log likelihood ratio.

The regularization parameter `C` has no hidden default. It is part of the locked experiment configuration and must be selected without final-test labels. Transformation of final scores consumes scores only, not labels.

## Probability metrics

When Brier score, NLL or ECE are reported, calibrated LLRs are converted to posterior probabilities using an explicitly declared reference target prior. The empirical fraction of genuine trials in the benchmark is not interpreted as real-world prevalence.

## Calibration evidence

Subject to the remaining independent implementation checks, the planned evidence is:

- `C_llr` on held-out calibrated LLRs;
- `C_llr_min` as the discrimination component under optimal monotonic calibration;
- `C_llr_cal = C_llr - C_llr_min`;
- Brier and NLL as complementary proper scoring rules;
- ECE and reliability diagrams as descriptive complementary evidence.

ECE alone cannot support the headline calibration claim.

## Stress and missing-modality conditions

Q3 requires condition-aware calibration analysis. Clean-condition calibration alone does not establish calibration robustness.

The final protocol distinguishes two experiments when feasible:

1. one calibrator fitted on development data and transferred unchanged across clean, degraded and modality-subset conditions;
2. condition-specific or subset-specific recalibration, only when a separate held-out calibration partition exists for that condition.

These answer different scientific questions and are reported separately.

## Leakage rule

Final-test labels may be used to compute evaluation statistics such as `C_llr` and `C_llr_min`. They are not used to fit the deployable calibrator, choose `C`, choose the reporting prior, select a model, or repair score orientation. Any such use removes confirmatory status from the affected calibration result.
