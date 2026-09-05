# Metric Validation Record v0.1

**Purpose:** document the external definitions used to validate the dataset-agnostic evaluation core before any biometric experiment.

## 1. Score orientation

Repository convention: larger scores support the genuine/target hypothesis. All verification and likelihood-ratio metrics use this convention.

## 2. `C_llr`

Reference definition checked against:

- N. Brümmer and E. de Villiers, *The BOSARIS Toolkit: Theory, Algorithms and Code for Surviving the New DCF*, arXiv:1304.2865.
- Idiap/Bob `bob.measure.calibration.cllr` documentation/source, which implements the BOSARIS definition.
- Forensic likelihood-ratio guidance that gives the same base-2 form and defines `C_llr^cal = C_llr - C_llr^min`.

For target log-likelihood ratios `l_t` and non-target log-likelihood ratios `l_n`:

`C_llr = 1/(2 N_t) sum log2(1 + exp(-l_t)) + 1/(2 N_n) sum log2(1 + exp(l_n))`.

Implementation uses `numpy.logaddexp` for numerical stability.

Validation cases committed in `tests/test_calibration_metrics.py`:

- zero LLR system -> `C_llr = 1`;
- symmetric one-target/one-non-target closed-form example;
- perfect-ranking `C_llr^min = 0`;
- all-tied scores -> `C_llr^min = 1`;
- fully reversed ranking -> monotonic PAV/isotonic optimum collapses to uninformative `C_llr^min = 1`.

## 3. `C_llr^min`

BOSARIS/Bob computes the minimum cost after a pool-adjacent-violators (PAV) monotonic calibration. The repository uses `sklearn.isotonic.IsotonicRegression`, which solves the same monotonic least-squares/PAV problem, then converts fitted posterior probabilities to LLRs after subtracting empirical prior log-odds.

Important interpretation: `min_cllr` is an **evaluation statistic**, not a deployable calibrator. It may use evaluation labels to measure discrimination under the best monotonic mapping; the actual model calibration used for headline `C_llr` must be fitted on development/validation data only.

Before Gate 5 is finally closed, numerical parity will also be checked on a fixed score vector against an independent BOSARIS-compatible implementation when environment compatibility permits.

## 4. EER variants

Two explicitly named variants are implemented:

- `eer`: intersection of the ordinary empirical ROC polyline with FAR=FRR;
- `eer_rocch`: EER on the upper ROC convex hull, following the BOSARIS interpretation.

The distinction matters for degenerate/harmful score orderings. In a perfectly reversed toy ranking, empirical interpolated EER is 1.0, whereas ROCCH-EER is 0.5 because the convex hull can fall back to the randomized no-information diagonal.

The final manuscript must state which EER definition is primary. The current protocol preference is ROCCH-EER for inferential reporting, with ordinary empirical EER retained as a sensitivity check if required for comparability with prior biometric papers.

## 5. TAR@FAR

`tar_at_far` selects the best **observed** ROC point satisfying `achieved FAR <= target FAR`; it does not interpolate to an empirically unattainable low FAR. This is a deliberate finite-sample conservatism rule.

Dataset lock must therefore verify that the number of impostor trials provides enough resolution for each requested FAR. Unsupported low-FAR targets will be removed before final evaluation rather than estimated optimistically.

## 6. Probability calibration metrics

Brier score, binary negative log-likelihood and equal-width ECE are implemented as complementary summaries. ECE bin count is a protocol parameter that must be fixed before final test analysis.

These probability metrics are not used to infer real-world prevalence. Their interpretation is conditional on the benchmark trial construction and calibration prior.

## 7. Open validation items

The metric core is not yet considered final until:

1. fixed-vector numerical parity for `C_llr` and `C_llr^min` is checked against an independent BOSARIS-compatible implementation;
2. ROCCH-EER is cross-checked on non-trivial score vectors against a trusted implementation;
3. clustered-bootstrap routines are validated on synthetic repeated-subject trial structures;
4. the final FAR grid is locked only after dataset trial counts are known.
