# Statistical Analysis Plan v0.1

**Study:** *Deep Learning Approaches for Multimodal Biometrics*

This document is a preregistration-style design. Numerical choices that depend on the selected dataset (number of subjects, exact FAR grid, severity values, seeds) remain `TO LOCK` until dataset audit, but the inferential structure is fixed now.

## 1. Experimental unit and dependence

Verification trials are not independent when multiple trials share a biometric subject. Therefore:

- raw trial count is **not** treated as the biological/statistical sample size;
- uncertainty estimation and paired comparisons must preserve subject-level dependence;
- technical reruns and latency repeats are not independent subjects.

Primary uncertainty procedure: **subject-clustered bootstrap**, resampling test identities with replacement and retaining the corresponding genuine/impostor trial structure under a deterministic reconstruction rule.

The exact clustered-trial reconstruction algorithm will be tested on synthetic data before use.

## 2. Repeated model training

The final number of independent training seeds is `TO LOCK` based on compute feasibility, with a target of at least five for headline trainable fusion families when practicable.

Seeds are fixed before final testing and shared across families.

Reported quantities distinguish:
- between-seed training variability;
- within-test-set statistical uncertainty;
- technical timing variability.

These are not pooled indiscriminately.

## 3. Discrimination metrics

Primary:
- Equal Error Rate (EER), lower is better;
- TAR at predeclared FAR operating points, higher is better.

Secondary:
- ROC-AUC;
- DET/ROC curves;
- threshold-dependent FAR/FRR where operationally interpretable.

The FAR grid is fixed only after the number of available impostor trials is known. A FAR target is inadmissible if the empirical trial count cannot resolve it responsibly.

## 4. Calibration metrics

Calibration is evaluated after fitting a common monotonic logistic score calibration on validation trials only.

Primary biometric-specific measures, provided implementation validation succeeds:
- `C_llr`;
- `C_llr_min` (discrimination component);
- `C_llr_cal = C_llr - C_llr_min` (calibration loss).

Complementary probability measures:
- negative log-likelihood;
- Brier score;
- Expected Calibration Error (ECE), with the binning rule fixed before test evaluation;
- reliability diagram.

Rationale: face/biometric score-calibration literature uses log-likelihood-ratio calibration and `C_llr`; generic ECE alone is insufficient as the sole calibration claim.

Important boundary: probability-based calibration metrics are conditional on the benchmark trial construction and prior assumptions. They are not interpreted as real-world prevalence probabilities.

## 5. Robustness metrics

For method `m`, condition `c`, and clean condition `0`:

- absolute degradation: `Delta_abs(m,c) = metric(m,c) - metric(m,0)` with sign convention documented;
- relative degradation: normalized by the clean metric when numerically stable;
- aggregate degradation score: area under the performance-versus-severity curve using the same severity grid for all methods.

For lower-is-better metrics such as EER, robustness loss is defined so larger positive values always mean worse degradation.

## 6. Q1 planned contrasts

The primary contrast family is predeclared:

1. best unimodal vs equal-score fusion;
2. best unimodal vs best classical fusion;
3. best classical score fusion vs deep score fusion;
4. classical feature fusion vs deep feature fusion;
5. quality-aware classical fusion vs quality-aware gated DL fusion when the same quality inputs are available;
6. best classical fusion vs each retained headline DL family.

For each contrast report:
- paired point difference;
- 95% clustered-bootstrap confidence interval;
- paired bootstrap probability / two-sided inferential test as appropriate;
- corrected p-value for the planned family when a p-value is used;
- practical effect size in the native metric.

No conclusion of superiority is based on overlapping/non-overlapping error bars alone.

## 7. Q2 multidimensional comparison

No single metric determines the winner.

Dimensions:
- discrimination;
- robustness;
- calibration;
- computational cost.

### 7.1 Dimension-specific ranking

Rank each family within each dimension, retaining uncertainty.

### 7.2 Point-estimate Pareto frontier

A family is Pareto-dominated if another is at least as good on all locked dimensions and strictly better on at least one.

### 7.3 Bootstrap dominance probability

For each pair `(A,B)`, compute across bootstrap replicates the probability that A dominates B over the locked metric vector.

This gives an uncertainty-aware alternative to declaring a deterministic frontier from noisy point estimates.

### 7.4 Scalar utility

Not primary. If used, weights and normalization are frozen before test evaluation and sensitivity analysis is mandatory.

## 8. Q3 rank stability and reversal

For every stress condition/severity:

- recompute family rankings;
- compute Kendall rank correlation against the clean ranking;
- identify pairwise rank reversals;
- estimate bootstrap support for each reversal;
- report whether the clean-condition preferred family remains non-dominated.

A rank reversal is scientifically reported even when it is unfavorable to the most complex DL family.

## 9. Multiplicity control

Primary planned inferential families use **Holm correction** unless a stronger domain-specific reason emerges before final lock.

Exploratory tests are labeled exploratory and are not mixed with confirmatory claims.

No post-hoc subset of favorable comparisons is promoted to primary evidence.

## 10. Hyperparameter/model-selection isolation

Test labels are consumed only after:
- architecture list frozen;
- tuning budgets exhausted/frozen;
- model checkpoints selected on development data;
- calibration fitted on validation data;
- thresholds selected on validation data;
- stress severities fixed;
- statistical comparison list frozen.

Any accidental premature test access triggers a new test split if the dataset permits it; otherwise the affected result cannot be called confirmatory.

## 11. Missing runs and failures

A planned run that fails due to memory, instability, or convergence is recorded.

Rules:
- implementation bugs are fixed and rerun with traceability;
- resource failure is reported as part of feasibility/cost if it persists under the locked budget;
- a failed seed is not silently discarded and replaced until the average improves;
- exclusions require a predeclared technical criterion or an explicit deviation note.

## 12. Timing analysis

Latency repetitions quantify measurement noise, not inferential sample size.

Protocol:
- fixed device;
- fixed software stack;
- fixed precision;
- fixed batch size (batch 1 primary);
- warm-up runs excluded by rule;
- synchronized GPU events when needed;
- report median, interquartile range, and tail percentile over technical repeats.

Hardware changes create a new timing stratum and are not pooled with prior results.

## 13. Minimum sample-size feasibility checks

Before dataset lock, verify:
- enough test identities for clustered uncertainty estimation;
- enough genuine cross-session trials;
- enough impostor trials to resolve the lowest FAR target;
- enough training identities relative to trainable fusion capacity;
- enough validation trials for calibration without reusing test data.

If these conditions fail, the dataset can be used for pilot/secondary evidence but not as the sole basis for the headline claims.

## 14. Statistical software validation

Before scientific use, metric/statistic implementations will be validated against:
- analytic toy examples;
- a trusted external implementation where available;
- invariance tests (score monotonicity where applicable, label permutation sanity checks, identical-score equality checks);
- deterministic seeded bootstrap tests.

## 15. Reporting rule

The manuscript reports exact numerical evidence only after the corresponding raw score files and regeneration scripts exist. Tables and figures must be generated programmatically from frozen result files.
