# Statistical Analysis Plan v0.2

**Study:** *Deep Learning Approaches for Multimodal Biometrics*

This document is a preregistration-style design. Numerical choices that depend on the selected dataset (number of subjects, exact FAR grid, severity values, seeds) remain `TO LOCK` until dataset audit, but the inferential structure is fixed now.

## 1. Experimental unit and dependence

Verification trials are not independent when multiple trials share a biometric subject, enrollment template, probe sample, session, or impostor identity. Therefore:

- raw trial count is **not** treated as the biological/statistical sample size;
- uncertainty estimation and paired comparisons must preserve the trial dependence induced by biometric identities;
- technical reruns and latency repeats are not independent subjects.

For an explicitly **subject-centric** verification protocol in which every trial belongs to one predeclared anchor identity, the primary uncertainty procedure is a subject-clustered bootstrap: sample test identities/clusters with replacement and retain the corresponding trial blocks under a deterministic reconstruction rule.

For dense **symmetric all-vs-all** impostor protocols, one-way clustering is not assumed valid because an impostor pair depends on two identities. Such a design must use a validated subsets/multiway bootstrap or a deterministic identity-resampling reconstruction that preserves both sides of the pair. This choice is locked only after the final trial-generation scheme is known.

Biometric-specific precedent for structured resampling is documented in `docs/bootstrap_protocol.md`, including Bolle, Ratha & Pankanti's subsets bootstrap (CVIU 2004, DOI `10.1016/j.cviu.2003.08.002`).

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

The repository distinguishes ordinary empirical ROC-polyline EER from ROC-convex-hull EER. The primary variant is locked before final testing and named explicitly in all tables.

Secondary:
- ROC-AUC;
- DET/ROC curves;
- threshold-dependent FAR/FRR where operationally interpretable.

The FAR grid is fixed only after the number and dependence structure of available impostor trials is known. A FAR target is inadmissible if the empirical trial count cannot resolve it responsibly.

## 4. Calibration metrics

Calibration is evaluated after fitting a common monotonic/logistic score calibration on validation trials only.

Primary biometric-specific measures, provided implementation validation succeeds:
- `C_llr`;
- `C_llr_min` (discrimination component under optimal monotonic calibration);
- `C_llr_cal = C_llr - C_llr_min` (calibration loss).

Complementary probability measures:
- negative log-likelihood;
- Brier score;
- Expected Calibration Error (ECE), with the binning rule fixed before test evaluation;
- reliability diagram.

Rationale: biometric score-calibration literature uses likelihood-ratio calibration and `C_llr`; generic ECE alone is insufficient as the sole calibration claim.

Important boundary: probability-based calibration metrics are conditional on the benchmark trial construction and prior assumptions. They are not interpreted as real-world prevalence probabilities.

For Q3, calibration is recomputed/reported by locked modality-availability subset and stress condition where sample size permits. One aggregate clean-condition calibration number cannot support a claim of reliable operation under missing modalities.

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
- 95% dependence-aware bootstrap confidence interval;
- paired cluster-level randomization/permutation test when its exchangeability assumptions match the final subject-centric design;
- Holm-adjusted p-value for the predeclared contrast family when p-values are used;
- practical effect size in the native metric.

The repository implementation swaps complete subject-cluster score blocks between systems for a paired randomization test. For small numbers of clusters it enumerates all swap assignments; otherwise it uses a seeded Monte-Carlo test with the +1 correction. This implementation is not used for dense symmetric all-vs-all protocols unless the dependence treatment is upgraded appropriately.

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

All methods are evaluated on matched bootstrap replicates. For each pair `(A,B)`, compute the probability that A Pareto-dominates B over the locked metric vector.

Also report the probability that each method lies on the non-dominated frontier across bootstrap replicates. This prevents a noisy point-estimate frontier from being presented as deterministic evidence.

### 7.4 Scalar utility

Not primary. If used, weights and normalization are frozen before test evaluation and sensitivity analysis is mandatory.

## 8. Q3 rank stability and reversal

For every stress condition/severity:

- recompute family rankings;
- compute Kendall tau-b against the clean ranking, preserving ties;
- identify pairwise rank reversals;
- estimate bootstrap support for each reversal when valid bootstrap replicates exist;
- report whether the clean-condition Pareto-optimal set remains non-dominated;
- report subset-specific calibration where supported by sample size.

A rank reversal is scientifically reported even when it is unfavorable to the most complex DL family.

## 9. Multiplicity control

Primary planned inferential families use **Holm correction** unless a stronger domain-specific reason emerges before final lock. The correction operates only on a predeclared family of tests; it is not a device for selecting favorable comparisons.

Exploratory tests are labeled exploratory and are not mixed with confirmatory claims.

No post-hoc subset of favorable comparisons is promoted to primary evidence.

## 10. Hyperparameter/model-selection isolation

Test labels are consumed only after:
- architecture/fusion-family list frozen;
- tuning budgets exhausted/frozen;
- model checkpoints selected on development data;
- deployable calibration fitted on validation data;
- thresholds selected on validation data;
- stress severities fixed;
- statistical comparison list frozen.

Evaluation-only statistics such as `C_llr_min` may use test labels by definition to characterize the score ranking under an optimal monotonic mapping; they are never reused as deployable calibrators or model-selection signals.

Any accidental premature test access for model development triggers a new test split if the dataset permits it; otherwise the affected result cannot be called confirmatory.

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
- enough test identities for dependence-aware uncertainty estimation;
- enough genuine cross-session trials;
- enough impostor trials to resolve the lowest FAR target;
- enough training identities relative to trainable fusion capacity;
- enough validation trials for calibration without reusing test data;
- enough observations in each missing-modality/stress subset for the intended calibration and ranking analyses.

If these conditions fail, the dataset can be used for pilot/secondary evidence but not as the sole basis for the headline claims.

## 14. Statistical software validation

Before scientific use, metric/statistic implementations are validated against:
- analytic toy examples;
- a trusted external implementation where available;
- invariance/sanity tests;
- deterministic seeded cluster-bootstrap tests;
- identical-system paired tests yielding zero effect;
- exact small-cluster randomization cases;
- known Holm-adjustment examples;
- synthetic Pareto and rank-reversal cases.

The final dependence-aware bootstrap is additionally checked after the dataset/trial structure is frozen. This is necessary because correct resampling depends on how subjects participate in genuine and impostor comparisons.

## 15. Reporting rule

The manuscript reports exact numerical evidence only after the corresponding raw score files and regeneration scripts exist. Tables and figures must be generated programmatically from frozen result files.
