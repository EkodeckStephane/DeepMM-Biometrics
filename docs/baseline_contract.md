# Baseline Contract v0.1

**Study:** *Deep Learning Approaches for Multimodal Biometrics*

The purpose of the baseline set is to make Q1 falsifiable. A deep fusion method cannot be credited for a gain that can already be obtained by score normalization, tuned linear weighting, logistic score fusion, direct feature access, or simple quality weighting.

## Unimodal anchors

For every selected biometric modality, the benchmark uses one strong reproducible unimodal encoder/matcher pipeline. Track I freezes these representations so all eligible fusion mechanisms receive the same unimodal evidence.

The best unimodal system is a mandatory comparator. Beating only the weaker modality does not establish a multimodal gain.

## C1 — Equal normalized score fusion

Repository class: `EqualScoreFusion`.

- fit per-modality normalization on development scores only;
- assign equal weights to all modalities;
- no class labels are used to fit the fusion rule;
- missing modalities are handled only through the separate missingness protocol.

Purpose: minimum-complexity multimodal reference.

## C2 — Validation-weighted score fusion

Repository class: `WeightedScoreFusion`.

- z-score normalization frozen from development data;
- non-negative weights constrained to the simplex;
- exhaustive finite grid search;
- objective and grid fixed before final test evaluation;
- deterministic tie-breaking favors the less extreme weight vector;
- candidate-count cap prevents hidden uncontrolled optimization.

Purpose: tests whether a claimed nonlinear/deep score-fusion gain survives a well-tuned linear weighting baseline.

## C3 — Logistic score fusion

Repository class: `LogisticScoreFusion`.

- regularized logistic regression;
- development-only score normalization and fitting;
- decision function used as the native fused verification score;
- genuine-class probability may be used only under the declared calibration/evaluation contract.

This is a **classical learned fusion** baseline, not deep learning.

Purpose: strong low-capacity learned score fusion.

## C4 — Standardized feature concatenation

Repository class: `StandardizedConcatFusion`.

- aligned modality embeddings are standardized using training-only statistics;
- each modality block is L2-normalized before concatenation so dimensionality alone does not give one modality larger vector energy;
- modality blocks receive equal scaling;
- concatenated embedding is L2-normalized and compared with the same predeclared matcher, initially cosine unless modality-specific evidence requires another matched rule.

Purpose: tests whether access to feature-level evidence alone explains a gain attributed to a deep feature-fusion head.

Row alignment is a scientific requirement. The dataset adapter must define what constitutes one multimodal template/sample unit; independent unimodal samples are not silently paired merely to satisfy the API.

## C5 — Classical quality-aware score fusion

Repository class: `QualityWeightedScoreFusion`.

For score `z_m` and externally supplied quality `q_m`, the baseline uses dynamic weights proportional to `q_m^gamma`, with `gamma` selected from a frozen development-data grid.

Rules:

- the same quality variables must be available to any DL gate used in the corresponding headline contrast;
- the quality extractor is either common/frozen across methods or its contribution is analyzed separately;
- a quality value is not used implicitly as a missing-modality marker unless the missingness protocol explicitly defines that behavior;
- the gamma grid and selection objective are frozen before test evaluation.

Purpose: a DL quality gate must beat a meaningful classical quality-aware comparator, not only equal weighting.

## Fair tuning rule

The exact tuning budget is locked after the final family set and compute budget are known. The governing rules are already fixed:

1. every trainable/tunable method receives a documented search space and budget;
2. classical baselines are not intentionally under-tuned;
3. a deep method is not granted repeated manual redesign after observing final test behavior;
4. test labels are never used to select baseline weights, regularization, quality exponent, network hyperparameters, checkpoints, thresholds or calibration;
5. pilot runs may detect broken implementations, but cannot be mined to create a favorable final comparator set.

## Primary Q1 contrasts supported by the baseline set

- best unimodal vs C1/C2/C3;
- C1 vs D1 deep score fusion;
- C2/C3 vs D1 deep score fusion;
- C4 vs D2 deep feature fusion;
- C5 vs D3 learned quality-aware/gated fusion;
- best classical baseline vs each retained headline DL family.

The article reports the whole planned contrast family, including negative or null results.

## Not yet frozen

The following remain dependent on the final SOTA/data regime:

- whether classical LDA/PLDA/SVM feature fusion is required in addition to C4;
- exact score-normalization alternatives beyond z-score;
- regularization grid for C3;
- quality-estimator definition and quality gamma grid;
- tuning budget per family;
- whether more than two/three modalities are used in the headline benchmark.

No extra baseline will be added solely because it produces a favorable pilot ranking.
