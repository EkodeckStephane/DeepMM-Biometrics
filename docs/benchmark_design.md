# Controlled Benchmark Design v0.1

**Title:** *Deep Learning Approaches for Multimodal Biometrics*

## 1. Scientific object

The study is a **controlled empirical benchmark of multimodal biometric fusion families**. Its primary scientific contribution is not a new named architecture. It is a matched protocol designed to determine:

- what deep learning adds beyond unimodal systems and strong classical fusion;
- which deep-fusion family gives the best multidimensional trade-off;
- whether that conclusion survives degradation and missing modalities.

This is a working novelty hypothesis until Gate 4 is closed by the systematic SOTA search.

## 2. Two-track experimental design

### Track I — Controlled fusion benchmark (primary)

Purpose: isolate the contribution of the **fusion mechanism**.

Rules:
- the same pretrained/frozen unimodal encoders feed all eligible fusion methods;
- the same embeddings/local features, splits, genuine/impostor trials, calibration data, and test trials are reused;
- only the fusion block differs;
- the tuning budget is matched by family;
- model selection uses development data only.

Track I is the primary evidence for Q1 and Q2 because it prevents a stronger backbone from masquerading as a better fusion strategy.

### Track II — Full-system optimization (secondary)

Purpose: estimate attainable system performance when each family is allowed a predeclared, matched fine-tuning budget.

Rules:
- encoder initialization is identical where the modality permits it;
- each family receives the same number of tuning trials and the same training-data access;
- differences in resolution/preprocessing required by an architecture are reported as part of system cost;
- Track II cannot overturn a Track-I conclusion about *fusion mechanism* without being explicitly described as a full-system effect.

Track II may be omitted if data volume or compute makes a fair comparison impossible.

## 3. Verification protocol

The target task is **identity verification**, not closed-set classification.

For a subject-disjoint split:

- training identities are used to fit encoders/fusion models;
- validation identities are used for hyperparameter selection, score normalization, probability/LLR calibration, and operating thresholds;
- test identities remain unseen until the final locked evaluation.

All headline systems must be scored on the exact same frozen test trial list.

## 4. Trial construction

The dataset adapter must produce:

- `train_subjects`;
- `val_subjects`;
- `test_subjects`;
- modality-specific sample manifests;
- genuine verification trials;
- impostor verification trials;
- acquisition/session metadata where available.

### Genuine trials

Where repeated sessions exist, prioritize **cross-session** genuine trials for the primary evaluation. Same-session trials may be reported separately.

### Impostor trials

Impostor sampling is generated once from a fixed seed and frozen. The number of impostor trials must support the lowest predeclared FAR operating point with adequate empirical resolution.

No system receives a different test-pair distribution.

## 5. Q1 experiment blocks

### Q1-A — unimodal versus multimodal

Compare every multimodal family with:
- modality A only;
- modality B only;
- best single modality.

Primary effect:
`multimodal metric - best-unimodal metric`, with direction adjusted for metrics where lower is better.

### Q1-B — classical versus DL fusion

Primary controlled contrasts:
- equal score sum vs deep score fusion;
- validation-weighted/logistic fusion vs deep score fusion;
- classical feature concatenation vs deep feature fusion;
- quality-aware classical weighting vs learned gated fusion when comparable quality variables exist.

### Q1-C — benefit-cost decomposition

For each family, report discrimination gain together with:
- additional trainable parameters in the fusion block;
- additional MACs/FLOPs where valid;
- latency overhead beyond the shared encoders;
- peak-memory overhead.

This prevents “better” from meaning only “more compute.”

## 6. Q2 experiment block

Q2 compares deep families on four locked dimensions:

1. **performance** — EER and predeclared TAR@FAR;
2. **robustness** — degradation loss and aggregate degradation curve;
3. **calibration** — calibrated score quality;
4. **cost** — fusion overhead and total inference cost.

Primary decision mechanism:
- per-dimension estimates and confidence intervals;
- Pareto frontier;
- uncertainty-aware dominance analysis.

There is no primary arbitrary weighted average.

## 7. Q3 degradation matrix

For two selected modalities A and B, every eligible model is evaluated under:

1. clean A + clean B;
2. degraded A + clean B;
3. clean A + degraded B;
4. degraded A + degraded B;
5. A present + B missing;
6. A missing + B present.

### Candidate image-domain corruptions

Only modality-valid corruptions are retained:
- Gaussian/defocus blur;
- additive noise;
- resolution/downsampling loss;
- JPEG compression;
- under/over-exposure or contrast shift;
- localized occlusion.

Severity levels are fixed before final evaluation.

### Q3 headline outcomes

- absolute performance drop from clean;
- relative performance drop;
- rank correlation with clean condition;
- pairwise rank reversals;
- worst-case condition within the predeclared stress set.

The scientific question is whether the identity of the preferred family changes, not simply whether all methods become worse.

## 8. Missing-modality comparison

Every primary family receives a predeclared missing-input policy.

Two analyses are separated:

### Native/fallback resilience

Evaluate the model using the simplest legal fallback for its architecture without retraining specifically for missingness.

### Trained missingness resilience

Where supported, retrain with the same modality-dropout schedule / availability-mask policy across eligible families.

This separation answers whether robustness comes from the *fusion family itself* or from explicit missing-modality training.

Dedicated reconstruction/generation methods may form a secondary Q3 comparison if implementation and data feasibility permit.

## 9. Calibration contract

Raw similarity scores are not automatically probabilities.

For every system:

1. keep the native score for discrimination metrics;
2. fit the same class of monotonic logistic calibration on validation trials only;
3. map scores to calibrated log-likelihood-ratio/probability form as appropriate;
4. evaluate calibration on untouched test trials.

Primary calibration metrics will include biometric-relevant log-likelihood-ratio calibration measures where implementation is validated, plus Brier/NLL and ECE as complementary probability summaries.

Calibration results are explicitly conditional on the frozen benchmark trial construction and prior assumptions; they are not claimed as population prevalence estimates.

## 10. Efficiency contract

Report two cost views:

### Fusion-only overhead

Measured after embeddings/local tokens are available.

### End-to-end cost

Includes modality encoders and fusion.

For each:
- parameters;
- MACs/FLOPs where counting is well-defined;
- CPU latency;
- GPU latency if a GPU environment is used;
- peak memory;
- model size.

Latency measurements use warm-up, fixed batch size, fixed precision, synchronized GPU timing where applicable, and repeated technical runs. Technical timing repeats are not treated as independent statistical subjects.

## 11. Tuning-budget fairness

Each trainable fusion family receives a fixed maximum number of model-selection trials. Search spaces are documented before the final run.

A family is not allowed substantially more manual tuning because initial results are disappointing.

Pilot runs may be used to detect broken implementations. Once the final protocol is frozen, pilot-driven architecture changes are prohibited for headline comparisons.

## 12. Anti-leakage tests required before training

Automated tests must verify:
- subject disjointness across train/validation/test;
- no duplicate file/hash across splits;
- no test label enters preprocessing fit, normalization, calibration, threshold selection, early stopping, or hyperparameter search;
- test trial list is immutable after lock;
- any pretrained encoder training corpus overlap that can be identified is disclosed.

## 13. Minimum evidence package

For every headline model/seed:
- config;
- software versions;
- seed;
- split hash;
- trial-list hash;
- model/checkpoint hash where feasible;
- per-trial native scores;
- calibrated scores;
- aggregate metrics;
- runtime metrics;
- training log;
- failure status if the run did not complete.

Failed or negative runs are not silently removed if they affect the planned comparison.
