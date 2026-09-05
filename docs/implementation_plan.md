# Implementation Plan v0.1

The implementation starts with **dataset-agnostic infrastructure**. No architecture is allowed to become the organizing principle of the project.

## Work package A — evaluation core

Implement and test:
- binary verification trial schema;
- EER;
- ROC-AUC;
- conservative TAR@FAR;
- Brier score, NLL, ECE;
- `C_llr` once LLR calibration is validated;
- subject-disjoint split checks;
- duplicate/leakage checks;
- immutable split/trial hashing.

**Exit criterion:** deterministic unit tests and toy analytical cases pass.

## Work package B — common model interfaces

Define a common interface for:
- unimodal encoders;
- embedding/token extractors;
- score-level fusion;
- feature-level fusion;
- availability masks;
- quality variables;
- fusion-only cost accounting.

**Exit criterion:** classical and DL heads consume the same canonical batch/trial representation.

## Work package C — classical baselines

Implement first:
- equal normalized score sum;
- validation-weighted score fusion;
- regularized logistic score fusion;
- classical feature concatenation baseline where sample size supports it.

**Exit criterion:** synthetic sanity tests show expected invariances and no test-label access.

## Work package D — deep fusion heads

Implement in the frozen order:
- deep score MLP;
- deep feature fusion;
- quality-aware gated fusion;
- one attention/Transformer representative after token interface is justified.

The purpose is to compare *families*, not to maximize architecture count.

## Work package E — robustness harness

Implement modality-aware corruptions behind one deterministic API:
- blur;
- additive noise;
- downsampling;
- compression;
- exposure/contrast;
- localized occlusion.

Severity parameters are configuration values, not hard-coded after results are seen.

## Work package F — missing-modality harness

Implement:
- availability masks;
- deterministic fallback;
- common modality-dropout training policy;
- availability-aware gating/missing tokens.

Representation reconstruction/generation is deferred until SOTA and compute feasibility justify it.

## Work package G — reproducibility and evidence generation

Every experiment emits:
- configuration snapshot;
- environment metadata;
- split/trial hashes;
- seed;
- raw per-trial scores;
- aggregate metrics;
- timing/cost output;
- failure state;
- table/figure-ready machine-readable result.

## Work package H — automated scientific audits

Before final campaign:
- leakage audit;
- metric unit-test audit;
- baseline-fairness audit;
- model/tuning-budget audit;
- claim/evidence manifest validation.

After final campaign:
- numerical cross-check;
- reference audit;
- Senior Reviewer Q1 prescreen;
- Q1 Gates 1–10 audit.

## Current implementation boundary

We can complete A–B and most of C before a dataset is locked. We should **not** optimize high-capacity fusion architectures or choose image-specific preprocessing until the primary data regime is known.
