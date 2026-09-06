# Implementation Plan v1.1

Gate 4 is closed at **PASS-POSITIONING** and the bounded V1 Q1--Q3 campaign is complete. This plan records the implemented evidence chain and the remaining article/V2 boundary.

## Work package A — evaluation core

Implemented and tested:
- binary verification trial schema;
- empirical EER and ROCCH-EER distinction;
- ROC-AUC;
- conservative TAR@FAR;
- Brier score, NLL, ECE;
- `C_llr`, `C_llr_min`, calibration-loss decomposition;
- held-out affine logistic score calibration;
- subject-disjoint split checks;
- duplicate/leakage checks;
- immutable split/trial/score hashing;
- run-manifest provenance.

**Status:** **V1 CORE COMPLETE.** Held-out calibration is executed. Dense symmetric impostor trials and unresolved person mapping still exclude person-population inference; this is a scope boundary, not an omitted V1 calculation.

## Work package B — common evidence/model interfaces

Implemented:
- explicit score and embedding information strata;
- canonical `ScoreEvidence` and `EmbeddingEvidence` objects;
- explicit enrollment/probe availability for embedding trials;
- explicit score-level availability;
- canonical zero placeholders for unavailable evidence, forbidding NaN/sentinel missingness;
- optional predeclared quality variables;
- method-information registry for C1/C2/C3/C4/C5/D1/D2/D3S/D3F;
- compatibility checks preventing a method from silently receiving a richer information tier;
- rule that labels never enter transform-time evidence objects.

Still data-dependent:
- encoder/token extractor interface and actual checkpoint semantics;
- token/local-feature contract for a possible D4 attention/Transformer representative;
- final fusion-only cost-accounting instrumentation on target hardware.

**Status:** **V1 SCORE/EMBEDDING AND COST CONTRACTS COMPLETE.** Token-level methods remain outside confirmatory V1.

## Work package C — classical baselines

Implemented:
- equal normalized score sum (C1);
- validation-weighted score fusion (C2);
- regularized logistic score fusion (C3);
- controlled feature concatenation (C4);
- classical quality-weighted score fusion (C5).

Synthetic sanity tests enforce deterministic behavior, correct modality dimensions, held-out transform semantics and quality handling.

**Status:** **IMPLEMENTED AND EVALUATED IN V1.** Final parameters and tuning opportunities were frozen before final access.

## Work package D — deep fusion heads

Gate-4-locked implementation order:
- D1 compact nonlinear score fusion;
- D2 compact nonlinear feature fusion;
- D3S/D3F quality/availability-aware learned gating;
- D4 attention/Transformer only if meaningful comparable token/local features are exposed by the selected upstream encoders.

Implementation will use a common family API and frozen search budgets. The goal is to compare families, not to maximize architecture count.

**Status:** **D1/D2/D3S IMPLEMENTED, SELECTED, CALIBRATED, AND FINALLY EVALUATED.** D3F/D4 remain outside the confirmatory V1 family set.

## Work package E — robustness harness

Planned modality-aware corruptions behind one deterministic API:
- blur;
- additive noise;
- downsampling;
- compression;
- exposure/contrast;
- localized occlusion.

Severity parameters remain configuration values and will be frozen before final test inspection.

**Status:** **V1 BLUR/CONTRAST PLAN IMPLEMENTED AND EXECUTED.** Other corruption families remain possible V2 extensions, not post-final V1 additions.

## Work package F — missing-modality harness

Planned:
- explicit availability masks (contract implemented);
- deterministic fallback;
- common modality-dropout training policy;
- availability-aware gating/missing representations.

Representation reconstruction/generation remains secondary unless the final SOTA/data/compute regime justifies a faithful comparison.

**Status:** **M0 FALLBACK IMPLEMENTED AND EXECUTED.** All fusion families tie when reduced to the same available unimodal evidence; a learned missingness advantage is not claimed.

## Work package G — reproducibility and evidence generation

Implemented infrastructure already supports:
- configuration/split/trial/score hashing;
- seed linkage;
- immutable run provenance;
- raw ordered per-trial score validation;
- failure-state semantics.

Completed for V1:
- environment and workflow provenance;
- checkpoint/model hashes;
- fusion-only timing/cost output;
- ordered trials, raw/calibrated scores and aggregate records;
- deterministic CSV, LaTeX-table and PGFPlots regeneration.

**Status:** **V1 EVIDENCE PACKAGE COMPLETE.**

## Work package H — automated scientific audits

Active/pre-final:
- leakage audit;
- metric tests;
- bibliographic registry/BibTeX/doc-DOI consistency;
- baseline-fairness contract;
- family-information compatibility;
- claim/evidence manifest planning.

Final campaign/submission:
- numerical cross-check (**complete for the V1 evidence package**);
- claim-level reference audit and correction/retraction refresh;
- Senior Reviewer prescreen;
- Q1 Gates 1–10 audit.

## Immediate execution order after V1 evidence closure

1. integrate the generated V1 tables/figures and drafted Results/Discussion into the complete article;
2. perform the final claim-level reference audit and submission-time SOTA refresh;
3. run the Senior Reviewer and title/number/terminology consistency audits;
4. retain G10 as NO-GO until those editorial checks pass;
5. obtain/audit the complete NUPT-FPV archive for independent V2 replication with verified person mapping and dependence-aware inference.

## Current implementation boundary

**GO:** bounded V1 reporting and independent V2 preparation.
**NO-GO:** post-final V1 model/configuration/condition changes, universal DL-superiority claims, person-population inference from the public identifiers, or complete-dataset claims from the subset.
