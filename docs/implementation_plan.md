# Implementation Plan v1.0

Gate 4 is closed at **PASS-POSITIONING**. Implementation remains dataset-conscious: infrastructure and low-capacity controlled baselines may advance now, while high-capacity/token-specific architectures wait for the selected data and encoder regime.

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

**Status:** **CORE IMPLEMENTED; external calibration parity/data-dependent inference locks remain.** Issue #6 still blocks final calibration claims, and issue #5 blocks a convenient one-way bootstrap for dense symmetric impostor protocols.

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

**Status:** **SCORE/EMBEDDING CONTRACT IMPLEMENTED; TOKEN/COST SUBCONTRACT OPEN.** See `docs/fusion_evidence_contract.md` and issue #7.

## Work package C — classical baselines

Implemented:
- equal normalized score sum (C1);
- validation-weighted score fusion (C2);
- regularized logistic score fusion (C3);
- controlled feature concatenation (C4);
- classical quality-weighted score fusion (C5).

Synthetic sanity tests enforce deterministic behavior, correct modality dimensions, held-out transform semantics and quality handling.

**Status:** **IMPLEMENTED FOR PILOT.** Final parameter/tuning fairness is frozen after the real data dimensionality and train/development counts are known.

## Work package D — deep fusion heads

Gate-4-locked implementation order:
- D1 compact nonlinear score fusion;
- D2 compact nonlinear feature fusion;
- D3S/D3F quality/availability-aware learned gating;
- D4 attention/Transformer only if meaningful comparable token/local features are exposed by the selected upstream encoders.

Implementation will use a common family API and frozen search budgets. The goal is to compare families, not to maximize architecture count.

**Status:** **NEXT DATASET-AGNOSTIC CODING TARGET = common neural-head interface and D1/D2/D3 skeletons.** No final training/tuning yet.

## Work package E — robustness harness

Planned modality-aware corruptions behind one deterministic API:
- blur;
- additive noise;
- downsampling;
- compression;
- exposure/contrast;
- localized occlusion.

Severity parameters remain configuration values and will be frozen before final test inspection.

**Status:** **API can be implemented next; exact corruption families/severities remain modality/data dependent.**

## Work package F — missing-modality harness

Planned:
- explicit availability masks (contract implemented);
- deterministic fallback;
- common modality-dropout training policy;
- availability-aware gating/missing representations.

Representation reconstruction/generation remains secondary unless the final SOTA/data/compute regime justifies a faithful comparison.

**Status:** **MASK CONTRACT IMPLEMENTED; operational policies/training axis OPEN.**

## Work package G — reproducibility and evidence generation

Implemented infrastructure already supports:
- configuration/split/trial/score hashing;
- seed linkage;
- immutable run provenance;
- raw ordered per-trial score validation;
- failure-state semantics.

Still required for real experiments:
- environment snapshot;
- checkpoint/model hash;
- target-hardware timing/cost output;
- generated aggregate-result records;
- table/figure regeneration scripts.

**Status:** **ADVANCED INFRASTRUCTURE; REAL-RUN ARTIFACTS OPEN.**

## Work package H — automated scientific audits

Active/pre-final:
- leakage audit;
- metric tests;
- bibliographic registry/BibTeX/doc-DOI consistency;
- baseline-fairness contract;
- family-information compatibility;
- claim/evidence manifest planning.

Final campaign/submission:
- numerical cross-check;
- claim-level reference audit and correction/retraction refresh;
- Senior Reviewer prescreen;
- Q1 Gates 1–10 audit.

## Immediate execution order after Gate 4

1. finish common neural-head interfaces without committing to a high-capacity architecture;
2. implement deterministic robustness and missingness harness APIs;
3. build an end-to-end **synthetic pipeline test** that exercises trials → evidence → fusion → metrics → calibration → paired statistics → Pareto/rank output, explicitly marked as CI evidence only;
4. when ready, screen directly accessible genuine multimodal datasets against the frozen inclusion contract;
5. lock real data, modality pair/subset, upstream encoders and trial topology;
6. resolve issues #5, #6 and #7 from the actual data/protocol;
7. run a pilot on development data only;
8. freeze hyperparameter budgets, stress severities, seeds/runs and confirmatory family list;
9. only then authorize final-test evaluation.

## Current implementation boundary

**GO:** interfaces, deterministic harnesses, synthetic end-to-end validation, dataset-access screening, development-only pilot preparation.  
**NO-GO:** final-test model selection, post-hoc family additions, token/Transformer privilege, or image-specific severity choices before data/protocol lock.
