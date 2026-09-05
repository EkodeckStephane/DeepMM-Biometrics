# DeepMM-Biometrics — Q1 Scientific Article Gate Matrix

This project adapts the owner-supplied **Q1 Scientific-Article Editorial Gates** to the present biometrics study.

The project now has two explicitly separated validation stages:

- **V1:** bounded study on the official public NUPT-FPV subset already available;
- **V2:** later full-scale validation on the complete 33,600-image NUPT-FPV archive after official access is requested and obtained.

Closing a V1 gate never implies that the corresponding V2 person-level validation is complete.

| Gate | DeepMM-Biometrics requirement | V1 current status | Evidence required for V1 PASS |
|---|---|---|---|
| **G1 — Scientific object first** | The paper is organized around Q1-Q3, not around a named model or repository. | **PASS-DESIGN** | Frozen research questions + controlled benchmark protocol exist. |
| **G2 — Claim/evidence alignment** | Every conclusion about DL benefit, best family, calibration, robustness, or missing modalities maps to direct measurements. Negative results retained. | **PASS-DESIGN / EVIDENCE OPEN** | `claim_evidence_matrix.md` and planned contrasts exist; final claim-evidence matrix must link each headline statement to validated V1 run manifests/raw outputs. |
| **G3 — Final scientific narrative** | Final article contains the final valid study, not development/debug/audit history. | **RULE FIXED** | Editorial audit of final manuscript. |
| **G4 — Novelty/current SOTA** | No architectural-priority claim. Positioning must show what matched fusion-mechanism comparison remains unresolved. | **PASS-POSITIONING — 2026-09-05** | `docs/sota_matrix_v1.0.md` and `literature/gate4_search_log.md` freeze the bounded contribution after targeted falsification. No `first`/`only` wording is authorized; submission-time SOTA refresh remains mandatory. |
| **G5 — Experimental validity** | V1 uses genuine same-instance fingerprint/finger-vein evidence, image-sample-disjoint fit/selection/calibration/final roles, identical frozen trials across methods, held-out calibration, matched baselines and explicitly bounded inference. | **V1 PUBLIC SUBSET VERIFIED / PROTOCOL FREEZE IN PROGRESS** | The real public archive has passed the initial 800-file structure audit. V1 PASS still requires final trial hashes, frozen encoder/preprocessing/search budgets, final corruption/missingness plan, calibration procedure, compute protocol and an inference statement that does not treat the 20 public instance IDs as independent humans. |
| **G6 — Article structure/prose** | Problem → gap → comparison framework → protocol → results Q1/Q2/Q3 → discussion/limits. | **PLANNED** | Final manuscript structural audit. |
| **G7 — Scope/operational claims** | No claim of universal superiority, deployment readiness, spoof resistance, fairness, demographic validity, or complete-NUPT generalization unless directly measured. | **RULE FIXED** | Scope/threats/validity audit. |
| **G8 — Reproducibility** | Code, configs, public-subset scanner, trial manifests, seeds, raw scores, hashes, environment and regeneration scripts. | **ADVANCED INFRASTRUCTURE / REAL V1 EVIDENCE OPEN** | CI, metrics, classical baselines, D1/D2/D3S/D3F, fit/selection/calibration/final firewall, deterministic trainer, checkpoint hashing, public NUPT adapter, trial generation and real-subset smoke CI are versioned. Frozen encoder checkpoints/configs, V1 raw scores, stress outputs and end-to-end regeneration remain. |
| **G9 — Bibliographic/editorial hygiene** | Every reference exists and supports its citing sentence; same title/numbers across manuscript and submission files. | **OPEN / MACHINE CONSISTENCY ACTIVE** | Registry/BibTeX CI plus final claim-level source audit, correction/retraction check and manuscript/submission consistency audit. |
| **G10 — Submission readiness** | No contribution/evidence mismatch, unfair comparison, article-code-data contradiction, unverified reference, or venue-scope mismatch. | **NO-GO NOW** | Reviewer Senior final pass + all V1 prior gates closed. |

## V1 identity semantics and hard boundary

The public NUPT-FPV directories `001`–`020` are used as **biometric finger-instance identities**. The official public documentation does not establish how those 20 identifiers map to the 140 human volunteers in the complete database. Therefore:

- V1 may evaluate enrolled-instance verification on those public identities;
- V1 must not describe them as 20 independent persons, subjects, participants or volunteers;
- V1 must not report person-population confidence intervals or person-level hypothesis tests;
- instance-cluster resampling, if reported, is a sensitivity analysis for this bounded public benchmark, not human-population inference;
- the complete 140-volunteer/840-finger validation is explicitly deferred to V2 and appears in V1 limitations/perspectives.

See `docs/v1_v2_dataset_strategy.md`.

## Biometrics-specific hard stops for V1

The project must stop or reframe before V1 final scoring if any of the following holds:

1. Fingerprint and finger-vein evidence cannot be matched unambiguously at the public instance/session/capture level.
2. The public subset is too small for a selected fusion family and that family cannot be fairly constrained/frozen; such a family must be removed or made exploratory before final scoring.
3. The “best approach” criterion is defined only after final results are seen.
4. Classical and unimodal baselines receive less tuning/evaluation opportunity than the DL methods used for headline claims.
5. Missing-modality robustness is claimed without explicit missing-modality evaluation.
6. Calibration is inferred from discrimination metrics rather than measured directly, or the deployable calibrator sees final labels during fitting/tuning.
7. Raw neural logits are reported/interpreted as calibrated likelihood ratios without explicit held-out calibration.
8. Robustness corruptions/severities are selected after observing which model benefits.
9. Cost comparisons use different hardware, batch size, precision, or measurement procedures without correction/disclosure.
10. The 20 public biometric-instance identifiers, technical repetitions, or verification-pair counts are presented as independent human subjects.
11. Fit, model-selection, score-calibration and final-test **image samples** overlap.
12. Any final-test image/score is inspected to select model family, encoder, hyperparameters, corruption severity, calibration strategy or metric weighting.
13. A one-way cluster bootstrap is presented as person-level inference despite unresolved public instance-to-person mapping.
14. A model-family ranking is declared universal or representative of the complete NUPT-FPV database from V1 alone.
15. A Transformer/attention/missing-modality mechanism is presented as novelty even though that mechanism is established prior art.
16. Fixed-model classical score/feature fusion is presented as novel despite the OU-MB precedent.
17. `C_llr`, score calibration or calibration loss is presented as a new biometric concept rather than an established evaluation axis.
18. Any principal result lacks a frozen trial hash, score hash and run provenance record linking it to code/configuration/seed/condition.
19. A later paper is found that satisfies the complete locked Gate-4 contribution contract and the positioning is not reopened/revised.
20. The manuscript silently calls the V1 public subset “the NUPT-FPV dataset” in a way that implies use of all 33,600 images.

## Reviewer-Senior prescreen questions for V1

Before the V1 article draft is considered mature, a strict reviewer must be able to answer:

- What exactly does DL improve relative to classical and unimodal systems on the public NUPT-FPV benchmark?
- On which metrics does it fail to improve?
- Is an observed gain attributable to fusion rather than a stronger upstream representation?
- Which family is Pareto-optimal under the predeclared criteria, and how stable is that result under the bounded V1 sensitivity analysis?
- Does the ranking survive controlled degradation and explicit single-modality absence?
- Is calibration fitted outside final scoring and evaluated under the relevant availability/quality conditions?
- Are raw neural outputs clearly distinguished from calibrated LLR evidence?
- Are the 20 identities consistently called **public biometric instances**, not independent human subjects?
- Are fitting, early stopping/model selection, score calibration and final testing separated at the image-sample level?
- Is any architectural complexity unsupported by measurable benefit?
- Can every principal table/figure be regenerated from validated manifests and raw score outputs?
- Does the limitations section explicitly identify the unresolved person mapping and the complete 33,600-image V2 validation?

## Current scientific status

**Gate 4 remains PASS-POSITIONING. V1 is now authorized to proceed on the official public NUPT-FPV subset under a bounded public-instance verification protocol. The complete 33,600-image database is no longer a V1 blocker; it is the planned V2 scale/person-level validation. GO for freezing the V1 trial hashes, upstream encoder/preprocessing path, search budgets, stress plan and then running development without final-score inspection. NO-GO for V1 final claims until those remaining Gate-5 locks are frozen and audited.**
