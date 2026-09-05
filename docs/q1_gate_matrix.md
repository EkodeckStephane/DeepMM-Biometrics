# DeepMM-Biometrics — Q1 Scientific Article Gate Matrix

This project adapts the owner-supplied **Q1 Scientific-Article Editorial Gates** to the present biometrics study.

| Gate | DeepMM-Biometrics requirement | Current status | Evidence required for PASS |
|---|---|---|---|
| **G1 — Scientific object first** | The paper is organized around Q1-Q3, not around a named model or repository. | **PASS-DESIGN** | Frozen research questions + controlled benchmark protocol exist. |
| **G2 — Claim/evidence alignment** | Every conclusion about DL benefit, best family, calibration, robustness, or missing modalities maps to direct measurements. Negative results retained. | **PASS-DESIGN / EVIDENCE OPEN** | `claim_evidence_matrix.md` and planned contrasts exist; final claim-evidence matrix must link each headline statement to validated run manifests/raw outputs. |
| **G3 — Final scientific narrative** | Final article contains the final valid study, not development/debug/audit history. | **RULE FIXED** | Editorial audit of final manuscript. |
| **G4 — Novelty/current SOTA** | No architectural-priority claim. Positioning must show what matched fusion-mechanism comparison remains unresolved. | **IN PROGRESS — CRITICAL, BOUNDARY STRONGLY NARROWED** | `sota_matrix_v0.6.md` now incorporates historical quality/cost/missingness/calibration work and OU-MB's fixed-model score/feature fusion benchmark. Remaining targeted falsification searches must still exclude a benchmark combining representative DL fusion families, calibration, stress/missingness, cost and uncertainty-aware rank/Pareto stability before wording is locked. |
| **G5 — Experimental validity** | Real multimodal identity correspondence, leakage-free splits, matched baselines, correct experimental unit, dependence-aware uncertainty, paired tests, multiplicity correction. | **DESIGN/INFRASTRUCTURE READY; DATA-DEPENDENT LOCK OPEN** | Trial/score manifest validation, subject-split checks, paired statistics, calibration protocol and hard failure rules exist. Final dataset, trial topology, dense-impostor resampling rule, FAR grid, calibration partitions, corruption severities and run count remain to lock. |
| **G6 — Article structure/prose** | Problem → gap → comparison framework → protocol → results Q1/Q2/Q3 → discussion/limits. | **PLANNED** | Final manuscript structural audit. |
| **G7 — Scope/operational claims** | No claim of universal superiority, deployment readiness, spoof resistance, fairness, or security improvement unless directly measured. | **RULE FIXED** | Scope/threats/validity audit. |
| **G8 — Reproducibility** | Code, configs, split manifests, trials, seeds, raw scores, hashes, environment, regeneration scripts; restricted datasets referenced rather than redistributed. | **ADVANCED INFRASTRUCTURE / FINAL EVIDENCE OPEN** | CI, metrics, baselines, statistics, split hashing, frozen trial/score contracts and immutable run-manifest provenance are versioned. Final real-data manifests, configs, checkpoints, environments, raw scores and regeneration scripts remain. |
| **G9 — Bibliographic/editorial hygiene** | Every reference exists and supports its citing sentence; same title/numbers across manuscript and submission files. | **OPEN / MACHINE CONSISTENCY ACTIVE** | `literature/sota_registry.csv` and `references.bib` are CI-synchronized; final claim-level source audit, correction/retraction check and manuscript/submission consistency audit remain. |
| **G10 — Submission readiness** | No contribution/evidence mismatch, unfair comparison, article-code-data contradiction, unverified reference, or journal-scope mismatch. | **NO-GO NOW** | Reviewer Senior final pass + all prior gates closed. |

## Biometrics-specific hard stops

The project must stop or reframe before the final campaign if any of the following holds:

1. Primary multimodal identities are not genuinely matched across modalities.
2. The dataset is too small to compare the chosen model families fairly and no justified frozen/pretrained-encoder strategy is possible.
3. The “best approach” criterion is defined only after test results are seen.
4. Classical and unimodal baselines receive less tuning/evaluation budget than the DL methods used for headline claims.
5. Missing-modality robustness is claimed without explicit missing-modality evaluation.
6. Calibration is inferred from discrimination metrics rather than measured directly, or a deployable calibrator is fitted/tuned with final-test labels.
7. Robustness uses corruptions/severities selected after seeing which model benefits.
8. Cost comparisons use different hardware, batch size, precision, or measurement procedures without correction/disclosure.
9. Technical training repetitions or raw verification-pair counts are treated as independent biometric subjects.
10. A one-way cluster bootstrap is used on dense symmetric all-vs-all impostor pairs without validating the dependence structure.
11. A model-family ranking is declared universal from a single dataset without bounded claims or generalization evidence.
12. A Transformer/attention/missing-modality mechanism is presented as novelty even though that mechanism is already established prior art.
13. Fixed-model classical score/feature fusion is presented as novel despite the OU-MB 2026 baseline precedent.
14. `C_llr`, score calibration or calibration loss is presented as a new biometric concept rather than an established evaluation axis.
15. Any principal result lacks a frozen trial hash, score hash and run provenance record linking it to code/configuration/seed/condition.

## Reviewer-Senior prescreen questions

Before any article draft is considered mature, a strict reviewer must be able to answer:

- What exactly does DL improve relative to classical and unimodal systems?
- On which metrics does it fail to improve?
- Is the observed gain attributable to **fusion**, rather than a stronger backbone/representation?
- What does the DL-family comparison add beyond OU-MB's fixed-model mean/weighted-score/concatenation baselines?
- Which family is Pareto-optimal under the locked criteria, and with what uncertainty?
- Are apparent gains larger than dependence-aware uncertainty and statistically defensible after multiplicity control?
- Does the ranking survive degradation and missing modalities?
- Is calibration evaluated under the relevant modality-availability/quality conditions, with held-out fitting?
- Are the data truly multimodal at subject level?
- Is any architectural complexity unsupported by measurable benefit?
- Can every principal table/figure be regenerated from validated run manifests and raw score outputs?

## Current scientific status

**GO for infrastructure completion, Gate-4 closure and pilot preparation. NO-GO for a final training campaign or manuscript claims.**
