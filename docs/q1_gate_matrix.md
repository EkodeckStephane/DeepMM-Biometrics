# DeepMM-Biometrics — Q1 Scientific Article Gate Matrix

This project adapts the owner-supplied **Q1 Scientific-Article Editorial Gates** to the present biometrics study.

| Gate | DeepMM-Biometrics requirement | Current status | Evidence required for PASS |
|---|---|---|---|
| **G1 — Scientific object first** | The paper is organized around Q1-Q3, not around a named model or repository. | **PASS-DESIGN** | Frozen research questions + controlled benchmark protocol exist. |
| **G2 — Claim/evidence alignment** | Every conclusion about DL benefit, best family, calibration, robustness, or missing modalities maps to direct measurements. Negative results retained. | **PASS-DESIGN / EVIDENCE OPEN** | `claim_evidence_matrix.md` and planned contrasts exist; final claim-evidence matrix must link each headline statement to validated run manifests/raw outputs. |
| **G3 — Final scientific narrative** | Final article contains the final valid study, not development/debug/audit history. | **RULE FIXED** | Editorial audit of final manuscript. |
| **G4 — Novelty/current SOTA** | No architectural-priority claim. Positioning must show what matched fusion-mechanism comparison remains unresolved. | **PASS-POSITIONING — 2026-09-05** | `docs/sota_matrix_v1.0.md` and `literature/gate4_search_log.md` freeze the bounded contribution after targeted falsification against OU-MB, LUTBIO, calibration, flexible/missing-modality, adaptation and adversarial fusion precedents. No `first`/`only` wording is authorized; submission-time SOTA refresh remains mandatory. |
| **G5 — Experimental validity** | Real multimodal identity correspondence, leakage-free splits, matched baselines, correct experimental unit, dependence-aware uncertainty, paired tests, multiplicity correction. | **P1 ACCESS CANDIDATE SELECTED; ARCHIVE/DATA LOCK OPEN** | NUPT-FPV is the current P1 access candidate. `docs/dataset_feasibility.md` v0.3 and `docs/dataset_lock_decision_v0.1.md` define person-level grouping, archive audit and stop conditions. PASS still requires obtained/verified raw topology, frozen split/trial design, dependence-aware resampling rule, FAR grid, held-out calibration partition, corruption severities, model/tuning budget and final run count. |
| **G6 — Article structure/prose** | Problem → gap → comparison framework → protocol → results Q1/Q2/Q3 → discussion/limits. | **PLANNED** | Final manuscript structural audit. |
| **G7 — Scope/operational claims** | No claim of universal superiority, deployment readiness, spoof resistance, fairness, or security improvement unless directly measured. | **RULE FIXED** | Scope/threats/validity audit. |
| **G8 — Reproducibility** | Code, configs, split manifests, trials, seeds, raw scores, hashes, environment, regeneration scripts; restricted datasets referenced rather than redistributed. | **ADVANCED INFRASTRUCTURE / FINAL EVIDENCE OPEN** | CI, metrics, classical baselines, neural fusion modules/contracts, statistics, split hashing, frozen trial/score contracts and immutable run-manifest provenance are versioned. Final real-data manifests, configs, checkpoints, environments, raw scores and regeneration scripts remain. |
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
9. Technical training repetitions, finger instances, or raw verification-pair counts are treated as independent human subjects.
10. For NUPT-FPV or another multi-finger dataset, different fingers from the same volunteer cross outer train/development/calibration/test partitions.
11. A one-way cluster bootstrap is used on dense symmetric all-vs-all impostor pairs without validating the dependence structure.
12. A model-family ranking is declared universal from a single dataset without bounded claims or generalization evidence.
13. A Transformer/attention/missing-modality mechanism is presented as novelty even though that mechanism is already established prior art.
14. Fixed-model classical score/feature fusion is presented as novel despite the OU-MB 2026 baseline precedent.
15. `C_llr`, score calibration or calibration loss is presented as a new biometric concept rather than an established evaluation axis.
16. Any principal result lacks a frozen trial hash, score hash and run provenance record linking it to code/configuration/seed/condition.
17. A later paper is found that satisfies the complete locked Gate-4 contribution contract and the positioning is not reopened/revised.

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
- Are all outer partitions disjoint at the human-subject level, including datasets with multiple fingers/instances per person?
- Is any architectural complexity unsupported by measurable benefit?
- Can every principal table/figure be regenerated from validated run manifests and raw score outputs?

## Current scientific status

**Gate 4 is closed at PASS-POSITIONING. NUPT-FPV is now the P1 access candidate, but Gate 5 remains open until the delivered archive and full trial/statistical contract are audited and frozen. GO for access acquisition, archive-audit tooling, neural training infrastructure and non-final pilots. NO-GO for any confirmatory final-test campaign or article performance conclusion.**
