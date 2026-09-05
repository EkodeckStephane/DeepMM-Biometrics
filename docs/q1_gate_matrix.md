# DeepMM-Biometrics — Q1 Scientific Article Gate Matrix

This project adapts the owner-supplied **Q1 Scientific-Article Editorial Gates** to the present biometrics study.

| Gate | DeepMM-Biometrics requirement | Current status | Evidence required for PASS |
|---|---|---|---|
| **G1 — Scientific object first** | The paper is organized around Q1-Q3, not around a named model or repository. | **PASS-DESIGN** | Frozen research questions + controlled benchmark protocol exist. |
| **G2 — Claim/evidence alignment** | Every conclusion about DL benefit, best family, calibration, robustness, or missing modalities maps to direct measurements. Negative results retained. | **PASS-DESIGN / EVIDENCE OPEN** | Q1-Q3 evidence map and planned contrasts exist; final claim-evidence matrix must link to raw outputs. |
| **G3 — Final scientific narrative** | Final article contains the final valid study, not development/debug/audit history. | **RULE FIXED** | Editorial audit of final manuscript. |
| **G4 — Novelty/current SOTA** | No architectural-priority claim. Positioning must show what matched family-level comparison is missing in current literature. | **IN PROGRESS — CRITICAL** | `sota_matrix_v0.2.md` + `sota_search_protocol.md` exist; systematic representative-current-work search must be completed before a novelty lock. |
| **G5 — Experimental validity** | Real multimodal identity correspondence, leakage-free splits, matched baselines, correct experimental unit, uncertainty, paired tests, multiplicity correction. | **DESIGN-READY / DATA OPEN** | Controlled two-track benchmark + statistical analysis plan + anti-leakage core exist; final dataset, splits, trial list and run count remain to lock. |
| **G6 — Article structure/prose** | Problem → gap → comparison framework → protocol → results Q1/Q2/Q3 → discussion/limits. | **PLANNED** | Final manuscript structural audit. |
| **G7 — Scope/operational claims** | No claim of universal superiority, deployment readiness, spoof resistance, fairness, or security improvement unless directly measured. | **RULE FIXED** | Scope/threats/validity audit. |
| **G8 — Reproducibility** | Code, configs, split manifests, seeds, raw scores, hashes, environment, regeneration scripts; restricted datasets referenced rather than redistributed. | **STARTED** | Package skeleton + metric and leakage tests exist; final experiment manifests/raw evidence/regeneration scripts remain. |
| **G9 — Bibliographic/editorial hygiene** | Every reference exists and supports its citing sentence; same title/numbers across manuscript and submission files. | **OPEN** | DOI/source audit + cross-file consistency audit. |
| **G10 — Submission readiness** | No contribution/evidence mismatch, unfair comparison, article-code-data contradiction, unverified reference, or journal-scope mismatch. | **NO-GO NOW** | Reviewer Senior final pass + all prior gates closed. |

## Biometrics-specific hard stops

The project must stop or reframe before the final campaign if any of the following holds:

1. Primary multimodal identities are not genuinely matched across modalities.
2. The dataset is too small to compare the chosen model families fairly and no justified frozen/pretrained-encoder strategy is possible.
3. The “best approach” criterion is defined only after test results are seen.
4. Classical and unimodal baselines receive less tuning/evaluation budget than the DL methods used for headline claims.
5. Missing-modality robustness is claimed without explicit missing-modality evaluation.
6. Calibration is inferred from discrimination metrics rather than measured directly.
7. Robustness uses corruptions/severities selected after seeing which model benefits.
8. Cost comparisons use different hardware, batch size, precision, or measurement procedures without correction/disclosure.
9. Technical training repetitions are treated as independent biometric subjects.
10. A model-family ranking is declared universal from a single dataset without bounded claims or generalization evidence.

## Reviewer-Senior prescreen questions

Before any article draft is considered mature, a strict reviewer must be able to answer:

- What exactly does DL improve relative to classical and unimodal systems?
- On which metrics does it fail to improve?
- Which family is Pareto-optimal under the locked criteria?
- Are apparent gains larger than uncertainty and statistically defensible?
- Does the ranking survive degradation and missing modalities?
- Are the data truly multimodal at subject level?
- Is any architectural complexity unsupported by measurable benefit?
- Can every principal table/figure be regenerated from committed scripts and raw outputs?

## Current scientific status

**GO for infrastructure, SOTA closure and pilot preparation. NO-GO for a final training campaign or manuscript claims.**
