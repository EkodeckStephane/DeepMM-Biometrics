# DeepMM-Biometrics — Q1 Scientific Article Gate Matrix

This project adapts the owner-supplied **Q1 Scientific-Article Editorial Gates** to the present biometrics study.

| Gate | DeepMM-Biometrics requirement | Current status | Evidence required for PASS |
|---|---|---|---|
| **G1 — Scientific object first** | The paper is organized around Q1-Q3, not around a named model or repository. | **PASS-DESIGN** | Frozen research questions + controlled benchmark protocol exist. |
| **G2 — Claim/evidence alignment** | Every conclusion about DL benefit, best family, calibration, robustness, or missing modalities maps to direct measurements. Negative results retained. | **PASS-DESIGN / EVIDENCE OPEN** | `claim_evidence_matrix.md` and planned contrasts exist; final claim-evidence matrix must link each headline statement to raw outputs. |
| **G3 — Final scientific narrative** | Final article contains the final valid study, not development/debug/audit history. | **RULE FIXED** | Editorial audit of final manuscript. |
| **G4 — Novelty/current SOTA** | No architectural-priority claim. Positioning must show what matched fusion-mechanism comparison remains unresolved. | **IN PROGRESS — CRITICAL, BOUNDARY NARROWED** | `sota_matrix_v0.4.md` now identifies the closest controlled precedents and a falsification matrix; the systematic venue sweep/full-text claim verification must still be completed before novelty wording is locked. |
| **G5 — Experimental validity** | Real multimodal identity correspondence, leakage-free splits, matched baselines, correct experimental unit, dependence-aware uncertainty, paired tests, multiplicity correction. | **DESIGN-READY / DATA-DEPENDENT LOCK OPEN** | Two-track benchmark + statistical plan v0.2 + metric/anti-leakage core + cluster bootstrap/permutation/Pareto/rank utilities exist. Final dataset, trial-dependence structure, bootstrap form, splits, FAR grid and run count remain to lock. |
| **G6 — Article structure/prose** | Problem → gap → comparison framework → protocol → results Q1/Q2/Q3 → discussion/limits. | **PLANNED** | Final manuscript structural audit. |
| **G7 — Scope/operational claims** | No claim of universal superiority, deployment readiness, spoof resistance, fairness, or security improvement unless directly measured. | **RULE FIXED** | Scope/threats/validity audit. |
| **G8 — Reproducibility** | Code, configs, split manifests, seeds, raw scores, hashes, environment, regeneration scripts; restricted datasets referenced rather than redistributed. | **ACTIVE** | Package + CI + metrics + classical fusion + split/hash validation + statistical utilities are versioned; final experiment manifests/raw evidence/checkpoint hashes/regeneration scripts remain. |
| **G9 — Bibliographic/editorial hygiene** | Every reference exists and supports its citing sentence; same title/numbers across manuscript and submission files. | **OPEN / SOURCE AUDIT RUNNING** | DOI/official-source verification for the final SOTA + cross-file consistency audit. |
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
9. Technical training repetitions or raw verification-pair counts are treated as independent biometric subjects.
10. A one-way cluster bootstrap is used on dense symmetric all-vs-all impostor pairs without validating the dependence structure.
11. A model-family ranking is declared universal from a single dataset without bounded claims or generalization evidence.
12. A Transformer/attention/missing-modality mechanism is presented as novelty even though that mechanism is already established prior art.

## Reviewer-Senior prescreen questions

Before any article draft is considered mature, a strict reviewer must be able to answer:

- What exactly does DL improve relative to classical and unimodal systems?
- On which metrics does it fail to improve?
- Is the observed gain attributable to **fusion**, rather than a stronger backbone/representation?
- Which family is Pareto-optimal under the locked criteria, and with what uncertainty?
- Are apparent gains larger than dependence-aware uncertainty and statistically defensible after multiplicity control?
- Does the ranking survive degradation and missing modalities?
- Is calibration reported by relevant modality-availability subset rather than only globally?
- Are the data truly multimodal at subject level?
- Is any architectural complexity unsupported by measurable benefit?
- Can every principal table/figure be regenerated from committed scripts and raw outputs?

## Current scientific status

**GO for infrastructure, Gate-4 closure and pilot preparation. NO-GO for a final training campaign or manuscript claims.**
