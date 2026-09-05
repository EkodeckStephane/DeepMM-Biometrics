# V1 Public NUPT-FPV Protocol — Frozen Data/Trial Layer

**Scope:** V1 only — official public NUPT-FPV subset.  
**Human-person mapping:** unresolved and not inferred.  
**Verification identity:** public biometric instance (`001`–`020`).

## Real archive audit

The `nupt-public-smoke` workflow checks out the official `REN382333467/NUPT-FPV` repository and audits the real files, rather than relying only on the publication/README description.

Audit PASS facts:

| Property | Audited V1 public subset |
|---|---:|
| Public biometric-instance IDs | 20 (`001`–`020`) |
| Sessions | 2 |
| Captures per instance/modality/session | 10 |
| Fingerprint files | 400 |
| Finger-vein files | 400 |
| Total files | 800 |
| Fingerprint BMP metadata | 300 × 400, 8 bit |
| Finger-vein BMP metadata | 300 × 450, 8 bit |
| Cross-modality `(instance, session, capture)` naming alignment | PASS |
| Human instance→volunteer mapping | **UNRESOLVED** |

The public files therefore reveal an important modality-size difference: fingerprint is 300×400 while finger vein is 300×450 in the audited public archive. V1 preprocessing must use the actual file geometry; it must not silently coerce source metadata to the README's single reported size.

### Dataset manifest hash

`be7d83e353476e50a6193d77e47d7f176ab0a9cb805f81cb8b8a87f79368238c`

This hash is generated from the ordered, validated metadata manifest for the 800 public files. Any path/sample/session/modality metadata change changes the digest.

## Frozen clean role firewall

No image file appears in more than one role.

| Role | Enrollment evidence | Probe evidence | Purpose |
|---|---|---|---|
| `fit` | session 1, captures 01–02 | session 1, captures 03–05 | fit classical/DL fusion parameters |
| `selection` | session 1, capture 06 | session 1, capture 07 | model/hyperparameter selection and early stopping |
| `calibration` | session 1, capture 08 | session 1, capture 09 | held-out post-hoc score→LLR calibration |
| `final` | session 1, capture 10 | session 2, captures 01–10 | untouched final cross-session verification |

All methods consume the same ordered trials for a given role/condition.

## Frozen clean trial counts and hashes

### Fit

- total: **2,400**
- genuine: **120**
- impostor: **2,280**
- SHA-256: `8125265b7407bdfdef2507b3ee6592625d6fc65e41657ad6ea8dd04989514eb7`

### Selection

- total: **400**
- genuine: **20**
- impostor: **380**
- SHA-256: `482086a4d3d7a57a68cdd9362363399a8b56174d3218f0b3eed138a7fe6d4fb4`

### Calibration

- total: **400**
- genuine: **20**
- impostor: **380**
- SHA-256: `f6104160a138bccebc1f4b03fd5012be1712027d41c44ad5f916a8c34639ca88`

### Final

- total: **4,000**
- genuine: **200**
- impostor: **3,800**
- SHA-256: `3b60ce30d0d496c35aefe0bf0b8c48f868cb3befba0c1fdfb52986645293f324`

The final trial set is cross-session and must remain untouched until the remaining V1 Gate-5 decisions (encoder/preprocessing, fusion search budgets, quality/stress plan, calibration policy and cost protocol) are frozen.

## Statistical boundary

The 20 public identifiers are not known to be 20 independent human volunteers. Therefore the trial count is never treated as an independent sample size, and public-instance clusters are not presented as human-subject clusters.

V1 may report:

- descriptive discrimination and calibration metrics on the frozen final trials;
- paired method differences on exactly the same trials;
- public-instance-cluster resampling/rank/Pareto analysis clearly labelled **benchmark sensitivity analysis**;
- no human-population confidence interval or human-subject significance claim.

The complete 140-volunteer/840-finger person-level inference is a V2 objective.

## Audit provenance

The first complete real-data protocol audit passed in GitHub Actions run `33984079128`, job `public-subset-structure`, against official NUPT-FPV commit `e9c9421cfe648e71ea0a689de295bc1d77df6a91` and DeepMM commit `05647d9ff8da5bf33fcc8fad5d72847428b17006`.
