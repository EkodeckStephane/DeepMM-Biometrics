# DeepMM-Biometrics — V1 / V2 Dataset Strategy

## Decision

The project now separates two publication stages.

### V1 — public NUPT-FPV subset

Version 1 will be completed using only the NUPT-FPV data that are already publicly available in the official repository.

The official public subset currently exposes:

- fingerprint and finger-vein modalities;
- two acquisition sessions;
- public biometric-instance identifiers `001`–`020`;
- ten captures per public instance, modality and session;
- 800 image files in total (400 fingerprint + 400 finger-vein).

The real-data CI audit of the official repository has passed with manifest hash:

`be7d83e353476e50a6193d77e47d7f176ab0a9cb805f81cb8b8a87f79368238c`

The public documentation does **not** establish how identifiers `001`–`020` map to the 140 human volunteers of the complete database. V1 therefore treats these identifiers only as **public biometric-instance identities**. It does not describe them as 20 independent persons and does not make population-level human-subject claims.

### V2 — complete 33,600-image NUPT-FPV database

The complete database will be pursued only after V1 is scientifically and technically closed. The already prepared access-request email remains unsent until then.

V2 will use the complete database only after:

- official access/release conditions are accepted;
- the complete archive is obtained;
- the 840 finger instances are authoritatively mapped to 140 human volunteers;
- person-disjoint train/model-selection/calibration/final-test partitions are frozen;
- the complete archive passes the same manifest, corruption, session and modality audits used by V1;
- final trial dependence and inference are upgraded to the human-subject level.

## V1 scientific scope

V1 is not a surrogate for the complete-data study. It is a bounded experiment answering Q1–Q3 **for the public 20-instance NUPT-FPV benchmark only**.

The V1 article may support claims about:

- the behavior of classical versus deep fusion mechanisms under identical public-instance evidence;
- discrimination/calibration changes on frozen public-instance verification trials;
- controlled quality degradation and explicit single-modality absence;
- computation cost under a matched environment;
- whether fusion-family rankings change across those controlled conditions.

V1 may **not** support claims that require known independent human subjects, including:

- population-level generalization to NUPT-FPV volunteers;
- demographic/fairness conclusions;
- person-level bootstrap or person-level hypothesis tests;
- biological-subject independence of the 20 public identities;
- performance representative of the complete 140-volunteer/840-finger database.

## V1 identity and verification semantics

The recognition target is a **biometric finger-instance identity** represented by the public directory identifier. Genuine trials compare different captures of the same public instance. Impostor trials compare different public instance identifiers.

Because multiple public instance identifiers could theoretically belong to the same human volunteer, dependence between nominal instance clusters cannot be ruled out. V1 therefore uses instance-cluster resampling only as a **sensitivity analysis**, not as a confidence interval for a human population. Confirmatory person-level inference is deferred to V2.

## V1 sample firewall

The final exact split is generated deterministically from session/capture metadata and frozen before any final-score inspection. The default V1 design reserves disjoint image samples for four roles:

1. fusion/model fitting;
2. model selection / early stopping;
3. held-out score calibration;
4. final verification.

The final-test role uses cross-session evidence and must not reuse image files used for fit/selection/calibration. All methods receive the same ordered trials within a role.

## Article wording

The manuscript must state explicitly that the experimental database is the **publicly released NUPT-FPV subset**, not the complete NUPT-FPV database. The complete 33,600-image database is reported under limitations and perspectives as the planned V2 external/scale validation.

No manuscript table, abstract sentence or conclusion may silently replace “public biometric instances” by “subjects”, “participants”, or “volunteers”.

## Governance consequence

Gate 5 for V1 is judged against this bounded public-instance protocol. Gate 5 for V2 remains a separate, stricter person-level lock. Closing V1 must not be represented as closing the complete-dataset validation problem.
