# Dataset Lock Decision v0.1

**Status:** access decision only — **not a scientific dataset lock**.

## Decision

The first dataset to pursue is **NUPT-FPV** (fingerprint + finger vein). This decision is based on the currently verified public evidence and the project constraints, not on any model performance result.

### Rationale

NUPT-FPV currently offers the strongest practical/scientific combination for the primary verification study:

- genuine paired fingerprint/finger-vein acquisition;
- 140 human volunteers;
- six fingers per volunteer;
- repeated acquisition: 20 samples per finger across two sessions;
- 33,600 reported images in total;
- modest 300 × 400 image size;
- explicit author contact and release-agreement workflow for research access;
- enough repeated structure to design session-aware genuine trials and controlled degradation/missing-modality tests.

The choice does **not** imply that fingerprint + finger vein is universally preferable to face + fingerprint or any other pair. It is a data-regime decision for a controlled scientific benchmark.

## Biological unit and identity semantics

The project distinguishes:

- `person_id`: human volunteer — the highest-level grouping unit for train/development/calibration/test separation;
- `finger_id`: a biometric identity instance nested inside a person;
- `modality`: fingerprint or finger vein;
- `session`: acquisition session;
- `capture`: repeated acquisition within a session.

All six fingers from one volunteer remain in the same outer partition. Treating fingers as independent subjects across train/test would create biological-subject leakage and is prohibited.

## Provisional trial design to audit after access

The archive audit will determine the exact final design, but the preferred structure is:

1. outer person-disjoint partition into model-development and final-test people;
2. development people further separated for tuning and calibration as needed;
3. genuine trials use the same finger identity across distinct captures, preferably crossing sessions for the primary verification condition when sample topology permits;
4. impostor trials use different human volunteers, with finger-selection rules frozen before final testing;
5. all methods consume the exact same frozen trials;
6. Q3 stress conditions transform only the designated probe/evidence modality according to a pre-hashed stress plan;
7. missing-modality tests use explicit availability masks rather than fabricated scores/features.

No final proportion, FAR operating point, bootstrap unit, or impostor-density rule is locked until the raw topology is verified.

## Access/audit stop conditions

NUPT-FPV will be rejected as primary data if any of the following is found after acquisition:

- fingerprint and finger-vein identity correspondence cannot be reconstructed unambiguously;
- sessions/capture indices are not sufficiently reliable for the planned verification protocol;
- access terms prohibit the intended noncommercial benchmark/reproducibility workflow;
- archive corruption or missingness materially changes the reported topology;
- person-disjoint splitting leaves insufficient development/calibration/test support;
- the trial dependence structure cannot support defensible uncertainty analysis at the planned scale.

## Fallback/generalization order

1. **SDUMLA-HMT** — recognized multimodal collection; useful for cross-pair/generalization; access by request.
2. **LUTBIO** — 306-subject, nine-modality resource; application required and same-subject interpretation must be confirmed from protocol/archive.
3. **OU-MB** — 1,099 subjects and eleven modalities; scientifically strong but current raw-data access route must be confirmed.
4. **BioSecure/BiosecurID** — strong historical datasets; current access path still requires verification.
5. **FaciaVox** — face+voice candidate; currently secondary because access is restricted and the record reports approximately 126 GB for 100 participants.

## Transition required for `DATASET-LOCKED`

The status can change from `P1 ACCESS CANDIDATE` to `DATASET-LOCKED` only after:

- access terms are recorded;
- archive audit passes;
- subject/session/capture manifest is generated;
- split logic is frozen and validated;
- trial-generation policy is frozen;
- calibration partition is frozen;
- dependence-aware statistical plan is selected;
- encoder/fusion feasibility is confirmed;
- final stress/missingness plan is hashed;
- no final-test labels/scores have been used for any of these choices.
