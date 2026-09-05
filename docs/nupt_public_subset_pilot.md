# NUPT-FPV public-subset technical pilot

**Status:** technical/infrastructure evidence only — **not scientific dataset evidence**.

## Source basis

The official NUPT-FPV repository states that the complete database contains 840 finger instances from 140 volunteers, with fingerprint and finger-vein acquisition, 10 repetitions in each of two sessions, and 33,600 images in total. It also states that the public repository provides fingerprint and finger-vein data for two sessions, numbered 1–20, while researchers needing the complete database must contact the authors.

The public GitHub tree independently confirms the following archive layout for the exposed subset:

```text
image/
  Session1/
    Fingerprint/
      001/001_01.bmp ...
      ...
    FingerVein/
      001/001_01.bmp ...
      ...
  Session2/
    Fingerprint/
      ...
    FingerVein/
      ...
```

The visible modality/session roots each expose instance directories `001` through `020`; representative directory inspection confirms capture names `*_01.bmp` through `*_10.bmp`.

## Critical identity limitation

The public documentation does **not** establish how exposed instance identifiers `001`–`020` map to the 140 human volunteers in the full database. DeepMM therefore does not infer that:

- one public directory equals one independent human subject;
- six consecutive instance IDs necessarily belong to one person;
- directory arithmetic can reconstruct the full volunteer/finger mapping.

The adapter `src/deepmm/datasets/nupt_fpv.py` deliberately assigns the sentinel human identifier `UNRESOLVED_PUBLIC_NUPT_PERSON` unless a separately verified instance-to-person mapping is supplied. `assert_nupt_person_mapping_resolved()` then blocks these records from scientific person-disjoint splitting.

This restriction is intentional: the public subset can validate software topology without creating false biological sample-size evidence.

## Technical checks performed by the pilot

The real-data smoke workflow `.github/workflows/nupt-public-smoke.yml` checks out the official public NUPT-FPV repository in an ephemeral CI workspace and executes `scripts/audit_nupt_public_subset.py`.

The audit requires:

- exactly two source sessions (`Session1`, `Session2`);
- fingerprint and finger-vein modality directories in each session;
- 20 exposed instance identifiers;
- 10 captures per session/modality/instance, yielding 800 public image records;
- structurally aligned `(session, capture)` keys between fingerprint and finger vein;
- no malformed or cross-instance filenames;
- a deterministic DeepMM metadata-manifest hash;
- the human-person mapping to remain explicitly unresolved.

A successful workflow therefore means **the parser and metadata contracts agree with the current official public repository layout**. It does not demonstrate recognition performance, subject-level independence, acquisition simultaneity, or scientific validity of a final split.

## What this pilot unlocks

Before the full restricted/requested archive arrives, DeepMM can now validate:

1. archive parsing and canonical modality naming;
2. session/capture extraction;
3. cross-modality structural completeness checks;
4. metadata-manifest hashing;
5. explicit refusal to confuse finger-instance IDs with human-subject IDs;
6. CI reproducibility against a real upstream public archive rather than only synthetic fixtures.

## What remains blocked

The following remain **NO-GO** until the complete archive and verified biological mapping are obtained:

- person-disjoint train/model-selection/calibration/final-test splits;
- scientific genuine/impostor trial construction;
- final dependence-aware bootstrap/randomization choice;
- biometric performance reporting;
- encoder/model selection justified from the complete data regime;
- Gate-5 dataset lock;
- final Q1/Q2/Q3 experiments.

## Transition criterion

NUPT-FPV may move from `P1 ACCESS CANDIDATE` to `PROVISIONAL EXPERIMENTAL CANDIDATE` only after the official complete archive is obtained and an auditable instance-to-human mapping is established from authoritative source material or the delivered dataset metadata. No inference from the public `001`–`020` naming convention is sufficient.
