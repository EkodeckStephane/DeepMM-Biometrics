# Frozen Trial Manifest Contract v0.1

**Study:** *Deep Learning Approaches for Multimodal Biometrics*

The trial list is part of the scientific protocol, not an implementation detail. Every system entering a paired comparison must be scored on the same ordered verification trials unless the manuscript explicitly labels a separate protocol.

## 1. Required trial fields

Every trial contains:

- `trial_id`: immutable unique identifier;
- `label`: literal `1` for genuine or `0` for impostor;
- `anchor_subject_id`: subject-centric cluster identifier used only where the frozen statistical protocol justifies a one-way subject cluster;
- `enrollment_subject_id`;
- `probe_subject_id`;
- `enrollment_sample_id`;
- `probe_sample_id`;
- `condition_id`: e.g. `clean`, a locked corruption/severity condition, or a missing-modality condition.

Additional metadata such as session, sensor, modality availability, corruption family and severity are retained in the record and participate in the trial-manifest hash.

## 2. Identity semantics

Hard validation rules:

1. genuine (`label=1`) requires equal enrollment/probe subject identity;
2. impostor (`label=0`) requires different enrollment/probe identities;
3. exact same-sample self-comparisons are rejected in the primary protocol;
4. one sample ID cannot be assigned to different subjects within a manifest;
5. `trial_id` values are unique;
6. the primary manifest contains both genuine and impostor trials.

Violating any rule invalidates the manifest before metrics are computed.

## 3. Frozen ordering

Trial order is scientifically significant because all systems must remain paired row-for-row. The SHA-256 `trial_manifest_hash` therefore changes when a trial is inserted, deleted, modified, or reordered. Dictionary key order is canonicalized and does not affect the hash.

## 4. Score contract

Every system output contains exactly one finite score for every frozen `trial_id`, in the same order. Missing trials, extra trials, reordered IDs and NaN/Inf scores are rejected.

A model failure to produce a score is not silently dropped. It must be represented as a documented run failure, failure-to-acquire outcome, or a separately predeclared protocol.

Each accepted score file receives a `score_manifest_hash`. Changing any score or score-record metadata changes that digest.

## 5. Reproducibility chain

A final experiment record should minimally bind:

```text
split_hash
trial_manifest_hash
config_hash
checkpoint_hash
score_manifest_hash
code_commit
seed
condition_id
```

The paper's principal tables and figures must be regenerated from score files that satisfy this contract.

## 6. Statistical boundary

`anchor_subject_id` does **not** imply that a one-way clustered bootstrap is always valid. In dense symmetric impostor protocols, a non-match trial depends on both enrollment and probe identities. Such trials require a subject-subsets / multiway reconstruction procedure rather than pretending that assigning one anchor removes the second-subject dependence.

The final uncertainty procedure is frozen only after the dataset and trial construction are known.

## 7. Gate consequences

This contract supports Gate 5 (paired valid evaluation) and Gate 8 (traceability). It does not by itself make those gates PASS: dataset identity correspondence, trial construction, statistical dependence handling and independent metric validation must still be closed.
