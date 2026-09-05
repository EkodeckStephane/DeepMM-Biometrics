# Experiment Run Manifest Contract v0.1

**Study:** *Deep Learning Approaches for Multimodal Biometrics*

Every numerical result retained for Q1–Q3 must be traceable to one immutable run manifest. A table cell without a valid provenance chain is not final evidence.

## Required fields

```text
run_id
method_id
family
seed
condition_id
code_commit
split_hash
trial_manifest_hash
config_hash
score_manifest_hash
```

Rules:

- `code_commit` is a full 40-character Git commit SHA; abbreviated SHAs are rejected for final evidence;
- all artifact hashes are lowercase SHA-256 digests;
- `seed` is a non-negative integer and booleans are rejected;
- `condition_id` binds the run to clean, degradation or missing-modality protocol state;
- `method_id` identifies the exact implementation/configuration and `family` identifies its scientific comparison family.

Optional immutable artifacts may include:

```text
dataset_manifest_hash
checkpoint_hash
environment_hash
```

A non-trainable classical method does not fabricate a checkpoint hash: the field is simply omitted.

## Evidence chain

The intended chain is:

```text
subject split
  -> split_hash
ordered trials
  -> trial_manifest_hash
method configuration
  -> config_hash
code state
  -> code_commit
optional trained weights
  -> checkpoint_hash
ordered system scores
  -> score_manifest_hash
run metadata
  -> run_manifest_hash
```

The run hash changes if any retained metadata or evidence hash changes. This makes accidental post-hoc substitution detectable.

## Manuscript rule

Principal manuscript tables and figures must be generated from validated run manifests plus their score files. Hand-entered headline numbers are not accepted as final evidence.

The run manifest guarantees provenance, not scientific validity. A perfectly hashed experiment can still be invalid if the dataset pairing, split, trials, calibration, statistical units or comparison budget are wrong. Gate 5 and Gate 8 therefore require both provenance and methodological validity.
