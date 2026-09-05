# Data handling and manifest policy

This repository does **not** contain raw biometric data.

Restricted or request-only datasets remain outside Git. DeepMM records only the metadata required to make the scientific protocol auditable: sample identifiers, human-subject grouping, nested biometric-instance identifiers, modality/session/capture metadata, dataset-local file references and optional file hashes.

## Required manifest columns

| Column | Meaning |
|---|---|
| `sample_id` | globally unique sample identifier inside the DeepMM manifest |
| `person_id` | human volunteer identifier; outer split grouping unit |
| `instance_id` | biometric instance nested in the person, e.g. `left_index` |
| `modality` | canonical modality name, e.g. `fingerprint` or `finger_vein` |
| `session_id` | acquisition session identifier |
| `capture_id` | repeated acquisition identifier within the source protocol |
| `relative_path` | dataset-root-relative path; never a user-specific absolute path |

Optional provenance columns include `file_sha256` and `file_size_bytes`.

## NUPT-FPV rule

For the current P1 access candidate, six fingers belong to each human volunteer. `person_id` therefore controls train/development/calibration/test disjointness; `instance_id` identifies the finger. Different fingers from one person may **not** be distributed across outer partitions.

## Archive audit

After lawful access is obtained, create a private/local CSV following `manifest_template.csv`, then run for example:

```bash
python scripts/audit_dataset_manifest.py path/to/manifest.csv \
  --modalities fingerprint,finger_vein \
  --min-samples 20 \
  --min-sessions 2
```

The exact `--min-samples` value must reflect the delivered archive and source protocol; it is not a substitute for checking missing/corrupt records. Add `--require-capture-alignment` only if the source documentation and archive prove that the modalities share one-to-one session/capture indices.

The audit emits a deterministic dataset-manifest SHA-256 and fails if required modality/session completeness is not met.

## Prohibited repository content

Do not commit:

- raw face/fingerprint/finger-vein/voice/iris or other biometric samples;
- signed dataset agreements containing personal data;
- credentials, private download links or access tokens;
- absolute local file paths containing researcher/account information;
- derived artifacts whose redistribution is prohibited by source terms.
