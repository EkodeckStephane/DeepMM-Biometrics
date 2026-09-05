import copy

import pytest

from deepmm.validation import (
    assert_person_partition_disjointness,
    audit_multimodal_topology,
    dataset_manifest_hash,
    validate_dataset_records,
)


def _rows():
    rows = []
    for person in ("P001", "P002"):
        for finger in ("left_index", "right_index"):
            for modality in ("fingerprint", "finger_vein"):
                for session in ("1", "2"):
                    for capture in ("1", "2"):
                        sample = f"{person}-{finger}-{modality}-s{session}-c{capture}"
                        rows.append(
                            {
                                "sample_id": sample,
                                "person_id": person,
                                "instance_id": finger,
                                "modality": modality,
                                "session_id": session,
                                "capture_id": capture,
                                "relative_path": f"{person}/{finger}/{modality}/s{session}_{capture}.png",
                                "file_size_bytes": 123,
                            }
                        )
    return rows


def test_dataset_manifest_normalizes_and_hash_is_order_independent():
    rows = _rows()
    normalized = validate_dataset_records(rows)
    assert len(normalized) == 32
    assert normalized[0]["modality"] == "fingerprint"
    assert dataset_manifest_hash(rows) == dataset_manifest_hash(list(reversed(rows)))


def test_dataset_manifest_rejects_duplicate_ids_paths_and_unsafe_paths():
    rows = _rows()
    duplicate_id = copy.deepcopy(rows)
    duplicate_id[1]["sample_id"] = duplicate_id[0]["sample_id"]
    with pytest.raises(ValueError, match="duplicate sample_id"):
        validate_dataset_records(duplicate_id)

    duplicate_path = copy.deepcopy(rows)
    duplicate_path[1]["relative_path"] = duplicate_path[0]["relative_path"]
    with pytest.raises(ValueError, match="duplicate relative_path"):
        validate_dataset_records(duplicate_path)

    unsafe = copy.deepcopy(rows)
    unsafe[0]["relative_path"] = "../outside.png"
    with pytest.raises(ValueError, match="dataset-local relative path"):
        validate_dataset_records(unsafe)


def test_topology_audit_requires_both_modalities_and_two_sessions():
    summary = audit_multimodal_topology(
        _rows(),
        required_modalities=("fingerprint", "finger_vein"),
        min_samples_per_modality=4,
        min_sessions_per_modality=2,
        require_capture_alignment=True,
    )
    assert summary["complete"] is True
    assert summary["n_people"] == 2
    assert summary["n_instances"] == 4
    assert summary["n_complete_instances"] == 4
    assert summary["samples_by_modality"] == {"finger_vein": 16, "fingerprint": 16}


def test_topology_audit_exposes_missing_modality_not_silent_drop():
    rows = [
        row
        for row in _rows()
        if not (
            row["person_id"] == "P002"
            and row["instance_id"] == "right_index"
            and row["modality"] == "finger_vein"
        )
    ]
    summary = audit_multimodal_topology(
        rows,
        required_modalities=("fingerprint", "finger_vein"),
        min_samples_per_modality=4,
        min_sessions_per_modality=2,
    )
    assert summary["complete"] is False
    assert summary["n_incomplete_instances"] == 1
    incomplete = summary["incomplete_instances"][0]
    assert incomplete["person_id"] == "P002"
    assert incomplete["instance_id"] == "right_index"
    assert any("finger_vein" in reason for reason in incomplete["reasons"])


def test_person_partition_check_detects_different_fingers_in_different_splits():
    rows = _rows()
    assignment = {row["sample_id"]: "train" for row in rows}
    # Move one finger of the same person to test: this must be rejected even though
    # finger instance IDs are distinct.
    for row in rows:
        if row["person_id"] == "P001" and row["instance_id"] == "right_index":
            assignment[row["sample_id"]] = "test"
    with pytest.raises(ValueError, match="person-level partition leakage"):
        assert_person_partition_disjointness(rows, assignment)


def test_person_partition_check_accepts_person_disjoint_assignment():
    rows = _rows()
    assignment = {
        row["sample_id"]: ("train" if row["person_id"] == "P001" else "test")
        for row in rows
    }
    result = assert_person_partition_disjointness(rows, assignment)
    assert result == {"P001": ("train",), "P002": ("test",)}
