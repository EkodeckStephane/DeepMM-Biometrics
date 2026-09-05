from pathlib import Path

import pytest

from deepmm.datasets import (
    NUPT_PUBLIC_UNRESOLVED_PERSON,
    assert_nupt_person_mapping_resolved,
    scan_nupt_fpv,
)
from deepmm.validation import audit_multimodal_topology


def _build_tree(root: Path, instances=("001", "002"), captures=("01", "02")) -> None:
    for session in ("Session1", "Session2"):
        for modality in ("Fingerprint", "FingerVein"):
            for instance in instances:
                directory = root / "image" / session / modality / instance
                directory.mkdir(parents=True, exist_ok=True)
                for capture in captures:
                    (directory / f"{instance}_{capture}.bmp").write_bytes(b"BMtechnical")


def test_public_subset_scan_keeps_human_mapping_unresolved(tmp_path):
    _build_tree(tmp_path)
    rows = scan_nupt_fpv(tmp_path)
    assert len(rows) == 16
    assert {row["person_id"] for row in rows} == {NUPT_PUBLIC_UNRESOLVED_PERSON}
    assert {row["person_mapping_status"] for row in rows} == {"unresolved"}
    assert {row["instance_id"] for row in rows} == {"001", "002"}
    with pytest.raises(ValueError, match="technical-only evidence"):
        assert_nupt_person_mapping_resolved(rows)


def test_verified_instance_to_person_mapping_enables_person_resolution(tmp_path):
    _build_tree(tmp_path)
    rows = scan_nupt_fpv(
        tmp_path,
        instance_to_person={"001": "P001", "002": "P002"},
    )
    assert {row["person_id"] for row in rows} == {"P001", "P002"}
    assert {row["person_mapping_status"] for row in rows} == {"resolved"}
    assert_nupt_person_mapping_resolved(rows)


def test_cross_modality_session_topology_is_auditable(tmp_path):
    _build_tree(tmp_path)
    rows = scan_nupt_fpv(tmp_path)
    summary = audit_multimodal_topology(
        rows,
        required_modalities=("fingerprint", "finger_vein"),
        min_samples_per_modality=4,
        min_sessions_per_modality=2,
        require_capture_alignment=True,
    )
    assert summary["complete"] is True
    assert summary["n_instances"] == 2
    assert summary["samples_by_modality"] == {"finger_vein": 8, "fingerprint": 8}


def test_missing_modality_directory_fails_loudly(tmp_path):
    _build_tree(tmp_path)
    target = tmp_path / "image" / "Session2" / "FingerVein"
    for child in target.rglob("*"):
        if child.is_file():
            child.unlink()
    for child in sorted(target.rglob("*"), reverse=True):
        if child.is_dir():
            child.rmdir()
    target.rmdir()
    with pytest.raises(ValueError, match="missing NUPT-FPV modality directory"):
        scan_nupt_fpv(tmp_path)


def test_filename_instance_mismatch_is_rejected(tmp_path):
    _build_tree(tmp_path, instances=("001",), captures=("01",))
    file_path = tmp_path / "image" / "Session1" / "Fingerprint" / "001" / "001_01.bmp"
    file_path.rename(file_path.with_name("002_01.bmp"))
    with pytest.raises(ValueError, match="does not match directory"):
        scan_nupt_fpv(tmp_path)


def test_incomplete_verified_mapping_is_rejected(tmp_path):
    _build_tree(tmp_path)
    with pytest.raises(ValueError, match="mapping is missing NUPT instance"):
        scan_nupt_fpv(tmp_path, instance_to_person={"001": "P001"})
