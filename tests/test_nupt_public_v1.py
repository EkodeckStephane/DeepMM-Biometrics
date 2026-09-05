from pathlib import Path

import pytest

from deepmm.datasets import generate_v1_trials, scan_nupt_fpv, v1_trial_summary


def _build_public_tree(root: Path, n_instances: int = 3) -> None:
    for session in (1, 2):
        for modality in ("Fingerprint", "FingerVein"):
            for number in range(1, n_instances + 1):
                instance = f"{number:03d}"
                directory = root / "image" / f"Session{session}" / modality / instance
                directory.mkdir(parents=True, exist_ok=True)
                for capture in range(1, 11):
                    (directory / f"{instance}_{capture:02d}.bmp").write_bytes(b"BMtechnical")


def test_v1_roles_are_sample_disjoint_and_final_is_cross_session(tmp_path):
    _build_public_tree(tmp_path)
    rows = scan_nupt_fpv(tmp_path)

    roles = {role: generate_v1_trials(rows, role) for role in ("fit", "selection", "calibration", "final")}
    used = {}
    for role, trials in roles.items():
        sample_ids = {
            sample_id
            for trial in trials
            for sample_id in (trial["enrollment_sample_id"], trial["probe_sample_id"])
        }
        used[role] = sample_ids
        assert all(trial["identity_scope"] == "public_biometric_instance" for trial in trials)

    names = tuple(used)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert used[left].isdisjoint(used[right])

    assert {trial["enrollment_session_id"] for trial in roles["final"]} == {"1"}
    assert {trial["probe_session_id"] for trial in roles["final"]} == {"2"}
    assert {trial["enrollment_capture_id"] for trial in roles["final"]} == {"10"}
    assert {trial["probe_capture_id"] for trial in roles["final"]} == {f"{i:02d}" for i in range(1, 11)}


def test_v1_expected_trial_counts_for_three_instances(tmp_path):
    _build_public_tree(tmp_path, n_instances=3)
    rows = scan_nupt_fpv(tmp_path)
    summary = v1_trial_summary(rows)

    # fit: 3x3 identity matrix x (2 enrollment captures x 3 probe captures)
    assert summary["roles"]["fit"]["n_trials"] == 54
    assert summary["roles"]["fit"]["n_genuine"] == 18
    assert summary["roles"]["fit"]["n_impostor"] == 36

    # selection/calibration: one enrollment/probe capture pair per identity pair.
    for role in ("selection", "calibration"):
        assert summary["roles"][role]["n_trials"] == 9
        assert summary["roles"][role]["n_genuine"] == 3
        assert summary["roles"][role]["n_impostor"] == 6

    # final: 3x3 identity matrix x ten session-2 probes.
    assert summary["roles"]["final"]["n_trials"] == 90
    assert summary["roles"]["final"]["n_genuine"] == 30
    assert summary["roles"]["final"]["n_impostor"] == 60

    for role in summary["roles"].values():
        assert len(role["trial_manifest_hash"]) == 64


def test_v1_rejects_missing_required_final_capture(tmp_path):
    _build_public_tree(tmp_path, n_instances=2)
    missing = tmp_path / "image" / "Session2" / "FingerVein" / "001" / "001_10.bmp"
    missing.unlink()
    rows = scan_nupt_fpv(tmp_path)
    with pytest.raises(ValueError, match="missing required V1 evidence unit"):
        generate_v1_trials(rows, "final")


def test_v1_role_name_is_strict(tmp_path):
    _build_public_tree(tmp_path, n_instances=2)
    rows = scan_nupt_fpv(tmp_path)
    with pytest.raises(ValueError, match="unknown V1 role"):
        generate_v1_trials(rows, "test")
