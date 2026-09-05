import pytest

from deepmm.training import FinalTestFirewall


def test_firewall_requires_four_distinct_partition_roles_and_hashes_deterministically():
    firewall = FinalTestFirewall("fit", "selection", "calibration", "final_test")
    assert firewall.assert_development_partition("fit") == "fit"
    assert firewall.assert_development_partition("selection") == "selection"
    assert firewall.assert_development_partition("calibration") == "calibration"
    assert len(firewall.protocol_hash) == 64
    assert firewall.protocol_hash == FinalTestFirewall(
        "fit", "selection", "calibration", "final_test"
    ).protocol_hash


def test_firewall_rejects_final_test_and_unknown_partitions():
    firewall = FinalTestFirewall("fit", "selection", "calibration", "final_test")
    with pytest.raises(ValueError, match="final-test partition"):
        firewall.assert_development_partition("final_test")
    with pytest.raises(ValueError, match="unknown development partition"):
        firewall.assert_development_partition("other")


def test_firewall_rejects_role_reuse():
    with pytest.raises(ValueError, match="must be distinct"):
        FinalTestFirewall("development", "development", "calibration", "test")
