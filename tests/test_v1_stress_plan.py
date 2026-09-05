from deepmm.robustness import StressKind, v1_stress_plan, v1_stress_plan_hash


def test_v1_stress_plan_is_complete_and_hashed():
    plan = v1_stress_plan()
    assert len(plan) == 15
    assert plan[0].kind is StressKind.CLEAN
    assert sum(c.kind is StressKind.CORRUPTION for c in plan) == 12
    assert sum(c.kind is StressKind.MISSING for c in plan) == 2
    assert len(v1_stress_plan_hash()) == 64


def test_v1_corruptions_are_probe_only_and_modality_specific():
    plan = v1_stress_plan()
    corruptions = [c for c in plan if c.kind is StressKind.CORRUPTION]
    for condition in corruptions:
        assert len(condition.target_modalities) == 1
        assert dict(condition.parameters)["scope"] == "probe_only"

    ids = {condition.condition_id for condition in plan}
    for modality in ("fingerprint", "finger_vein"):
        for rank in (1, 2, 3):
            assert f"{modality}-blur-{rank}" in ids
            assert f"{modality}-contrast-{rank}" in ids
    assert {"missing-fingerprint", "missing-finger-vein"}.issubset(ids)
