import pytest

from deepmm.robustness import (
    StressCondition,
    StressKind,
    clean_condition,
    stress_plan_hash,
    validate_stress_plan,
)


def _plan():
    return (
        clean_condition(),
        StressCondition(
            "face_blur_s1",
            StressKind.CORRUPTION,
            ("face",),
            "blur",
            1,
            (("sigma", 1.0),),
        ),
        StressCondition(
            "face_blur_s2",
            StressKind.CORRUPTION,
            ("face",),
            "blur",
            2,
            (("sigma", 2.0),),
        ),
        StressCondition("face_missing", StressKind.MISSING, ("face",), "missing", 1),
    )


def test_valid_stress_plan_hash_is_deterministic():
    plan = validate_stress_plan(_plan(), ("face", "finger"))
    assert len(plan) == 4
    assert stress_plan_hash(plan, ("face", "finger")) == stress_plan_hash(
        plan, ("face", "finger")
    )


def test_parameter_order_does_not_change_condition_hash():
    a = (
        clean_condition(),
        StressCondition(
            "noise",
            StressKind.CORRUPTION,
            ("face",),
            "noise",
            1,
            (("sigma", 0.1), ("clip", True)),
        ),
    )
    b = (
        clean_condition(),
        StressCondition(
            "noise",
            StressKind.CORRUPTION,
            ("face",),
            "noise",
            1,
            (("clip", True), ("sigma", 0.1)),
        ),
    )
    assert stress_plan_hash(a, ("face", "finger")) == stress_plan_hash(
        b, ("face", "finger")
    )


def test_corruption_requires_positive_severity_and_target():
    with pytest.raises(ValueError):
        StressCondition("bad", StressKind.CORRUPTION, (), "blur", 1)
    with pytest.raises(ValueError):
        StressCondition("bad", StressKind.CORRUPTION, ("face",), "blur", 0)


def test_missingness_is_categorical_and_cannot_remove_every_modality():
    with pytest.raises(ValueError, match="severity_rank=1"):
        StressCondition("missing", StressKind.MISSING, ("face",), "missing", 2)

    plan = (
        clean_condition(),
        StressCondition(
            "all_missing",
            StressKind.MISSING,
            ("face", "finger"),
            "missing",
            1,
        ),
    )
    with pytest.raises(ValueError, match="cannot remove every modality"):
        validate_stress_plan(plan, ("face", "finger"))


def test_unknown_modality_is_rejected():
    plan = (
        clean_condition(),
        StressCondition("iris_missing", StressKind.MISSING, ("iris",), "missing", 1),
    )
    with pytest.raises(ValueError, match="unknown modalities"):
        validate_stress_plan(plan, ("face", "finger"))


def test_duplicate_condition_ids_and_duplicate_severity_are_rejected():
    duplicate_ids = (
        clean_condition(),
        StressCondition("clean", StressKind.MISSING, ("face",), "missing", 1),
    )
    with pytest.raises(ValueError, match="IDs must be unique"):
        validate_stress_plan(duplicate_ids, ("face", "finger"))

    duplicate_severity = (
        clean_condition(),
        StressCondition("a", StressKind.CORRUPTION, ("face",), "blur", 1),
        StressCondition("b", StressKind.CORRUPTION, ("face",), "blur", 1),
    )
    with pytest.raises(ValueError, match="duplicate corruption severity"):
        validate_stress_plan(duplicate_severity, ("face", "finger"))
