"""Frozen Q3 stress plan for the bounded public-subset V1 campaign.

Only probe images are corrupted. Enrollment evidence remains clean. The plan uses
simple deterministic degradations whose semantics are unambiguous in grayscale
biometric imagery: Gaussian blur and contrast attenuation. Missingness is an
explicit availability condition, not an image corruption or NaN sentinel.
"""

from __future__ import annotations

from .conditions import StressCondition, StressKind, clean_condition, stress_plan_hash, validate_stress_plan


V1_MODALITIES = ("fingerprint", "finger_vein")


def v1_stress_plan() -> tuple[StressCondition, ...]:
    conditions: list[StressCondition] = [clean_condition()]

    for modality in V1_MODALITIES:
        for rank, radius in enumerate((1.0, 2.0, 3.0), start=1):
            conditions.append(
                StressCondition(
                    condition_id=f"{modality}-blur-{rank}",
                    kind=StressKind.CORRUPTION,
                    target_modalities=(modality,),
                    operator="gaussian_blur",
                    severity_rank=rank,
                    parameters=(("radius", radius), ("scope", "probe_only")),
                )
            )
        for rank, factor in enumerate((0.75, 0.50, 0.25), start=1):
            conditions.append(
                StressCondition(
                    condition_id=f"{modality}-contrast-{rank}",
                    kind=StressKind.CORRUPTION,
                    target_modalities=(modality,),
                    operator="contrast_scale",
                    severity_rank=rank,
                    parameters=(("factor", factor), ("scope", "probe_only")),
                )
            )

    conditions.extend(
        [
            StressCondition(
                condition_id="missing-fingerprint",
                kind=StressKind.MISSING,
                target_modalities=("fingerprint",),
                operator="missing",
                severity_rank=1,
            ),
            StressCondition(
                condition_id="missing-finger-vein",
                kind=StressKind.MISSING,
                target_modalities=("finger_vein",),
                operator="missing",
                severity_rank=1,
            ),
        ]
    )
    return validate_stress_plan(conditions, V1_MODALITIES)


def v1_stress_plan_hash() -> str:
    return stress_plan_hash(v1_stress_plan(), V1_MODALITIES)
