"""Deterministic image corruption operators for the frozen V1 Q3 plan."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from deepmm.robustness import StressCondition, StressKind


def apply_v1_corruption(image: Image.Image, condition: StressCondition, modality: str) -> Image.Image:
    """Apply one V1 corruption to a probe image.

    Clean and non-targeted corruption conditions return an independent copy.
    Missingness is not rendered into pixels; it must be represented by the explicit
    availability mask in the fusion evidence contract.
    """
    modality = str(modality).strip()
    if condition.kind is StressKind.MISSING:
        raise ValueError("missing modality must use availability masks, not pixel corruption")
    source = image.convert("L")
    if condition.kind is StressKind.CLEAN or modality not in condition.target_modalities:
        return source.copy()

    params = dict(condition.parameters)
    if params.get("scope") != "probe_only":
        raise ValueError("V1 image corruption must declare probe_only scope")
    if condition.operator == "gaussian_blur":
        radius = float(params["radius"])
        if radius <= 0:
            raise ValueError("Gaussian blur radius must be positive")
        return source.filter(ImageFilter.GaussianBlur(radius=radius))
    if condition.operator == "contrast_scale":
        factor = float(params["factor"])
        if not (0 < factor < 1):
            raise ValueError("V1 contrast factor must lie in (0,1)")
        return ImageEnhance.Contrast(source).enhance(factor)
    raise ValueError(f"unsupported V1 corruption operator {condition.operator!r}")


def load_and_apply_v1_corruption(
    path: str | Path,
    condition: StressCondition,
    modality: str,
) -> Image.Image:
    with Image.open(path) as image:
        return apply_v1_corruption(image, condition, modality)
