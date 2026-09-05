"""Label-free image-quality proxy shared by V1 C5 and D3 fusion families.

This is a bounded technical quality cue, not a standardized biometric quality
measure. It combines grayscale contrast and gradient energy. Robust normalization
is fitted on V1 fit-role images only, separately for each modality.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image


V1_QUALITY_MODALITIES = ("fingerprint", "finger_vein")


def raw_quality_features(path: str | Path) -> np.ndarray:
    """Return [contrast_std, gradient_rms] from an 8-bit grayscale image."""
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    if gray.ndim != 2 or min(gray.shape) < 2 or not np.all(np.isfinite(gray)):
        raise ValueError("quality input must be a finite non-trivial grayscale image")
    contrast = float(np.std(gray))
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    # Equal directional contribution avoids geometry-dependent weighting.
    gradient_rms = float(np.sqrt(0.5 * (np.mean(dx * dx) + np.mean(dy * dy))))
    return np.asarray([contrast, gradient_rms], dtype=np.float64)


@dataclass(frozen=True)
class RobustQualityScale:
    low: tuple[float, float]
    high: tuple[float, float]
    lower_quantile: float = 0.05
    upper_quantile: float = 0.95
    available_floor: float = 1e-3

    def __post_init__(self) -> None:
        low = np.asarray(self.low, dtype=np.float64)
        high = np.asarray(self.high, dtype=np.float64)
        if low.shape != (2,) or high.shape != (2,) or not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError("quality scale bounds must be finite 2-vectors")
        if np.any(high <= low):
            raise ValueError("quality scale high bounds must exceed low bounds")
        if not (0 <= self.lower_quantile < self.upper_quantile <= 1):
            raise ValueError("invalid robust quality quantiles")
        if not (0 < self.available_floor < 1):
            raise ValueError("available_floor must lie in (0,1)")

    def score_features(self, features) -> float:
        values = np.asarray(features, dtype=np.float64)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError("quality features must be a finite 2-vector")
        low = np.asarray(self.low, dtype=np.float64)
        high = np.asarray(self.high, dtype=np.float64)
        scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
        adjusted = self.available_floor + (1.0 - self.available_floor) * scaled
        return float(np.sqrt(adjusted[0] * adjusted[1]))

    def as_dict(self) -> dict[str, object]:
        return {
            "low": list(self.low),
            "high": list(self.high),
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "available_floor": self.available_floor,
        }


class V1QualityModel:
    """Per-modality robust quality normalization frozen from fit-role images."""

    def __init__(self, scales: Mapping[str, RobustQualityScale]):
        keys = tuple(sorted(str(key).strip() for key in scales))
        if set(keys) != set(V1_QUALITY_MODALITIES):
            raise ValueError(f"quality scales must cover exactly {V1_QUALITY_MODALITIES}")
        self.scales = {key: scales[key] for key in V1_QUALITY_MODALITIES}

    @classmethod
    def fit(cls, modality_paths: Mapping[str, list[str | Path]]) -> "V1QualityModel":
        scales: dict[str, RobustQualityScale] = {}
        for modality in V1_QUALITY_MODALITIES:
            paths = list(modality_paths.get(modality, ()))
            if len(paths) < 10:
                raise ValueError(f"at least 10 fit images are required for {modality}")
            matrix = np.stack([raw_quality_features(path) for path in paths], axis=0)
            low = np.quantile(matrix, 0.05, axis=0)
            high = np.quantile(matrix, 0.95, axis=0)
            # Degenerate data are scientifically unusable for a quality gate rather
            # than silently patched with arbitrary scaling.
            if np.any(high <= low):
                raise ValueError(f"degenerate fit quality distribution for {modality}")
            scales[modality] = RobustQualityScale(tuple(low), tuple(high))
        return cls(scales)

    def image_quality(self, modality: str, path: str | Path, *, available: bool = True) -> float:
        modality = str(modality).strip()
        if modality not in self.scales:
            raise ValueError(f"unknown modality {modality!r}")
        if not available:
            return 0.0
        return self.scales[modality].score_features(raw_quality_features(path))

    def trial_quality(
        self,
        modality: str,
        enrollment_path: str | Path,
        probe_path: str | Path,
        *,
        available: bool = True,
    ) -> float:
        """Geometric mean of enrollment and probe image quality for one modality."""
        if not available:
            return 0.0
        q_enroll = self.image_quality(modality, enrollment_path)
        q_probe = self.image_quality(modality, probe_path)
        return float(np.sqrt(q_enroll * q_probe))

    def as_dict(self) -> dict[str, object]:
        return {
            "definition": "sqrt(robust_contrast * robust_gradient); pair=sqrt(enroll*probe)",
            "fit_scope": "V1 fit-role images only; per modality; no labels",
            "scales": {modality: self.scales[modality].as_dict() for modality in V1_QUALITY_MODALITIES},
        }

    @property
    def model_hash(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
