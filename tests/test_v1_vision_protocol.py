import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image

from deepmm.robustness import v1_stress_plan
from deepmm.vision.quality import V1QualityModel, raw_quality_features
from deepmm.vision.stress import apply_v1_corruption


def _save_pattern(path, *, contrast=1.0, shift=0):
    y, x = np.mgrid[0:64, 0:64]
    base = ((x * 4 + y * 3 + shift) % 256).astype(np.float64)
    array = np.clip(127.5 + contrast * (base - 127.5), 0, 255).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def test_quality_model_is_label_free_positive_and_hashable(tmp_path):
    modality_paths = {"fingerprint": [], "finger_vein": []}
    for modality_index, modality in enumerate(modality_paths):
        for index in range(12):
            path = tmp_path / f"{modality}-{index}.png"
            _save_pattern(path, contrast=0.45 + 0.04 * index, shift=modality_index * 13 + index)
            modality_paths[modality].append(path)

    model = V1QualityModel.fit(modality_paths)
    assert len(model.model_hash) == 64
    for modality, paths in modality_paths.items():
        score = model.image_quality(modality, paths[5])
        assert 0.0 < score <= 1.0
        pair = model.trial_quality(modality, paths[4], paths[6])
        assert 0.0 < pair <= 1.0
        assert model.image_quality(modality, paths[5], available=False) == 0.0
        assert model.trial_quality(modality, paths[4], paths[6], available=False) == 0.0


def test_blur_reduces_gradient_and_contrast_scale_reduces_contrast(tmp_path):
    path = tmp_path / "pattern.png"
    _save_pattern(path, contrast=1.0)
    with Image.open(path) as source:
        source = source.convert("L")
        clean = np.asarray(source, dtype=np.uint8)
        plan = {condition.condition_id: condition for condition in v1_stress_plan()}
        blurred = apply_v1_corruption(source, plan["fingerprint-blur-3"], "fingerprint")
        contrasted = apply_v1_corruption(source, plan["fingerprint-contrast-3"], "fingerprint")
        untouched = apply_v1_corruption(source, plan["fingerprint-blur-3"], "finger_vein")

    clean_path = tmp_path / "clean.png"
    blur_path = tmp_path / "blur.png"
    contrast_path = tmp_path / "contrast.png"
    untouched_path = tmp_path / "untouched.png"
    Image.fromarray(clean).save(clean_path)
    blurred.save(blur_path)
    contrasted.save(contrast_path)
    untouched.save(untouched_path)

    clean_features = raw_quality_features(clean_path)
    blur_features = raw_quality_features(blur_path)
    contrast_features = raw_quality_features(contrast_path)
    assert blur_features[1] < clean_features[1]
    assert contrast_features[0] < clean_features[0]
    assert np.array_equal(np.asarray(untouched), clean)


def test_missing_condition_cannot_be_rendered_as_pixels(tmp_path):
    path = tmp_path / "pattern.png"
    _save_pattern(path)
    missing = next(c for c in v1_stress_plan() if c.condition_id == "missing-fingerprint")
    with Image.open(path) as image:
        with pytest.raises(ValueError, match="availability masks"):
            apply_v1_corruption(image, missing, "fingerprint")
