import numpy as np
import pytest

from deepmm.fusion.features import StandardizedConcatFusion, cosine_similarity_rows


def _blocks():
    a = np.array(
        [
            [1.0, 0.0, 2.0],
            [2.0, 1.0, 1.0],
            [3.0, 2.0, 0.0],
            [4.0, 3.0, -1.0],
        ]
    )
    b = np.array(
        [
            [10.0, 0.0],
            [11.0, 2.0],
            [13.0, 3.0],
            [16.0, 5.0],
        ]
    )
    return [a, b]


def test_standardized_concat_returns_unit_norm_fused_embeddings():
    model = StandardizedConcatFusion()
    fused = model.fit_transform(_blocks())
    assert fused.shape == (4, 5)
    assert np.allclose(np.linalg.norm(fused, axis=1), 1.0)
    assert cosine_similarity_rows(fused, fused) == pytest.approx(np.ones(4))


def test_each_nonzero_modality_block_has_equal_pre_final_energy():
    model = StandardizedConcatFusion().fit(_blocks())
    fused = model.transform(_blocks())
    # Both standardized blocks are non-zero for these rows. Equal block scaling
    # makes the two modality portions contribute equal squared norm (=0.5) before
    # the final normalization, which remains 1 here.
    assert np.sum(fused[:, :3] ** 2, axis=1) == pytest.approx(np.full(4, 0.5))
    assert np.sum(fused[:, 3:] ** 2, axis=1) == pytest.approx(np.full(4, 0.5))


def test_feature_fusion_rejects_dimension_and_alignment_mismatch():
    model = StandardizedConcatFusion().fit(_blocks())
    a, b = _blocks()
    with pytest.raises(ValueError, match="dimensions differ"):
        model.transform([a[:, :2], b])
    with pytest.raises(ValueError, match="same aligned rows"):
        StandardizedConcatFusion().fit([a, b[:-1]])


def test_cosine_rejects_zero_norm_rows():
    with pytest.raises(ValueError, match="zero-norm"):
        cosine_similarity_rows([[0.0, 0.0]], [[1.0, 0.0]])
