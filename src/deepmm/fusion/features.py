"""Classical feature-level fusion baselines for aligned multimodal templates."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _embedding_blocks(blocks: Sequence, *, expected_dims=None) -> list[np.ndarray]:
    if not isinstance(blocks, (list, tuple)) or len(blocks) < 2:
        raise ValueError("blocks must contain at least two modality embedding matrices")
    arrays = [np.asarray(block, dtype=np.float64) for block in blocks]
    n = arrays[0].shape[0] if arrays[0].ndim == 2 else -1
    for i, block in enumerate(arrays):
        if block.ndim != 2 or block.shape[0] == 0 or block.shape[1] == 0:
            raise ValueError(f"modality block {i} must be a non-empty 2-D matrix")
        if block.shape[0] != n:
            raise ValueError("all modality blocks must contain the same aligned rows")
        if not np.all(np.isfinite(block)):
            raise ValueError("embedding blocks must be finite; missing modalities require an explicit policy")
    if expected_dims is not None:
        dims = tuple(block.shape[1] for block in arrays)
        if dims != tuple(expected_dims):
            raise ValueError("embedding dimensions differ from fitted model")
    return arrays


def cosine_similarity_rows(left, right) -> np.ndarray:
    """Return row-wise cosine similarity for two equal embedding matrices."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or a.shape[0] == 0:
        raise ValueError("left and right must be equal non-empty 2-D matrices")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("embeddings must be finite")
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    if np.any(an <= np.finfo(float).eps) or np.any(bn <= np.finfo(float).eps):
        raise ValueError("cosine similarity is undefined for zero-norm rows")
    return np.sum(a * b, axis=1) / (an * bn)


class StandardizedConcatFusion:
    """Low-capacity classical feature concatenation for Track-I comparison.

    The baseline fits only per-feature mean/standard deviation on **training**
    embeddings. At transform time each modality block is standardized, each block
    is L2-normalized so a higher-dimensional modality does not dominate merely by
    dimensionality, the blocks are concatenated with equal block scaling, and the
    final fused vector is L2-normalized.

    This baseline contains no neural fusion head and no test-label fitting. It is
    intended to answer whether access to feature-level evidence alone explains a
    gain attributed to a deep feature-fusion model.

    Row alignment is a dataset-adapter responsibility: each row must represent the
    same logical multimodal template/sample unit across all modality blocks.
    """

    def fit(self, blocks: Sequence) -> "StandardizedConcatFusion":
        arrays = _embedding_blocks(blocks)
        means: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        dims: list[int] = []
        for block in arrays:
            mean = block.mean(axis=0)
            scale = block.std(axis=0, ddof=0)
            # Constant coordinates carry no discriminative variation; map them to
            # zero after centering rather than failing the whole embedding block.
            scale = np.where(scale <= np.finfo(np.float64).eps, 1.0, scale)
            means.append(mean)
            scales.append(scale)
            dims.append(block.shape[1])
        self.means_ = means
        self.scales_ = scales
        self.dims_ = tuple(dims)
        self.n_modalities_ = len(arrays)
        return self

    def transform(self, blocks: Sequence) -> np.ndarray:
        if not hasattr(self, "dims_"):
            raise RuntimeError("fit must be called before transform")
        arrays = _embedding_blocks(blocks, expected_dims=self.dims_)
        if len(arrays) != self.n_modalities_:
            raise ValueError("number of modalities differs from fitted model")

        normalized_blocks: list[np.ndarray] = []
        block_scale = 1.0 / np.sqrt(float(self.n_modalities_))
        for block, mean, scale in zip(arrays, self.means_, self.scales_):
            z = (block - mean) / scale
            norms = np.linalg.norm(z, axis=1, keepdims=True)
            # A row that is exactly at every training mean has no direction in the
            # standardized block; retain it as a zero contribution rather than
            # injecting an arbitrary vector.
            safe = np.where(norms > np.finfo(float).eps, norms, 1.0)
            normalized_blocks.append(block_scale * z / safe)

        fused = np.concatenate(normalized_blocks, axis=1)
        norms = np.linalg.norm(fused, axis=1, keepdims=True)
        if np.any(norms <= np.finfo(float).eps):
            raise ValueError("fused embedding has zero norm for at least one row")
        return fused / norms

    def fit_transform(self, blocks: Sequence) -> np.ndarray:
        return self.fit(blocks).transform(blocks)
