"""PyTorch implementations of Gate-4-locked neural fusion families.

The modules in this file deliberately expose *models only*. Optimizer choice,
training schedule, validation selection, seeds, and search budgets are governed by
``neural_contracts.py`` and the final experiment protocol. This separation prevents
training convenience from silently changing the scientific comparison.

All forward methods return a higher-is-more-genuine scalar score/logit per trial.
They do not consume labels at inference time.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


def _activation(name: str) -> nn.Module:
    key = str(name).strip().lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"unsupported activation {name!r}")
    return _ACTIVATIONS[key]()


def _mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int, *, activation: str, dropout: float) -> nn.Sequential:
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim and output_dim must be positive")
    hidden = tuple(int(v) for v in hidden_dims)
    if not hidden or any(v <= 0 for v in hidden):
        raise ValueError("hidden_dims must contain positive integers")
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError("dropout must lie in [0, 1)")

    layers: list[nn.Module] = []
    dims = (int(input_dim),) + hidden
    for left, right in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(left, right))
        layers.append(_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))
    layers.append(nn.Linear(dims[-1], int(output_dim)))
    return nn.Sequential(*layers)


def _score_matrix(scores: Tensor) -> Tensor:
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 2:
        raise ValueError("scores must have shape [batch, modalities>=2]")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")
    return scores


def _availability(mask: Tensor, shape: torch.Size) -> Tensor:
    if mask.shape != shape:
        raise ValueError(f"availability must have shape {tuple(shape)}")
    if mask.dtype != torch.bool:
        raise ValueError("availability must be a Boolean tensor")
    if torch.any(mask.sum(dim=1) == 0):
        raise ValueError("each trial must retain at least one available modality")
    return mask


def _quality(quality: Tensor, shape: torch.Size, availability: Tensor) -> Tensor:
    if quality.shape != shape:
        raise ValueError(f"quality must have shape {tuple(shape)}")
    if not torch.isfinite(quality).all() or torch.any(quality < 0):
        raise ValueError("quality must be finite and non-negative")
    if torch.any(quality.masked_select(~availability) != 0):
        raise ValueError("quality for unavailable modalities must be zero")
    return quality


def parameter_count(module: nn.Module) -> int:
    """Return trainable parameter count for reproducibility/cost manifests."""
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


class ScoreMLPFusion(nn.Module):
    """D1: compact nonlinear fusion over a frozen vector of unimodal scores."""

    method_id = "D1"

    def __init__(self, n_modalities: int, hidden_dims: Sequence[int], *, activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        if n_modalities < 2:
            raise ValueError("n_modalities must be >= 2")
        self.n_modalities = int(n_modalities)
        self.network = _mlp(self.n_modalities, hidden_dims, 1, activation=activation, dropout=dropout)

    def forward(self, scores: Tensor) -> Tensor:
        x = _score_matrix(scores)
        if x.shape[1] != self.n_modalities:
            raise ValueError("score modality dimension differs from the configured model")
        return self.network(x).squeeze(-1)


class FeatureFusionMLP(nn.Module):
    """D2: shared nonlinear feature-fusion encoder with cosine verification.

    Enrollment and probe evidence pass through the same encoder. This preserves a
    verification interpretation and avoids turning D2 into an unconstrained
    pairwise classifier. Upstream per-modality normalization is performed by the
    dataset/feature pipeline and is frozen before final testing.
    """

    method_id = "D2"

    def __init__(
        self,
        modality_dims: Sequence[int],
        hidden_dims: Sequence[int],
        *,
        fused_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = tuple(int(v) for v in modality_dims)
        if len(dims) < 2 or any(v <= 0 for v in dims):
            raise ValueError("modality_dims must contain at least two positive dimensions")
        if fused_dim <= 0:
            raise ValueError("fused_dim must be positive")
        self.modality_dims = dims
        self.fused_dim = int(fused_dim)
        self.encoder = _mlp(sum(dims), hidden_dims, self.fused_dim, activation=activation, dropout=dropout)

    def _concat(self, blocks: Sequence[Tensor]) -> Tensor:
        if len(blocks) != len(self.modality_dims):
            raise ValueError("one embedding block is required per modality")
        batch = None
        checked: list[Tensor] = []
        for index, (block, dim) in enumerate(zip(blocks, self.modality_dims)):
            if block.ndim != 2 or block.shape[1] != dim:
                raise ValueError(f"modality {index} must have shape [batch, {dim}]")
            if batch is None:
                batch = block.shape[0]
                if batch == 0:
                    raise ValueError("embedding blocks must be non-empty")
            elif block.shape[0] != batch:
                raise ValueError("embedding blocks must share the same batch size")
            if not torch.isfinite(block).all():
                raise ValueError("embedding blocks must be finite")
            checked.append(block)
        return torch.cat(checked, dim=1)

    def encode(self, blocks: Sequence[Tensor]) -> Tensor:
        fused = self.encoder(self._concat(blocks))
        return F.normalize(fused, p=2, dim=1, eps=1e-12)

    def forward(self, enrollment: Sequence[Tensor], probe: Sequence[Tensor]) -> Tensor:
        left = self.encode(enrollment)
        right = self.encode(probe)
        if left.shape != right.shape:
            raise ValueError("enrollment and probe fused embeddings must align")
        return torch.sum(left * right, dim=1)


class ScoreQualityGate(nn.Module):
    """D3S: quality/availability-conditioned learned gate over unimodal scores."""

    method_id = "D3S"

    def __init__(self, n_modalities: int, hidden_dims: Sequence[int], *, activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        if n_modalities < 2:
            raise ValueError("n_modalities must be >= 2")
        self.n_modalities = int(n_modalities)
        # Gate sees quality and an explicit availability bit per modality.
        self.gate = _mlp(2 * self.n_modalities, hidden_dims, self.n_modalities, activation=activation, dropout=dropout)
        # A global affine mapping permits verification-loss training without
        # changing the modality weights themselves into unconstrained score heads.
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def weights(self, quality: Tensor, availability: Tensor) -> Tensor:
        availability = _availability(availability, quality.shape)
        quality = _quality(quality, availability.shape, availability)
        gate_input = torch.cat([quality, availability.to(dtype=quality.dtype)], dim=1)
        logits = self.gate(gate_input)
        logits = logits.masked_fill(~availability, torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=1)

    def forward(self, scores: Tensor, quality: Tensor, availability: Tensor) -> Tensor:
        x = _score_matrix(scores)
        if x.shape[1] != self.n_modalities:
            raise ValueError("score modality dimension differs from the configured model")
        availability = _availability(availability, x.shape)
        if torch.any(x.masked_select(~availability) != 0):
            raise ValueError("unavailable score slots must use the canonical zero placeholder")
        weights = self.weights(quality, availability)
        fused = torch.sum(weights * x, dim=1)
        return self.scale * fused + self.bias


class FeatureQualityGate(nn.Module):
    """D3F: quality/availability-conditioned gate over modality feature projections.

    The same trial-specific modality weights are applied to enrollment and probe
    projections. Weights are derived from probe quality plus *joint* availability,
    so an unavailable modality cannot influence either side of the verification
    comparison. Each modality may have a different upstream embedding dimension.
    """

    method_id = "D3F"

    def __init__(
        self,
        modality_dims: Sequence[int],
        *,
        projection_dim: int,
        gate_hidden_dims: Sequence[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = tuple(int(v) for v in modality_dims)
        if len(dims) < 2 or any(v <= 0 for v in dims):
            raise ValueError("modality_dims must contain at least two positive dimensions")
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        self.modality_dims = dims
        self.n_modalities = len(dims)
        self.projection_dim = int(projection_dim)
        self.projections = nn.ModuleList([nn.Linear(dim, self.projection_dim) for dim in dims])
        self.gate = _mlp(
            2 * self.n_modalities,
            gate_hidden_dims,
            self.n_modalities,
            activation=activation,
            dropout=dropout,
        )

    def _blocks(self, blocks: Sequence[Tensor]) -> list[Tensor]:
        if len(blocks) != self.n_modalities:
            raise ValueError("one embedding block is required per modality")
        batch = None
        out: list[Tensor] = []
        for index, (block, dim) in enumerate(zip(blocks, self.modality_dims)):
            if block.ndim != 2 or block.shape[1] != dim:
                raise ValueError(f"modality {index} must have shape [batch, {dim}]")
            if batch is None:
                batch = block.shape[0]
                if batch == 0:
                    raise ValueError("embedding blocks must be non-empty")
            elif block.shape[0] != batch:
                raise ValueError("embedding blocks must share the same batch size")
            if not torch.isfinite(block).all():
                raise ValueError("embedding blocks must be finite")
            out.append(block)
        return out

    def forward(
        self,
        enrollment: Sequence[Tensor],
        probe: Sequence[Tensor],
        quality: Tensor,
        enrollment_availability: Tensor,
        probe_availability: Tensor,
    ) -> Tensor:
        enroll = self._blocks(enrollment)
        query = self._blocks(probe)
        if enroll[0].shape[0] != query[0].shape[0]:
            raise ValueError("enrollment and probe batch sizes must match")
        shape = torch.Size((enroll[0].shape[0], self.n_modalities))
        enroll_av = _availability(enrollment_availability, shape)
        probe_av = _availability(probe_availability, shape)
        joint_av = enroll_av & probe_av
        if torch.any(joint_av.sum(dim=1) == 0):
            raise ValueError("each trial must retain at least one jointly available modality")
        quality = _quality(quality, shape, probe_av)
        # Quality for a modality that is probe-available but enrollment-missing is
        # intentionally removed before gating because it cannot contribute jointly.
        joint_quality = quality * joint_av.to(dtype=quality.dtype)
        gate_input = torch.cat([joint_quality, joint_av.to(dtype=quality.dtype)], dim=1)
        logits = self.gate(gate_input)
        logits = logits.masked_fill(~joint_av, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)

        projected_enroll: list[Tensor] = []
        projected_probe: list[Tensor] = []
        for projector, left, right in zip(self.projections, enroll, query):
            projected_enroll.append(F.normalize(projector(left), p=2, dim=1, eps=1e-12))
            projected_probe.append(F.normalize(projector(right), p=2, dim=1, eps=1e-12))

        left_stack = torch.stack(projected_enroll, dim=1)
        right_stack = torch.stack(projected_probe, dim=1)
        fused_left = torch.sum(weights.unsqueeze(-1) * left_stack, dim=1)
        fused_right = torch.sum(weights.unsqueeze(-1) * right_stack, dim=1)
        fused_left = F.normalize(fused_left, p=2, dim=1, eps=1e-12)
        fused_right = F.normalize(fused_right, p=2, dim=1, eps=1e-12)
        return torch.sum(fused_left * fused_right, dim=1)
