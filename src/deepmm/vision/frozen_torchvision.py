"""Frozen, directly accessible torchvision encoders for the bounded V1 study.

V1 deliberately avoids learning an image backbone from the 20 public NUPT-FPV
instance IDs. The selected upstream encoders are frozen ImageNet models so every
classical and neural fusion family consumes identical unimodal evidence.

The primary and sensitivity backbones are declared by explicit weight enums rather
than ``DEFAULT`` aliases so a future torchvision release cannot silently change the
scientific representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)


EncoderName = Literal["resnet18_imagenet1k_v1", "mobilenet_v3_small_imagenet1k_v1"]


@dataclass(frozen=True)
class FrozenEncoderSpec:
    encoder_id: str
    architecture: str
    weights_id: str
    embedding_dim: int
    input_policy: str
    role: str

    def as_dict(self) -> dict[str, object]:
        return {
            "encoder_id": self.encoder_id,
            "architecture": self.architecture,
            "weights_id": self.weights_id,
            "embedding_dim": self.embedding_dim,
            "input_policy": self.input_policy,
            "role": self.role,
        }

    @property
    def spec_hash(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


_SPECS: dict[str, FrozenEncoderSpec] = {
    "resnet18_imagenet1k_v1": FrozenEncoderSpec(
        encoder_id="resnet18_imagenet1k_v1",
        architecture="torchvision.resnet18",
        weights_id="ResNet18_Weights.IMAGENET1K_V1",
        embedding_dim=512,
        input_policy="grayscale->RGB; official weight transforms; L2 output",
        role="primary",
    ),
    "mobilenet_v3_small_imagenet1k_v1": FrozenEncoderSpec(
        encoder_id="mobilenet_v3_small_imagenet1k_v1",
        architecture="torchvision.mobilenet_v3_small",
        weights_id="MobileNet_V3_Small_Weights.IMAGENET1K_V1",
        embedding_dim=576,
        input_policy="grayscale->RGB; official weight transforms; L2 output",
        role="representation-sensitivity",
    ),
}


def frozen_encoder_spec(name: str) -> FrozenEncoderSpec:
    key = str(name).strip().lower()
    if key not in _SPECS:
        raise ValueError(f"unsupported frozen encoder {name!r}")
    return _SPECS[key]


def _state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        descriptor = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        raw = tensor.numpy().tobytes(order="C")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


class FrozenTorchvisionEncoder:
    """Frozen grayscale-image encoder producing L2-normalized embeddings."""

    def __init__(self, name: str, *, device: str | torch.device = "cpu") -> None:
        self.spec = frozen_encoder_spec(name)
        self.device = torch.device(device)

        if self.spec.encoder_id == "resnet18_imagenet1k_v1":
            weights = ResNet18_Weights.IMAGENET1K_V1
            model = resnet18(weights=weights)
            model.fc = nn.Identity()
        elif self.spec.encoder_id == "mobilenet_v3_small_imagenet1k_v1":
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            model = mobilenet_v3_small(weights=weights)
            model.classifier = nn.Identity()
        else:  # pragma: no cover - spec lookup prevents this.
            raise AssertionError(self.spec.encoder_id)

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval().to(self.device)
        self.model = model
        self.transform = weights.transforms()
        self.weight_state_hash = _state_hash(model)

    def encode_image(self, path: str | Path) -> np.ndarray:
        image_path = Path(path)
        with Image.open(image_path) as source:
            # NUPT-FPV public files are 8-bit grayscale BMPs. Explicit conversion
            # avoids palette/mode differences entering the frozen encoder.
            image = source.convert("L").convert("RGB")
            batch = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            embedding = self.model(batch)
            if embedding.ndim != 2 or embedding.shape != (1, self.spec.embedding_dim):
                raise RuntimeError(
                    f"unexpected {self.spec.encoder_id} embedding shape {tuple(embedding.shape)}"
                )
            embedding = F.normalize(embedding, p=2, dim=1, eps=1e-12)
        result = embedding[0].detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("encoder produced non-finite evidence")
        return result

    def encode_paths(self, paths: list[str | Path]) -> np.ndarray:
        if not paths:
            raise ValueError("paths must not be empty")
        return np.stack([self.encode_image(path) for path in paths], axis=0)
