"""Deterministic PyTorch development harness for DeepMM neural fusion heads.

This module is intentionally restricted to *development* partitions. It supports
training and early stopping but has no final-test API and performs no score
calibration. Calibration remains a separate held-out operation.

The default scientific comparison is not encoded here: optimizer values, model
widths, epoch counts and search spaces must be frozen in experiment configuration
after the real dataset/encoder dimensionality is audited.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from deepmm.fusion.neural_contracts import TrainingBudget
from deepmm.metrics import eer, roc_auc
from .contracts import FinalTestFirewall


Batch = Mapping[str, Any]
BatchFactory = Callable[[], Iterable[Batch]]
ForwardBatch = Callable[[nn.Module, Batch], Tensor]
SelectionFunction = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class TorchOptimizerConfig:
    """One explicit development optimizer configuration.

    Values are required rather than silently treated as final scientific defaults.
    """

    optimizer: str
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float | None = None
    deterministic_algorithms: bool = True

    def __post_init__(self) -> None:
        optimizer = str(self.optimizer).strip().lower()
        if optimizer not in {"adam", "adamw"}:
            raise ValueError("optimizer must be 'adam' or 'adamw'")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.gradient_clip_norm is not None:
            if not np.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
                raise ValueError("gradient_clip_norm must be finite and positive when set")
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "learning_rate", float(self.learning_rate))
        object.__setattr__(self, "weight_decay", float(self.weight_decay))
        if self.gradient_clip_norm is not None:
            object.__setattr__(self, "gradient_clip_norm", float(self.gradient_clip_norm))

    def as_dict(self) -> dict[str, object]:
        return {
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip_norm": self.gradient_clip_norm,
            "deterministic_algorithms": self.deterministic_algorithms,
        }


@dataclass(frozen=True)
class DevelopmentFitResult:
    seed: int
    epochs_completed: int
    best_epoch: int
    selection_metric_id: str
    best_selection_value: float
    checkpoint_hash: str
    history: tuple[dict[str, float], ...]

    def __post_init__(self) -> None:
        if self.epochs_completed < 1 or self.best_epoch < 1:
            raise ValueError("epoch counts must be positive")
        if self.best_epoch > self.epochs_completed:
            raise ValueError("best_epoch cannot exceed epochs_completed")
        if not np.isfinite(self.best_selection_value):
            raise ValueError("best_selection_value must be finite")
        if len(self.checkpoint_hash) != 64:
            raise ValueError("checkpoint_hash must be SHA-256")


def set_torch_seed(seed: int, *, deterministic_algorithms: bool) -> None:
    """Seed Python, NumPy and PyTorch for a declared technical repeat."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(deterministic_algorithms))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic_algorithms)


def checkpoint_state_hash(module: nn.Module) -> str:
    """Hash tensor state deterministically without relying on torch.save bytes."""
    digest = hashlib.sha256()
    state = module.state_dict()
    if not state:
        raise ValueError("module state_dict must not be empty")
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"state entry {name!r} is not a Tensor")
        value = tensor.detach().cpu().contiguous()
        descriptor = json.dumps(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        raw = value.numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _labels(batch: Batch, label_key: str, *, device: torch.device) -> Tensor:
    if label_key not in batch:
        raise ValueError(f"batch is missing label key {label_key!r}")
    labels = batch[label_key]
    if not isinstance(labels, Tensor):
        labels = torch.as_tensor(labels)
    labels = labels.to(device=device, dtype=torch.float32)
    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("labels must be a non-empty 1-D tensor")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("binary training labels must be 0/1")
    return labels


def _optimizer(module: nn.Module, config: TorchOptimizerConfig) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    cls = torch.optim.Adam if config.optimizer == "adam" else torch.optim.AdamW
    return cls(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _builtin_selection(objective: str) -> tuple[str, SelectionFunction, bool]:
    objective = str(objective).strip().lower()
    if objective == "eer":
        return "eer", lambda y, score: float(eer(y, score)[0]), True
    if objective == "auc":
        return "auc", lambda y, score: float(roc_auc(y, score)), False
    raise ValueError(
        f"training-budget objective {objective!r} requires an explicit held-out selection function; "
        "raw neural logits are not silently treated as calibrated LLRs or composite scores"
    )


def _evaluate(
    module: nn.Module,
    batches: BatchFactory,
    forward_batch: ForwardBatch,
    *,
    label_key: str,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray]:
    module.eval()
    total_loss = 0.0
    total_count = 0
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches():
            labels = _labels(batch, label_key, device=device)
            scores = forward_batch(module, batch)
            if not isinstance(scores, Tensor):
                raise TypeError("forward_batch must return a Tensor")
            scores = scores.to(device=device, dtype=torch.float32)
            if scores.ndim != 1 or scores.shape != labels.shape:
                raise ValueError("model scores must be a 1-D tensor matching labels")
            if not torch.isfinite(scores).all():
                raise ValueError("model scores must be finite")
            loss = loss_fn(scores, labels)
            total_loss += float(loss.detach()) * labels.numel()
            total_count += int(labels.numel())
            labels_all.append(labels.detach().cpu().numpy().astype(np.int8, copy=False))
            scores_all.append(scores.detach().cpu().numpy().astype(np.float64, copy=False))
    if total_count == 0:
        raise ValueError("batch factory produced no development examples")
    y = np.concatenate(labels_all)
    s = np.concatenate(scores_all)
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("development selection data must contain both classes")
    return total_loss / total_count, y, s


def fit_binary_score_model(
    module: nn.Module,
    *,
    train_batches: BatchFactory,
    selection_batches: BatchFactory,
    forward_batch: ForwardBatch,
    budget: TrainingBudget,
    seed: int,
    optimizer_config: TorchOptimizerConfig,
    firewall: FinalTestFirewall,
    train_partition: str,
    selection_partition: str,
    device: str | torch.device = "cpu",
    label_key: str = "labels",
    selection_function: SelectionFunction | None = None,
    selection_metric_id: str | None = None,
    selection_minimize: bool | None = None,
) -> DevelopmentFitResult:
    """Fit one neural score model using development data only.

    Binary cross-entropy with logits is used as the optimization loss. The model-
    selection metric is defined independently. EER/AUC are available as built-ins;
    calibration/composite selection requires an explicit function so raw logits are
    never mislabeled as LLR-calibrated evidence.
    """
    if seed not in budget.seeds:
        raise ValueError("seed is outside the declared TrainingBudget seed set")
    if firewall.assert_development_partition(train_partition) != firewall.fit_partition:
        raise ValueError("train_partition must equal firewall.fit_partition")
    if firewall.assert_development_partition(selection_partition) != firewall.selection_partition:
        raise ValueError("selection_partition must equal firewall.selection_partition")
    if train_partition == selection_partition:
        raise ValueError("training and model-selection partitions must differ")

    if selection_function is None:
        metric_id, selection_function, minimize = _builtin_selection(budget.tuning_objective)
        if selection_metric_id is not None or selection_minimize is not None:
            raise ValueError("do not override metric metadata when using the built-in selector")
    else:
        metric_id = str(selection_metric_id or "").strip()
        if not metric_id:
            raise ValueError("selection_metric_id is required with a custom selection_function")
        if selection_minimize is None:
            raise ValueError("selection_minimize is required with a custom selection_function")
        minimize = bool(selection_minimize)

    torch_device = torch.device(device)
    set_torch_seed(seed, deterministic_algorithms=optimizer_config.deterministic_algorithms)
    module.to(torch_device)
    optimizer = _optimizer(module, optimizer_config)
    loss_fn = nn.BCEWithLogitsLoss()

    best_value: float | None = None
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, budget.max_epochs + 1):
        module.train()
        train_loss_total = 0.0
        train_count = 0
        for batch in train_batches():
            labels = _labels(batch, label_key, device=torch_device)
            optimizer.zero_grad(set_to_none=True)
            scores = forward_batch(module, batch)
            if not isinstance(scores, Tensor):
                raise TypeError("forward_batch must return a Tensor")
            scores = scores.to(device=torch_device, dtype=torch.float32)
            if scores.ndim != 1 or scores.shape != labels.shape:
                raise ValueError("model scores must be a 1-D tensor matching labels")
            if not torch.isfinite(scores).all():
                raise ValueError("model scores must be finite")
            loss = loss_fn(scores, labels)
            if not torch.isfinite(loss):
                raise ValueError("training loss became non-finite")
            loss.backward()
            if optimizer_config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    module.parameters(), optimizer_config.gradient_clip_norm
                )
            optimizer.step()
            train_loss_total += float(loss.detach()) * labels.numel()
            train_count += int(labels.numel())
        if train_count == 0:
            raise ValueError("train batch factory produced no examples")

        dev_loss, dev_labels, dev_scores = _evaluate(
            module,
            selection_batches,
            forward_batch,
            label_key=label_key,
            device=torch_device,
            loss_fn=loss_fn,
        )
        value = float(selection_function(dev_labels, dev_scores))
        if not np.isfinite(value):
            raise ValueError("selection metric returned a non-finite value")

        history.append(
            {
                "epoch": float(epoch),
                "train_bce": train_loss_total / train_count,
                "selection_bce": float(dev_loss),
                "selection_metric": value,
            }
        )

        improved = (
            best_value is None
            or (value < best_value if minimize else value > best_value)
        )
        if improved:
            best_value = value
            best_epoch = epoch
            best_state = deepcopy(module.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= budget.early_stopping_patience:
                break

    if best_state is None or best_value is None:
        raise RuntimeError("no valid model state was selected")
    module.load_state_dict(best_state)
    module.to(torch_device)

    return DevelopmentFitResult(
        seed=seed,
        epochs_completed=len(history),
        best_epoch=best_epoch,
        selection_metric_id=metric_id,
        best_selection_value=float(best_value),
        checkpoint_hash=checkpoint_state_hash(module),
        history=tuple(history),
    )
