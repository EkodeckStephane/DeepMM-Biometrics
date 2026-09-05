import numpy as np
import pytest


torch = pytest.importorskip("torch")

from deepmm.fusion.neural_contracts import TrainingBudget
from deepmm.fusion.neural_torch import ScoreMLPFusion
from deepmm.training import FinalTestFirewall
from deepmm.training.torch_fit import (
    TorchOptimizerConfig,
    checkpoint_state_hash,
    fit_binary_score_model,
)


def _data():
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(80, 2, generator=generator)
    y = (x[:, 0] + 0.8 * x[:, 1] > 0).to(torch.float32)
    return x, y


def _factory(x, y, indices, *, batch_size=16):
    indices = tuple(int(value) for value in indices)

    def factory():
        for start in range(0, len(indices), batch_size):
            chosen = indices[start : start + batch_size]
            yield {
                "scores": x[list(chosen)],
                "labels": y[list(chosen)],
            }

    return factory


def _forward(model, batch):
    return model(batch["scores"])


def _budget(objective="eer"):
    return TrainingBudget(
        max_epochs=12,
        early_stopping_patience=2,
        max_candidate_configs=1,
        seeds=(7,),
        tuning_objective=objective,
        max_training_runs=1,
    )


def _optimizer():
    return TorchOptimizerConfig(
        optimizer="adamw",
        learning_rate=0.02,
        weight_decay=0.0,
        deterministic_algorithms=True,
    )


def test_fit_is_deterministic_and_restores_best_checkpoint():
    x, y = _data()
    train = _factory(x, y, range(0, 56))
    selection = _factory(x, y, range(56, 80))
    firewall = FinalTestFirewall("fit", "selection", "calibration", "final_test")

    torch.manual_seed(999)
    model_a = ScoreMLPFusion(2, (8,))
    initial_state = {name: tensor.clone() for name, tensor in model_a.state_dict().items()}

    torch.manual_seed(999)
    model_b = ScoreMLPFusion(2, (8,))
    for name, tensor in initial_state.items():
        model_b.state_dict()[name].copy_(tensor)

    result_a = fit_binary_score_model(
        model_a,
        train_batches=train,
        selection_batches=selection,
        forward_batch=_forward,
        budget=_budget(),
        seed=7,
        optimizer_config=_optimizer(),
        firewall=firewall,
        train_partition="fit",
        selection_partition="selection",
    )
    result_b = fit_binary_score_model(
        model_b,
        train_batches=train,
        selection_batches=selection,
        forward_batch=_forward,
        budget=_budget(),
        seed=7,
        optimizer_config=_optimizer(),
        firewall=firewall,
        train_partition="fit",
        selection_partition="selection",
    )

    assert result_a.checkpoint_hash == result_b.checkpoint_hash
    assert result_a.best_epoch == result_b.best_epoch
    assert result_a.best_selection_value == pytest.approx(result_b.best_selection_value)
    assert checkpoint_state_hash(model_a) == result_a.checkpoint_hash
    assert 1 <= result_a.best_epoch <= result_a.epochs_completed <= 12
    assert result_a.selection_metric_id == "eer"


def test_final_test_cannot_be_used_as_training_or_selection_partition():
    x, y = _data()
    train = _factory(x, y, range(0, 56))
    selection = _factory(x, y, range(56, 80))
    firewall = FinalTestFirewall("fit", "selection", "calibration", "final_test")
    model = ScoreMLPFusion(2, (4,))

    with pytest.raises(ValueError, match="final-test partition"):
        fit_binary_score_model(
            model,
            train_batches=train,
            selection_batches=selection,
            forward_batch=_forward,
            budget=_budget(),
            seed=7,
            optimizer_config=_optimizer(),
            firewall=firewall,
            train_partition="final_test",
            selection_partition="selection",
        )


def test_cllr_objective_requires_explicit_calibrated_selector():
    x, y = _data()
    train = _factory(x, y, range(0, 56))
    selection = _factory(x, y, range(56, 80))
    firewall = FinalTestFirewall("fit", "selection", "calibration", "final_test")
    model = ScoreMLPFusion(2, (4,))

    with pytest.raises(ValueError, match="raw neural logits are not silently treated as calibrated LLRs"):
        fit_binary_score_model(
            model,
            train_batches=train,
            selection_batches=selection,
            forward_batch=_forward,
            budget=_budget("cllr"),
            seed=7,
            optimizer_config=_optimizer(),
            firewall=firewall,
            train_partition="fit",
            selection_partition="selection",
        )


def test_custom_selector_must_declare_direction_and_id():
    x, y = _data()
    train = _factory(x, y, range(0, 56))
    selection = _factory(x, y, range(56, 80))
    firewall = FinalTestFirewall("fit", "selection", "calibration", "final_test")
    model = ScoreMLPFusion(2, (4,))

    def separation(labels: np.ndarray, scores: np.ndarray) -> float:
        return float(scores[labels == 1].mean() - scores[labels == 0].mean())

    result = fit_binary_score_model(
        model,
        train_batches=train,
        selection_batches=selection,
        forward_batch=_forward,
        budget=_budget("composite_development_only"),
        seed=7,
        optimizer_config=_optimizer(),
        firewall=firewall,
        train_partition="fit",
        selection_partition="selection",
        selection_function=separation,
        selection_metric_id="toy_separation_debug_only",
        selection_minimize=False,
    )
    assert result.selection_metric_id == "toy_separation_debug_only"
    assert np.isfinite(result.best_selection_value)
