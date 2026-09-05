#!/usr/bin/env python3
"""Run the locked V1 fit+selection campaign on the public NUPT-FPV subset.

This script is deliberately unable to generate calibration or final trials. It
opens only samples referenced by the frozen `fit` and `selection` roles, freezes
quality normalization on fit images, fits classical comparators, and executes the
predeclared D1/D2/D3S neural search. Its output is development evidence only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from deepmm.datasets import build_v1_evidence_units, generate_v1_trials, scan_nupt_fpv
from deepmm.fusion import (
    EqualScoreFusion,
    LogisticScoreFusion,
    QualityWeightedScoreFusion,
    WeightedScoreFusion,
)
from deepmm.fusion.classical import zscore_fit, zscore_transform
from deepmm.fusion.features import StandardizedConcatFusion, cosine_similarity_rows
from deepmm.fusion.neural_torch import (
    FeatureFusionMLP,
    ScoreMLPFusion,
    ScoreQualityGate,
    parameter_count,
)
from deepmm.metrics import eer, roc_auc
from deepmm.training.contracts import FinalTestFirewall
from deepmm.training.torch_fit import fit_binary_score_model
from deepmm.training.v1_public_config import (
    V1_BATCH_SIZE,
    V1_NEURAL_BUDGET,
    V1_NEURAL_CANDIDATES,
    V1_OPTIMIZER,
    V1_PRIMARY_ENCODER,
    V1_REPORTING_SEED,
    V1_SELECTION_RULE,
    V1_TRAINING_LOCK_SHA256,
    assert_v1_training_lock,
)
from deepmm.validation import dataset_manifest_hash, trial_manifest_hash
from deepmm.vision import FrozenTorchvisionEncoder, V1QualityModel


FIREWALL = FinalTestFirewall(
    fit_partition="fit",
    selection_partition="selection",
    calibration_partition="calibration",
    final_test_partition="final",
)


def _metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    value, _ = eer(labels, scores)
    return {"eer": float(value), "auc": float(roc_auc(labels, scores))}


def _paths_by_sample(rows: list[dict[str, Any]], root: Path) -> dict[str, Path]:
    return {row["sample_id"]: root / row["relative_path"] for row in rows}


def _needed_sample_ids(trials: list[dict[str, Any]]) -> set[str]:
    fields = (
        "enrollment_fingerprint_sample_id",
        "enrollment_finger_vein_sample_id",
        "probe_fingerprint_sample_id",
        "probe_finger_vein_sample_id",
    )
    return {trial[field] for trial in trials for field in fields}


def _trial_arrays(
    trials: list[dict[str, Any]],
    embeddings: dict[str, np.ndarray],
    qualities: dict[str, float],
) -> dict[str, Any]:
    efp = np.stack([embeddings[t["enrollment_fingerprint_sample_id"]] for t in trials]).astype(np.float32)
    efv = np.stack([embeddings[t["enrollment_finger_vein_sample_id"]] for t in trials]).astype(np.float32)
    pfp = np.stack([embeddings[t["probe_fingerprint_sample_id"]] for t in trials]).astype(np.float32)
    pfv = np.stack([embeddings[t["probe_finger_vein_sample_id"]] for t in trials]).astype(np.float32)
    labels = np.asarray([t["label"] for t in trials], dtype=np.int8)
    scores = np.column_stack(
        [cosine_similarity_rows(efp, pfp), cosine_similarity_rows(efv, pfv)]
    ).astype(np.float64)
    quality = np.column_stack(
        [
            np.sqrt(
                np.asarray([qualities[t["enrollment_fingerprint_sample_id"]] for t in trials])
                * np.asarray([qualities[t["probe_fingerprint_sample_id"]] for t in trials])
            ),
            np.sqrt(
                np.asarray([qualities[t["enrollment_finger_vein_sample_id"]] for t in trials])
                * np.asarray([qualities[t["probe_finger_vein_sample_id"]] for t in trials])
            ),
        ]
    ).astype(np.float32)
    availability = np.ones((len(trials), 2), dtype=bool)
    return {
        "enrollment": [efp, efv],
        "probe": [pfp, pfv],
        "scores": scores,
        "quality": quality,
        "availability": availability,
        "labels": labels,
    }


def _fit_quality_model(rows: list[dict[str, Any]], root: Path) -> V1QualityModel:
    modality_paths: dict[str, list[Path]] = {"fingerprint": [], "finger_vein": []}
    for row in rows:
        if row["session_id"] == "1" and row["capture_id"] in {"01", "02", "03", "04", "05"}:
            modality = row["modality"]
            if modality in modality_paths:
                modality_paths[modality].append(root / row["relative_path"])
    if {key: len(value) for key, value in modality_paths.items()} != {
        "fingerprint": 100,
        "finger_vein": 100,
    }:
        raise ValueError("unexpected V1 fit-image count for quality normalization")
    return V1QualityModel.fit(modality_paths)


def _precompute_quality(
    rows: list[dict[str, Any]],
    root: Path,
    needed: set[str],
    model: V1QualityModel,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in needed:
            out[sample_id] = model.image_quality(
                row["modality"], root / row["relative_path"]
            )
    if set(out) != needed:
        raise ValueError("quality cache does not cover exactly the development samples")
    return out


def _fit_feature_baseline(
    rows: list[dict[str, Any]], embeddings: dict[str, np.ndarray]
) -> StandardizedConcatFusion:
    units = build_v1_evidence_units(rows)
    keys = [
        key
        for key in sorted(units)
        if key[1] == "1" and key[2] in {"01", "02", "03", "04", "05"}
    ]
    if len(keys) != 100:
        raise ValueError(f"expected 100 fit evidence units, found {len(keys)}")
    fp = np.stack([embeddings[units[key]["fingerprint_sample_id"]] for key in keys])
    fv = np.stack([embeddings[units[key]["finger_vein_sample_id"]] for key in keys])
    return StandardizedConcatFusion().fit([fp, fv])


def _balanced_fit_indices(labels: np.ndarray) -> np.ndarray:
    genuine = np.flatnonzero(labels == 1)
    impostor = np.flatnonzero(labels == 0)
    if genuine.size != 120 or impostor.size != 2280:
        raise ValueError("V1 fit role no longer has the frozen 120/2280 class counts")
    if impostor.size != 19 * genuine.size:
        raise ValueError("frozen V1 19:1 class ratio changed")
    repeated_genuine = np.tile(genuine, 19)
    # Deterministic alternating order gives every full 256-sized batch the same
    # class balance up to the final partial batch, without introducing a shuffle seed.
    paired = np.column_stack([impostor, repeated_genuine]).reshape(-1)
    if paired.size != 4560:
        raise AssertionError("unexpected balanced fit size")
    return paired


def _tensor(array, *, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype)


def _score_batch_factory(
    data: dict[str, Any], indices: np.ndarray, *, include_quality: bool
) -> Callable[[], Any]:
    scores = _tensor(data["scores"])
    labels = _tensor(data["labels"])
    quality = _tensor(data["quality"])
    availability = torch.as_tensor(data["availability"], dtype=torch.bool)
    idx = torch.as_tensor(indices, dtype=torch.long)

    def factory():
        for start in range(0, idx.numel(), V1_BATCH_SIZE):
            take = idx[start : start + V1_BATCH_SIZE]
            batch = {"scores": scores[take], "labels": labels[take]}
            if include_quality:
                batch["quality"] = quality[take]
                batch["availability"] = availability[take]
            yield batch

    return factory


def _feature_batch_factory(data: dict[str, Any], indices: np.ndarray) -> Callable[[], Any]:
    enrollment = [_tensor(block) for block in data["enrollment"]]
    probe = [_tensor(block) for block in data["probe"]]
    labels = _tensor(data["labels"])
    idx = torch.as_tensor(indices, dtype=torch.long)

    def factory():
        for start in range(0, idx.numel(), V1_BATCH_SIZE):
            take = idx[start : start + V1_BATCH_SIZE]
            yield {
                "enrollment": [block[take] for block in enrollment],
                "probe": [block[take] for block in probe],
                "labels": labels[take],
            }

    return factory


def _selection_scores(model: torch.nn.Module, family: str, data: dict[str, Any]) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        if family == "D1":
            scores = model(_tensor(data["scores"]))
        elif family == "D2":
            scores = model(
                [_tensor(block) for block in data["enrollment"]],
                [_tensor(block) for block in data["probe"]],
            )
        elif family == "D3S":
            scores = model(
                _tensor(data["scores"]),
                _tensor(data["quality"]),
                torch.as_tensor(data["availability"], dtype=torch.bool),
            )
        else:
            raise ValueError(f"unknown family {family}")
    return scores.detach().cpu().numpy().astype(np.float64)


def _build_model(family: str, candidate: dict[str, Any], embedding_dim: int) -> torch.nn.Module:
    common = {
        "activation": candidate["activation"],
        "dropout": candidate["dropout"],
    }
    if family == "D1":
        return ScoreMLPFusion(2, candidate["hidden_dims"], **common)
    if family == "D2":
        return FeatureFusionMLP(
            (embedding_dim, embedding_dim),
            candidate["hidden_dims"],
            fused_dim=int(candidate["fused_dim"]),
            **common,
        )
    if family == "D3S":
        return ScoreQualityGate(2, candidate["hidden_dims"], **common)
    raise ValueError(f"unknown family {family}")


def _forward_for_family(family: str):
    if family == "D1":
        return lambda model, batch: model(batch["scores"])
    if family == "D2":
        return lambda model, batch: model(batch["enrollment"], batch["probe"])
    if family == "D3S":
        return lambda model, batch: model(
            batch["scores"], batch["quality"], batch["availability"]
        )
    raise ValueError(f"unknown family {family}")


def _select_candidate(candidate_results: list[dict[str, Any]]) -> str:
    aggregates = []
    for candidate in candidate_results:
        eers = np.asarray([run["selection_metrics"]["eer"] for run in candidate["runs"]])
        aggregates.append(
            (
                float(np.median(eers)),
                float(np.mean(eers)),
                int(candidate["trainable_parameters"]),
                str(candidate["candidate_id"]),
            )
        )
    aggregates.sort()
    return aggregates[0][3]


def run(root: Path) -> dict[str, Any]:
    assert_v1_training_lock()
    rows = scan_nupt_fpv(root)
    fit_trials = generate_v1_trials(rows, "fit")
    selection_trials = generate_v1_trials(rows, "selection")
    needed = _needed_sample_ids(fit_trials) | _needed_sample_ids(selection_trials)
    paths = _paths_by_sample(rows, root)
    if not needed.issubset(paths):
        raise ValueError("development trial manifest refers to missing raw samples")

    encoder = FrozenTorchvisionEncoder(V1_PRIMARY_ENCODER)
    embeddings = {
        sample_id: encoder.encode_image(paths[sample_id]) for sample_id in sorted(needed)
    }
    quality_model = _fit_quality_model(rows, root)
    qualities = _precompute_quality(rows, root, needed, quality_model)

    fit = _trial_arrays(fit_trials, embeddings, qualities)
    selection = _trial_arrays(selection_trials, embeddings, qualities)

    # One frozen score normalization is reused by D1/D3S. C1/C2/C3/C5 fit the
    # equivalent development-only normalization inside their own implementations.
    score_mean, score_scale = zscore_fit(fit["scores"])
    fit_neural_scores = zscore_transform(fit["scores"], score_mean, score_scale)
    selection_neural_scores = zscore_transform(selection["scores"], score_mean, score_scale)
    fit_neural = dict(fit)
    selection_neural = dict(selection)
    fit_neural["scores"] = fit_neural_scores.astype(np.float32)
    selection_neural["scores"] = selection_neural_scores.astype(np.float32)

    c1 = EqualScoreFusion(normalize=True).fit(fit["scores"])
    c2 = WeightedScoreFusion(grid_step=0.05, objective="eer", normalize=True).fit(
        fit["scores"], fit["labels"]
    )
    c3 = LogisticScoreFusion(C=1.0, normalize=True).fit(fit["scores"], fit["labels"])
    c4 = _fit_feature_baseline(rows, embeddings)
    c5 = QualityWeightedScoreFusion(objective="eer", normalize=True).fit(
        fit["scores"], fit["quality"], fit["labels"]
    )

    c4_left = c4.transform(selection["enrollment"])
    c4_right = c4.transform(selection["probe"])
    classical_scores = {
        "U-FP": selection["scores"][:, 0],
        "U-FV": selection["scores"][:, 1],
        "C1": c1.transform(selection["scores"]),
        "C2": c2.transform(selection["scores"]),
        "C3": c3.transform(selection["scores"]),
        "C4": cosine_similarity_rows(c4_left, c4_right),
        "C5": c5.transform(selection["scores"], selection["quality"]),
    }

    balanced_indices = _balanced_fit_indices(fit["labels"])
    selection_indices = np.arange(selection["labels"].size, dtype=np.int64)
    neural: dict[str, Any] = {}

    for family, candidates in V1_NEURAL_CANDIDATES.items():
        family_results: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_runs: list[dict[str, Any]] = []
            trainable_parameters = None
            for seed in V1_NEURAL_BUDGET.seeds:
                model = _build_model(family, candidate, encoder.spec.embedding_dim)
                params = parameter_count(model)
                trainable_parameters = params if trainable_parameters is None else trainable_parameters
                if params != trainable_parameters:
                    raise AssertionError("parameter count changed across seeds")

                if family == "D2":
                    train_batches = _feature_batch_factory(fit_neural, balanced_indices)
                    selection_batches = _feature_batch_factory(selection_neural, selection_indices)
                else:
                    include_quality = family == "D3S"
                    train_batches = _score_batch_factory(
                        fit_neural, balanced_indices, include_quality=include_quality
                    )
                    selection_batches = _score_batch_factory(
                        selection_neural, selection_indices, include_quality=include_quality
                    )

                fit_result = fit_binary_score_model(
                    model,
                    train_batches=train_batches,
                    selection_batches=selection_batches,
                    forward_batch=_forward_for_family(family),
                    budget=V1_NEURAL_BUDGET,
                    seed=seed,
                    optimizer_config=V1_OPTIMIZER,
                    firewall=FIREWALL,
                    train_partition="fit",
                    selection_partition="selection",
                    device="cpu",
                )
                scores = _selection_scores(model, family, selection_neural)
                candidate_runs.append(
                    {
                        "seed": int(seed),
                        "best_epoch": int(fit_result.best_epoch),
                        "epochs_completed": int(fit_result.epochs_completed),
                        "checkpoint_hash": fit_result.checkpoint_hash,
                        "selection_metrics": _metric(selection["labels"], scores),
                    }
                )

            eers = np.asarray([run["selection_metrics"]["eer"] for run in candidate_runs])
            family_results.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "configuration": {
                        key: (list(value) if isinstance(value, tuple) else value)
                        for key, value in candidate.items()
                    },
                    "trainable_parameters": int(trainable_parameters),
                    "selection_eer_median": float(np.median(eers)),
                    "selection_eer_mean": float(np.mean(eers)),
                    "runs": candidate_runs,
                }
            )

        selected = _select_candidate(family_results)
        neural[family] = {
            "selected_candidate_id": selected,
            "selection_rule": V1_SELECTION_RULE,
            "candidates": family_results,
        }

    result = {
        "scope": "V1 development only: fit + selection",
        "calibration_images_read": False,
        "final_images_read": False,
        "training_lock_sha256": V1_TRAINING_LOCK_SHA256,
        "dataset_manifest_sha256": dataset_manifest_hash(rows),
        "fit_trial_manifest_sha256": trial_manifest_hash(fit_trials),
        "selection_trial_manifest_sha256": trial_manifest_hash(selection_trials),
        "n_unique_images_read": len(needed),
        "fit_original_class_counts": {"genuine": 120, "impostor": 2280},
        "fit_balanced_examples": int(balanced_indices.size),
        "reporting_seed": V1_REPORTING_SEED,
        "encoder": {
            "spec": encoder.spec.as_dict(),
            "spec_hash": encoder.spec.spec_hash,
            "weight_state_hash": encoder.weight_state_hash,
        },
        "quality_model_hash": quality_model.model_hash,
        "score_normalization": {
            "mean": score_mean.tolist(),
            "scale": score_scale.tolist(),
            "fit_only": True,
        },
        "classical_selection": {
            "metrics": {
                key: _metric(selection["labels"], scores)
                for key, scores in classical_scores.items()
            },
            "C2_weights_fingerprint_finger_vein": c2.weights_.tolist(),
            "C5_gamma": float(c5.gamma_),
        },
        "neural_selection": neural,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Root of official NUPT-FPV checkout")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    result = run(args.root)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
