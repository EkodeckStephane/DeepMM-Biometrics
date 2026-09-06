#!/usr/bin/env python3
"""Run the locked V1 held-out calibration campaign.

The workflow first reconstructs the selected reporting-seed neural systems using
*fit + selection only* and verifies their frozen checkpoint hashes. Calibration-role
images are opened only after all checkpoint hashes match. Final-role images are
never generated or opened by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from deepmm.calibration import LogisticLLRCalibrator
from deepmm.datasets import generate_v1_trials, scan_nupt_fpv
from deepmm.fusion import (
    EqualScoreFusion,
    LogisticScoreFusion,
    QualityWeightedScoreFusion,
    WeightedScoreFusion,
)
from deepmm.fusion.classical import zscore_fit, zscore_transform
from deepmm.fusion.features import cosine_similarity_rows
from deepmm.robustness import StressKind, v1_stress_plan, v1_stress_plan_hash
from deepmm.training.torch_fit import fit_binary_score_model
from deepmm.training.v1_public_config import (
    V1_NEURAL_BUDGET,
    V1_NEURAL_CANDIDATES,
    V1_OPTIMIZER,
    V1_PRIMARY_ENCODER,
    V1_REPORTING_SEED,
    assert_v1_training_lock,
)
from deepmm.training.v1_selection_lock import (
    V1_DATASET_MANIFEST_SHA256,
    V1_ENCODER_WEIGHT_STATE_HASH,
    V1_QUALITY_MODEL_HASH,
    V1_SELECTED_MODELS,
    assert_v1_selection_lock,
)
from deepmm.evaluation.v1_final_config import (
    V1_CALIBRATION_C,
    V1_FINAL_POLICY_SHA256,
    V1_STRESS_PLAN_SHA256,
    assert_v1_final_policy_lock,
)
from deepmm.validation import dataset_manifest_hash, trial_manifest_hash
from deepmm.vision import FrozenTorchvisionEncoder
from deepmm.vision.stress import apply_v1_corruption

# Reuse the exact already-executed development harness primitives rather than
# maintaining a second subtly different implementation of score construction.
from run_v1_development_training import (  # noqa: E402
    FIREWALL,
    _balanced_fit_indices,
    _build_model,
    _feature_batch_factory,
    _fit_feature_baseline,
    _fit_quality_model,
    _forward_for_family,
    _needed_sample_ids,
    _paths_by_sample,
    _precompute_quality,
    _score_batch_factory,
    _selection_scores,
    _trial_arrays,
)


V1_CALIBRATION_TRIAL_SHA256 = "f6104160a138bccebc1f4b03fd5012be1712027d41c44ad5f916a8c34639ca88"
SYSTEM_IDS = ("U-FP", "U-FV", "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3S")


def _sha256_scores(values: np.ndarray) -> str:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("score hash requires a non-empty finite 1-D vector")
    digest = hashlib.sha256()
    descriptor = json.dumps(
        {"dtype": "float64", "shape": list(x.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(descriptor)
    digest.update(x.astype("<f8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _calibrator_record(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    score_hash = _sha256_scores(scores)
    try:
        calibrator = LogisticLLRCalibrator(C=V1_CALIBRATION_C).fit(scores, labels)
    except ValueError as exc:
        return {
            "status": "failed",
            "reason": str(exc),
            "score_sha256": score_hash,
            "n_trials": int(labels.size),
            "n_genuine": int(np.sum(labels == 1)),
            "n_impostor": int(np.sum(labels == 0)),
        }
    payload = {
        "method": "class-balanced affine logistic LLR",
        "C": V1_CALIBRATION_C,
        "effective_target_prior": 0.5,
        "slope": float(calibrator.slope_),
        "intercept": float(calibrator.intercept_),
        "score_sha256": score_hash,
        "n_trials": int(labels.size),
        "n_genuine": int(np.sum(labels == 1)),
        "n_impostor": int(np.sum(labels == 0)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "status": "fitted",
        **payload,
        "calibrator_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _selected_candidate(family: str) -> dict[str, Any]:
    wanted = V1_SELECTED_MODELS[family]["candidate_id"]
    for candidate in V1_NEURAL_CANDIDATES[family]:
        if candidate["candidate_id"] == wanted:
            return candidate
    raise RuntimeError(f"selected candidate {wanted!r} is absent from the frozen training lock")


def _reconstruct_selected_models(
    fit: dict[str, Any],
    selection: dict[str, Any],
    *,
    embedding_dim: int,
) -> tuple[dict[str, torch.nn.Module], np.ndarray, np.ndarray]:
    score_mean, score_scale = zscore_fit(fit["scores"])
    fit_neural = dict(fit)
    selection_neural = dict(selection)
    fit_neural["scores"] = zscore_transform(fit["scores"], score_mean, score_scale).astype(np.float32)
    selection_neural["scores"] = zscore_transform(
        selection["scores"], score_mean, score_scale
    ).astype(np.float32)

    balanced_indices = _balanced_fit_indices(fit["labels"])
    selection_indices = np.arange(selection["labels"].size, dtype=np.int64)
    models: dict[str, torch.nn.Module] = {}

    for family in ("D1", "D2", "D3S"):
        candidate = _selected_candidate(family)
        model = _build_model(
            family,
            candidate,
            embedding_dim,
            seed=V1_REPORTING_SEED,
        )
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
        result = fit_binary_score_model(
            model,
            train_batches=train_batches,
            selection_batches=selection_batches,
            forward_batch=_forward_for_family(family),
            budget=V1_NEURAL_BUDGET,
            seed=V1_REPORTING_SEED,
            optimizer_config=V1_OPTIMIZER,
            firewall=FIREWALL,
            train_partition="fit",
            selection_partition="selection",
            device="cpu",
        )
        expected = V1_SELECTED_MODELS[family]["expected_checkpoint_hash"]
        if result.checkpoint_hash != expected:
            raise RuntimeError(
                f"{family} checkpoint reproduction failure before calibration access: "
                f"expected {expected}, got {result.checkpoint_hash}"
            )
        models[family] = model
    return models, score_mean, score_scale


def _fit_classical(fit: dict[str, Any], rows, embeddings):
    return {
        "C1": EqualScoreFusion(normalize=True).fit(fit["scores"]),
        "C2": WeightedScoreFusion(grid_step=0.05, objective="eer", normalize=True).fit(
            fit["scores"], fit["labels"]
        ),
        "C3": LogisticScoreFusion(C=1.0, normalize=True).fit(fit["scores"], fit["labels"]),
        "C4": _fit_feature_baseline(rows, embeddings),
        "C5": QualityWeightedScoreFusion(objective="eer", normalize=True).fit(
            fit["scores"], fit["quality"], fit["labels"]
        ),
    }


def _condition_evidence(
    *,
    condition,
    trials,
    clean_embeddings: dict[str, np.ndarray],
    clean_qualities: dict[str, float],
    paths: dict[str, Path],
    encoder: FrozenTorchvisionEncoder,
    quality_model,
) -> dict[str, Any]:
    if condition.kind is not StressKind.CORRUPTION:
        return _trial_arrays(trials, clean_embeddings, clean_qualities)

    embeddings = dict(clean_embeddings)
    qualities = dict(clean_qualities)
    target = condition.target_modalities[0]
    sample_field = (
        "probe_fingerprint_sample_id" if target == "fingerprint" else "probe_finger_vein_sample_id"
    )
    target_ids = sorted({trial[sample_field] for trial in trials})
    for sample_id in target_ids:
        with Image.open(paths[sample_id]) as source:
            corrupted = apply_v1_corruption(source, condition, target)
        embeddings[sample_id] = encoder.encode_pil(corrupted)
        qualities[sample_id] = quality_model.image_quality_pil(target, corrupted)
    return _trial_arrays(trials, embeddings, qualities)


def _complete_scores(
    data: dict[str, Any],
    *,
    classical: dict[str, Any],
    neural: dict[str, torch.nn.Module],
    score_mean: np.ndarray,
    score_scale: np.ndarray,
) -> dict[str, np.ndarray]:
    c4_left = classical["C4"].transform(data["enrollment"])
    c4_right = classical["C4"].transform(data["probe"])
    neural_data = dict(data)
    neural_data["scores"] = zscore_transform(data["scores"], score_mean, score_scale).astype(np.float32)
    return {
        "U-FP": data["scores"][:, 0],
        "U-FV": data["scores"][:, 1],
        "C1": classical["C1"].transform(data["scores"]),
        "C2": classical["C2"].transform(data["scores"]),
        "C3": classical["C3"].transform(data["scores"]),
        "C4": cosine_similarity_rows(c4_left, c4_right),
        "C5": classical["C5"].transform(data["scores"], data["quality"]),
        "D1": _selection_scores(neural["D1"], "D1", neural_data),
        "D2": _selection_scores(neural["D2"], "D2", neural_data),
        "D3S": _selection_scores(neural["D3S"], "D3S", neural_data),
    }


def _missing_scores(
    data: dict[str, Any],
    *,
    missing_modality: str,
    classical: dict[str, Any],
    neural: dict[str, torch.nn.Module],
    score_mean: np.ndarray,
    score_scale: np.ndarray,
) -> dict[str, np.ndarray | None]:
    missing_index = 0 if missing_modality == "fingerprint" else 1
    available_index = 1 - missing_index
    available_score = np.asarray(data["scores"][:, available_index], dtype=np.float64)
    out: dict[str, np.ndarray | None] = {
        "U-FP": None if missing_index == 0 else available_score,
        "U-FV": None if missing_index == 1 else available_score,
    }
    # Conservative M0 fallback for systems without native availability handling.
    for method in ("C1", "C2", "C3", "C4", "D1", "D2"):
        out[method] = available_score

    # C5: the one remaining quality weight is exactly one after availability
    # renormalization, so its frozen normalized available score is the M0 output.
    c5 = classical["C5"]
    normalized = zscore_transform(data["scores"], c5.mean_, c5.scale_)
    out["C5"] = normalized[:, available_index]

    neural_data = dict(data)
    scores = zscore_transform(data["scores"], score_mean, score_scale).astype(np.float32)
    availability = np.ones_like(data["availability"], dtype=bool)
    availability[:, missing_index] = False
    quality = np.asarray(data["quality"], dtype=np.float32).copy()
    quality[:, missing_index] = 0.0
    scores[:, missing_index] = 0.0
    neural_data["scores"] = scores
    neural_data["quality"] = quality
    neural_data["availability"] = availability
    out["D3S"] = _selection_scores(neural["D3S"], "D3S", neural_data)
    return out


def run(root: Path) -> dict[str, Any]:
    assert_v1_training_lock()
    assert_v1_selection_lock()
    assert_v1_final_policy_lock()
    if v1_stress_plan_hash() != V1_STRESS_PLAN_SHA256:
        raise RuntimeError("V1 stress plan differs from the frozen final policy")

    rows = scan_nupt_fpv(root)
    manifest_hash = dataset_manifest_hash(rows)
    if manifest_hash != V1_DATASET_MANIFEST_SHA256:
        raise RuntimeError("dataset manifest differs from the selection lock")

    fit_trials = generate_v1_trials(rows, "fit")
    selection_trials = generate_v1_trials(rows, "selection")
    dev_needed = _needed_sample_ids(fit_trials) | _needed_sample_ids(selection_trials)
    paths = _paths_by_sample(rows, root)

    encoder = FrozenTorchvisionEncoder(V1_PRIMARY_ENCODER)
    if encoder.weight_state_hash != V1_ENCODER_WEIGHT_STATE_HASH:
        raise RuntimeError("frozen encoder weight-state hash changed")
    dev_embeddings = {
        sample_id: encoder.encode_image(paths[sample_id]) for sample_id in sorted(dev_needed)
    }
    quality_model = _fit_quality_model(rows, root)
    if quality_model.model_hash != V1_QUALITY_MODEL_HASH:
        raise RuntimeError("V1 quality model changed")
    dev_qualities = _precompute_quality(rows, root, dev_needed, quality_model)

    fit = _trial_arrays(fit_trials, dev_embeddings, dev_qualities)
    selection = _trial_arrays(selection_trials, dev_embeddings, dev_qualities)
    classical = _fit_classical(fit, rows, dev_embeddings)
    neural, score_mean, score_scale = _reconstruct_selected_models(
        fit, selection, embedding_dim=encoder.spec.embedding_dim
    )

    # Hard firewall: calibration samples are referenced/opened only after every
    # selected neural checkpoint has been reconstructed and hash-verified above.
    calibration_trials_clean = generate_v1_trials(rows, "calibration")
    calibration_hash = trial_manifest_hash(calibration_trials_clean)
    if calibration_hash != V1_CALIBRATION_TRIAL_SHA256:
        raise RuntimeError("calibration trial manifest changed")
    calibration_needed = _needed_sample_ids(calibration_trials_clean)
    calibration_only = calibration_needed - dev_needed
    clean_embeddings = dict(dev_embeddings)
    clean_qualities = dict(dev_qualities)
    for sample_id in sorted(calibration_only):
        clean_embeddings[sample_id] = encoder.encode_image(paths[sample_id])
        row = next(row for row in rows if row["sample_id"] == sample_id)
        clean_qualities[sample_id] = quality_model.image_quality(
            row["modality"], paths[sample_id]
        )

    conditions: dict[str, Any] = {}
    for condition in v1_stress_plan():
        trials = generate_v1_trials(rows, "calibration", condition_id=condition.condition_id)
        data = _condition_evidence(
            condition=condition,
            trials=trials,
            clean_embeddings=clean_embeddings,
            clean_qualities=clean_qualities,
            paths=paths,
            encoder=encoder,
            quality_model=quality_model,
        )
        if condition.kind is StressKind.MISSING:
            raw_scores = _missing_scores(
                data,
                missing_modality=condition.target_modalities[0],
                classical=classical,
                neural=neural,
                score_mean=score_mean,
                score_scale=score_scale,
            )
        else:
            raw_scores = _complete_scores(
                data,
                classical=classical,
                neural=neural,
                score_mean=score_mean,
                score_scale=score_scale,
            )
        records: dict[str, Any] = {}
        for system_id in SYSTEM_IDS:
            scores = raw_scores[system_id]
            if scores is None:
                records[system_id] = {
                    "status": "unavailable",
                    "reason": f"{system_id} requires the missing modality",
                }
            else:
                records[system_id] = _calibrator_record(
                    np.asarray(scores, dtype=np.float64), data["labels"]
                )
        conditions[condition.condition_id] = {
            "condition": condition.as_dict(),
            "trial_manifest_sha256": trial_manifest_hash(trials),
            "calibrators": records,
        }

    clean_records = conditions["clean"]["calibrators"]
    failed_clean = [key for key, value in clean_records.items() if value["status"] != "fitted"]
    if failed_clean:
        raise RuntimeError(f"primary clean calibration failed for {failed_clean}")

    payload = {
        "scope": "V1 held-out calibration only; final role untouched",
        "final_images_read": False,
        "development_checkpoints_verified_before_calibration": True,
        "dataset_manifest_sha256": manifest_hash,
        "calibration_trial_manifest_sha256": calibration_hash,
        "training_final_policy_sha256": V1_FINAL_POLICY_SHA256,
        "stress_plan_sha256": v1_stress_plan_hash(),
        "encoder_weight_state_hash": encoder.weight_state_hash,
        "quality_model_hash": quality_model.model_hash,
        "reporting_seed": V1_REPORTING_SEED,
        "selected_checkpoint_hashes": {
            family: V1_SELECTED_MODELS[family]["expected_checkpoint_hash"]
            for family in ("D1", "D2", "D3S")
        },
        "calibration_C": V1_CALIBRATION_C,
        "conditions": conditions,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["calibration_evidence_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


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
