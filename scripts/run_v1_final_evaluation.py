#!/usr/bin/env python3
"""Execute the locked 4,000-trial V1 Q1--Q3 result campaign.

The final role is not even generated until the training, selection, calibration,
stress, policy and runner-integrity locks have all passed. No fit operation occurs
after final-role samples are referenced or opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any, Callable

import numpy as np
import sklearn
import torch

from deepmm.calibration import load_v1_calibration_lock, posterior_probability_from_llr
from deepmm.datasets import generate_v1_trials, scan_nupt_fpv
from deepmm.evaluation.cost import MeasurementContext, measure_latency
from deepmm.evaluation.v1_final_config import (
    V1_COST_POLICY,
    V1_ECE_BINS,
    V1_FAR_GRID,
    V1_FINAL_POLICY_SHA256,
    V1_FINAL_TRIAL_MANIFEST_SHA256,
    V1_PROBABILITY_REFERENCE_PRIOR,
    V1_STRESS_PLAN_SHA256,
    assert_v1_final_policy_lock,
)
from deepmm.evaluation.v1_final_execution_lock import assert_v1_final_script_lock
from deepmm.fusion.features import cosine_similarity_rows
from deepmm.fusion.neural_torch import parameter_count
from deepmm.metrics import (
    brier_score,
    cllr,
    cllr_calibration_loss,
    eer,
    eer_rocch,
    expected_calibration_error,
    min_cllr,
    negative_log_likelihood,
    roc_auc,
    tar_at_far,
)
from deepmm.robustness import StressKind, v1_stress_plan, v1_stress_plan_hash
from deepmm.stats import kendall_tau_b, non_dominated_mask, pairwise_rank_reversals
from deepmm.training.v1_public_config import V1_PRIMARY_ENCODER, V1_REPORTING_SEED, assert_v1_training_lock
from deepmm.training.v1_selection_lock import (
    V1_DATASET_MANIFEST_SHA256,
    V1_ENCODER_WEIGHT_STATE_HASH,
    V1_QUALITY_MODEL_HASH,
    V1_SELECTED_MODELS,
    V1_SELECTION_LOCK_SHA256,
    assert_v1_selection_lock,
)
from deepmm.validation import (
    dataset_manifest_hash,
    run_manifest_hash,
    score_manifest_hash,
    sha256_text,
    trial_manifest_hash,
    validate_run_manifest,
)
from deepmm.vision import FrozenTorchvisionEncoder

# These are the exact, already exercised fit/selection/calibration primitives.
from run_v1_calibration import (  # noqa: E402
    SYSTEM_IDS,
    _complete_scores,
    _condition_evidence,
    _fit_classical,
    _missing_scores,
    _reconstruct_selected_models,
)
from run_v1_development_training import (  # noqa: E402
    _fit_quality_model,
    _needed_sample_ids,
    _paths_by_sample,
    _precompute_quality,
    _trial_arrays,
)


FUSION_SYSTEM_IDS = ("C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3S")
FAMILY = {
    "U-FP": "unimodal_fingerprint",
    "U-FV": "unimodal_finger_vein",
    "C1": "classical_equal_score",
    "C2": "classical_weighted_score",
    "C3": "classical_logistic_score",
    "C4": "classical_feature",
    "C5": "classical_quality_score",
    "D1": "deep_score",
    "D2": "deep_feature",
    "D3S": "deep_quality_gate",
}


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _split_hash() -> str:
    # The exact final-role construction is already bound by the frozen trial hash;
    # this additional manifest field names its split semantics explicitly.
    return _canonical_hash(
        {
            "role": "final",
            "identity_scope": "public_biometric_instance",
            "enrollment": {"session": "1", "captures": ["10"]},
            "probe": {"session": "2", "captures": [f"{i:02d}" for i in range(1, 11)]},
        }
    )


def _llr(scores: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    if record.get("status") != "fitted":
        raise RuntimeError("attempted to use an unavailable calibration mapping")
    values = float(record["slope"]) * np.asarray(scores, dtype=np.float64) + float(record["intercept"])
    if not np.all(np.isfinite(values)):
        raise RuntimeError("calibration mapping produced non-finite LLRs")
    return values


def _metrics(labels: np.ndarray, raw: np.ndarray, llrs: np.ndarray) -> dict[str, Any]:
    empirical, threshold = eer(labels, raw)
    probability = posterior_probability_from_llr(
        llrs, target_prior=V1_PROBABILITY_REFERENCE_PRIOR
    )
    non = llrs[labels == 0]
    target = llrs[labels == 1]
    result: dict[str, Any] = {
        "rocch_eer": eer_rocch(labels, raw),
        "empirical_eer": empirical,
        "empirical_eer_threshold": threshold if np.isfinite(threshold) else None,
        "auc": roc_auc(labels, raw),
        "cllr": cllr(non, target),
        "min_cllr": min_cllr(non, target),
        "cllr_cal": cllr_calibration_loss(non, target),
        "brier": brier_score(labels, probability),
        "nll": negative_log_likelihood(labels, probability),
        "ece": expected_calibration_error(labels, probability, n_bins=V1_ECE_BINS),
    }
    result["tar_at_far"] = {
        str(target_far): {
            "tar": values[0],
            "achieved_far": values[1],
            "threshold": values[2] if np.isfinite(values[2]) else None,
        }
        for target_far in V1_FAR_GRID
        for values in [tar_at_far(labels, raw, target_far)]
    }
    return result


def _hardware_id() -> str:
    cpu = platform.processor().strip()
    if not cpu and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return f"{platform.system()}-{platform.machine()}-{cpu or 'unknown-cpu'}"


def _cost_callables(
    data: dict[str, Any], classical: dict[str, Any], neural: dict[str, torch.nn.Module]
) -> dict[str, Callable[[], object]]:
    n = V1_COST_POLICY["batch_size"]
    scores = np.asarray(data["scores"][:n], dtype=np.float64)
    quality = np.asarray(data["quality"][:n], dtype=np.float32)
    enrollment = [np.asarray(block[:n], dtype=np.float32) for block in data["enrollment"]]
    probe = [np.asarray(block[:n], dtype=np.float32) for block in data["probe"]]
    c4_left = [torch.as_tensor(block) for block in enrollment]
    c4_right = [torch.as_tensor(block) for block in probe]
    score_tensor = torch.as_tensor(scores, dtype=torch.float32)
    quality_tensor = torch.as_tensor(quality, dtype=torch.float32)
    availability_tensor = torch.ones_like(score_tensor, dtype=torch.bool)

    for model in neural.values():
        model.eval()

    return {
        "U-FP": lambda: scores[:, 0].copy(),
        "U-FV": lambda: scores[:, 1].copy(),
        "C1": lambda: classical["C1"].transform(scores),
        "C2": lambda: classical["C2"].transform(scores),
        "C3": lambda: classical["C3"].transform(scores),
        "C4": lambda: cosine_similarity_rows(
            classical["C4"].transform(enrollment), classical["C4"].transform(probe)
        ),
        "C5": lambda: classical["C5"].transform(scores, quality),
        "D1": lambda: neural["D1"](score_tensor),
        "D2": lambda: neural["D2"](c4_left, c4_right),
        "D3S": lambda: neural["D3S"](score_tensor, quality_tensor, availability_tensor),
    }


def _parameter_counts(classical: dict[str, Any], neural: dict[str, torch.nn.Module]) -> dict[str, Any]:
    dim = int(sum(classical["C4"].dims_))
    return {
        "U-FP": {"trainable": 0, "total": 0},
        "U-FV": {"trainable": 0, "total": 0},
        "C1": {"trainable": 0, "total": 4},
        "C2": {"trainable": 2, "total": 6},
        "C3": {"trainable": 3, "total": 7},
        "C4": {"trainable": 0, "total": 2 * dim},
        "C5": {"trainable": 1, "total": 5},
        **{
            key: {"trainable": parameter_count(model), "total": sum(p.numel() for p in model.parameters())}
            for key, model in neural.items()
        },
    }


def _measure_costs(data, classical, neural) -> dict[str, Any]:
    torch.set_num_threads(int(V1_COST_POLICY["num_threads"]))
    context = MeasurementContext(
        hardware_id=_hardware_id(),
        device=str(V1_COST_POLICY["device"]),
        batch_size=int(V1_COST_POLICY["batch_size"]),
        precision=str(V1_COST_POLICY["precision"]),
        scope=str(V1_COST_POLICY["primary_scope"]),
        num_threads=int(V1_COST_POLICY["num_threads"]),
    )
    counts = _parameter_counts(classical, neural)
    records: dict[str, Any] = {}
    with torch.no_grad():
        for system_id, fn in _cost_callables(data, classical, neural).items():
            summary, samples = measure_latency(
                fn,
                warmup=int(V1_COST_POLICY["warmup"]),
                repeats=int(V1_COST_POLICY["repeats"]),
            )
            records[system_id] = {
                "context": context.__dict__,
                "latency": {**summary.__dict__, "iqr_ms": summary.iqr_ms},
                "raw_latency_ms": samples.tolist(),
                "trainable_params": counts[system_id]["trainable"],
                "total_params": counts[system_id]["total"],
                "parameter_accounting": "stored fitted scalars for classical; exact torch parameters for neural",
            }
    return records


def _preflight(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Ordering is deliberate: every immutable lock precedes any final-role reference.
    assert_v1_training_lock()
    assert_v1_selection_lock()
    assert_v1_final_policy_lock()
    assert_v1_final_script_lock()
    calibration = load_v1_calibration_lock()
    if v1_stress_plan_hash() != V1_STRESS_PLAN_SHA256:
        raise RuntimeError("V1 stress plan differs from the frozen final policy")
    rows = scan_nupt_fpv(root)
    if dataset_manifest_hash(rows) != V1_DATASET_MANIFEST_SHA256:
        raise RuntimeError("dataset manifest differs from the frozen V1 selection lock")
    return rows, calibration


def run(root: Path, *, output_dir: Path, code_commit: str, preflight_only: bool = False) -> dict[str, Any]:
    rows, calibration = _preflight(root)
    if preflight_only:
        return {
            "status": "ready",
            "final_images_read": False,
            "dataset_manifest_sha256": dataset_manifest_hash(rows),
            "selection_lock_sha256": V1_SELECTION_LOCK_SHA256,
            "calibration_evidence_sha256": calibration["calibration_evidence_sha256"],
            "final_policy_sha256": V1_FINAL_POLICY_SHA256,
        }

    if len(code_commit) != 40 or any(c not in "0123456789abcdef" for c in code_commit):
        raise ValueError("code_commit must be a full lowercase 40-character Git SHA")

    # Reconstruct every fitted object from fit/selection only, then verify the
    # selected neural checkpoint hashes before the final role is generated.
    fit_trials = generate_v1_trials(rows, "fit")
    selection_trials = generate_v1_trials(rows, "selection")
    dev_needed = _needed_sample_ids(fit_trials) | _needed_sample_ids(selection_trials)
    paths = _paths_by_sample(rows, root)
    encoder = FrozenTorchvisionEncoder(V1_PRIMARY_ENCODER)
    if encoder.weight_state_hash != V1_ENCODER_WEIGHT_STATE_HASH:
        raise RuntimeError("frozen encoder weight-state hash changed")
    dev_embeddings = {sample_id: encoder.encode_image(paths[sample_id]) for sample_id in sorted(dev_needed)}
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

    # Final firewall opens here, after every training/calibration/configuration lock.
    final_trials_clean = generate_v1_trials(rows, "final")
    clean_hash = trial_manifest_hash(final_trials_clean)
    if clean_hash != V1_FINAL_TRIAL_MANIFEST_SHA256:
        raise RuntimeError("final trial manifest changed")
    labels_clean = np.asarray([row["label"] for row in final_trials_clean], dtype=np.int8)
    if (len(final_trials_clean), int(labels_clean.sum()), int(np.sum(labels_clean == 0))) != (4000, 200, 3800):
        raise RuntimeError("final trial counts differ from the frozen 4000/200/3800 design")

    final_needed = _needed_sample_ids(final_trials_clean)
    clean_embeddings = dict(dev_embeddings)
    clean_qualities = dict(dev_qualities)
    row_by_sample = {row["sample_id"]: row for row in rows}
    for sample_id in sorted(final_needed - dev_needed):
        clean_embeddings[sample_id] = encoder.encode_image(paths[sample_id])
        row = row_by_sample[sample_id]
        clean_qualities[sample_id] = quality_model.image_quality(row["modality"], paths[sample_id])

    output_dir.mkdir(parents=True, exist_ok=True)
    score_arrays: dict[str, np.ndarray] = {}
    trial_records: dict[str, list[dict[str, Any]]] = {}
    condition_results: dict[str, Any] = {}
    run_manifests: list[dict[str, Any]] = []
    clean_data = None

    for condition in v1_stress_plan():
        condition_id = condition.condition_id
        trials = generate_v1_trials(rows, "final", condition_id=condition_id)
        trial_records[condition_id] = trials
        data = _condition_evidence(
            condition=condition,
            trials=trials,
            clean_embeddings=clean_embeddings,
            clean_qualities=clean_qualities,
            paths=paths,
            encoder=encoder,
            quality_model=quality_model,
        )
        if condition_id == "clean":
            clean_data = data
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

        systems: dict[str, Any] = {}
        for system_id in SYSTEM_IDS:
            raw = raw_scores[system_id]
            if raw is None:
                systems[system_id] = {"status": "unavailable"}
                continue
            raw = np.asarray(raw, dtype=np.float64)
            primary = _llr(raw, calibration["conditions"]["clean"]["calibrators"][system_id])
            secondary = _llr(raw, calibration["conditions"][condition_id]["calibrators"][system_id])
            score_arrays[f"{condition_id}__{system_id}__raw"] = raw
            score_arrays[f"{condition_id}__{system_id}__primary_llr"] = primary
            score_arrays[f"{condition_id}__{system_id}__secondary_llr"] = secondary

            score_rows = [
                {"trial_id": trial["trial_id"], "score": float(score)}
                for trial, score in zip(trials, raw)
            ]
            raw_hash = score_manifest_hash(trials, score_rows)
            config_hash = _canonical_hash(
                {
                    "final_policy": V1_FINAL_POLICY_SHA256,
                    "selection_lock": V1_SELECTION_LOCK_SHA256,
                    "calibration_evidence": calibration["calibration_evidence_sha256"],
                    "condition": condition.as_dict(),
                    "system_id": system_id,
                }
            )
            manifest = {
                "run_id": f"v1-final-{code_commit}-{condition_id}-{system_id}",
                "method_id": system_id,
                "family": FAMILY[system_id],
                "seed": V1_REPORTING_SEED,
                "condition_id": condition_id,
                "code_commit": code_commit,
                "split_hash": _split_hash(),
                "trial_manifest_hash": trial_manifest_hash(trials),
                "config_hash": config_hash,
                "score_manifest_hash": raw_hash,
                "dataset_manifest_hash": V1_DATASET_MANIFEST_SHA256,
            }
            if system_id in V1_SELECTED_MODELS:
                manifest["checkpoint_hash"] = V1_SELECTED_MODELS[system_id]["expected_checkpoint_hash"]
            manifest = validate_run_manifest(manifest)
            run_manifests.append({**manifest, "run_manifest_hash": run_manifest_hash(manifest)})
            systems[system_id] = {
                "status": "complete",
                "raw_score_manifest_sha256": raw_hash,
                "primary_clean_calibrator": _metrics(data["labels"], raw, primary),
                "secondary_condition_calibrator": _metrics(data["labels"], raw, secondary),
            }
        condition_results[condition_id] = {
            "condition": condition.as_dict(),
            "trial_manifest_sha256": trial_manifest_hash(trials),
            "n_trials": len(trials),
            "n_genuine": int(np.sum(data["labels"] == 1)),
            "n_impostor": int(np.sum(data["labels"] == 0)),
            "systems": systems,
        }

    if clean_data is None:
        raise AssertionError("frozen stress plan omitted clean")
    costs = _measure_costs(clean_data, classical, neural)
    clean_metric = {
        system: condition_results["clean"]["systems"][system]["primary_clean_calibrator"]["rocch_eer"]
        for system in SYSTEM_IDS
    }
    best_unimodal = min(("U-FP", "U-FV"), key=clean_metric.get)
    best_classical = min(("C1", "C2", "C3", "C4", "C5"), key=clean_metric.get)
    q1_pairs = [
        ("C1", best_unimodal, "equal fusion vs best unimodal"),
        (best_classical, best_unimodal, "best classical vs best unimodal"),
        ("D1", min(("C1", "C2", "C3"), key=clean_metric.get), "deep vs best classical score fusion"),
        ("D2", "C4", "deep vs classical feature fusion"),
        ("D3S", "C5", "deep vs classical quality-aware fusion"),
        *((deep, best_classical, "headline DL family vs best classical") for deep in ("D1", "D2", "D3S")),
    ]
    q1 = {
        "metric": "clean rocch_eer; negative delta favors first system",
        "best_unimodal": best_unimodal,
        "best_classical": best_classical,
        "contrasts": [
            {"system_a": a, "system_b": b, "contrast": name, "delta": clean_metric[a] - clean_metric[b]}
            for a, b, name in q1_pairs
        ],
        "inference_boundary": "public biometric-instance point estimates; no person-population p-values or CIs",
    }

    q2_values = []
    q2_rows = []
    for system_id in FUSION_SYSTEM_IDS:
        stressed = [
            condition_results[c]["systems"][system_id]["primary_clean_calibrator"]["rocch_eer"]
            for c in condition_results
            if c != "clean"
        ]
        robustness_loss = float(np.mean(np.asarray(stressed) - clean_metric[system_id]))
        calibration_loss = condition_results["clean"]["systems"][system_id]["primary_clean_calibrator"]["cllr"]
        latency = costs[system_id]["latency"]["median_ms"]
        values = [clean_metric[system_id], robustness_loss, calibration_loss, latency]
        q2_values.append(values)
        q2_rows.append({
            "system_id": system_id,
            "clean_rocch_eer": values[0],
            "mean_stress_rocch_eer_loss": values[1],
            "clean_cllr": values[2],
            "fusion_latency_median_ms": values[3],
        })
    frontier = non_dominated_mask(q2_values, minimize=[True, True, True, True])
    q2 = {
        "criteria": ["clean_rocch_eer", "mean_stress_rocch_eer_loss", "clean_cllr", "fusion_latency_median_ms"],
        "directions": ["minimize"] * 4,
        "rows": [{**row, "pareto_non_dominated": bool(keep)} for row, keep in zip(q2_rows, frontier)],
        "rule": "point Pareto frontier; no post-hoc weighted composite",
    }

    clean_rank = np.asarray([clean_metric[s] for s in FUSION_SYSTEM_IDS], dtype=np.float64)
    q3_conditions: dict[str, Any] = {}
    for condition_id in condition_results:
        values = np.asarray([
            condition_results[condition_id]["systems"][s]["primary_clean_calibrator"]["rocch_eer"]
            for s in FUSION_SYSTEM_IDS
        ])
        reversals = pairwise_rank_reversals(clean_rank, values)
        tau = kendall_tau_b(clean_rank, values)
        q3_conditions[condition_id] = {
            # Tau-b is mathematically undefined when every compared method is tied.
            # Preserve that fact as JSON null rather than emitting a non-standard NaN.
            "kendall_tau_b_vs_clean": float(tau) if np.isfinite(tau) else None,
            "pairwise_rank_reversals": [
                [FUSION_SYSTEM_IDS[i], FUSION_SYSTEM_IDS[j]] for i, j in reversals
            ],
            "ranking": sorted(
                ({"system_id": s, "rocch_eer": float(v)} for s, v in zip(FUSION_SYSTEM_IDS, values)),
                key=lambda row: (row["rocch_eer"], row["system_id"]),
            ),
        }

    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "github_runner_name": os.environ.get("RUNNER_NAME"),
    }
    result = {
        "campaign": "V1 public NUPT-FPV Q1-Q3 final",
        "status": "complete",
        "final_images_read": True,
        "scope_boundary": "20 public biometric-instance identities; not independent humans; not full 33,600-image NUPT-FPV",
        "code_commit": code_commit,
        "dataset_manifest_sha256": V1_DATASET_MANIFEST_SHA256,
        "selection_lock_sha256": V1_SELECTION_LOCK_SHA256,
        "calibration_evidence_sha256": calibration["calibration_evidence_sha256"],
        "final_policy_sha256": V1_FINAL_POLICY_SHA256,
        "final_clean_trial_manifest_sha256": clean_hash,
        "environment": environment,
        "environment_sha256": _canonical_hash(environment),
        "conditions": condition_results,
        "cost": costs,
        "q1": q1,
        "q2": q2,
        "q3": {"systems": list(FUSION_SYSTEM_IDS), "conditions": q3_conditions},
    }

    np.savez_compressed(output_dir / "v1_final_scores.npz", **score_arrays)
    (output_dir / "v1_final_trials.json").write_text(
        json.dumps(trial_records, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    (output_dir / "v1_final_run_manifests.json").write_text(
        json.dumps(run_manifests, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "v1_final_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    bundle = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    (output_dir / "SHA256SUMS.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Root of official NUPT-FPV checkout")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v1-final"))
    parser.add_argument("--code-commit", default=os.environ.get("GITHUB_SHA", ""))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        args.root,
        output_dir=args.output_dir,
        code_commit=args.code_commit,
        preflight_only=args.preflight_only,
    )
    if args.preflight_only:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps({
            "campaign": result["campaign"],
            "status": result["status"],
            "code_commit": result["code_commit"],
            "final_clean_trial_manifest_sha256": result["final_clean_trial_manifest_sha256"],
            "output_dir": str(args.output_dir),
        }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
