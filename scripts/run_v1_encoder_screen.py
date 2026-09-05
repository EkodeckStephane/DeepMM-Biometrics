#!/usr/bin/env python3
"""Development-only V1 frozen-encoder screening on the public NUPT-FPV subset.

This script intentionally reads only V1 `fit` and `selection` image roles. It does
not read calibration or final images. Its purpose is to verify that directly
accessible frozen representations provide usable unimodal evidence before the
remaining confirmatory configuration is frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from deepmm.datasets import build_v1_evidence_units, generate_v1_trials, scan_nupt_fpv
from deepmm.fusion import EqualScoreFusion, LogisticScoreFusion, WeightedScoreFusion
from deepmm.fusion.features import StandardizedConcatFusion, cosine_similarity_rows
from deepmm.metrics import eer, roc_auc
from deepmm.vision import FrozenTorchvisionEncoder


DEFAULT_ENCODERS = (
    "resnet18_imagenet1k_v1",
    "mobilenet_v3_small_imagenet1k_v1",
)


def _metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    value, _ = eer(labels, scores)
    return {"eer": float(value), "auc": float(roc_auc(labels, scores))}


def _raw_sample_paths(rows: list[dict[str, Any]], root: Path) -> dict[str, Path]:
    return {row["sample_id"]: root / row["relative_path"] for row in rows}


def _needed_sample_ids(trials: list[dict[str, Any]]) -> set[str]:
    keys = (
        "enrollment_fingerprint_sample_id",
        "enrollment_finger_vein_sample_id",
        "probe_fingerprint_sample_id",
        "probe_finger_vein_sample_id",
    )
    return {trial[key] for trial in trials for key in keys}


def _trial_embeddings(
    trials: list[dict[str, Any]],
    embeddings: dict[str, np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    enroll_fp = np.stack([embeddings[t["enrollment_fingerprint_sample_id"]] for t in trials])
    enroll_fv = np.stack([embeddings[t["enrollment_finger_vein_sample_id"]] for t in trials])
    probe_fp = np.stack([embeddings[t["probe_fingerprint_sample_id"]] for t in trials])
    probe_fv = np.stack([embeddings[t["probe_finger_vein_sample_id"]] for t in trials])
    labels = np.asarray([t["label"] for t in trials], dtype=np.int8)
    score_matrix = np.column_stack(
        [
            cosine_similarity_rows(enroll_fp, probe_fp),
            cosine_similarity_rows(enroll_fv, probe_fv),
        ]
    )
    return [enroll_fp, enroll_fv], [probe_fp, probe_fv], score_matrix, labels


def _fit_feature_baseline(
    rows: list[dict[str, Any]],
    embeddings: dict[str, np.ndarray],
) -> StandardizedConcatFusion:
    units = build_v1_evidence_units(rows)
    fit_keys = [
        key
        for key in sorted(units)
        if key[1] == "1" and key[2] in {"01", "02", "03", "04", "05"}
    ]
    if len(fit_keys) != 100:
        raise ValueError(f"expected 100 fit evidence units, found {len(fit_keys)}")
    fp = np.stack([embeddings[units[key]["fingerprint_sample_id"]] for key in fit_keys])
    fv = np.stack([embeddings[units[key]["finger_vein_sample_id"]] for key in fit_keys])
    return StandardizedConcatFusion().fit([fp, fv])


def _score_feature_baseline(
    baseline: StandardizedConcatFusion,
    enrollment: list[np.ndarray],
    probe: list[np.ndarray],
) -> np.ndarray:
    left = baseline.transform(enrollment)
    right = baseline.transform(probe)
    return cosine_similarity_rows(left, right)


def run(root: Path, encoder_names: tuple[str, ...]) -> dict[str, Any]:
    rows = scan_nupt_fpv(root)
    fit_trials = generate_v1_trials(rows, "fit")
    selection_trials = generate_v1_trials(rows, "selection")
    # This set is deliberately limited to captures 01-07 of session 1 by the
    # frozen role contract. Calibration/final samples are not opened here.
    needed = _needed_sample_ids(fit_trials) | _needed_sample_ids(selection_trials)
    paths = _raw_sample_paths(rows, root)
    if not needed.issubset(paths):
        raise ValueError("trial manifest refers to missing raw samples")

    result: dict[str, Any] = {
        "scope": "development_only_fit_plus_selection",
        "calibration_images_read": False,
        "final_images_read": False,
        "n_unique_images_read": len(needed),
        "encoders": {},
    }

    for name in encoder_names:
        encoder = FrozenTorchvisionEncoder(name)
        embeddings = {
            sample_id: encoder.encode_image(paths[sample_id])
            for sample_id in sorted(needed)
        }
        fit_enroll, fit_probe, fit_scores, fit_labels = _trial_embeddings(fit_trials, embeddings)
        sel_enroll, sel_probe, sel_scores, sel_labels = _trial_embeddings(selection_trials, embeddings)

        c1 = EqualScoreFusion(normalize=True).fit(fit_scores)
        c2 = WeightedScoreFusion(grid_step=0.05, objective="eer", normalize=True).fit(
            fit_scores, fit_labels
        )
        c3 = LogisticScoreFusion(C=1.0, normalize=True).fit(fit_scores, fit_labels)
        c4 = _fit_feature_baseline(rows, embeddings)

        methods = {
            "U-FP": sel_scores[:, 0],
            "U-FV": sel_scores[:, 1],
            "C1-equal-score": c1.transform(sel_scores),
            "C2-weighted-score": c2.transform(sel_scores),
            "C3-logistic-score": c3.transform(sel_scores),
            "C4-standardized-concat": _score_feature_baseline(c4, sel_enroll, sel_probe),
        }
        result["encoders"][name] = {
            "spec": encoder.spec.as_dict(),
            "spec_hash": encoder.spec.spec_hash,
            "weight_state_hash": encoder.weight_state_hash,
            "fit_trials": len(fit_trials),
            "selection_trials": len(selection_trials),
            "c2_weights_fingerprint_finger_vein": c2.weights_.tolist(),
            "selection_metrics": {key: _metric(sel_labels, scores) for key, scores in methods.items()},
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Root of official NUPT-FPV checkout")
    parser.add_argument(
        "--encoders",
        default=",".join(DEFAULT_ENCODERS),
        help="Comma-separated frozen encoder IDs",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    encoders = tuple(value.strip() for value in args.encoders.split(",") if value.strip())
    if not encoders:
        parser.error("at least one encoder is required")

    result = run(args.root, encoders)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
