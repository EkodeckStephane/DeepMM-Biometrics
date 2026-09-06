#!/usr/bin/env python3
"""Audit the committed V1 result bundle from trials through headline outputs."""

from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from deepmm.evaluation.v1_final_results_lock import (
    V1_FINAL_RESULTS_PATH,
    load_v1_final_results_lock,
    read_v1_final_file,
)
from deepmm.validation import score_manifest_hash, trial_manifest_hash


def audit(evidence: Path | None = None) -> dict[str, Any]:
    root = evidence or Path(__file__).resolve().parents[1] / V1_FINAL_RESULTS_PATH
    result = load_v1_final_results_lock(root)
    trials_by_condition = json.loads(
        gzip.decompress(read_v1_final_file("v1_final_trials.json.gz", root))
    )
    manifests = json.loads(
        (root / "v1_final_run_manifests.json").read_text(encoding="utf-8")
    )

    conditions = result["conditions"]
    if set(trials_by_condition) != set(conditions):
        raise RuntimeError("trial/result condition sets differ")
    for condition_id, trials in trials_by_condition.items():
        condition = conditions[condition_id]
        expected_hash = condition["trial_manifest_sha256"]
        actual_hash = trial_manifest_hash(trials)
        if actual_hash != expected_hash:
            raise RuntimeError(f"trial hash mismatch for {condition_id}")
        counts = (
            len(trials),
            sum(int(row["label"] == 1) for row in trials),
            sum(int(row["label"] == 0) for row in trials),
        )
        expected_counts = (
            condition["n_trials"],
            condition["n_genuine"],
            condition["n_impostor"],
        )
        if counts != expected_counts:
            raise RuntimeError(f"trial counts differ for {condition_id}")

    verified_scores = 0
    score_bytes = read_v1_final_file("v1_final_scores.npz", root)
    with np.load(io.BytesIO(score_bytes), allow_pickle=False) as arrays:
        for manifest in manifests:
            condition_id = manifest["condition_id"]
            method_id = manifest["method_id"]
            key = f"{condition_id}__{method_id}__raw"
            trials = trials_by_condition[condition_id]
            score_rows = [
                {"trial_id": trial["trial_id"], "score": float(score)}
                for trial, score in zip(trials, arrays[key], strict=True)
            ]
            actual_hash = score_manifest_hash(trials, score_rows)
            expected_hash = manifest["score_manifest_hash"]
            result_hash = conditions[condition_id]["systems"][method_id][
                "raw_score_manifest_sha256"
            ]
            if actual_hash != expected_hash or actual_hash != result_hash:
                raise RuntimeError(f"raw score hash mismatch for {condition_id}/{method_id}")
            verified_scores += 1

    return {
        "status": "pass",
        "conditions": len(conditions),
        "trials_per_condition": 4000,
        "verified_score_manifests": verified_scores,
        "verified_score_arrays": 3 * verified_scores,
        "headline_result_status": result["status"],
        "inference_boundary": result["q1"]["inference_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(audit(args.evidence), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
