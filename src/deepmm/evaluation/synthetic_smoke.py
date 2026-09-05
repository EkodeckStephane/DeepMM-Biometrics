"""End-to-end synthetic smoke harness for the DeepMM evidence pipeline.

The generated values are **CI/debug data only**. They must never be reported as
scientific biometric results. The goal is to exercise development-only fitting,
held-out calibration transfer, clustered uncertainty, Pareto analysis, and ranking
stability before any real dataset is selected.
"""

from __future__ import annotations

import numpy as np

from deepmm.calibration import LogisticLLRCalibrator
from deepmm.fusion import (
    EqualScoreFusion,
    LogisticScoreFusion,
    QualityWeightedScoreFusion,
    WeightedScoreFusion,
)
from deepmm.metrics import cllr, eer, roc_auc
from deepmm.stats import (
    cluster_bootstrap_metric,
    kendall_tau_b,
    non_dominated_mask,
    pairwise_rank_reversals,
)


def _block(seed: int, n_subjects: int, *, stress: bool = False):
    rng = np.random.default_rng(seed)
    labels = np.tile(np.array([1, 1, 0, 0], dtype=np.int8), n_subjects)
    clusters = np.repeat(np.arange(n_subjects), 4)
    sign = 2.0 * labels.astype(np.float64) - 1.0

    subject_shift = rng.normal(0.0, 0.15, size=n_subjects)
    shift = np.repeat(subject_shift, 4)

    # Modality A is strong in clean conditions but is deliberately degraded in the
    # stress block. Modality B is weaker but comparatively stable. This is a
    # pipeline test, not a claim about any real biometric modality.
    a_signal = 1.45 if not stress else 0.35
    a_noise = 0.75 if not stress else 1.35
    b_signal = 0.95
    b_noise = 0.85

    score_a = a_signal * sign + shift + rng.normal(0.0, a_noise, labels.size)
    score_b = b_signal * sign - 0.25 * shift + rng.normal(0.0, b_noise, labels.size)
    scores = np.column_stack([score_a, score_b])

    q_a_center = 0.88 if not stress else 0.30
    quality_a = np.clip(rng.normal(q_a_center, 0.05, labels.size), 0.05, 1.0)
    quality_b = np.clip(rng.normal(0.78, 0.06, labels.size), 0.05, 1.0)
    quality = np.column_stack([quality_a, quality_b])
    return labels, scores, quality, clusters


def _systems(dev_scores, dev_quality, dev_labels):
    systems = {
        "C1": EqualScoreFusion().fit(dev_scores),
        "C2": WeightedScoreFusion(grid_step=0.1).fit(dev_scores, dev_labels),
        "C3": LogisticScoreFusion(C=1.0).fit(dev_scores, dev_labels),
        "C5": QualityWeightedScoreFusion(gamma_grid=(0.0, 0.5, 1.0, 2.0)).fit(
            dev_scores, dev_quality, dev_labels
        ),
    }
    return systems


def _transform(system_id, system, scores, quality):
    if system_id == "C5":
        return system.transform(scores, quality)
    return system.transform(scores)


def run_synthetic_smoke(*, n_subjects: int = 40, n_boot: int = 80) -> dict[str, object]:
    """Run a deterministic synthetic development/calibration/test pipeline.

    Returns machine-checkable diagnostics. No output from this function is valid
    evidence for Q1–Q3; the function exists only to catch plumbing and leakage bugs.
    """
    if not isinstance(n_subjects, int) or n_subjects < 10:
        raise ValueError("n_subjects must be an integer >= 10")
    if not isinstance(n_boot, int) or n_boot < 10:
        raise ValueError("n_boot must be an integer >= 10")

    dev_y, dev_x, dev_q, _ = _block(100, n_subjects, stress=False)
    cal_y, cal_x, cal_q, _ = _block(200, n_subjects, stress=False)
    test_y, clean_x, clean_q, clusters = _block(300, n_subjects, stress=False)
    stress_y, stress_x, stress_q, stress_clusters = _block(300, n_subjects, stress=True)

    # The stress generator uses the same subject/trial topology and label sequence.
    if not np.array_equal(test_y, stress_y) or not np.array_equal(clusters, stress_clusters):
        raise RuntimeError("synthetic clean/stress trial topology drifted")

    systems = _systems(dev_x, dev_q, dev_y)
    method_ids = tuple(systems)

    clean_eer: list[float] = []
    stress_eer: list[float] = []
    clean_auc: list[float] = []
    stress_auc: list[float] = []
    clean_cllr: list[float] = []
    stress_cllr: list[float] = []
    bootstrap_eer: dict[str, np.ndarray] = {}

    for method_id, system in systems.items():
        cal_scores = _transform(method_id, system, cal_x, cal_q)
        clean_scores = _transform(method_id, system, clean_x, clean_q)
        stressed_scores = _transform(method_id, system, stress_x, stress_q)

        calibrator = LogisticLLRCalibrator(C=10.0).fit(cal_scores, cal_y)
        clean_llr = calibrator.transform(clean_scores)
        stressed_llr = calibrator.transform(stressed_scores)

        e_clean, _ = eer(test_y, clean_scores)
        e_stress, _ = eer(test_y, stressed_scores)
        clean_eer.append(float(e_clean))
        stress_eer.append(float(e_stress))
        clean_auc.append(float(roc_auc(test_y, clean_scores)))
        stress_auc.append(float(roc_auc(test_y, stressed_scores)))
        clean_cllr.append(float(cllr(clean_llr[test_y == 0], clean_llr[test_y == 1])))
        stress_cllr.append(float(cllr(stressed_llr[test_y == 0], stressed_llr[test_y == 1])))

        bootstrap_eer[method_id] = cluster_bootstrap_metric(
            test_y,
            clean_scores,
            clusters,
            lambda y, s: eer(y, s)[0],
            n_boot=n_boot,
            seed=900 + len(bootstrap_eer),
        )

    clean_eer_a = np.asarray(clean_eer)
    stress_eer_a = np.asarray(stress_eer)
    clean_cllr_a = np.asarray(clean_cllr)
    stress_cllr_a = np.asarray(stress_cllr)

    # Two criteria are enough to exercise the non-dominance plumbing here. The
    # scientific Q2 run later adds locked robustness and measured cost dimensions.
    clean_pareto = non_dominated_mask(
        np.column_stack([clean_eer_a, clean_cllr_a]),
        minimize=[True, True],
    )
    stress_pareto = non_dominated_mask(
        np.column_stack([stress_eer_a, stress_cllr_a]),
        minimize=[True, True],
    )

    return {
        "synthetic_only": True,
        "method_ids": method_ids,
        "n_trials": int(test_y.size),
        "n_subject_clusters": int(np.unique(clusters).size),
        "clean_eer": clean_eer_a,
        "stress_eer": stress_eer_a,
        "clean_auc": np.asarray(clean_auc),
        "stress_auc": np.asarray(stress_auc),
        "clean_cllr": clean_cllr_a,
        "stress_cllr": stress_cllr_a,
        "clean_pareto": clean_pareto,
        "stress_pareto": stress_pareto,
        "eer_tau_b": float(kendall_tau_b(clean_eer_a, stress_eer_a)),
        "eer_rank_reversals": tuple(pairwise_rank_reversals(clean_eer_a, stress_eer_a)),
        "bootstrap_eer": bootstrap_eer,
    }
