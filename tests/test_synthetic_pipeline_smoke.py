import numpy as np

from deepmm.evaluation import run_synthetic_smoke


def test_synthetic_pipeline_exercises_clean_stress_calibration_and_uncertainty():
    result = run_synthetic_smoke(n_subjects=20, n_boot=20)
    assert result["synthetic_only"] is True
    assert result["method_ids"] == ("C1", "C2", "C3", "C5")
    assert result["n_trials"] == 80
    assert result["n_subject_clusters"] == 20

    for key in (
        "clean_eer",
        "stress_eer",
        "clean_auc",
        "stress_auc",
        "clean_cllr",
        "stress_cllr",
    ):
        values = result[key]
        assert values.shape == (4,)
        assert np.all(np.isfinite(values))

    assert result["clean_pareto"].shape == (4,)
    assert result["stress_pareto"].shape == (4,)

    # A fully tied metric ranking legitimately makes Kendall tau-b undefined. The
    # pipeline must preserve that state instead of manufacturing an ordering.
    assert result["eer_tau_defined"] == bool(np.isfinite(result["eer_tau_b"]))
    assert result["cllr_tau_defined"] == bool(np.isfinite(result["cllr_tau_b"]))
    assert result["cllr_tau_defined"] is True

    for method_id, replicates in result["bootstrap_eer"].items():
        assert method_id in result["method_ids"]
        assert replicates.shape == (20,)
        assert np.all(np.isfinite(replicates))


def test_synthetic_smoke_is_deterministic():
    a = run_synthetic_smoke(n_subjects=20, n_boot=20)
    b = run_synthetic_smoke(n_subjects=20, n_boot=20)
    for key in ("clean_eer", "stress_eer", "clean_cllr", "stress_cllr"):
        assert np.array_equal(a[key], b[key])
    assert a["eer_rank_reversals"] == b["eer_rank_reversals"]
    assert a["cllr_rank_reversals"] == b["cllr_rank_reversals"]
