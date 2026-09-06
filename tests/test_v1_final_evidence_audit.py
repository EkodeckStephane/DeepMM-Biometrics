from scripts.audit_v1_final_evidence import audit


def test_committed_v1_evidence_passes_claim_code_data_audit():
    summary = audit()
    assert summary == {
        "status": "pass",
        "conditions": 15,
        "trials_per_condition": 4000,
        "verified_score_manifests": 148,
        "verified_score_arrays": 444,
        "headline_result_status": "complete",
        "inference_boundary": (
            "public biometric-instance point estimates; "
            "no person-population p-values or CIs"
        ),
    }
