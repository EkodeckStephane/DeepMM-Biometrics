# Claim–Evidence Matrix v0.1

**Study:** *Deep Learning Approaches for Multimodal Biometrics*

This document freezes what evidence is required before any headline conclusion can appear in the manuscript. It implements Gate 2 and the Senior Reviewer promise–evidence discipline.

| Claim family | Minimum direct evidence | Evidence that is insufficient | Current status |
|---|---|---|---|
| **Q1-A — multimodal fusion improves over unimodal** | Same test identities/trials; comparison against each unimodal modality and the best unimodal; paired effect with uncertainty | Comparing to only the weaker modality; cross-paper numbers; training accuracy | OPEN |
| **Q1-B — DL fusion adds value beyond classical fusion** | Strong equal/weighted/logistic score fusion and classical feature fusion on the same frozen evidence; matched tuning budget; paired test | Comparing DL only with equal-weight sum; giving DL stronger encoders; unpaired mean±SD across seeds | OPEN |
| **Q1-C — claimed DL gain is practically worthwhile** | Performance delta reported together with parameters, latency/memory/compute overhead | Higher AUC alone; parameter count without measured runtime | OPEN |
| **Q2 — one DL family has the best trade-off** | Locked performance, robustness, calibration and cost dimensions; uncertainty; Pareto analysis; no post-hoc weights | Single scalar accuracy; arbitrary utility weights chosen after testing | OPEN |
| **Q2 — family A dominates family B** | A at least as good on every locked dimension and strictly better on ≥1, plus bootstrap/uncertainty support | Point-estimate dominance without uncertainty | OPEN |
| **Calibration is improved** | Held-out calibration procedure; `C_llr`/calibration loss when validated; Brier/NLL/ECE complementary; same test trials | EER/AUC improvement; softmax confidence alone | OPEN |
| **Quality-aware/gated fusion is robust** | Controlled degradations fixed before testing; clean-vs-stress deltas; matched classical quality-aware baseline | Natural-quality anecdotes; one corrupted example | OPEN |
| **Model handles missing modalities** | Explicit A-only and B-only evaluation with predeclared missing-input policy; optionally trained-missingness track separated from native fallback | Zeroing a modality without documenting policy; testing only complete inputs | OPEN |
| **Q3 — best family changes/does not change under stress** | Rankings for every locked stress condition; Kendall/rank-reversal analysis; uncertainty support; subset-specific calibration | Observing that every model degrades; one point-estimate rank table | OPEN |
| **Transformer/attention is justified** | Measurable benefit over simpler families after matched encoders and budget; cost reported | Architecture novelty; parameter count/complexity; attention visualization alone | OPEN |
| **Result generalizes beyond one dataset** | Replication on an independent real multimodal dataset or clearly bounded cross-dataset validation | Multiple random seeds on one dataset | OPEN |
| **Fusion learns cross-modal interaction** | Real same-subject multimodal data; appropriate ablation showing interaction mechanism matters | Chimeric identities; independent unimodal datasets paired by labels | OPEN |

## Hard wording rules

The final article must not use an unconditional statement such as “deep learning is superior”, “Transformer is best”, “multimodal is more robust”, or “the model handles missing modalities” unless the corresponding row above is closed by direct evidence.

Negative evidence is not a failure of the study. If the strongest classical fusion remains Pareto-optimal, if the best unimodal modality beats multimodal DL, or if rankings reverse under stress, that is a direct answer to Q1-Q3 and must remain in the manuscript.

## Evidence-state vocabulary

Senior-reviewer audits will label every headline promise as one of:

- demonstrated;
- partially demonstrated;
- observed only;
- specified only;
- simulated only;
- not demonstrated;
- contradicted;
- deferred;
- unverifiable.

No claim is promoted from “observed” to “demonstrated” solely because the point estimate is favorable.
