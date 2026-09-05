# V1 Calibration, Robustness and Final-Evaluation Lock

**Scope:** public NUPT-FPV V1 only.  
**Status:** frozen after development model selection and before inspection of calibration-role or final-role outcomes by the selected systems.

## Calibration

Every system first produces one scalar verification score with the convention higher = more genuine. The common deployable mapping is `LogisticLLRCalibrator(C=1.0)`, fitted only on the held-out calibration role. Class-balanced fitting imposes an effective target prior of 0.5. The calibrated decision output is treated as an approximate natural-log likelihood ratio.

`C=1.0` is fixed rather than tuned on the small calibration role. Final labels never choose the regularization strength, score orientation, reference prior or model. A non-positive fitted calibration slope is a protocol failure, not a reason to reverse scores post hoc.

Probability metrics use reference target prior 0.5. ECE uses 15 fixed equal-width bins. Headline calibration evidence consists of `C_llr`, `C_llr_min`, `C_llr_cal`, Brier score and NLL; ECE is complementary descriptive evidence.

Two Q3 calibration analyses are distinguished:

1. **primary transfer test:** fit the clean calibration mapping once and transfer it unchanged to every clean/degraded/missing final condition;
2. **secondary recalibration diagnostic:** fit a separate mapping on the matching held-out calibration condition, then apply it to the corresponding final condition.

The second analysis measures recoverable condition-specific calibration; it cannot replace the first robustness result.

## Frozen Q3 stress plan

Stress-plan SHA-256:

`6ba45461396f61dda720e7d289cdade98cac750cf9172b7502518428e022bbd3`

There are 15 conditions: clean, 12 probe-only corruptions and two probe-only single-modality absence conditions.

For each modality independently:

- Gaussian blur radii: 1.0, 2.0, 3.0;
- contrast factors: 0.75, 0.50, 0.25.

Corruptions affect probe images only; enrollment remains clean. Missingness is never rendered as a corrupted image and never encoded with NaN.

## Missing-modality policy

The missing-fingerprint and missing-finger-vein conditions model **probe-time/query-time sensor absence**. Enrollment evidence remains complete.

The primary M0 policy is intentionally conservative because the selected D1/D2 systems were not trained with modality dropout:

- U/C1/C2/C3/C4/D1/D2 fall back to the single available unimodal verification score;
- C5 uses its frozen quality-weighted rule renormalized over available evidence;
- D3S uses its explicit availability mask and canonical zero placeholder for the unavailable score/quality slot.

This policy measures native graceful degradation without introducing a post-selection missingness-trained replacement. A later M1 modality-dropout experiment, if performed, must be labelled secondary rather than silently replacing M0.

## Discrimination and Q2/Q3 analyses

Primary discrimination reports ROCCH-EER; empirical EER is retained as a sensitivity/comparability metric. AUC and TAR at FAR = 0.1, 0.01 and 0.001 are also frozen. With 3,800 final impostor trials, the smallest grid FAR remains above the one-impostor resolution 1/3800.

Q2 does not use a post-hoc weighted composite score. Performance, robustness, calibration and cost are examined by Pareto/non-dominance analysis. Q3 reports condition-wise metrics, Kendall tau-b ranking stability and explicit pairwise rank reversals.

## Cost protocol

The primary directly comparable cost measurement is fusion-only inference on CPU, float32, batch 256, two threads, with 20 warm-up calls and 200 retained repetitions in the same workflow/hardware context. All raw latency repetitions are preserved. End-to-end image-to-score timing is secondary/descriptive because all Track-I systems share the same frozen image encoder and it dominates the pipeline cost.

Parameter counts are exact where applicable. MACs/peak memory are reported only where the implemented measurement is meaningful and matched; unavailable cost components remain explicitly unavailable rather than estimated from incomparable environments.

## Final firewall

Final clean trial-manifest SHA-256:

`3b60ce30d0d496c35aefe0bf0b8c48f868cb3befba0c1fdfb52986645293f324`

The final evaluation script must verify the dataset, training, model-selection, stress-plan and final-policy locks before opening final-role images. Calibration coefficients and selected neural checkpoint hashes must already be frozen. Any change after final outcomes are visible creates a new exploratory protocol version.

Final-policy SHA-256:

`e9701015b541e9c7e4debccd01fd1f32affecc97abaf2996b8cf6c5811adbfb5`
