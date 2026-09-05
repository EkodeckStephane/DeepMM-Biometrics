# Fixed Research Questions and Evidence Map

**Study title:** *Deep Learning Approaches for Multimodal Biometrics*

These questions are frozen. Architectural choices and experimental design must serve them rather than replace them.

## Q1 — Contribution of deep learning

**Question.** What does deep learning actually contribute to multimodal biometrics compared with classical fusion approaches and unimodal systems?

### Required comparisons
- Best unimodal system for each selected modality.
- Classical score-level fusion.
- Classical feature-level fusion.
- Deep learned score fusion.
- Deep representation/feature fusion.
- Attention-based fusion.
- Transformer/cross-attention fusion when justified by the data regime.

### Required evidence
- Verification discrimination: EER, ROC-AUC, TAR at predeclared FAR operating points.
- Multimodal gain: paired delta relative to the best unimodal system.
- Calibration: ECE, Brier score, NLL, reliability curves.
- Robustness: degradation curves under controlled corruption.
- Missing-modality behavior: both modalities, modality A only, modality B only.
- Computational cost: trainable parameters, MACs/FLOPs where reproducible, peak inference memory, and latency under a fixed environment.

**Interpretation rule:** deep learning is not considered beneficial merely because one accuracy metric improves. Benefits and costs are reported dimension by dimension.

## Q2 — Best deep-fusion family

**Question.** Among deep-learning multimodal-fusion families, which provides the best trade-off among performance, robustness, calibration, and computational cost?

### Families to be considered after SOTA lock
1. Learned score fusion.
2. Deep feature concatenation/projection.
3. Gated or quality-aware fusion.
4. Bilinear/intermediate fusion where technically appropriate.
5. Cross-attention / multimodal Transformer.
6. Missing-modality-aware architectures if they represent a distinct family rather than an add-on.

### Decision rule
There will be no post-hoc scalar score designed to favor a method. The primary analysis will use:
- dimension-specific rankings;
- confidence intervals and paired comparisons;
- Pareto dominance/non-dominance over performance, robustness, calibration, and cost.

A scalar multi-criteria index may be added only if its normalization and weights are fixed before final test evaluation and accompanied by sensitivity analysis.

## Q3 — Stability under quality degradation and missing modalities

**Question.** Does the best approach remain the same when input quality degrades or when one modality is missing?

### Stress matrix
For each eligible model:
- clean A + clean B;
- degraded A + clean B;
- clean A + degraded B;
- degraded A + degraded B;
- A only;
- B only.

Candidate controlled degradations include blur, noise, resolution loss, compression, exposure/contrast change, and localized occlusion, subject to modality-specific validity.

### Primary Q3 outcome
The primary result is the **stability or reversal of model-family ranking** across acquisition conditions, not merely whether one proposed architecture is robust.

## Cross-question prohibitions

- Do not assume that a Transformer is the strongest family.
- Do not infer cross-modal learning value from chimeric identity pairing.
- Do not select the test metric or stress severity after observing final results.
- Do not call a point-estimate winner the best approach without uncertainty and cost analysis.
- Do not suppress negative results or rank reversals.
