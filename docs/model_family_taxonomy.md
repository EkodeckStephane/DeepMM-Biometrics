# Model-Family Taxonomy v0.1

**Study:** *Deep Learning Approaches for Multimodal Biometrics*

This taxonomy is designed to answer the fixed questions Q1–Q3. It is not a list of fashionable architectures. A family is retained only if it represents a materially different fusion mechanism that can be compared under a controlled verification protocol.

## 1. Comparison principle

The study separates two effects that are frequently confounded in the multimodal-biometric literature:

1. **representation quality** — how strong the unimodal encoders are;
2. **fusion quality** — what the multimodal combination mechanism adds once comparable unimodal evidence exists.

Accordingly, the primary benchmark will use **shared/frozen unimodal encoders** wherever technically possible. A secondary end-to-end/fine-tuning track may be added to measure attainable system performance, but it will not replace the controlled fusion comparison.

This separation is essential for Q1: a deep fusion head cannot be credited for an improvement that actually comes from a stronger encoder.

## 2. Tier U — unimodal anchors

For each selected modality:

- one strong reproducible encoder;
- a fixed embedding normalization procedure;
- a fixed matcher (cosine or another modality-appropriate matcher selected before final testing);
- thresholding/calibration fitted on validation only.

These systems provide the best-single-modality references for Q1.

## 3. Tier C — classical multimodal baselines

### C1. Normalized score sum

Input: per-modality verification scores.

Mechanism: normalize scores with parameters learned on the development split, then sum with equal weights.

Purpose: minimum-complexity multimodal reference.

### C2. Validation-weighted score fusion

Input: per-modality verification scores.

Mechanism: weights selected using validation data only, with a constrained low-dimensional search.

Purpose: tests whether adaptive weighting is needed at all.

### C3. Logistic score fusion

Input: per-modality scores, optionally modality-quality variables only if the same variables are available to all comparable quality-aware methods.

Mechanism: regularized logistic regression trained on development data.

Purpose: strong classical learned fusion baseline. It must not be mislabeled as deep learning.

### C4. Classical feature fusion

Input: fixed unimodal embeddings.

Mechanism: standardized concatenation followed by a non-deep metric/classifier when sample size supports it.

Purpose: separates the effect of feature-level access from the effect of deep nonlinear learning.

## 4. Tier D — deep-learning fusion families

### D1. Deep score fusion

Canonical representative: compact MLP over per-modality scores, availability masks, and predeclared quality variables.

Scientific question: does nonlinear score combination add measurable value beyond logistic/weighted fusion?

Cost expectation: low.

### D2. Deep feature fusion

Canonical representative: modality-specific linear projection -> concatenation -> residual MLP projection -> normalized fused embedding.

Scientific question: does jointly learned nonlinear feature interaction outperform classical concatenation and score fusion?

Cost expectation: low to moderate.

### D3. Gated / quality-aware fusion

Canonical representative: learned gate producing modality weights conditioned on embeddings, objective quality estimates, and availability masks. The fused representation is a normalized weighted combination of projected modality embeddings.

Scientific question: does dynamic trust allocation help when modality quality is heterogeneous?

Historical relevance: quality-dependent biometric fusion has long shown that automatically derived quality can matter; the deep model must therefore beat strong quality-aware classical baselines, not merely equal-weight fusion.

Cost expectation: low to moderate.

### D4. Interaction / bilinear-style fusion

Canonical representative: compact multiplicative interaction (e.g., low-rank bilinear or factorized interaction) between projected modality embeddings.

Scientific question: are second-order cross-modal interactions useful beyond concatenation/gating?

Inclusion rule: retain only if the final SOTA and data regime justify the extra family. It is optional rather than mandatory.

Cost expectation: moderate.

### D5. Attention / cross-attention fusion

Canonical representative: modality-specific token sequences projected to a common dimension, followed by symmetric or alternating cross-attention and pooled to a fused embedding.

Scientific question: does explicit cross-modal interaction improve verification under matched encoders and matched tuning budget?

Important constraint: attention must receive meaningful token sequences. A two-scalar or two-token construction created merely to label a method “attention” is not admissible.

Cost expectation: moderate to high.

### D6. Multimodal Transformer

Canonical representative: shared-dimensional modality tokens/local tokens plus modality/type embeddings, a compact Transformer encoder, and a fused verification embedding.

Scientific question: does a Transformer provide a favorable performance–robustness–calibration–cost trade-off rather than merely higher capacity?

Inclusion rule: parameter budget and tuning budget must be reported and matched as far as practicable.

Cost expectation: high relative to D1–D3.

## 5. Tier M — missing-modality mechanisms

Missing-modality handling is treated as a **second axis**, not automatically as a separate fusion family. This prevents Q2 from conflating fusion architecture with a robustness add-on.

### M0. Deterministic fallback

- score fusion: renormalize over available modalities;
- feature fusion: predeclared zero/mask strategy;
- unimodal fallback: use the available modality directly.

### M1. Modality dropout / masked training

Randomly mask a modality during training with a frozen probability schedule selected before final testing.

### M2. Learned missing token / availability-aware gating

Represent absence explicitly through an availability mask or learned missing embedding.

### M3. Representation reconstruction / shared-specific modeling

Reconstruct or infer missing-modality information in representation space.

### M4. Cross-modal generation

Generate missing-modality data/features from the observed modality.

M3/M4 are retained as dedicated Q3 families only if SOTA closure and compute/data feasibility justify a faithful implementation. They are not required to answer Q1.

## 6. Why these families are SOTA-aligned

Verified current literature supports the relevance of the taxonomy:

- Es-Sobbahi, Radouane, and Nafil, *IET Biometrics* (2025), DOI `10.1049/bme2/5055434`, documents the continuing dominance of feature- and score-level fusion in physiological multimodal biometrics.
- Ren et al., *IEEE TIFS* (2022), DOI `10.1109/TIFS.2022.3175599`, provides a paired fingerprint–finger-vein dataset and CNN-based multimodal benchmark, showing the importance of real paired acquisition and learned fusion.
- Fan et al., *IEEE TSMC: Systems* (2024), DOI `10.1109/TSMC.2024.3382877`, combines sequential decision logic with adaptive weighted palmprint/palm-vein fusion and explicitly treats recognition time as part of the design objective.
- Zheng et al., *Pattern Recognition Letters* (2025), DOI `10.1016/j.patrec.2025.06.017`, uses structured state-space encoders, cross-modal attention, and contrastive alignment for PPG–fingerprint verification.
- Pan et al., *IEEE TIFS* (2025), DOI `10.1109/TIFS.2025.3559802`, addresses missing-modality biometric recognition using hierarchical cross-modal generation and dynamic sparse fusion.
- Pan et al., *Digital Signal Processing* (2025), DOI `10.1016/j.dsp.2025.105003`, addresses missing modalities through shared-specific feature disentanglement and cross-modal feature transformation.
- Wu et al., *Transactions on Machine Learning Research* (2026), *Deep Multimodal Learning with Missing Modality: A Survey*, provides the broader MLMM taxonomy and confirms that missing-modality robustness is a distinct methodological problem.

These papers justify the *families*. They do **not** establish that any family is universally best.

## 7. Candidate benchmark set for the first implementation pass

The minimum controlled set is:

1. U-A: unimodal modality A;
2. U-B: unimodal modality B;
3. C1: equal score sum;
4. C2: validation-weighted score fusion;
5. C3: logistic score fusion;
6. D1: deep score MLP;
7. D2: deep feature fusion;
8. D3: quality-aware gated fusion;
9. D5/D6: one rigorously justified attention/Transformer representative.

This gives a sufficiently broad test of Q1 and Q2 without turning the study into an uncontrolled architecture zoo.

## 8. Family-selection stop rule

No additional family is added merely because it produces a better pilot result. After the SOTA lock, the family list is frozen before the final test campaign. Any later architecture is labeled exploratory and cannot replace the preregistered primary comparison.
