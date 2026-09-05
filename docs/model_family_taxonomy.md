# Model-Family Taxonomy v0.2

**Study:** *Deep Learning Approaches for Multimodal Biometrics*

This taxonomy serves Q1–Q3. It is intentionally organized by **information access and fusion mechanism**, not by fashionable architecture names. A family is retained only if it creates a scientifically distinct contrast under a common protocol.

## 1. Primary causal comparison principle

The study separates:

1. **representation quality** — information supplied by the unimodal encoders/matchers;
2. **fusion quality** — what the combination mechanism adds when the unimodal evidence is held fixed;
3. **information access** — whether a fusion method sees only scalar scores, full embeddings, local/token features, quality variables, or availability masks.

These effects must not be conflated.

The primary Track I therefore freezes the unimodal evidence wherever technically possible. Track II may fine-tune/end-to-end train systems to measure attainable performance, but Track II cannot support the causal statement that a gain came specifically from the fusion mechanism.

The 2026 OU-MB benchmark makes this distinction especially important: it already evaluates mean/weighted-score and normalized concatenation fusion using fixed modality-specific recognition models. DeepMM must extend, not rediscover, that baseline philosophy.

## 2. Fairness rule: compare within information-access strata first

A feature-level neural model has access to richer information than a score-only method. Therefore the confirmatory contrasts are stratified.

### Score-input stratum

All methods receive the exact same ordered per-modality scores and, when applicable, the same predeclared quality/availability variables.

Primary score-level contrast:

```text
C1 equal score
C2 weighted score
C3 logistic score
C5 classical quality-aware score
vs.
D1 nonlinear deep score fusion
D3-score learned gated/quality-aware score fusion
```

### Embedding-input stratum

All methods receive the same frozen unimodal embeddings.

Primary feature-level contrast:

```text
C4 normalized/standardized concatenation
vs.
D2 nonlinear feature fusion
D3-feature learned gated feature fusion
D4 attention/Transformer interaction (only when justified)
```

### Cross-stratum comparisons

Score-level versus feature-level systems may be compared for **overall system trade-off**, but a cross-stratum advantage is not automatically attributed to “better fusion”: it may arise from richer input information.

## 3. Tier U — unimodal anchors

For each selected modality:

- one strong reproducible encoder/matcher pipeline;
- frozen embedding normalization and score orientation;
- fixed enrollment/probe construction;
- thresholds and deployable calibration fitted on held-out development/calibration data only.

The strongest unimodal modality is the Q1 reference; beating only the weaker modality is insufficient.

## 4. Tier C — classical/model-agnostic baselines

### C1 — Equal normalized score fusion

Per-modality score normalization is fitted on development data only, then scores are averaged with equal weights.

Purpose: minimum-complexity multimodal reference and direct OU-MB-style baseline.

### C2 — Validation-weighted score fusion

Non-negative weights sum to one and are selected on development data under a frozen objective/grid.

Purpose: tests whether learned nonlinear score fusion adds anything beyond careful static weighting.

### C3 — Logistic score fusion

Regularized logistic regression over the same score vector.

Purpose: strong classical learned score-fusion baseline. It is not deep learning.

### C4 — Normalized classical feature fusion

Per-modality embeddings are normalized under a frozen rule, concatenated, optionally re-normalized, then matched with a fixed non-deep similarity/classifier.

Purpose: controls for the advantage of seeing embeddings rather than scalar scores. OU-MB 2026 provides a strong precedent for normalized concatenation with cosine matching.

### C5 — Classical quality-aware score fusion

Dynamic non-neural weighting from predeclared quality values, including deterministic renormalization when a modality is unavailable.

Purpose: quality-aware DL must beat a quality-aware classical comparator, because quality-dependent biometric fusion is established prior art.

## 5. Tier D — deep fusion families

### D1 — Nonlinear deep score fusion

Canonical representative: compact MLP receiving only the same scalar match scores used by C1–C3; availability/quality variables are excluded unless the corresponding classical comparison receives them too.

Scientific contrast:

> Does nonlinear score interaction improve upon weighted and logistic score fusion when the input evidence is identical?

The MLP capacity must remain deliberately small enough that the study tests nonlinear fusion rather than arbitrary overparameterization.

### D2 — Nonlinear feature fusion

Canonical representative:

```text
frozen embedding A -> fixed/common-dimension projection
frozen embedding B -> fixed/common-dimension projection
concatenate -> compact residual MLP -> normalized fused embedding
```

The trainable projection/fusion head is the independent variable; upstream unimodal encoders remain frozen in Track I.

Scientific contrast:

> Does learned nonlinear feature interaction improve upon normalized concatenation under the same frozen embeddings?

### D3 — Learned gated / quality-aware fusion

This family has two access-matched variants, not one privileged model:

- **D3-score:** gate sees scores plus the exact same objective quality/availability variables available to C5;
- **D3-feature:** gate sees frozen projected embeddings and the same quality/availability variables.

The gate produces normalized modality weights or reliability coefficients.

Scientific contrast:

> Does a learned dynamic reliability mapping add value beyond classical quality-aware weighting under heterogeneous quality or missing evidence?

Quality awareness, dynamic weighting and adaptive hybrid fusion are established prior art (Poh et al. 2009; Soleymani et al. 2022; Fan et al. 2024; DIRS 2025; AHFNet 2026), so D3 is a comparison family, not a novelty claim.

### D4 — Attention/Transformer interaction family

Retained as **one** high-capacity interaction family for the primary benchmark, not two redundant “attention” and “Transformer” rows.

Admission rule: D4 is included only when the chosen modality pipeline exposes semantically meaningful token/local feature sequences. A construction that converts two scalar scores into two fake “tokens” merely to use self-attention is inadmissible.

Possible canonical representative:

- modality-specific frozen token/local features;
- projection to common dimension;
- modality/type embeddings;
- one or a small number of symmetric cross-attention/Transformer blocks;
- pooled normalized fused embedding.

Scientific contrast:

> Does explicit token-level cross-modal interaction provide a benefit large enough to justify its added cost versus D2/D3 under matched upstream evidence?

FBR 2024, MPAD 2025 and PPG–fingerprint cross-attention work already establish attention-based biometric fusion as prior art. DeepMM therefore evaluates this family; it does not brand attention itself as a contribution.

### D5 — Multiplicative/bilinear interaction — optional

A low-rank bilinear/factorized interaction family is included only if final SOTA closure shows that it represents a distinct important mechanism not already adequately covered by D2/D4 **and** the dataset size supports the extra trainable capacity.

D5 is not part of the minimum confirmatory set at present.

## 6. Tier M — missing-modality mechanisms are a second experimental axis

Missingness handling is intentionally separated from the fusion-family label.

### M0 — Native/deterministic missingness policy

Applied without missing-modality training:

- unimodal fallback when only one modality is available;
- score fusion renormalizes over available evidence;
- feature methods use a predeclared mask/absence representation where required.

This measures **native graceful degradation**.

### M1 — Modality-dropout training

The same fusion family is retrained with a frozen modality-dropout schedule.

This measures the benefit of explicit missingness exposure during training and is treated as a factorial training condition, not as a new fusion family.

### M2 — Availability-aware learned absence representation

An explicit mask or learned missing token is allowed only for methods whose architecture requires it. The availability signal must be defined identically across comparable methods.

### M3 — Reconstruction/shared-specific modeling — secondary

SSFD-Net-style shared/specific representation reconstruction can be evaluated as an advanced Q3 comparator if faithful implementation and compute/data feasibility permit.

### M4 — Cross-modal generation — secondary

TIFS 2025-style cross-modal generation can be evaluated only as a separate advanced missing-modality comparator. It must not be conflated with the core Q1 question of whether DL fusion adds value under complete evidence.

M3/M4 are not required for the minimum Q1/Q2 benchmark.

## 7. Calibration is an evaluation axis, not a fusion family

Calibration is applied/evaluated under a common protocol after each method generates verification scores.

- deployable affine logistic calibration is fitted on held-out data only;
- `C_llr`, `C_llr_min`, calibration loss, Brier/NLL and descriptive ECE are reported according to the locked protocol;
- calibration transfer across clean/degraded/missing-subset conditions is separated from subset-specific recalibration.

Mandasari et al. 2014 and Susyanto et al. 2019 establish calibration/Cllr as prior biometric concepts; DeepMM's role is to integrate calibration consistently into the family comparison.

## 8. Minimum confirmatory family set

Subject to dataset feasibility and final Gate-4 falsification, the minimum set is:

| ID | Family | Primary input |
|---|---|---|
| U-A | unimodal modality A | modality A |
| U-B | unimodal modality B | modality B |
| C1 | equal normalized score fusion | scores |
| C2 | validation-weighted score fusion | scores |
| C3 | logistic score fusion | scores |
| C4 | normalized classical feature fusion | embeddings |
| C5 | classical quality-aware fusion | scores + quality/availability |
| D1 | nonlinear deep score fusion | scores |
| D2 | nonlinear feature fusion | embeddings |
| D3 | learned gated/quality-aware fusion | access-matched inputs |
| D4 | attention/Transformer interaction | meaningful local/token features, if available |

If D4 cannot be implemented without giving it different upstream information than all competitors, it moves to Track II or exploratory analysis rather than contaminating Track I.

## 9. Complexity/tuning fairness

Before final testing, each trainable family receives a documented tuning budget. The study reports at least:

- trainable parameter count;
- inference latency under identical hardware/batch/precision;
- peak memory when measurable;
- FLOPs/MACs where calculation is meaningful;
- number of tuning configurations/runs attempted.

A larger DL family does not receive unlimited hyperparameter search while classical baselines use defaults.

## 10. Family freeze rule

The family list is frozen **before final test evaluation**. Pilot results may reveal implementation bugs or infeasibility, but a new family is not added merely because it improves a disappointing result.

Any post-freeze architecture is exploratory and cannot silently replace the preregistered comparison.

## 11. Current status

The family *definitions and fairness strata* are ready for pilot implementation. Exact modality-specific inputs, D4 token feasibility, parameter budgets and hyperparameter grids remain `TO LOCK` after dataset/encoder feasibility is known.

This is intentionally not yet a final family lock.
