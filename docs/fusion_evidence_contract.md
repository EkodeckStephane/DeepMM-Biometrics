# Fusion Evidence Contract v1.0

**Gate-4 dependency:** `PASS-POSITIONING`  
**Purpose:** prevent representation privilege from being misreported as a fusion gain.

## 1. Scientific rule

The primary DeepMM experiment treats the **fusion mechanism** as the controlled independent variable. Consequently, methods are compared confirmatorily only when they receive the same class of upstream unimodal evidence.

Three information strata are recognized:

1. **score** — one matcher score per modality and trial;
2. **embedding** — paired enrollment/probe embeddings per modality;
3. **token/local-feature** — spatial/local/token representations required by attention/Transformer methods.

A method that receives embeddings or tokens is not allowed to claim a pure fusion improvement over a score-only method. Cross-stratum comparisons may be reported as system-level comparisons, but the manuscript must identify the information-access difference explicitly.

## 2. Canonical evidence objects

`src/deepmm/fusion/contracts.py` defines:

- `ScoreEvidence`;
- `EmbeddingEvidence`;
- `FusionMethodSpec`;
- `EvidenceTier`;
- the canonical confirmatory method registry.

Labels are deliberately absent from transform-time evidence objects. They enter only development/evaluation routines that are allowed to fit or score against labels.

## 3. Missingness representation

Missing evidence is never represented by `NaN`, infinity, an extreme sentinel score, or an undocumented embedding value.

For score evidence:

- `availability[n, m]` explicitly states whether modality `m` is available for trial `n`;
- unavailable score slots use the canonical value `0.0`;
- every trial retains at least one available modality.

For embedding evidence:

- enrollment and probe availability are represented separately;
- unavailable embedding vectors are exactly zero;
- every enrollment/probe side retains at least one modality.

The zero value is a serialization placeholder, **not evidence**. A missingness-aware model may use the explicit availability mask; a method not registered as missingness-aware is rejected on incomplete evidence.

## 4. Quality-variable symmetry

Quality-aware comparisons are fair only when competing methods receive the same predeclared quality variables.

- C5 and D3-score may both receive the same score-level quality matrix.
- D3-feature may receive the corresponding embedding-level quality variables.
- A non-quality method is not silently given quality features.
- Unavailable modality quality uses zero as a canonical placeholder.

Quality estimators themselves are part of the frozen upstream evidence contract. A model cannot obtain a stronger proprietary quality estimator and then attribute the gain solely to its fusion mechanism.

## 5. Gate-4-locked method strata

### Score-input confirmatory methods

| ID | Method | Quality access | Missingness access |
|---|---|---:|---:|
| C1 | equal normalized score fusion | no | no |
| C2 | validation-weighted score fusion | no | no |
| C3 | regularized logistic score fusion | no | no |
| C5 | classical quality-weighted score fusion | yes | no |
| D1 | compact nonlinear score fusion | no | no |
| D3S | learned score quality/availability gate | yes | yes |

### Embedding-input confirmatory methods

| ID | Method | Quality access | Missingness access |
|---|---|---:|---:|
| C4 | controlled feature concatenation | no | no |
| D2 | compact nonlinear feature fusion | no | no |
| D3F | learned feature quality/availability gate | yes | yes |

### Token/local-feature method

The attention/Transformer representative remains **conditionally confirmatory**. It moves into the primary comparison only if the selected upstream encoders expose meaningful, comparable token/local features without giving the Transformer a privileged representation that the embedding methods cannot access. Otherwise it is Track II / exploratory.

## 6. Missing-modality analysis is a second axis

The method family and missingness strategy are not conflated. The same base family may be evaluated under:

- complete evidence;
- deterministic fallback;
- explicit availability-aware operation;
- modality-dropout training, where applicable;
- later reconstruction/generation only if separately justified.

This prevents a model from being called a superior fusion family merely because it alone was trained with missing-modal examples.

## 7. Confirmatory comparison rules

A headline fusion-only contrast is admissible only when all of the following hold:

1. same subjects/splits;
2. same ordered trial manifest;
3. same unimodal checkpoints/features/scores upstream of fusion;
4. same modality set and preprocessing;
5. same information stratum;
6. same quality variables when quality-aware methods are compared;
7. comparable tuning budget;
8. no final-test labels used for model selection or calibration;
9. missingness access declared explicitly;
10. result linked to immutable run provenance.

Any violation changes the interpretation from **fusion-mechanism effect** to a broader **system-level effect**.

## 8. Implementation status

The score and embedding contracts, canonical placeholders, quality symmetry, method registry and compatibility checks are now implemented and unit-tested. Token/local-feature evidence is intentionally deferred until the selected encoder/data regime makes its semantics concrete.
