# DeepMM-Biometrics

Research repository for **Deep Learning Approaches for Multimodal Biometrics**.

## Scientific objective

This project is a clean-room restart. It does **not** reuse previous code, models, experimental results, protocol choices, or proposed architectures from earlier work.

The study is governed by three fixed research questions:

- **Q1.** What does deep learning actually contribute to multimodal biometrics compared with classical fusion approaches and unimodal systems?
- **Q2.** Among deep-learning multimodal-fusion families, which provides the best trade-off among performance, robustness, calibration, and computational cost?
- **Q3.** Does the best approach remain the same when input quality degrades or when one modality is missing?

## Primary scientific design

The headline experiment is a **controlled fusion-mechanism benchmark**, not a competition between named end-to-end architectures.

### Track I — controlled fusion benchmark (primary)

Strong unimodal evidence is shared/frozen wherever scientifically possible. Eligible fusion families receive the same person-disjoint splits, ordered trial lists, preprocessing outputs and matched development/tuning budget. The independent variable is the **fusion mechanism**.

### Track II — full-system optimization (secondary)

Selected systems may be fine-tuned end to end under a matched budget to estimate attainable full-system performance. Track II cannot be used to attribute an encoder/backbone gain to the fusion mechanism.

## Gate-4-locked scientific positioning

Gate 4 is closed at **PASS-POSITIONING (2026-09-05)**. The completed falsification search did not locate a biometric-specific study that simultaneously keeps upstream unimodal evidence controlled, compares representative classical and deep fusion mechanisms on identical person-disjoint verification evidence, jointly measures discrimination, held-out calibration, controlled degradation, modality absence and compute, and quantifies dependence-aware rank/Pareto stability across conditions.

This is an operational research gap, **not a proof of uniqueness**. No `first`, `only`, or universal `best` claim is authorized. Gate 4 is reopened if a later equivalent benchmark is found, and a submission-time SOTA refresh is mandatory.

The locked contribution statement is:

> **DeepMM-Biometrics is a controlled multimodal-biometric verification study that holds unimodal evidence and trials fixed while comparing representative classical and deep fusion mechanisms, jointly evaluates discrimination, held-out biometric calibration, controlled degradation, single-modality absence and computational cost, and quantifies dependence-aware changes in family ranking and Pareto non-dominance across stress conditions.**

See `docs/sota_matrix_v1.0.md` and `literature/gate4_search_log.md`.

## Controlled comparator families and information strata

Fusion-only claims are made **within information strata**.

### Score-input stratum

- C1 equal normalized score fusion;
- C2 validation-weighted score fusion;
- C3 regularized logistic score fusion;
- C5 classical quality-weighted score fusion;
- D1 compact nonlinear score fusion;
- D3S learned quality/availability-aware score gate.

### Embedding-input stratum

- C4 controlled feature concatenation;
- D2 compact nonlinear feature fusion;
- D3F learned quality/availability-aware feature gate.

### Token/local-feature stratum

One attention/Transformer representative is confirmatory only if the selected frozen upstream encoders expose meaningful comparable local/token features. Otherwise it is Track II/exploratory.

`src/deepmm/fusion/contracts.py` enforces these information tiers, explicit availability masks and quality-access rules. A richer-input method cannot be credited with a pure fusion gain over a poorer-input method.

## Neural implementation status

The framework-independent search/budget contract lives in `src/deepmm/fusion/neural_contracts.py`. A CPU-testable PyTorch backend now implements the four currently eligible neural families in `src/deepmm/fusion/neural_torch.py`:

- D1 `ScoreMLPFusion`;
- D2 `FeatureFusionMLP`, using one shared enrollment/probe encoder followed by cosine verification;
- D3S `ScoreQualityGate`, with explicit quality/availability-conditioned masked weights;
- D3F `FeatureQualityGate`, with per-modality projections and joint-availability gating.

Optimizer choice, final hidden widths, final training epochs and the final search grid are **not** hard-coded into these model classes. Those choices remain governed by the preregisterable training-budget contract and must be frozen only after the primary data/encoder dimensionality is known.

PyTorch is an optional dependency (`.[neural]`) and has its own CPU CI workflow. The ordinary non-neural scientific infrastructure remains usable without PyTorch.

## Evaluation and reproducibility infrastructure

The repository implements/tests infrastructure for:

- empirical EER and ROC-convex-hull EER as distinct quantities;
- ROC-AUC and conservative TAR@FAR;
- Brier score, NLL, ECE, `C_llr`, `C_llr_min`, and calibration loss;
- held-out affine logistic score-to-LLR calibration;
- subject/split leakage checks and immutable split/trial/score hashing;
- strict trial-to-score ordering and provenance manifests;
- dependence-aware subject-cluster bootstrap for compatible subject-centric trials;
- paired cluster-level randomization tests and Holm correction;
- Pareto non-dominance and bootstrap dominance probability;
- Kendall tau-b and pairwise rank-reversal analysis;
- explicit score/embedding evidence contracts;
- deterministic missing-modality masks/fallback utilities;
- hashable preregistration-style clean/corruption/missing stress plans;
- matched hardware/batch/precision/scope cost-measurement records and raw latency retention;
- metadata-only dataset archive manifests with person/instance/session/capture structure;
- person-level partition leakage detection for multi-instance datasets;
- an end-to-end synthetic smoke harness covering development-only fitting, held-out calibration transfer, clustered uncertainty, Pareto analysis and clean-vs-stress ranking logic.

The synthetic harness is **CI/debug evidence only** and can never be used as a scientific result.

A one-way subject-cluster analysis is **not** assumed valid for dense symmetric all-vs-all impostor trials. The final dependence-aware resampling method is locked after the dataset/trial construction is known.

## Missingness and robustness discipline

Missing evidence is represented explicitly; `NaN`, infinity and sentinel values are forbidden as hidden modality-availability signals. Unavailable score/embedding slots use canonical zero serialization placeholders plus explicit availability masks.

Final corruption operators and severities remain data/modality dependent. Their condition IDs, targets, parameters and severity order will be frozen and hashed before final-test inspection. Missing-modality handling is a second experimental axis rather than automatically a separate fusion family.

## Dataset and completed V1 boundary

V1 is complete on the official public NUPT-FPV fingerprint/finger-vein subset:
800 images associated with 20 public biometric-instance identifiers. Those
identifiers are not described as independent people because the public material
does not establish their mapping to the volunteers in the complete archive.

The full 33,600-image archive remains V2. It requires verified person-to-finger
mapping, person-disjoint partitions, and a dependence-aware inferential design.
V1 therefore reports bounded public-instance point estimates rather than
person-population confidence intervals or hypothesis tests. Restricted raw
biometric images are not committed.

## Literature traceability

- `literature/sota_registry.csv` contains the machine-readable verified SOTA registry.
- `literature/references.bib` contains synchronized BibTeX metadata.
- `literature/gate4_search_log.md` records the frozen Gate-4 falsification search.
- `docs/sota_matrix_v1.0.md` records the locked positioning and forbidden novelty claims.
- `scripts/validate_literature.py` checks registry/BibTeX consistency offline.
- `scripts/validate_doc_references.py` rejects DOI mentions in project documentation that are not registered.

“Code not located” means that the targeted search did not locate an official implementation; it is never treated as proof that no implementation exists.

## Repository map

```text
DeepMM-Biometrics/
├── .github/workflows/        # standard + optional PyTorch CPU CI
├── data/                     # metadata schema only; no raw restricted biometrics
├── docs/                     # protocol, SOTA, Gates, dataset/access contracts
├── literature/               # verified registry + BibTeX + Gate-4 search log
├── manuscript/sections/      # V1 Results and Discussion/limitations source
├── results/v1/               # generated CSV, LaTeX tables and PGFPlots figures
├── scripts/                  # literature, documentation and dataset-manifest audits
├── src/deepmm/
│   ├── calibration/          # held-out score calibration
│   ├── evaluation/           # cost contract + synthetic pipeline smoke harness
│   ├── fusion/               # classical + neural fusion, evidence/missingness contracts
│   ├── metrics/              # discrimination + calibration metrics
│   ├── robustness/           # frozen stress-condition contract
│   ├── stats/                # bootstrap, paired inference, Pareto/rank analysis
│   └── validation/           # dataset/split/trial/hash/provenance controls
├── tests/                    # scientific software validation
├── pyproject.toml
└── README.md
```

## Current Q1-Gate status

| Gate | Status |
|---|---|
| G1 — scientific object | **PASS-DESIGN** |
| G2 — claim/evidence alignment | **PASS-V1 (bounded claims)** |
| G3 — final scientific narrative | **V1 Results/Discussion drafted; full article open** |
| G4 — current SOTA / novelty | **PASS-POSITIONING — 2026-09-05** |
| G5 — experimental validity | **PASS-V1 EXECUTION / population inference excluded** |
| G8 — reproducibility | **PASS-V1 EVIDENCE PACKAGE** |
| G9 — bibliography/editorial hygiene | **OPEN / machine consistency active** |
| G10 — submission readiness | **NO-GO** |

## V1 result package

The final campaign ran on commit
`480c1f4e67757e4789b270b5ea12ecd0e9eac16b` and contains 15 conditions with
4,000 trials each. C4 has the lowest clean ROCCH-EER (0.2115); D2 is the best
deep family (0.2865) but does not beat C4. C4 remains first in all 12 frozen
blur/contrast conditions. Under the M0 missing-modality policy, fusion methods
tie because each receives the same available unimodal evidence.

Run `PYTHONPATH=src python scripts/audit_v1_final_evidence.py` to validate the
committed provenance/trial/score chain and
`PYTHONPATH=src python scripts/generate_v1_result_assets.py` to regenerate the
tables and PGFPlots figures. See `results/v1/README.md` and
`docs/v1_claim_code_data_audit.md`.

**GO:** complete the full article around the bounded V1 findings and prepare V2
as an independent replication/generalization stage. **NO-GO:** post-final V1
retuning, universal DL-superiority wording, person-population inference, or
claims about the complete archive from this subset.
