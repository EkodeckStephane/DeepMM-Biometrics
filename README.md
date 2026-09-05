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

## Dataset direction — P1 access candidate selected

**NUPT-FPV (fingerprint + finger vein) is the current P1 access candidate. It is not yet the locked primary dataset.** The public project documentation reports 140 human volunteers, six fingers per volunteer, 20 acquisitions per finger across two sessions and 33,600 images in total. The final scientific lock requires obtaining the official archive and independently auditing its identity/session/capture topology and access terms.

For a multi-finger dataset, the outer biological grouping unit is the **human volunteer**, not the finger. All fingers from one volunteer must remain in the same train/development/calibration/test partition. `person_id` and nested `instance_id` are therefore separate fields in the dataset manifest contract.

Current fallback/generalization order is recorded in `docs/dataset_feasibility.md` and `docs/dataset_lock_decision_v0.1.md`. Restricted raw biometric data are never committed unless source terms explicitly permit redistribution.

After lawful access, a local metadata manifest can be audited with `scripts/audit_dataset_manifest.py`; `data/manifest_template.csv` documents the required schema. The tool validates multimodal/session completeness and emits a deterministic dataset-manifest SHA-256 without reading biometric pixels.

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
| G2 — claim/evidence alignment | **PASS-DESIGN / evidence open** |
| G3 — final scientific narrative | **rule fixed** |
| G4 — current SOTA / novelty | **PASS-POSITIONING — 2026-09-05** |
| G5 — experimental validity | **P1 ACCESS CANDIDATE SELECTED / archive & data lock open** |
| G8 — reproducibility | **ADVANCED INFRASTRUCTURE / real evidence open** |
| G9 — bibliography/editorial hygiene | **OPEN / machine consistency active** |
| G10 — submission readiness | **NO-GO** |

## Immediate next boundary

**GO:** obtain NUPT-FPV through the official research route, audit its delivered archive, complete dataset-agnostic neural training infrastructure, freeze person-disjoint development/calibration/test logic after topology verification, and run non-final pilots without final-test access.

**NO-GO:** confirmatory final-test model selection, post-hoc family additions, treating fingers as independent human subjects, unvalidated dense-trial inference, test-fitted calibration, token/Transformer information privilege, or final corruption severities before the real data/protocol lock.

No scientific biometric performance result is currently claimed.
