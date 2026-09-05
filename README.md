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

Strong unimodal evidence is shared/frozen wherever scientifically possible. Eligible fusion families receive the same subject-disjoint splits, trial lists, preprocessing outputs, validation/test evidence and matched tuning budget. The independent variable is the **fusion mechanism**.

### Track II — full-system optimization (secondary)

Selected systems may be fine-tuned end to end under a matched budget to estimate attainable full-system performance. Track II cannot be used to attribute an encoder/backbone gain to the fusion mechanism.

## Current controlled comparator families

Classical baseline infrastructure already includes:

- equal normalized score fusion;
- simplex-weighted score fusion selected on development data;
- regularized logistic score fusion;
- standardized equal-energy feature concatenation;
- classical quality-aware score fusion.

Deep families remain SOTA-governed and are not yet frozen: deep score fusion, deep feature fusion, quality-aware/gated fusion, and one rigorously justified attention/Transformer representative form the current minimum candidate set. Missing-modality handling is a second experimental axis rather than automatically a separate fusion family.

## Evaluation contract

The repository already implements/tests infrastructure for:

- empirical EER and ROC-convex-hull EER as distinct quantities;
- ROC-AUC and conservative TAR@FAR;
- Brier score, NLL, ECE, `C_llr`, `C_llr_min`, and calibration loss;
- subject/split leakage checks and immutable split/trial hashing;
- dependence-aware subject-cluster bootstrap for compatible subject-centric trials;
- paired cluster-level randomization tests and Holm correction;
- Pareto non-dominance and bootstrap dominance probability;
- Kendall tau-b and pairwise rank-reversal analysis.

A one-way subject-cluster analysis is **not** assumed valid for dense symmetric all-vs-all impostor trials. The final dependence-aware resampling method will be locked after the dataset/trial construction is known.

## SOTA and novelty discipline

Gate 4 remains open. Strong prior work already covers quality-dependent fusion, cost-sensitive fusion, quality-aware deep fusion, attention/flexible biometrics, missing-modality robustness, and controlled backbone comparisons. Therefore none of these mechanisms alone is treated as the contribution.

The current contribution candidate is the matched comparison of **fusion mechanisms themselves**, followed by multidimensional performance–robustness–calibration–cost analysis and family-ranking/Pareto stability under degradation and modality absence.

This remains a working novelty hypothesis until the systematic current-work falsification search is closed. No `first`, `only`, or universal `best` claim is permitted at this stage.

## Literature traceability

- `literature/sota_registry.csv` contains the current machine-readable verified SOTA seed.
- `literature/references.bib` contains synchronized BibTeX metadata.
- `scripts/validate_literature.py` checks DOI/key/year/title consistency offline.
- `docs/sota_search_log_2026-09-05.md`, `docs/sota_matrix_v0.4.md`, and `docs/sota_reproducibility_audit.md` record the novelty-boundary and reproducibility searches.

“Code not located” means that the targeted search did not locate an official implementation; it is never treated as proof that no implementation exists.

## Dataset policy

Dataset locking is intentionally deferred. Directly and lawfully accessible real multimodal datasets are preferred, but accessibility cannot override subject-level identity correspondence, leakage-free splitting, adequate repeated samples, and trial/statistical feasibility. Chimeric identities are not admissible as primary evidence for learned cross-modal interaction.

Restricted raw biometric data will not be committed to this repository unless redistribution terms explicitly allow it.

## Repository map

```text
DeepMM-Biometrics/
├── .github/workflows/        # CI
├── docs/                     # research protocol, SOTA, Gates, statistical contracts
├── literature/               # verified registry + BibTeX
├── scripts/                  # validation/regeneration utilities
├── src/deepmm/
│   ├── fusion/               # controlled classical baselines
│   ├── metrics/              # discrimination + calibration metrics
│   ├── stats/                # bootstrap, paired inference, Pareto/rank analysis
│   └── validation/           # split/leakage/hash controls
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
| G4 — current SOTA / novelty | **OPEN — critical, boundary narrowed** |
| G5 — experimental validity | **DESIGN-READY / dataset-dependent lock open** |
| G8 — reproducibility | **ACTIVE** |
| G9 — bibliography/editorial hygiene | **OPEN / source audit running** |
| G10 — submission readiness | **NO-GO** |

No scientific performance result is currently claimed. The project is ready for continued Gate-4 closure, baseline/SOTA reproducibility audit, experiment-schema preparation, and dataset feasibility work when needed; it is **not** yet ready for a final training campaign or article conclusion.
