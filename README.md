# DeepMM-Biometrics

Research repository for **Deep Learning Approaches for Multimodal Biometrics**.

## Scientific objective

This project is a clean-room restart. It does **not** reuse previous code, models, experimental results, protocol choices, or proposed architectures from earlier work.

The study is governed by three fixed research questions:

- **Q1.** What does deep learning actually contribute to multimodal biometrics compared with classical fusion approaches and unimodal systems?
- **Q2.** Among deep-learning multimodal-fusion families, which provides the best trade-off among performance, robustness, calibration, and computational cost?
- **Q3.** Does the best approach remain the same when input quality degrades or when one modality is missing?

## Research principles

1. The scientific questions precede any architecture or software artifact.
2. No architecture is assumed to be best in advance.
3. Classical, unimodal, and deep multimodal baselines must be evaluated under matched protocols.
4. Negative results and ranking reversals are retained.
5. Dataset identity correspondence must be verified; synthetic/chimeric identity pairing cannot be used as primary evidence for learned cross-modal interactions.
6. Performance, robustness, calibration, and efficiency are evaluated separately before any global trade-off analysis.
7. Statistical units, confidence intervals, paired comparisons, and multiplicity handling are defined before the final campaign.
8. Every headline claim must map to direct evidence.
9. References, datasets, code, raw results, tables, figures, and manuscript claims must remain traceable and mutually consistent.

## Initial repository structure

```text
DeepMM-Biometrics/
├── README.md
├── docs/
│   ├── research_questions.md
│   ├── research_protocol.md
│   ├── dataset_feasibility.md
│   ├── sota_seed.md
│   └── q1_gate_matrix.md
├── configs/
├── data/
│   ├── README.md
│   ├── manifests/
│   └── splits/
├── src/
│   ├── unimodal/
│   ├── classical_fusion/
│   ├── deep_fusion/
│   ├── attention_fusion/
│   ├── transformer_fusion/
│   ├── robustness/
│   ├── missing_modalities/
│   └── evaluation/
├── experiments/
├── tests/
├── results/
│   ├── raw/
│   └── processed/
├── figures/
├── manuscript/
│   ├── article/
│   └── thesis/
└── audits/
```

Directories will be instantiated when they contain versionable files. Raw biometric datasets will not be committed unless their licenses explicitly permit redistribution.

## Current status

**Research reset / protocol-design stage.**

No experimental result is currently claimed. Dataset selection, model-family selection, evaluation metrics, and statistical analysis are being locked before implementation.
