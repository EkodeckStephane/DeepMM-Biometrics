# Research Protocol v0.1

## 1. Scope

The project evaluates deep-learning approaches for **multimodal biometric verification**. The empirical modality pair(s) will be locked only after a dataset-feasibility audit confirms real subject-level correspondence, adequate sample structure, lawful/research access, and sufficient data for fair model comparison.

Face + fingerprint is a priority candidate because it directly matches the motivating use case, but it is **not yet locked** as the sole empirical pair.

## 2. Experimental hierarchy

### Tier A — Unimodal anchors
For each selected modality, implement a strong, current, reproducible unimodal representation and verification pipeline. These define the best-single-modality reference required by Q1.

### Tier B — Classical multimodal baselines
At minimum:
- normalized score sum;
- tuned weighted score fusion using validation data only;
- feature concatenation followed by a non-deep classifier or metric model when dimensionality and sample size allow it.

### Tier C — Deep multimodal families
Candidates are selected by verified SOTA coverage, not by preference:
- learned score fusion;
- deep feature fusion;
- gated/quality-aware fusion;
- intermediate/bilinear fusion;
- attention/cross-attention;
- multimodal Transformer;
- explicit missing-modality learning if methodologically distinct.

## 3. Fair-comparison contract

All methods compared for a headline conclusion must share, as far as scientifically possible:
- identical subject-disjoint train/validation/test partitions;
- identical genuine/impostor pair-generation policy;
- identical test pairs;
- frozen preprocessing for shared inputs;
- fixed evaluation code;
- comparable hyperparameter-selection budget;
- no use of test labels for threshold selection, calibration, model selection, or early stopping.

Where architectures require distinct preprocessing or input resolution, the difference must be documented and treated as part of the model family rather than hidden.

## 4. Statistical design

Before the final test campaign, freeze:
- independent experimental unit;
- seed policy;
- number of seeds/runs;
- confidence-interval procedure;
- paired hypothesis tests for planned model comparisons;
- multiple-comparison correction;
- operating FAR points;
- robustness severities;
- missing-modality scenarios;
- Pareto dimensions and directionality.

Technical repeats are not independent biological subjects.

## 5. Primary metric families

### Discrimination
- Equal Error Rate (EER)
- ROC-AUC
- TAR at predeclared FAR values

### Calibration
- Expected Calibration Error (ECE)
- Brier score
- Negative log-likelihood
- reliability diagrams

### Robustness
- absolute and relative degradation from the clean condition
- area under a degradation-performance curve when defined consistently
- rank stability / rank reversal across severities

### Missing modalities
- complete input
- modality A absent
- modality B absent
- performance loss from the complete condition

### Efficiency
- trainable parameters
- MACs/FLOPs when the counting procedure is valid
- peak inference memory
- batch-1 latency under a frozen hardware/software environment

## 6. Q2 trade-off analysis

The primary trade-off result is Pareto-based. A method is dominated if another method is at least as good on every locked criterion and strictly better on at least one.

A scalar utility score is secondary and only admissible if defined before final testing, with sensitivity analysis over plausible weighting choices.

## 7. Dataset validity rules

Primary cross-modal evidence requires verified subject correspondence across modalities. A dataset assembled by pairing independent unimodal identities cannot establish that a model learned biologically grounded or acquisition-grounded cross-modal interactions.

Synthetic/chimeric pairings may be used only for explicitly labeled secondary analyses whose conclusions are bounded accordingly.

## 8. Robustness protocol

Corruptions must be modality-appropriate and applied at predeclared severity levels. Candidate image degradations:
- Gaussian or defocus blur;
- additive noise;
- downsampling/resolution loss;
- JPEG compression;
- under/over-exposure or contrast change;
- localized occlusion.

No corruption should be included merely because it makes one architecture look better.

## 9. Reproducibility

The repository will preserve:
- code and tests;
- environment lock/requirements;
- dataset acquisition instructions and licenses, not restricted raw datasets;
- immutable split manifests;
- experiment configs;
- seeds;
- raw scores and aggregate outputs;
- scripts regenerating all tables and figures;
- model/checkpoint hashes where storage permits;
- claim-to-evidence traceability.

## 10. Stop conditions before expensive training

No large training campaign starts until all are true:
1. SOTA matrix is current enough to justify model families and baselines.
2. Primary dataset identity correspondence is verified.
3. Dataset access and redistribution constraints are documented.
4. Split and pair-generation code passes leakage tests.
5. Metrics and statistical plan are frozen.
6. Q1/Q2/Q3 each maps directly to planned experiments.

## 11. Manuscript discipline

The final article answers the scientific questions. It will not narrate repository construction, debugging history, failed preliminary runs, or internal audit chronology except where a failure mode is itself scientific evidence.
