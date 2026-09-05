# Verified SOTA Matrix v0.6 — Gate-4 Candidate Positioning

**Study:** *Deep Learning Approaches for Multimodal Biometrics*  
**Cutoff:** 2026-09-05

This version incorporates two findings that materially tighten the contribution boundary: (i) **OU-MB (IEEE T-BIOM 2026)** already evaluates multiple model-agnostic fusion strategies while keeping modality-specific recognition models fixed, and (ii) biometric calibration and calibrated multimodal score fusion have an established literature. This remains a falsification document, not a priority proof.

## 1. Closest controlled-fusion precedent: OU-MB

Xu et al., *OU-MB: The OU Multimodal Biometric Database and Its Performance Evaluation*, IEEE T-BIOM 2026, DOI `10.1109/TBIOM.2026.3710514`, is now a critical comparator for DeepMM.

The accepted author version reports:

- 1,099 subjects and eleven biometric modalities;
- genuinely multimodal collection from the same participant population;
- an intra-database protocol with disjoint training/test subjects for modalities without large external training resources;
- a unified fusion experiment on a 550-subject test set;
- fixed modality-specific recognition models serving as unimodal baselines;
- score-level mean fusion;
- two-modality weighted-sum score fusion with weights varied from 0 to 1 in steps of 0.05;
- feature-level fusion by per-modality L2 normalization, concatenation, another L2 normalization, then cosine matching;
- no additional projection layer or trainable fusion network in that fusion baseline;
- Rank-1 and EER, with FRR at FAR = 1%, 0.1%, and 0.01% in the broader verification evaluation.

### Consequence

DeepMM can **no longer present “holding modality models fixed and comparing score/feature fusion” as novel by itself**. OU-MB already provides a strong, large-scale, true-multimodal precedent for exactly that baseline philosophy.

What OU-MB does **not attempt**, according to its stated scope, is an exhaustive exploration of advanced fusion strategies. The paper explicitly identifies learning-based fusion, dynamic modality weighting and modality selection under varying acquisition conditions as future investigations. Its fusion section is designed as a database baseline, not as a controlled benchmark across representative deep fusion families.

Therefore DeepMM's remaining object is narrower:

> hold the unimodal evidence fixed **and extend the comparison from classical/model-agnostic fusion to representative deep fusion mechanisms**, then measure calibration, controlled degradation, missing modalities, compute, and uncertainty-aware ranking/Pareto stability on the same frozen verification evidence.

## 2. Calibration is an established biometric evaluation concept

Calibration cannot be presented as a new metric contribution.

Mandasari et al., *IET Biometrics* 2014, DOI `10.1049/iet-bmt.2013.0066`, evaluate linear and categorical score calibration in face recognition and use `C_llr` to separate verification/discrimination quality from calibration quality.

Susyanto et al., *IET Biometrics* 2019, DOI `10.1049/iet-bmt.2018.5106`, study likelihood-ratio-based **biometric score-level fusion** using parametric copulas and explicitly report `C_llr`, `C_llr_min`, and calibration loss behavior in their applications.

The 2026 *Information Fusion* flexible-biometrics survey further identifies **subset-conditional calibration and calibrated comparable operating points under variable modality subsets** as an open evaluation/deployment direction.

### Consequence

DeepMM's contribution is not “we use Cllr”. Its potential contribution is the **systematic use of calibration as one locked dimension in a controlled fusion-family benchmark**, including calibration transfer when quality or modality availability changes.

## 3. Updated novelty-falsification matrix

Legend: `Y` = directly established in the extracted source; `P` = partial/related; `—` = not established in the current extraction.

| Work | Fixed/common unimodal evidence while fusion varies | Representative classical + DL fusion families | Calibration measured | Controlled degradation | Missing modality | Cost | Uncertainty-aware rank/Pareto stability |
|---|---:|---:|---:|---:|---:|---:|---:|
| Poh et al., TIFS 2009 | Y at supplied-score level | P — 22 score-fusion systems, pre-DL | — | Y/device quality | Y/FTA-FTM | Y | — |
| Poh et al., TIFS 2010 | Y at supplied-score level | P | — | — | Y | — | — |
| Mandasari et al., IET Biom. 2014 | unimodal face system | — | Y | P/categorical image conditions | — | — | — |
| Susyanto et al., IET Biom. 2019 | Y at score-input level | P — several trained/non-trained score fusion comparators | Y (`C_llr`, `C_llr_min`) | — | — | — | — |
| Maity et al., JIPS 2020 | P | — | — | — | Y | — | — |
| Edwards & Hossain, TAI 2021 | P | P/serial fusion | — | — | P | Y | — |
| Soleymani et al., T-BIOM 2022 | — | — proposed family | — | Y/quality | P | — | — |
| FBR, CVPR 2024 | — end-to-end architecture | — proposed family | — | — | P/flexible recognition | — | — |
| SSFD-Net, DSP 2025 | — | — proposed family | — | — | Y | — | — |
| Pan et al., TIFS 2025 | — | — proposed family | — | P | Y | — | — |
| DIRS, KBS 2025 | — | — proposed family | — | P | — | P | — |
| MPAD, EAAI 2025 | — | — proposed family | — | — | — | Y | — |
| AHFNet, TIFS 2026 | — | — proposed family | — | Y | Y | — | — |
| **OU-MB, T-BIOM 2026** | **Y** | **P — classical/model-agnostic mean, weighted-sum, concat; no deep fusion-family benchmark** | — | — | — | — | — |
| Alazawi et al., 2026 | No — backbone is the main varying factor | P | — | — | — | Y | — |

The current extraction still contains **no row that closes all columns**. That supports continuation but is not a “first” proof.

## 4. Current contribution hypothesis

The strongest defensible hypothesis is now:

> **A controlled fusion-mechanism benchmark that extends fixed/common unimodal evidence beyond classical score/feature fusion to representative deep fusion families, while evaluating on identical frozen trials the joint trade-off among discrimination, biometric score calibration, controlled quality degradation, missing-modality operation and computational cost, and quantifying uncertainty-aware changes in rank and Pareto non-dominance across stress conditions.**

This is narrower and stronger than earlier formulations because it explicitly concedes that:

1. fixed-model classical fusion benchmarking already exists at scale (OU-MB);
2. quality/cost fusion benchmarking predates deep learning (Poh et al.);
3. missing-data fusion predates deep learning (Poh et al. 2010);
4. DL with missing modalities exists (Maity et al. 2020 and later work);
5. calibration and `C_llr` are established biometric concepts;
6. attention/gating/distillation/reconstruction/routing are existing architectural mechanisms;
7. a simple EER–latency Pareto plot is not novel.

## 5. Research-question implications

### Q1

The key contrast becomes **strongest classical/model-agnostic fusion versus representative deep fusion under the same unimodal evidence**, not merely multimodal versus unimodal.

### Q2

The “best compromise” must be multidimensional and uncertainty-aware. A two-dimensional accuracy/latency frontier or a single test EER is insufficient.

### Q3

The key question is not whether missing modalities can be handled. It is whether the **relative ordering and non-dominance structure of fusion families changes** when evidence quality or modality availability changes, with condition-aware calibration included.

## 6. Gate-4 remaining falsification searches

Before `PASS-POSITIONING`:

- search for any biometric paper that feeds **the same frozen embeddings/scores** into classical, MLP/gated, attention and Transformer-style fusion under one protocol;
- search explicitly for **calibration transfer/subset-conditional calibration** in multimodal/flexible biometrics;
- search for **rank reversal or Pareto stability** across biometric quality/missingness stress;
- inspect OU-MB final publication/update status and access conditions when dataset selection becomes active;
- finish reproducibility/code status for closest advanced-fusion methods;
- verify correction/retraction status of all headline references before Gate 9 lock.

## 7. Current Gate-4 decision

**Gate 4 = OPEN — close to a defensible positioning, but not yet PASS.**

**Je ne peux pas confirmer qu’aucun article existant ne réalise exactement le benchmark complet proposé.** The searches completed so far have not located one.
