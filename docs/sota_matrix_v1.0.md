# Verified SOTA Matrix v1.0 — Gate-4 Locked Positioning

**Study:** *Deep Learning Approaches for Multimodal Biometrics*  
**Cutoff:** 2026-09-05  
**Gate-4 status:** **PASS-POSITIONING**

This document freezes the scientific positioning used to proceed with implementation. It deliberately avoids priority language. The corresponding search/falsification record is `literature/gate4_search_log.md`.

## 1. What the literature already owns

The following ideas are established prior art and cannot be treated as DeepMM headline novelty:

1. multimodal improvement over unimodal systems;
2. score-, feature-, decision- and rank-level fusion;
3. quality-dependent and cost-sensitive biometric fusion;
4. missing-score and missing-modality fusion;
5. deep multimodal biometric fusion;
6. quality-aware and reliability-aware neural fusion;
7. attention/cross-attention/Transformer fusion;
8. dynamic routing/gating and reconstruction/generation under missing modalities;
9. fixed upstream biometric models followed by model-agnostic score/feature fusion;
10. biometric score calibration and likelihood-ratio-based fusion;
11. computational-efficiency analysis;
12. stress-dependent changes in which backbone/fusion configuration performs best.

## 2. Closest precedents

| Work | Most relevant established property | Remaining difference from DeepMM's locked experiment |
|---|---|---|
| Poh et al., TIFS/Pattern Recognition 2009–2010 | fixed supplied scores; many fusion systems; quality, cost, acquisition failures and missing values | pre-DL; not a matched classical-vs-deep fusion-family benchmark with modern calibration/rank stability |
| Mandasari et al., IET Biometrics 2014 | held-out biometric score calibration and `C_llr` | unimodal calibration study, not fusion-family comparison |
| Susyanto et al., IET Biometrics 2019 | calibrated likelihood-ratio multibiometric score fusion | no broad deep-family/stress/cost benchmark |
| Soleymani et al., T-BIOM 2022 | quality-aware deep multimodal fusion | proposed method, not representative family comparison under fixed evidence |
| FBR, CVPR 2024 | attention alignment/prompt tuning for flexible biometrics | end-to-end architecture; does not isolate fusion-family effect under common upstream evidence |
| Yang et al., Information Fusion 2025 (LUTBIO) | real-subject nine-modality fusion plus sensor-quality analysis | broad baseline/data study rather than matched classical+deep family benchmark |
| Ryu et al., IEEE Access 2025 | feature-vs-score fusion plus adaptation | virtual heterogeneous pairing; no controlled deep-family benchmark |
| SSFD-Net / HCMIG / UMR-Net / related 2025 work | explicit missing-modality modeling/reconstruction | proposed missingness methods, not Q2 family-ranking experiment |
| OU-MB, IEEE T-BIOM 2026 | **fixed modality-specific models**, mean/weighted score fusion and normalized feature concatenation on 1,099 true multimodal subjects | strongest philosophical precedent, but deliberately no trainable advanced fusion benchmark |
| AHFNet, IEEE TIFS 2026 | adaptive fusion under unreliable/missing modalities | dedicated architecture rather than matched family comparison |
| Yoon et al., Expert Systems 2026 | backbone/fusion configurations under FGSM/PGD; winners can depend on modality/stress | backbone and fusion vary together; no held-out calibration/missingness/cost/Pareto family contract |
| Tiong et al., Information Fusion 2026 | flexible biometrics and subset-aware evaluation/calibration agenda | survey/taxonomy, not the matched empirical benchmark |
| Alazawi et al., 2026 | controlled subject-disjoint backbone comparison plus multiple score-fusion rules and cost | controlled variable is the backbone, not the fusion mechanism |

## 3. Locked gap

The literature search did not locate a biometric-specific study that simultaneously satisfies all of the following:

- genuinely paired multimodal subjects;
- common/frozen unimodal evidence sufficient to isolate fusion effects;
- representative classical and deep **fusion mechanisms** as the controlled variable;
- identical subject-disjoint verification trials across methods;
- discrimination plus held-out score calibration;
- controlled modality degradation;
- explicit single-modality absence;
- comparable computational-cost measurement;
- paired/dependence-aware uncertainty;
- explicit rank/Pareto stability analysis across clean and stressed conditions.

This is an operational research gap, not an absence proof.

## 4. Locked contribution statement

> **DeepMM-Biometrics is a controlled multimodal-biometric verification study that holds unimodal evidence and trials fixed while comparing representative classical and deep fusion mechanisms, jointly evaluates discrimination, held-out biometric calibration, controlled degradation, single-modality absence and computational cost, and quantifies dependence-aware changes in family ranking and Pareto non-dominance across stress conditions.**

No stronger priority wording is authorized by Gate 4.

## 5. Q1–Q3 after Gate-4 closure

### Q1 — measurable DL contribution

Q1 must decompose the observed effect into:

- multimodal gain over **every** unimodal anchor and especially the best unimodal system;
- strongest classical fusion gain over the best unimodal system;
- deep-fusion gain over the strongest classical fusion **within the same information stratum**;
- the calibration/robustness/cost price paid for any discrimination gain.

A negative or mixed DL result is scientifically valid.

### Q2 — family trade-off

The confirmatory conclusion is not a universal winner. It is the clean-condition and aggregate-condition structure of non-dominated methods over locked dimensions, with within-stratum comparisons and uncertainty. Any scalar utility, if eventually required, must be preregistered before final test access.

### Q3 — stability under degraded or missing evidence

Q3 specifically tests whether clean-condition ordering survives:

- modality-A degradation;
- modality-B degradation;
- both degraded where scientifically meaningful;
- modality A absent;
- modality B absent;
- multiple preregistered severity levels.

Primary evidence includes Kendall tau-b/rank reversals, paired metric differences, subset-specific calibration, and probability of remaining Pareto non-dominated.

## 6. Confirmatory family boundary

The final family set remains data-dependent in its token/feature details, but the scientific strata are now locked.

### Score-input stratum

- C1 equal normalized score fusion;
- C2 validation-weighted score fusion;
- C3 regularized logistic score fusion;
- C5 classical quality-weighted score fusion;
- D1 compact nonlinear score fusion;
- D3-score learned quality/availability gate, if the same quality variables are available to C5.

### Embedding-input stratum

- C4 standardized/L2-controlled feature concatenation;
- D2 compact nonlinear feature fusion;
- D3-feature learned quality/availability gate.

### Token/local-feature stratum

One attention/Transformer representative is confirmatory only if the selected unimodal encoders expose meaningful comparable tokens/local features without giving that method a privileged upstream representation. Otherwise it moves to secondary Track II.

## 7. Gate-4 hard constraints carried into experiments

1. No method can receive richer upstream evidence and then be credited with a pure fusion gain over a poorer-input method.
2. No missing-modality architecture can define the whole novelty claim.
3. No calibration metric can be fitted on final-test labels.
4. No clean-condition winner can be declared universally best.
5. No family may be added to the confirmatory set after final-test inspection because it looked promising in pilot results.
6. A later equivalent benchmark finding reopens Gate 4.
7. A submission-time SOTA refresh is mandatory even though Gate 4 is now closed for implementation.
