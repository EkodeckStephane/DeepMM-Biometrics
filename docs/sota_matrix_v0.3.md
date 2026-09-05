# Verified SOTA Matrix v0.3

**Cutoff:** 2026-09-05.

This working matrix extends v0.2 with the closest 2024-2026 precedents to Q1-Q3. It is still not the final systematic review and therefore cannot support a priority claim such as “first”.

## 1. Closest precedents to the DeepMM-Biometrics questions

| Work | Venue/year | What it already establishes | What it does not settle for our Q1-Q3 |
|---|---|---|---|
| Edwards & Hossain, DOI `10.1109/TAI.2021.3064003` | IEEE TAI, 2021 | Deep Siamese scoring combined with serial multibiometric fusion on true multimodal data; explicitly studies authentication performance against acquisition/decision cost. | Does not compare representative deep fusion families under one frozen fusion-only benchmark; missing-modality robustness and score calibration are not the joint comparison object. |
| Soleymani et al., DOI `10.1109/TBIOM.2021.3131664` | IEEE T-BIOM, 2022 | Quality-aware deep multimodal recognition; weakly supervised modality/sample quality weighting; face/iris/fingerprint; verification metrics include AUC/EER/TAR@FAR. | Does not establish that quality-aware fusion is globally best across deep score, deep feature, attention/Transformer and strong classical fusion under a matched budget. |
| Ren et al., DOI `10.1109/TIFS.2022.3175599` | IEEE TIFS, 2022 | Real paired fingerprint-finger-vein dataset and deep multimodal benchmark with two acquisition sessions. | Benchmark is not designed to jointly rank broad fusion families by discrimination, calibration, degradation robustness, missingness and compute. |
| A. El_Rahman & Alluhaidan, DOI `10.1371/journal.pone.0291084` | PLOS ONE, 2024 | Directly compares CNN and traditional classifiers with parallel/serial multimodal systems and different fusion levels. | Primary multimodal evidence uses a virtual/chimeric ECG-fingerprint construction; therefore it cannot close our real-subject matched-fusion question. It also does not jointly evaluate calibration, controlled missing modalities and cost. |
| Fan et al., DOI `10.1109/TSMC.2024.3382877` | IEEE TSMC: Systems, 2024 | Adaptive weighted multimodal hand recognition with recognition-time efficiency as an explicit design objective. | Not a broad family-level fusion benchmark and not a calibration/missingness study. |
| Lu, Wu & Bao, DOI `10.1016/j.engappai.2025.110865` | Engineering Applications of AI, 2025 | Face-fingerprint multilevel spatial/channel attention plus knowledge distillation; explicitly targets model compression/resource-constrained deployment and evaluates on two real multimodal datasets. | Does not answer whether attention/distillation is Pareto-superior to representative deep score, feature, gating and Transformer families under one matched protocol. |
| Zheng et al., DOI `10.1016/j.patrec.2025.06.017` | Pattern Recognition Letters, 2025 | SSM encoders + cross-modal attention + contrastive alignment for PPG-fingerprint verification; single- and dual-session evaluation. | Strong architecture paper rather than a broad fusion-family benchmark. |
| Chitrapu et al., Scientific Reports article 14244, 2026 | Scientific Reports, 2026 | Face-fingerprint trust-adaptive feature fusion using confidence/quality weighting, MobileNetV2 and channel attention. | The paper explicitly states that CASIA-FaceV5 and CASIA-FingerprintV5 have no official same-subject correspondence, so this cannot serve as decisive evidence for biologically matched cross-modal fusion. Calibration is not the central evaluation axis. |
| Rajkumar & Yuvasini, DOI `10.1007/s10044-026-01739-3` | Pattern Analysis and Applications, 2026 | Reliability-aware attention/gating under non-stationary sensing, including low illumination, acoustic noise and missing modalities; compares against unimodal/static/heuristic fusion. | Very close to Q3, but still a proposed adaptive architecture rather than a matched comparison of representative fusion families with explicit calibration and computational Pareto analysis. |
| Wu et al., DOI `10.1109/TIFS.2026.3700801` (AHFNet) | IEEE TIFS, 2026 | Directly addresses unreliable and missing multimodal hand biometrics with adaptive hybrid fusion. | Makes Q3 novelty highly competitive: our contribution cannot be “we handle missing/degraded modalities”. We must study **ranking stability across families**, not propose another missingness architecture as the headline claim. |
| Alazawi et al., DOI `10.24017/science.2026.2.2` | Kurdistan Journal of Applied Research, 2026 | Controlled comparison of several deep backbones for multimodal biometric verification with score-level fusion. | Compares backbone architectures while fusion level is fixed; it does not isolate and compare multiple fusion mechanisms across the four Q2 dimensions. |
| Tiong et al., DOI `10.1016/j.inffus.2026.104267` | Information Fusion, 2026 | Defines flexible biometrics for variable modality subsets; emphasizes graceful degradation, subset-conditional calibration and standardized reporting under partial/asymmetric evidence. | This substantially raises the bar for Q3. Our benchmark must include subset-specific calibration and cannot treat simple missing-modality accuracy as a novel contribution. |
| Xu et al., DOI `10.1109/TBIOM.2026.3710514` | IEEE T-BIOM, 2026 | OU-MB: 1,099 subjects and 11 modalities; broad modern multimodal data resource. | Dataset paper rather than a fusion-family comparison, but a potentially important future validation resource subject to access and protocol feasibility. |

## 2. Implications for Q1

Q1 cannot be framed as “does deep learning improve multimodal biometrics?” in an unconditional sense. The literature already contains cases where deep multimodal models improve over selected traditional or unimodal baselines, as well as cases where the strongest single modality is already very competitive.

The defensible Q1 contribution is therefore a **matched decomposition**:

- representation gain from strong unimodal encoders;
- fusion gain relative to the best unimodal modality;
- nonlinear/deep fusion gain relative to strong classical score/feature fusion;
- the cost paid for that gain;
- conditions under which the gain disappears or reverses.

## 3. Implications for Q2

The literature already separately studies quality-aware fusion, attention, distillation, adaptive weighting, serial cost-sensitive fusion and Transformer-like interaction. Therefore no single one of these mechanisms can be presented as novel merely because it is implemented.

Q2 remains scientifically useful only if the benchmark compares representative families under a **common fusion-only contract**, ideally with shared frozen unimodal encoders and the same trials/tuning budget.

Primary family set remains:

1. equal normalized score sum;
2. validation-weighted score fusion;
3. logistic score fusion;
4. deep score MLP;
5. deep feature fusion;
6. quality-aware/gated fusion;
7. one rigorously justified cross-attention/Transformer representative.

Optional bilinear or dedicated reconstruction families are secondary unless Gate-4 closure shows they are indispensable to the current taxonomy.

## 4. Implications for Q3

Q3 is no longer novel simply because it includes degraded or missing modalities. Recent TIFS and Pattern Analysis & Applications work already addresses those problems directly, and the 2026 Information Fusion survey formalizes flexible biometrics and explicitly calls for graceful degradation and subset-conditional calibration.

The distinctive question must be:

> **Does the ranking and Pareto structure of fusion families remain stable when quality or modality availability changes?**

Required Q3 evidence is therefore comparative and family-level:

- clean-condition family ranking;
- rank correlation across stress conditions;
- statistically supported rank reversals;
- subset-specific calibration rather than one global calibration score;
- whether the clean-condition Pareto-optimal family remains non-dominated;
- the robustness gained per unit of additional compute.

## 5. Revised provisional gap

The previous broad wording is narrowed.

**Provisional gap after the 2026 search pass:**

> We have not yet found a biometric-specific study that isolates the **fusion mechanism** by feeding representative classical and deep fusion families with matched unimodal evidence and then evaluates the same frozen verification trials across discrimination, biometric score calibration, controlled quality degradation, arbitrary single-modality absence, and computational cost, with explicit statistical analysis of family-rank stability and Pareto reversals.

This remains a working hypothesis, not a priority claim.

## 6. Gate-4 falsification conditions

Gate 4 must remain OPEN if a current paper is found that simultaneously satisfies all of the following:

1. real subject-level multimodal pairing;
2. representative classical + deep fusion families, not merely multiple backbones;
3. common encoders or another convincing control separating encoder quality from fusion quality;
4. same subject-disjoint trials and comparable tuning budget;
5. discrimination + score calibration + controlled degradation + missing-modality evaluation + compute;
6. direct analysis of rank stability/reversal or equivalent multi-condition family comparison;
7. uncertainty/statistics sufficient for the headline family ranking.

If such a paper exists, the article contribution must be repositioned before final experiments.

## 7. References requiring full-text extraction before final Gate-4 lock

Highest priority for deeper extraction:

- Soleymani et al., T-BIOM 2022, quality-aware multimodal recognition;
- Edwards & Hossain, IEEE TAI 2021, deep serial fusion and cost;
- Wu et al., TIFS 2026, AHFNet;
- Tiong et al., Information Fusion 2026, flexible biometrics survey;
- Rajkumar & Yuvasini, Pattern Analysis and Applications 2026;
- Lu et al., EAAI 2025, MPAD;
- Alazawi et al., 2026 cross-architecture evaluation;
- the closest recent benchmark papers discovered in the remaining T-BIOM/TIFS/IJCB search.

No manuscript novelty wording is frozen before this extraction is complete.
