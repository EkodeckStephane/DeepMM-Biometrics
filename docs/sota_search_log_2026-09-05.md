# Gate-4 SOTA Search Log — 2026-09-05

**Study:** *Deep Learning Approaches for Multimodal Biometrics*

This log records the targeted novelty-falsification pass. It is not a PRISMA claim of exhaustive retrieval. Its purpose is to make the current Gate-4 reasoning inspectable and to preserve what still needs to be searched.

## Search targets

The pass combined variants of:

- `multimodal biometric`, `multibiometric`, `biometric fusion`;
- `deep learning`, `feature fusion`, `score fusion`, `attention`, `cross-attention`, `Transformer`, `quality-aware`;
- `missing modality`, `unreliable modality`, `flexible biometrics`, `degradation`;
- `calibration`, `C_llr`, `uncertainty`, `computational cost`, `efficiency`;
- `benchmark`, `controlled comparison`, `subject-disjoint`, `fusion strategy`.

Priority sources/venues were IEEE TIFS, IEEE T-BIOM, IJCB, CVPR, Pattern Recognition/Pattern Recognition Letters, Information Fusion, Engineering Applications of AI, IET Biometrics and direct publisher/institutional records.

## High-value retained records from this pass

### Poh et al. — TIFS 2009

**Title:** *Benchmarking Quality-Dependent and Cost-Sensitive Score-Level Multimodal Biometric Fusion Algorithms*  
**DOI:** `10.1109/TIFS.2009.2034885`

Why retained: direct benchmark precedent. The BioSecure campaign received 22 fusion-system submissions and explicitly evaluated changing quality, failure-to-acquire/failure-to-match and acquisition/computation/hardware cost. This prevents us from treating quality/cost-aware benchmarking as a new idea.

### Soleymani et al. — T-BIOM 2022

**Title:** *Quality-Aware Multimodal Biometric Recognition*  
**DOI:** `10.1109/TBIOM.2021.3131664`

Why retained: strong quality-aware deep-fusion prior art over face/iris/fingerprint. It occupies the “learned quality weighting” territory that a DeepMM gated model might otherwise overclaim.

### Tiong et al. — CVPR 2024

**Title:** *Flexible Biometrics Recognition: Bridging the Multimodality Gap through Attention Alignment and Prompt Tuning*

Why retained: top-venue flexible/cross-modality biometric framework using Multimodal Fusion Attention and Multimodal Prompt Tuning in a Vision Transformer. Code is publicly referenced by the CVF page. Attention/flexible recognition therefore cannot be the headline novelty.

### Gu et al. — IJCB 2025

**Title:** *A Mutual Distillation Learning Framework for Multimodal Biometric Recognition with Uncertain Missing Modality*  
**DOI:** `10.1109/IJCB65343.2025.11410967`

Why retained: direct missing-modality biometric work using missing-sample augmentation and mutual knowledge distillation. Missing-modality training is an established research direction, not a unique DeepMM contribution.

### Zheng et al. — Pattern Recognition Letters 2025

**Title:** *Multimodal biometric authentication using camera-based PPG and fingerprint fusion*  
**DOI:** `10.1016/j.patrec.2025.06.017`

Why retained: current cross-modal attention and contrastive-alignment example for biometric verification, including single- and dual-session evaluation.

### Wu et al. — TIFS 2026

**Title:** *AHFNet: An Adaptive Hybrid Fusion Network for Robust Multimodal Hand Biometrics under Unreliable Modalities*  
**DOI:** `10.1109/TIFS.2026.3700801`

Why retained: direct robust fusion under partial missingness and poor modality quality. It sharply narrows Q3 novelty to comparative family-ranking stability rather than architecture-level robustness.

### Tiong et al. — Information Fusion 2026

**Title:** *From unimodal to flexible: A survey of generalized biometric systems*  
**DOI:** `10.1016/j.inffus.2026.104267`

Why retained: formalizes flexible biometrics and emphasizes variable modality sets, deployment-realistic benchmarks, subset-conditional calibration and standardized reporting. Q3 must meet this evaluation bar.

### Alazawi, Habeeb & Almaliki — 2026

**Title:** *Cross-Architecture Evaluation of Deep Learning Models for Multimodal Biometric Verification with Score-Level Fusion*  
**DOI:** `10.24017/science.2026.2.2`

Why retained: currently the closest **controlled-comparison** precedent found. It uses subject-disjoint LUTBIO/XJTU evaluation, ResNet50/EfficientNetV2-S/Swin-T, EER/AUC/TAR@FAR=0.1%, five score-level fusion rules, and reports selected confidence/effect-size and cost information. Crucially, it holds many protocol factors fixed to study **backbone architecture**; score-level fusion remains the multimodal combination level. DeepMM-Biometrics therefore protects the inverse controlled experiment: hold unimodal evidence fixed and vary **fusion mechanism**.

### Es-Sobbahi, Radouane & Nafil — IET Biometrics 2025

**Title:** *Multimodal Biometrics: A Review of Handcrafted and AI-Based Fusion Approaches*  
**DOI:** `10.1049/bme2/5055434`

Why retained: systematic review of 29 peer-reviewed studies integrating traditional and AI-based approaches across fusion levels. The authors explicitly identify difficulty comparing methods/best practices because techniques and fusion levels are not integrated under common comparisons. This supports—but does not prove—the relevance of a controlled family-level benchmark.

## Statistical-methodology record

### Bolle, Ratha & Pankanti — CVIU 2004

**Title:** *Error analysis of pattern recognition systems—the subsets bootstrap*  
**DOI:** `10.1016/j.cviu.2003.08.002`

Why retained: biometric-specific dependence-aware bootstrap precedent. The subsets bootstrap samples structured blocks/subsets to account for dependent match scores and derives uncertainty for ROC/FAR/FRR/EER. This is the basis for rejecting an IID trial bootstrap in the final benchmark.

## Current falsification outcome

The search has located strong precedents for each component separately:

- controlled benchmarking;
- score-fusion benchmarking;
- quality-aware fusion;
- cost-sensitive fusion;
- deep feature/attention fusion;
- flexible/cross-modality recognition;
- explicit missing-modality training;
- computational-cost comparisons;
- dependence-aware biometric uncertainty.

It has **not yet located** one study that jointly freezes comparable unimodal evidence, varies representative classical and deep **fusion mechanisms**, evaluates the same subject-disjoint verification trials over discrimination + calibration + controlled degradation + modality absence + compute, and tests family-level rank/Pareto stability with dependence-aware uncertainty.

This is a **search result to date, not proof of absence**. Gate 4 remains OPEN.

## Next searches required

- Full-text sweep for T-BIOM/TIFS 2020–2026 using fusion-mechanism terms rather than only modality names.
- IJCB/ICB 2020–2026 for comparative fusion and incomplete-modality papers.
- Search specifically for `calibration`/`likelihood ratio` combined with multimodal/flexible biometrics.
- Search for `shared encoder`, `frozen encoder`, `fixed embeddings`, or equivalent fusion-ablation benchmark designs.
- Code-availability audit for the closest modern baselines.
- Correction/retraction/PubPeer-type audit before final bibliography lock.
