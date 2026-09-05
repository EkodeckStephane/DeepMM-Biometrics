# Verified SOTA Matrix v0.2

**Cutoff used for this working pass:** 2026-09-05.

This is a curated, verified working matrix, not yet the final systematic review. Every entry below has had its existence and core metadata checked against a publisher, official proceedings page, DBLP, or an institutional record. The final Gate-4 matrix will expand coverage and check the exact claim supported by each citation.

## A. Biometric-specific fusion and benchmarking

| Ref | Venue/year | Modalities | Method/fusion family | Missing modality | Quality/robustness | Cost/efficiency | Role in our study |
|---|---|---|---|---|---|---|---|
| Poh et al., DOI `10.1109/TIFS.2009.2034885` | IEEE TIFS, 2009 | Face + fingerprint + iris (score/quality benchmark) | Quality-dependent and cost-sensitive score fusion benchmark | Failure-to-acquire/failure-to-match considered in cost-sensitive setting | Explicit quality-dependent evaluation; quality-aware fusion performed strongly | Explicit acquisition/computation/hardware cost discussion | Historical strong baseline: quality and cost were already legitimate fusion objectives before modern DL |
| Poh, Bourlai, Kittler, DOI `10.1016/j.patcog.2009.09.011` | Pattern Recognition, 2010 | BioSecure multimodal scores | Test bed for quality-dependent, client-specific, cost-sensitive fusion | Indirectly through system/failure setting | Explicit quality metadata | Explicit cost-sensitive benchmark | Benchmark-design precedent |
| Edwards & Hossain, DOI `10.1109/TAI.2021.3064003` | IEEE TAI, 2021 | Face + fingerprint + palm | Siamese DL match-score generation + serial fusion | Serial decision may stop before all modalities are acquired; not the same as arbitrary sensor loss | Uses a reject/uncertainty region for sequential decisions | Explicit AUC–average-number-of-stages trade-off | Important precedent for Q1/Q2: deep biometric fusion has already been evaluated jointly with acquisition/decision cost on a real multimodal dataset |
| Ren et al., DOI `10.1109/TIFS.2022.3175599` | IEEE TIFS, 2022 | Fingerprint + finger vein | CNN multimodal benchmark (FPV-Net) on simultaneously collected paired data | Not primary focus | Realistic repeated acquisitions across 2 sessions | Benchmark paper; hardware/training described | Strong paired-data and deep-fusion benchmark anchor |
| Fan et al., DOI `10.1109/TSMC.2024.3382877` | IEEE TSMC: Systems, 2024 | Palmprint + palm vein | Hybrid two-stage recognition + adaptive weighted fusion | Not primary focus | Uses uncertainty subset in coarse/fine recognition | Time efficiency is an explicit objective | Evidence that “best” fusion can be a performance–cost trade-off, not raw accuracy only |
| Zheng et al., DOI `10.1016/j.patrec.2025.06.017` | Pattern Recognition Letters, 2025 | Camera PPG + fingerprint | SSM encoders + cross-modal attention + contrastive distribution alignment | Not primary focus | Single-session and two-session verification | Not primary contribution | Current evidence for attention/cross-modal interaction in biometric verification |
| Lu, Wu & Bao, DOI `10.1016/j.engappai.2025.110865` | Engineering Applications of AI, 2025 | Face + fingerprint | Multilevel parallel spatial/channel attention + knowledge distillation | Not primary focus | Evaluated on two real-world multimodal biometric datasets | Model compression/resource-constrained deployment is a central objective | Direct face–fingerprint evidence that attention must be judged together with model efficiency; also identifies XJTU and SDUMLA-HMT as real multimodal evaluation datasets |
| Gimba et al., DOI `10.1007/s10791-025-09775-z` | Discover Computing, 2025 | Face + fingerprint | CNN-based multimodal authentication | Not central | Claims robustness to common acquisition issues | Not central | Useful face–fingerprint example showing multimodal accuracy need not exceed every unimodal accuracy; metric choice matters |

## B. Missing-modality biometric recognition

| Ref | Venue/year | Modalities | Missing-modality strategy | Fusion mechanism | Verified significance for Q3 |
|---|---|---|---|---|---|
| Pan et al., DOI `10.1109/TIFS.2025.3559802` | IEEE TIFS, 2025 | Palmprint + palm vein | Hierarchical cross-modal image generation | Dynamic sparse feature fusion | Strong journal evidence that missing modalities require explicit design; also addresses changing image quality |
| Pan et al., DOI `10.1016/j.dsp.2025.105003` | Digital Signal Processing, 2025 | Palmprint + palm vein | Shared-specific feature disentanglement + cross-modal feature transformation | Reconstructed missing features + multimodal recognition | Direct biometric missing-modality method with experiments on 3 benchmark datasets |
| Gu et al., DOI `10.1109/IJCB65343.2025.11410967` | IJCB, 2025 | Multimodal biometrics | Mutual-distillation framework for uncertain missing modality | Knowledge transfer / missingness-aware learning | Top biometric-conference evidence that uncertain modality availability is an active evaluation axis |
| Gu et al., DOI `10.1007/978-981-95-6123-0_5` | CCBR, 2025 | Multimodal biometrics | Unified multimodal representation network | Unified representation under missing modality | Additional architecture-focused missing-modality evidence |
| Yuan et al., DOI `10.1016/j.eswa.2025.130645` | Expert Systems with Applications, 2026 | Multi-modal hand biometrics | Progressive collaborative adversarial learning + modality-responsive interaction | Dynamic fusion with modality availability indicators | Recent evidence for arbitrary missing-modality combinations and scalability beyond two inputs |

## C. Surveys that constrain our taxonomy

| Ref | Venue/year | Scope | Key verified use |
|---|---|---|---|
| Es-Sobbahi, Radouane, Nafil, DOI `10.1049/bme2/5055434` | IET Biometrics, 2025 | Systematic review of physiological multimodal biometric recognition combining handcrafted and AI-based approaches | 29 peer-reviewed studies; feature- and score-level fusion dominate; supports inclusion of strong classical baselines |
| Li et al., DOI `10.1016/j.inffus.2024.102418` | Information Fusion, 2024 | Hand-based multimodal biometric fusion review | Reviews multiple fusion levels and challenges; supports broad family taxonomy beyond one architecture |
| Wu et al., *Deep Multimodal Learning with Missing Modality: A Survey* | TMLR, 2026 | General deep multimodal learning under missing modalities | Provides a broad taxonomy separating imputation/reconstruction, representation-focused methods, architecture-focused strategies, distillation/ensemble and related families |
| Tang, Liang, Zhu, DOI `10.1016/j.sigpro.2023.109165` | Signal Processing, 2023 | Deep multimodal sensor fusion | Reviews adaptive, generative, discriminative, algorithm-unrolling and Transformer fusion; useful methodological context outside biometrics |

## D. Calibration and uncertainty anchors for biometric verification

These are not multimodal-biometrics papers, but they are methodologically important because Q2 explicitly includes calibration.

| Ref | Venue/year | Contribution | Use here |
|---|---|---|---|
| Shi & Jain, *Probabilistic Face Embeddings* | ICCV, 2019 | Represents a face embedding as a distribution; learned variance estimates data uncertainty and can influence matching/fusion | Establishes biometric relevance of uncertainty-aware embeddings |
| Mandasari et al., DOI `10.1049/iet-bmt.2013.0066` | IET Biometrics, 2014 | Score calibration in face recognition; discusses log-likelihood-ratio cost `C_llr` and decomposition into discrimination/calibration components | Supports use of biometric-specific calibration metrics rather than ECE alone |
| Salvador et al., *FairCal: Fairness Calibration for Face Verification* | ICLR, 2022 | Post-training calibration for face verification | Confirms probability calibration is a meaningful verification-stage problem and must use held-out data |
| Li et al., DOI `10.1016/j.imavis.2022.104429` | Image and Vision Computing, 2022 | Efficient probabilistic face embedding / uncertainty estimation and temperature-based fusion weighting | Links uncertainty, quality-aware fusion and computational efficiency |

## E. General multimodal benchmarking precedent

| Ref | Venue/year | Use |
|---|---|---|
| Xue et al., DOI `10.1609/aaai.v40i32.39963` | AAAI, 2026 | MULTIBENCH++ argues that lack of unified evaluation standards leads to biased fusion comparisons; provides broad cross-domain multimodal benchmarking precedent. It is not biometric-specific. |

## F. What the verified literature already rules out

The following statements are **not defensible** at this stage:

- “Transformers are the best multimodal biometric fusion method.”
- “Deep feature fusion always beats score fusion.”
- “Multimodal always beats the best unimodal system.”
- “A model that is best on clean inputs remains best with a missing modality.”
- “Higher accuracy implies better calibration.”
- “More complex fusion is justified without measuring cost.”

The reviewed literature contains diverse methods, modalities, datasets, objectives, and protocols, so cross-paper point estimates cannot answer Q1–Q3 reliably.

## G. Provisional unresolved gap

**Working hypothesis, not yet a priority claim:**

> Current multimodal-biometric literature contains strong individual methods, surveys, historical quality/cost benchmarks, a serial deep-fusion performance–cost study, and recent attention/distillation systems, but we have not yet located a recent biometric-specific study that compares representative classical and deep fusion families under one matched verification protocol while jointly evaluating discrimination, score calibration, controlled quality degradation, missing-modality behavior, and computational cost.

This gap must remain marked **PROVISIONAL** until the systematic Gate-4 search is completed.

If a direct recent benchmark with the same scope is found, the contribution must be narrowed or repositioned rather than ignored.

## H. Search work still required before Gate 4 can pass

1. Search IEEE T-BIOM, TIFS, TPAMI, TSMC, Pattern Recognition, Information Fusion, Neurocomputing, Signal Processing, Digital Signal Processing, Expert Systems with Applications, IET Biometrics, Engineering Applications of AI, and IEEE TAI for 2020–2026.
2. Search IJCB/ICB, CVPR/ICCV/ECCV where biometric multimodal fusion is materially addressed.
3. Verify whether any study has a matched family-level benchmark spanning classical + deep + attention/Transformer fusion.
4. Verify whether calibration is reported in multimodal biometrics beyond isolated application papers.
5. Audit code/data availability for candidate exact SOTA baselines.
6. Check retractions/corrections/expressions of concern for every critical source before manuscript submission.
