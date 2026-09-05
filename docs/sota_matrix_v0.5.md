# Verified SOTA Matrix v0.5 — Fusion-Mechanism Novelty Falsification

**Study:** *Deep Learning Approaches for Multimodal Biometrics*  
**Cutoff:** 2026-09-05

This version extends v0.4 with historical missing-score fusion and 2025 missing-modality/dynamic-fusion work. It is a **falsification document**, not an exhaustive systematic review. “Not located” means the current search has not found the property; it does not prove absence from the literature.

## 1. Strongest lesson from the updated SOTA

The novelty boundary is substantially older than current deep-learning papers suggest.

### Quality, cost and unavailable evidence were benchmarked before deep fusion

Poh et al. (*IEEE TIFS*, 2009, DOI `10.1109/TIFS.2009.2034885`) benchmarked 22 score-level fusion systems in the BioSecure DS2 campaign under **quality-dependent** and **cost-sensitive** conditions. The cost-sensitive evaluation included restricted computation and failures such as failure-to-acquire/failure-to-match. Acquisition time, computation time and hardware/maintenance cost were explicitly part of the study.

Poh et al. (*IEEE TIFS*, 2010, DOI `10.1109/TIFS.2010.2053535`) then addressed **missing match scores/modalities** with SVM fusion and neutral-point substitution on BioSecure DS2 scores.

Consequently DeepMM-Biometrics cannot claim novelty merely because one benchmark contains quality, cost and missing evidence.

### Modern DL work occupies the architecture space densely

The current verified lineage includes:

- Edwards & Hossain, *IEEE TAI* 2021: deep Siamese match-score generation plus serial fusion and performance/acquisition-cost trade-offs on a true multimodal database;
- Soleymani et al., *IEEE T-BIOM* 2022: quality-aware deep multimodal fusion;
- Tiong et al., CVPR 2024: attention/alignment/prompt-based flexible biometric recognition;
- El_Rahman & Alluhaidan, *PLOS ONE* 2024: traditional-versus-CNN multimodal comparisons, but using ECG and fingerprint databases assembled across sources;
- Artabaz & Sliman, *Scientific Reports* 2025: handcrafted versus EfficientNet multimodal hand-feature fusion with an efficiency/accuracy framing;
- Pan et al., *Digital Signal Processing* 2025 (SSFD-Net): shared/specific feature disentanglement and missing-modality reconstruction;
- Pan et al., *IEEE TIFS* 2025: hierarchical cross-modal image generation for missing-modality recognition;
- Yang et al., *Knowledge-Based Systems* 2025 (DIRS): dynamic interaction units and a soft router selecting fusion paths according to sample complexity;
- Lu et al., *EAAI* 2025: face–fingerprint multilevel attention plus knowledge distillation/efficiency;
- Gu et al., IJCB 2025: mutual distillation with uncertain missing modality;
- Wu et al., *IEEE TIFS* 2026 (AHFNet): adaptive robust fusion under unreliable/missing modalities;
- Tiong et al., *Information Fusion* 2026: flexible biometrics and subset-aware evaluation/calibration as a generalized research direction.

Therefore attention, quality weighting, routing/gating, missing-modality recovery, distillation, flexible modality subsets and efficiency are **prior-art mechanisms**. DeepMM must compare these families; it cannot treat implementing any one of them as the headline contribution.

## 2. Pairing validity remains a differentiator

Recent papers also demonstrate why the dataset contract must be strict.

Artabaz & Sliman report their fused result on MS-PolyU and FVC2006-DB1_A, described as containing 300 and 140 users respectively. The currently extracted article text does not establish that those databases contain the same physical persons across modalities. Such evidence is useful for pipeline comparison but cannot by itself demonstrate learning of real subject-level cross-modal interactions.

Chitrapu et al. (*Scientific Reports*, 2026, DOI `10.1038/s41598-026-43252-x`) are even more explicit: CASIA-FaceV5 and CASIA-FingerprintV5 both use labels 0–499, but the article states that CASIA provides **no official confirmation that the subjects are identical**. Their labels are used for evaluation consistency rather than to assert a direct paired-identity mapping.

DeepMM therefore retains a hard stop: **headline cross-modal interaction claims require genuine subject-level multimodal correspondence.**

## 3. Novelty-falsification matrix

Legend: `Y` = directly established by the current extraction; `P` = partial/related; `—` = not established by the current extraction.

| Work | Multiple fusion strategies/families | Encoder strength isolated from fusion | Calibration axis | Controlled quality/stress | Missing/unavailable modality | Cost/efficiency | Rank/Pareto stability under stress |
|---|---:|---:|---:|---:|---:|---:|---:|
| Poh et al., TIFS 2009 | Y at score-fusion benchmark level | Y at common supplied-score level | — | Y | Y | Y | — |
| Poh et al., TIFS 2010 | P | Y at fixed score-input level | — | — | Y | — | — |
| Edwards & Hossain, TAI 2021 | P — serial rules | P | — | — | P — sequential subset acquisition | Y | — |
| Soleymani et al., T-BIOM 2022 | — proposed family | — | — | Y/quality-aware | P | — | — |
| FBR, CVPR 2024 | — proposed architecture | — | — | — | P/flexible recognition | — | — |
| El_Rahman & Alluhaidan, PLOS ONE 2024 | P | — | — | — | — | — | — |
| Artabaz & Sliman, Sci Rep 2025 | P | — | — | — | — | Y/efficiency analysis | — |
| SSFD-Net, DSP 2025 | — | — | — | — | Y | — | — |
| Pan et al., TIFS 2025 | — | — | — | P | Y | — | — |
| DIRS, KBS 2025 | — | — | — | P/sample-complexity adaptation | — | P | — |
| MPAD, EAAI 2025 | — | — | — | — | — | Y | — |
| Gu et al., IJCB 2025 | — | — | — | — | Y | — | — |
| AHFNet, TIFS 2026 | — | — | — | Y | Y | — | — |
| Chitrapu et al., Sci Rep 2026 | — | — | — | P | — | Y/runtime | — |
| Alazawi et al., 2026 | P — multiple backbones + score rules | **No: backbone is the main varying factor** | — | — | — | Y | — |

No extracted row currently satisfies the complete DeepMM contract. This supports continued investigation but **does not establish a “first” claim**.

## 4. What remains potentially distinctive

The current falsifiable contribution hypothesis is:

> **Control the unimodal evidence strongly enough to make the fusion mechanism the independent variable; compare representative classical and deep fusion families on the same frozen subject-disjoint verification trials; jointly measure discrimination, biometric score calibration, controlled degradation, single-modality absence and computational cost; then quantify uncertainty-aware changes in family ranking and Pareto structure across stress conditions.**

This is deliberately a benchmark contribution rather than an architecture claim.

### Why the shared/frozen-encoder track is central

Alazawi et al. 2026 already show that backbone choice can change biometric results under a controlled score-fusion experiment. If every DeepMM fusion family receives a different encoder, any apparent “fusion gain” becomes inseparable from representation strength. The primary track must therefore share/freeze unimodal evidence wherever technically possible; a secondary end-to-end track may measure attainable system performance but cannot replace the controlled fusion experiment.

## 5. Consequences for Q1

Q1 is decomposed into:

1. multimodal versus **best** unimodal performance;
2. strongest classical fusion versus best unimodal;
3. representative DL fusion versus **strongest classical fusion** under matched evidence;
4. DL benefit after calibration, robustness and computational overhead are exposed;
5. separate secondary accounting of representation/fine-tuning gain.

A classical method remaining Pareto-optimal, or the best unimodal system winning under some conditions, is a direct Q1 result and must remain in the manuscript.

## 6. Consequences for Q2

A family cannot be called “best” from EER or AUC alone. Q2 requires locked dimensions and directions, uncertainty-aware Pareto/non-dominance analysis and no post-hoc utility weights. At minimum the family set must include strong score fusion, feature fusion, nonlinear/deep score or feature fusion, quality-aware/gated fusion and a justified attention/Transformer representative.

The presence of DIRS, FBR, MPAD and quality-aware prior art means the representative DL mechanisms should be **minimal canonical family implementations or faithful reproducible baselines**, rather than unnecessary new branded architectures.

## 7. Consequences for Q3

Missing-modality robustness itself is already heavily occupied. Q3 must measure **comparative stability**:

- clean-condition family ordering;
- the same locked family set at every corruption severity;
- modality-A-only and modality-B-only conditions;
- subset-specific calibration;
- Kendall tau-b and explicit pairwise reversals;
- bootstrap support for meaningful reversals;
- probability of remaining Pareto non-dominated under each stress condition;
- robustness gain per unit of extra cost where meaningful.

## 8. Statistical implication

The trial count is not the biological sample size. Bolle, Ratha & Pankanti (CVIU 2004) established subset-based bootstrap methods specifically to account for dependence among biometric scores. DeepMM therefore keeps separate protocols for:

- subject-centric trial constructions where one-way identity clusters are defensible;
- dense symmetric impostor constructions where a score depends on two identities and a validated subject-subsets/joint-person reconstruction is required.

Issue #5 blocks a convenient but unvalidated multiway formula from becoming headline inference before the exact trial topology is known.

## 9. Current Gate-4 decision

**Gate 4 = OPEN, but the contribution boundary is materially narrowed.**

To move to `PASS-POSITIONING`, the remaining work is:

1. finish the targeted T-BIOM/TIFS/IJCB/ICB/Pattern Recognition/Information Fusion/KBS/EAAI sweep;
2. complete claim-level full-text extraction for the nearest rows;
3. specifically search for benchmarks that **freeze encoders/embeddings and vary fusion mechanisms**;
4. specifically search biometric calibration under partial/missing modalities;
5. complete code/reproducibility status for candidate numerical baselines;
6. verify correction/retraction status before final bibliography lock;
7. rerun the falsification matrix before any manuscript novelty wording is frozen.

**Je ne peux pas confirmer qu’aucun article existant ne réalise exactement ce benchmark complet.** The present search has not located one.
