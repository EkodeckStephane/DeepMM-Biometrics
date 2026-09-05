# SOTA Reproducibility Audit v0.1

**Study:** *Deep Learning Approaches for Multimodal Biometrics*

This audit separates three questions that are often conflated:

1. Is the paper scientifically relevant to the novelty boundary?
2. Is official/public code available?
3. Can the method be reproduced faithfully under the DeepMM controlled benchmark without changing its scientific meaning?

“Not located” means exactly that: the current search did not locate an official implementation. It does **not** prove that no code exists.

## 1. Flexible Biometrics Recognition (FBR) — CVPR 2024

**Paper:** Tiong et al., *Flexible Biometrics Recognition: Bridging the Multimodality Gap through Attention Alignment and Prompt Tuning*, CVPR 2024, pp. 267–276.

**Official code:** verified public repository `MIS-DevWorks/FBR`.

Repository evidence checked:
- repository describes itself as the official/source code for the CVPR 2024 paper;
- MIT license;
- Python implementation;
- training/evaluation entry point documented;
- pretrained-model link documented;
- README lists VGGFace2 and MAAD for training and gives PyTorch compatibility information.

**DeepMM role:** **reference implementation to audit**, not an automatic drop-in baseline. FBR is designed for face/periocular/soft-biometric flexible recognition. A faithful adaptation to another modality pair may materially change its assumptions. We may reuse architectural principles only if the controlled family representative remains scientifically equivalent and the adaptation is documented.

## 2. NUPT-FPV benchmark — IEEE TIFS 2022

**Paper:** Ren et al., *A Dataset and Benchmark for Multimodal Biometric Recognition Based on Fingerprint and Finger Vein*, TIFS 17 (2022), 2030–2043, DOI `10.1109/TIFS.2022.3175599`.

**Public repository:** verified `REN382333467/NUPT-FPV`.

Repository evidence checked:
- 140 volunteers;
- six fingers per subject;
- 20 acquisitions per finger over two sessions;
- 16,800 fingerprint + 16,800 finger-vein images reported;
- same collection is explicitly described as a multimodal fingerprint/finger-vein database;
- README states free use but also instructs researchers to contact the authors for the complete dataset.

**Access classification:** metadata/repository directly accessible; **complete biometric data still require contact according to the current README**. Therefore NUPT-FPV does not satisfy the user's preferred “direct download with no request” criterion at present, although it remains a scientifically strong candidate/generalization resource.

## 3. Quality-Aware Multimodal Biometric Recognition — T-BIOM 2022

**Paper:** Soleymani et al., DOI `10.1109/TBIOM.2021.3131664`.

**Code status in this pass:** **official implementation not located** through the targeted paper/author/repository search.

**DeepMM consequence:** do not claim exact reproduction yet. Its scientific role is to define the prior-art boundary for quality-aware deep fusion and to inform the D3 family representative. Before selecting it as an exact SOTA baseline, perform a deeper author/project-page/code search and inspect full implementation details.

## 4. AHFNet — IEEE TIFS 2026

**Paper:** Rong Wu, Zhongxia Zhang, Mingxing Zhang, Zhengchun Zhou, *AHFNet: An Adaptive Hybrid Fusion Network for Robust Multimodal Hand Biometrics Under Unreliable Modalities*, TIFS 21 (2026), 5691–5705, DOI `10.1109/TIFS.2026.3700801`.

**Project/article record:** located and metadata verified.

**Code status in this pass:** **public GitHub repository not located** by the targeted repository search.

**DeepMM consequence:** AHFNet remains a critical Q3 prior-art comparator at the conceptual/results level. It is not yet approved as a code-reuse baseline. If no official code is found, any reimplementation must be labeled as our reimplementation and validated against reported architecture/protocol details rather than presented as author code.

## 5. Multimodal PPG–fingerprint cross-attention — Pattern Recognition Letters 2025

**Paper:** Zheng et al., DOI `10.1016/j.patrec.2025.06.017`.

**Accessible paper path:** publisher metadata and preprint located.

**Code status in this pass:** Papers With Code currently lists no implementation; no official GitHub implementation was located in the targeted search.

**DeepMM consequence:** useful D5/cross-attention prior art, but not currently a reproducible exact baseline.

## 6. MPAD — Engineering Applications of Artificial Intelligence 2025

**Paper:** Lu, Wu & Bao, *Multilevel parallel attention knowledge distillation for multimodal biometric recognition*, DOI `10.1016/j.engappai.2025.110865`.

**Scientific relevance:** direct face–fingerprint attention and model-compression/efficiency work.

**Code status in this pass:** **not located**.

**DeepMM consequence:** important novelty/cost boundary. Exact implementation remains pending before it can be called a reproduced SOTA baseline.

## 7. Cross-Architecture Evaluation — 2026

**Paper:** Alazawi, Habeeb & Almaliki, DOI `10.24017/science.2026.2.2`.

**Full-text accessibility:** located through the journal/author-hosted search path.

**Code status in this pass:** **official repository not located**.

**DeepMM consequence:** this paper is primarily a methodological precedent for controlled comparison, not a model to copy. Its most useful design lesson is to isolate the independent variable. DeepMM deliberately inverts that variable: shared/frozen unimodal evidence, varying fusion mechanism.

## 8. Baseline-selection rule derived from this audit

A method can enter the final experiment table in one of three statuses:

### A — Author implementation
Official code/checkpoints found, license permits use, and core behavior is verified.

### B — Faithful reimplementation
No usable author code, but the paper contains enough detail for an independent implementation. The manuscript must label it as a reimplementation and report validation against any reproducible published reference points.

### C — Conceptual SOTA boundary only
Paper is scientifically important but cannot be implemented faithfully enough under available information/resources. It remains in Related Work/SOTA positioning but is not used for a numerical “we beat the paper” claim.

A C-status work cannot be represented numerically by an invented surrogate bearing the paper's name.

## 9. Current reproducibility decision

- FBR: **A candidate**, official code verified; adaptation validity still to assess.
- NUPT-FPV: **dataset/benchmark repository verified**, complete-data access still requires contact according to README.
- Soleymani quality-aware: **B/C pending deeper implementation search**.
- AHFNet: **B/C pending**.
- Zheng PPG–fingerprint: **B/C pending**.
- MPAD: **B/C pending**.
- Alazawi controlled architecture benchmark: **methodological precedent; numerical reproduction optional rather than required for our fusion-mechanism claim**.

This status table will be updated before the model-family list is frozen.
