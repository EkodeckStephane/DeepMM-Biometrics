# Verified SOTA Seed v0.1

This is a **seed**, not the final state of the art. Every paper included in the final manuscript will undergo existence/metadata/support verification before citation.

## 1. Recent systematic review of multimodal biometrics

H. Es-Sobbahi, M. Radouane, K. Nafil, **“Multimodal Biometrics: A Review of Handcrafted and AI-Based Fusion Approaches,”** *IET Biometrics*, 2025, article 5055434. DOI: https://doi.org/10.1049/bme2/5055434

Relevant verified points:
- review covers traditional and AI-based multimodal biometric recognition;
- 29 peer-reviewed studies were synthesized;
- feature-level and score-level fusion dominate the reviewed literature;
- the review explicitly organizes methods across fusion levels and physiological traits.

**Use in this project:** establishes why Q1 must compare DL against strong classical fusion rather than treating DL as the default comparator.

## 2. Missing-modality deep multimodal learning

R. Wu, H. Wang, H.-T. Chen, G. Carneiro, **“Deep Multimodal Learning with Missing Modality: A Survey,”** 2024, arXiv:2409.07825. https://arxiv.org/abs/2409.07825

Relevant points:
- missing modalities arise from sensor limitations, cost, privacy, and data loss;
- the survey organizes deep multimodal methods specifically designed to remain useful when modalities are unavailable.

**Use in this project:** provides the general multimodal-learning foundation for Q3; biometric-specific methods still need to be identified separately.

## 3. Modern cross-modal attention in biometrics

X. X. Zheng et al., **“Multimodal biometric authentication using camera-based PPG and fingerprint fusion,”** *Pattern Recognition Letters*, 2025, vol. 197. DOI: https://doi.org/10.1016/j.patrec.2025.06.017

Verified from the publisher abstract:
- fingerprint and camera-based PPG are encoded by structured state-space encoders;
- a cross-modal attention mechanism is used;
- a contrastive objective aligns feature distributions;
- both single-session and two-session verification are evaluated.

**Use in this project:** evidence that attention-based fusion remains an active biometric family, but it does not establish universal superiority.

## 4. Recent face-fingerprint CNN fusion example

U. A. Gimba et al., **“Enhancing biometric authentication through multimodal approach combining face and fingerprint recognition using convolutional neural networks (CNN),”** *Discover Computing*, 2025, 28:246. DOI: https://doi.org/10.1007/s10791-025-09775-z

Verified from the publisher page:
- combines face and fingerprint;
- uses CNN-based feature extraction and multimodal fusion;
- reports that the multimodal accuracy is not numerically higher than both unimodal accuracies, while FAR/FRR behavior is used to argue complementary value.

**Use in this project:** reinforces the need for Q1 to distinguish overall accuracy from specific verification-error improvements and robustness.

## 5. Strong paired multimodal benchmark outside face-fingerprint

H. Ren, L. Sun, J. Guo, C. Han, **“A Dataset and Benchmark for Multimodal Biometric Recognition Based on Fingerprint and Finger Vein,”** *IEEE Transactions on Information Forensics and Security*, vol. 17, pp. 2030–2043, 2022. DOI: https://doi.org/10.1109/TIFS.2022.3175599

Verified from the project repository/paper metadata:
- paired fingerprint and finger-vein data;
- 140 volunteers;
- two sessions;
- 33,600 images total;
- provides a benchmark for multimodal biometric recognition.

**Use in this project:** candidate secondary modality pair if face-fingerprint datasets are too small for a defensible family-level comparison, and useful evidence about proper paired multimodal benchmarking.

## 6. Historical fusion foundation to retain

A. Ross, K. Nandakumar, A. K. Jain and related fusion literature remains methodologically relevant for sensor/feature/score/decision fusion taxonomy. Exact references and which historical works remain necessary will be fixed during the formal SOTA review rather than copied automatically from secondary sources.

## Search agenda before Gate-4 lock

The full review must identify and verify, preferably from Q1/Q2 journals and top biometric/vision venues:

1. classical score-level and feature-level biometric fusion baselines still considered strong;
2. deep learned score fusion;
3. deep feature/intermediate fusion;
4. gated and quality-aware fusion;
5. cross-attention and Transformer-based biometric fusion;
6. missing-modality-aware biometric methods;
7. calibration and uncertainty estimation in biometric verification;
8. robustness evaluation under sensor/image degradation;
9. efficiency-aware multimodal biometric comparisons;
10. studies that compare multiple fusion families under a single matched protocol.

## Gate-4 rule

No claim such as “Transformer is best,” “deep fusion is superior,” “first,” “novel,” or “state of the art” is permitted until the representative-current-work matrix is closed and the claim is supported by direct evidence.
