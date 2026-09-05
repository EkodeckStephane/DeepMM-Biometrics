# Gate-4 Falsification Search Log — Frozen 2026-09-05

**Study:** *Deep Learning Approaches for Multimodal Biometrics*

## Purpose

This log records the targeted search used to decide whether the proposed scientific positioning is sufficiently distinct to proceed. It is a **novelty-falsification exercise**, not a proof that no related paper exists. The study therefore forbids manuscript wording such as *first*, *only*, or *state-of-the-art* unless a later submission-time audit independently supports it.

## Search scope

The primary search window was 2020–2026, supplemented by older foundational work when it defines a still-relevant benchmark, calibration method, fusion paradigm, or dependence-aware statistical method.

Targeted source families included IEEE T-BIOM, IEEE TIFS, IEEE TAI, IEEE Access, CVPR/IJCB/ICB/CCBR, Pattern Recognition and related biometric venues, Information Fusion, Digital Signal Processing, Engineering Applications of Artificial Intelligence, Knowledge-Based Systems, Expert Systems with Applications, Expert Systems, IET Biometrics, Scientific Reports, and publisher/metadata records for relevant current papers. Current surveys were used to expand terminology and candidate lineages; novelty decisions rely on original studies whenever available.

## Falsification queries

Searches were organized around combinations of the following concepts:

- multimodal biometric / multibiometric / biometric fusion;
- fixed encoder / frozen embedding / common embedding / common score input;
- score fusion / feature fusion / learned fusion / MLP / gating / quality-aware fusion;
- attention / cross-attention / Transformer / dynamic routing;
- calibration / likelihood ratio / `C_llr` / Brier / ECE;
- degraded quality / sensor quality / noise / adversarial perturbation;
- missing modality / incomplete modality / arbitrary modality subset;
- cost / latency / efficiency / energy;
- Pareto / rank stability / rank reversal / robustness ranking;
- benchmark / controlled comparison / subject-disjoint verification.

The central falsification question was:

> Is there already a biometric-specific study that holds unimodal evidence sufficiently constant, compares representative classical and deep **fusion mechanisms** under the same verification protocol, jointly measures discrimination, calibration, controlled degradation, missing modalities and compute, and then quantifies uncertainty-aware changes in family ranking or Pareto structure across stress conditions?

## Closest located precedents and boundary decisions

### OU-MB — fixed upstream models plus model-agnostic fusion

Xu et al., IEEE T-BIOM 2026, DOI `10.1109/TBIOM.2026.3710514`, is the strongest direct precedent for the philosophy of holding modality-specific models fixed. It provides a real-subject multimodal database with 1,099 participants and eleven modalities and evaluates mean score fusion, two-modality weighted-sum score fusion, and L2-normalized feature concatenation without a trainable fusion network.

**Boundary consequence:** fixed-model score/feature fusion is not novel. DeepMM must extend the controlled variable from classical/model-agnostic fusion to representative deep fusion families while keeping upstream evidence and trials matched.

### LUTBIO — broad real-subject fusion and sensor-quality analysis

Yang et al., *Information Fusion* 2025, DOI `10.1016/j.inffus.2025.102945`, evaluates a nine-modality, 306-subject database, multiple fusion levels, and sensitivity to sensor type/quality.

**Boundary consequence:** broad multi-trait fusion and sensor-quality analysis are not novel. DeepMM's stress contribution must concern comparative family stability under a locked verification/calibration protocol rather than merely showing that fusion is robust.

### Alazawi et al. — controlled backbone comparison with score fusion

Alazawi et al. 2026, DOI `10.24017/science.2026.2.2`, controls the pipeline while varying ResNet50, EfficientNetV2-S and Swin-T and evaluates multiple score-fusion rules under subject-disjoint verification.

**Boundary consequence:** controlled architectural comparison already exists. DeepMM inverts the controlled variable: the primary experiment fixes upstream unimodal evidence and varies the fusion mechanism.

### Yoon et al. — backbone/fusion vulnerability and stress-dependent winners

Yoon, Cho and Choi, *Expert Systems* 2026, DOI `10.1111/exsy.70299`, explicitly studies combinations of VGGNet, ResNet, ViT and BEiT with general/hierarchical/dense fusion under FGSM/PGD perturbations. The reported best configurations change by modality and attack intensity, and weighted-mean probability fusion can outperform feature concatenation in robustness.

**Boundary consequence:** stress-dependent fusion/backbone ranking is already observable prior art. DeepMM cannot claim novelty from the existence of ranking changes alone. Its distinct object is a **matched fusion-family experiment** with dependence-aware uncertainty, calibration, missingness, cost, and explicit rank/Pareto stability analysis.

### Ryu et al. — feature/score fusion plus adaptation

Ryu et al., IEEE Access 2025, DOI `10.1109/ACCESS.2025.3599907`, compares feature- and score-level fusion and then evaluates adaptation strategies for face plus keystroke authentication. The multimodal combinations are constructed from separate source datasets rather than a naturally paired real-subject heterogeneous corpus.

**Boundary consequence:** feature-versus-score comparison and adaptation are not novel; real subject pairing remains a hard validity requirement for DeepMM headline cross-modal claims.

### AuthFormer — variable modality combinations with cross-attention/GRN

Yang, Meng and Zhang, arXiv 2024, DOI `10.48550/arXiv.2411.05395`, proposes AuthFormer with cross-attention and a gated residual network and reports experiments across different quantities/combinations of LUTBIO/XJTU modalities. This is a preprint in the frozen Gate-4 record, not a peer-reviewed journal/conference anchor.

**Boundary consequence:** adaptive Transformer fusion over variable modality combinations cannot be presented as a new mechanism.

### Flexible biometrics and modern missing-modality work

Tiong et al., *Information Fusion* 2026, DOI `10.1016/j.inffus.2026.104267`, formalizes flexible biometrics over variable modality sets and emphasizes subset-aware evaluation/calibration. Wu et al., IEEE TIFS 2026, DOI `10.1109/TIFS.2026.3700801`, and multiple 2025–2026 missing-modality studies already address unreliable or absent modalities directly.

**Boundary consequence:** handling missing modalities is not the contribution. The confirmatory Q3 object is whether **relative fusion-family ordering and non-dominance change** when quality or availability changes.

### Calibration prior art

Mandasari et al., *IET Biometrics* 2014, DOI `10.1049/iet-bmt.2013.0066`, and Susyanto et al., *IET Biometrics* 2019, DOI `10.1049/iet-bmt.2018.5106`, establish biometric score calibration and likelihood-ratio fusion with `C_llr`/`C_llr_min`-style analysis.

**Boundary consequence:** calibration metrics are not novel. DeepMM uses calibration as a locked dimension of the matched benchmark and studies transfer across stress/modality subsets.

## Gate-4 falsification matrix

| Required property of the DeepMM contribution | Closest located evidence | Already closed by prior work? |
|---|---|---:|
| Real subject-level multimodal data | OU-MB, LUTBIO, multiple hand datasets | Yes |
| Fixed/common upstream evidence while fusion varies | OU-MB for mean/weighted/concat | Partly |
| Representative **classical + deep fusion families** under that fixed evidence | Located papers cover pieces, not the full matched family benchmark | **No located study closes this** |
| Subject-disjoint/common verification trials | OU-MB/Alazawi and others provide relevant precedents | Partly |
| Biometric calibration as an explicit benchmark axis | Established separately | Yes separately |
| Controlled quality/stress | Poh/LUTBIO/Yoon/reliability-aware studies | Yes separately |
| Missing/variable modalities | Established extensively | Yes separately |
| Cost/efficiency | Established separately in several studies | Yes separately |
| Dependence-aware paired uncertainty for the family comparison | Statistical precedent exists; not located as part of the complete fusion benchmark | Partly |
| Explicit family-rank and Pareto stability across stress, with calibration and cost | No located biometric study satisfied the complete contract | **Not located** |

## Gate-4 decision

**PASS-POSITIONING as of 2026-09-05.**

This means the project has a sufficiently bounded, falsifiable contribution to justify implementation and pilot work. It does **not** mean absence of all equivalent work has been mathematically proved.

The locked contribution wording is:

> **DeepMM-Biometrics is a controlled multimodal-biometric verification study that holds unimodal evidence and trials fixed while comparing representative classical and deep fusion mechanisms, jointly evaluates discrimination, held-out biometric calibration, controlled degradation, single-modality absence and computational cost, and quantifies dependence-aware changes in family ranking and Pareto non-dominance across stress conditions.**

## Forbidden novelty wording

Unless a later independent audit changes the evidence, the manuscript must not claim novelty from any of the following alone:

- multimodal versus unimodal improvement;
- score-level or feature-level fusion;
- fixed upstream models;
- quality-aware fusion;
- attention/cross-attention/Transformer fusion;
- missing-modality handling;
- adaptive routing/gating;
- calibration or `C_llr`;
- efficiency/latency measurement;
- a two-dimensional accuracy–latency Pareto plot;
- the observation that rankings can change under stress.

## Re-open rule

Gate 4 is immediately reopened if a later paper is found that satisfies the complete locked contribution contract above, or if the final selected data/encoder regime prevents a fair fixed-evidence comparison. A submission-time SOTA refresh remains mandatory.
