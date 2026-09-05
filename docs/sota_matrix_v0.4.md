# Verified SOTA Matrix v0.4 — Closest-Precedent and Novelty-Boundary Pass

**Cutoff:** 2026-09-05.

This version focuses on papers most capable of falsifying the current DeepMM-Biometrics contribution. It does **not** claim exhaustive coverage and cannot support wording such as “first”, “only”, or “state of the art”. The correct scientific question at this stage is whether a prior study already performs the same controlled *fusion-mechanism* comparison under the same multidimensional evaluation contract.

## 1. Closest precedents

| Work | Controlled variable / contribution | Data / task facts relevant here | Dimensions already addressed | Why it does **not yet** close Q1–Q3 |
|---|---|---|---|---|
| Poh et al., *IEEE TIFS* 2009, DOI `10.1109/TIFS.2009.2034885` | Benchmark of **22 score-level fusion systems** under quality-dependent and cost-sensitive conditions | BioSecure access-control campaign; face, fingerprint, iris; about 500-person target setting | Fusion accuracy, quality changes, failure-to-acquire/failure-to-match, acquisition/computation/hardware cost | Very strong historical fusion-algorithm benchmark, but predates representative DL fusion families and does not provide the modern shared-encoder classical-vs-DL family comparison with calibration and rank/Pareto stability. |
| Soleymani et al., *IEEE T-BIOM* 2022, DOI `10.1109/TBIOM.2021.3131664` | Weakly supervised **quality-aware deep fusion**, two quality/aggregation blocks and task-specific losses | Face, iris, fingerprint; three multimodal datasets | EER/AUC/TAR-style verification performance and quality-aware fusion | Establishes quality-aware DL prior art; it is a proposed method rather than a matched comparison of representative fusion families under a common fusion-only contract. |
| Tiong et al., CVPR 2024, *Flexible Biometrics Recognition: Bridging the Multimodality Gap through Attention Alignment and Prompt Tuning* | Vision-Transformer framework using Multimodal Fusion Attention and Multimodal Prompt Tuning | Face, periocular, soft biometrics; intra- and cross-modality recognition; four benchmark datasets; source code released | Flexible/cross-modality recognition and attention-based interaction | Makes attention/flexible recognition non-novel by itself. It does not answer which fusion family is best when encoders/trials/tuning budget are deliberately matched across classical and deep mechanisms. |
| Lu, Wu & Bao, *Engineering Applications of AI* 2025, DOI `10.1016/j.engappai.2025.110865` | Face–fingerprint multilevel spatial/channel attention plus knowledge distillation | Real multimodal evaluation datasets | Performance and model/resource efficiency | Directly occupies attention + efficiency territory; not a broad fusion-family benchmark with missingness/calibration/rank-stability analysis. |
| Wu et al., *IEEE TIFS* 2026, DOI `10.1109/TIFS.2026.3700801` (AHFNet) | Adaptive hybrid fusion for unreliable and partially missing multimodal hand biometrics | Multimodal hand recognition on multiple public datasets | Missing/unreliable modality robustness and adaptive fusion | Makes “handling missing/degraded modalities” non-novel. The DeepMM question must concern **comparative family ranking under stress**, not another dedicated robustness architecture. |
| Tiong et al., *Information Fusion* 2026, DOI `10.1016/j.inffus.2026.104267` | Survey formalizing **flexible biometrics** as modality-agnostic recognition over variable modality sets | Verification/identification/retrieval; partial/asymmetric enrollment/query settings | Taxonomy, flexible-modality evaluation, deployment agenda, subset-conditional calibration and standardized reporting | Raises the Q3 bar: subset-specific calibration and partial-modality protocols are requirements, not standalone novelty. It is a survey rather than the matched empirical fusion-family benchmark proposed here. |
| Alazawi, Habeeb & Almaliki, *Kurdistan Journal of Applied Research* 2026, DOI `10.24017/science.2026.2.2` | **Closest controlled empirical precedent:** holds protocol factors fixed while varying ResNet50, EfficientNetV2-S, Swin-T | LUTBIO and XJTU; subject-disjoint verification; face/fingerprint/palmprint; common preprocessing/training conditions as described by authors | EER, AUC, TAR@FAR=0.1%; five score-fusion rules; effect sizes/selected bootstrap CIs; training/inference cost | Critically, the primary controlled variable is the **backbone architecture** while multimodal combination remains score-level. It does not invert the experiment by freezing matched unimodal evidence and making the **fusion mechanism** (classical score, deep score, feature, gating, attention/Transformer) the controlled independent variable. It also does not jointly evaluate calibration, controlled degradation, arbitrary modality absence, and Pareto/rank reversal. |

## 2. The Alazawi et al. 2026 study changes our design in a useful way

This paper is close enough that it must shape the final article rather than be treated as a routine citation.

Its design asks approximately:

> **With the evaluation pipeline controlled, which deep backbone is preferable across biometric modalities when score-level fusion is used?**

The DeepMM-Biometrics primary track must ask the complementary question:

> **With strong unimodal evidence controlled, which fusion mechanism is preferable across performance, robustness, calibration and cost?**

Accordingly, our independent variable is the fusion mechanism, not the encoder family. Shared/frozen unimodal encoders are therefore not merely a reproducibility convenience; they are the central confound-control required to distinguish our experiment from the closest controlled architecture benchmark.

The Alazawi paper also reports modality-dependent backbone ranking. That directly cautions us against allowing different preferred backbones to enter each fusion method without a separate analysis: doing so would confound representation quality with fusion quality.

## 3. What is already prior art and cannot be our novelty claim

The following are independently occupied by prior work and must not be presented as headline novelty:

- score-level multimodal fusion benchmarking;
- quality-dependent fusion;
- cost-sensitive/sequential fusion;
- deep multimodal biometric fusion;
- attention/cross-modal interaction;
- quality-aware deep fusion;
- Transformer-based flexible biometric recognition;
- explicit missing/unreliable-modality fusion;
- comparing CNN and Transformer backbones under a controlled verification protocol;
- reporting EER/AUC/TAR plus computational cost.

Any proposed contribution based only on one of these items would fail Gate 4.

## 4. Current novelty candidate

The current contribution candidate is **not an architecture**. It is a controlled empirical study of the *fusion mechanism*.

> **Candidate scientific contribution:** a matched multimodal-biometric verification benchmark that holds unimodal evidence and trials fixed while comparing representative classical and deep fusion mechanisms, then tests whether their multidimensional performance–robustness–calibration–cost Pareto structure and ranking remain stable under controlled quality degradation and single-modality absence.

This wording is intentionally descriptive rather than a priority claim.

## 5. Required decomposition of Q1

Q1 is answered through four separable effects:

1. **Multimodal gain:** fused system versus every unimodal anchor and the best unimodal system.
2. **Deep-fusion gain:** DL mechanism versus strong classical score/feature fusion with identical unimodal evidence.
3. **Representation gain:** secondary Track-II comparison if encoders are allowed to fine-tune; never confused with fusion-mechanism gain.
4. **Benefit–cost boundary:** whether any measured DL gain survives its additional parameters, latency/memory/compute, calibration and stress losses.

A result in which classical fusion remains non-dominated is a valid answer, not a failed experiment.

## 6. Required strengthening of Q3

Because flexible biometrics and missing-modality fusion already exist, Q3 is specifically about **comparative stability**:

- family ranking on complete/clean inputs;
- family ranking under each locked degradation severity;
- ranking when modality A is absent;
- ranking when modality B is absent;
- Kendall tau-b against clean ranking;
- pairwise rank reversals with bootstrap support;
- subset-conditional calibration;
- probability each method remains Pareto non-dominated under each condition.

This is stronger and more falsifiable than claiming that one architecture “is robust”.

## 7. Gate-4 falsification matrix

A prior study would materially close the present gap only if it jointly satisfies the following conditions.

| Condition | Strongest located precedent | Satisfied by that precedent? |
|---|---|---|
| Real subject-level multimodal evidence | Multiple works above | **Yes, by several papers** |
| Classical + representative DL **fusion mechanisms** compared | Poh 2009 covers many score fusion systems; modern papers cover individual DL mechanisms | **Not yet located jointly** |
| Encoder/representation strength controlled while fusion varies | Alazawi 2026 controls many factors but varies backbone instead | **Not located in the required direction** |
| Same subject-disjoint trials across fusion families | Controlled architecture studies come close | **Not yet verified for the full required family set** |
| Discrimination + biometric calibration | Flexible-biometrics survey calls for calibration; many architecture papers report discrimination | **Not yet located jointly for the family benchmark** |
| Controlled quality degradation | Poh 2009 and recent reliability-aware work address quality | **Yes separately; not jointly with the full family benchmark** |
| Arbitrary/single-modality absence | Recent flexible/AHFNet work | **Yes separately; not jointly with the full family benchmark** |
| Computational cost | Poh 2009, Edwards 2021, Lu 2025, Alazawi 2026 | **Yes separately** |
| Paired uncertainty / appropriate dependent-trial inference | Some studies report bootstrap/statistics | **Not yet verified as a complete benchmark property** |
| Rank/Pareto stability across stress conditions | Current search | **Not located** |

**Current Gate-4 status: OPEN.** We have not located a paper satisfying all rows, but this is **not proof that none exists**. No “first” or “only” wording is permitted until the systematic venue sweep and claim-level verification are complete.

## 8. Statistical-methodology boundary

Biometric score dependence is itself established prior art. Bolle, Ratha & Pankanti, *CVIU* 2004, DOI `10.1016/j.cviu.2003.08.002`, introduced the **subsets bootstrap**, resampling structured subsets/blocks to account for dependence in biometric score data and derive confidence regions for ROC/FAR/FRR/EER.

Therefore the final study will not treat the raw number of verification pairs as the number of independent subjects. The final bootstrap form is locked only after the trial-construction scheme is known. A simple one-way subject cluster bootstrap is admissible for an explicitly subject-centric protocol; dense symmetric all-vs-all impostor trials require a validated subsets/multiway or deterministic identity-resampling reconstruction.

See `docs/bootstrap_protocol.md`.

## 9. Remaining Gate-4 work

Before Gate 4 can receive PASS:

1. finish the 2020–2026 venue sweep for T-BIOM, TIFS, Pattern Recognition, Information Fusion, EAAI, IJCB/ICB and directly relevant CVPR/ICCV/ECCV work;
2. extract full methods/evaluation details for the closest papers rather than relying on title/abstract alone;
3. search specifically for studies that freeze encoders and vary fusion mechanisms;
4. search specifically for multimodal biometric **calibration** under missing/partial modalities;
5. audit code/data availability for the closest reproducible baselines;
6. verify correction/retraction status of critical references before manuscript submission;
7. freeze the novelty wording only after the above falsification test remains unmet.
