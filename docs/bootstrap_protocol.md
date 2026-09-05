# Biometric Bootstrap and Paired-Inference Protocol v0.1

**Study:** *Deep Learning Approaches for Multimodal Biometrics*

This document fixes the dependence problem before a final dataset/trial protocol is chosen. Verification trials are not automatically independent observations: many trials can share the same biometric subject, enrollment template, probe sample, session, or impostor identity.

## 1. Literature anchors

The bootstrap design is grounded in biometric-specific uncertainty work rather than a generic IID bootstrap.

- R. M. Bolle, N. K. Ratha, S. Pankanti, **“Error analysis of pattern recognition systems—the subsets bootstrap,”** *Computer Vision and Image Understanding*, 93(1), 1–33, 2004. DOI: `10.1016/j.cviu.2003.08.002`. The paper explicitly introduces subset/block resampling to account for dependence in biometric match/non-match scores.
- S. L. Cheng, R. J. Micheals, J. Lu, **NISTIR 7740**, 2010, compares parametric and non-parametric confidence intervals for operational biometric FAR/FRR and validates bootstrap-based uncertainty estimates.
- C. Vivaracho-Pascual, A. Simon-Hurtado, E. Manso-Martinez, **“Improving biometric recognition by means of score ratio … A benchmarking study,”** *IET Biometrics* 10 (2021), 127–141. DOI: `10.1049/bme2.12011`. Its bootstrap explicitly samples subjects with replacement and then samples genuine/impostor trials within the selected subject.

These sources establish that raw trial count must not be treated as an IID biological sample size.

## 2. Trial-table requirement

Every final verification trial must carry enough provenance to reconstruct its dependence structure. Minimum fields:

```text
trial_id
label                  # genuine / impostor
anchor_subject_id      # cluster used by subject-centric inference
enrollment_subject_id
probe_subject_id
enrollment_sample_id
probe_sample_id
session metadata       # when available
condition_id            # clean / corruption / missingness state
```

The exact meaning of `anchor_subject_id` must be frozen with the dataset adapter. It cannot be invented after results are observed.

## 3. Primary bootstrap mode for subject-centric trial protocols

When trials are naturally organized around a claimant/probe identity and every trial belongs to one predeclared subject cluster:

1. sample test subjects/clusters with replacement;
2. retain the complete block of trials for each selected subject;
3. optionally resample genuine and impostor trials inside that subject only if the locked protocol requires the two-stage design;
4. evaluate every compared method on the **same resampled trial indices**;
5. repeat using a fixed recorded bootstrap seed;
6. report percentile 95% CIs and paired metric-difference distributions.

The repository implementation is `deepmm.stats.cluster_bootstrap_metric` and `paired_cluster_bootstrap_difference`.

## 4. Critical limitation: symmetric all-vs-all impostor trials

A one-way subject cluster is not automatically valid when an impostor trial is a symmetric pair `(subject_i, subject_j)` and both identities contribute dependence. In that case, simply assigning the pair to one arbitrary identity can understate uncertainty.

**Hard rule:** if the final benchmark uses symmetric all-vs-all or dense cross-subject pairing, the primary CI procedure will be upgraded to a validated subsets bootstrap, multiway/pigeonhole-style bootstrap, or a deterministic identity-resampling reconstruction that preserves both sides of the pair. The current one-way implementation is not sufficient evidence for that design.

This decision must be made **before** final test evaluation, once the selected dataset and trial-generation scheme are known.

## 5. Paired comparisons

For methods A and B, inference is based on the bootstrap distribution of

`Delta = metric(A on replicate) - metric(B on the same replicate)`.

Independent bootstrap CIs for A and B are not used as the primary significance test. The paired design preserves the shared test identities and trials.

For lower-is-better metrics (EER, C_llr, latency where inferential comparison is justified), negative `Delta` favors A. For higher-is-better metrics (TAR@FAR), positive `Delta` favors A. Direction is always reported explicitly.

## 6. Multiple comparisons

The preregistered Q1 contrast family uses Holm correction when p-values are reported. Exploratory pairwise analyses are labeled exploratory. A favorable subset of comparisons cannot be promoted post hoc to the confirmatory family.

## 7. Training seeds versus test-set uncertainty

Two sources of variability remain distinct:

- **training variability:** independent model seeds/checkpoints;
- **evaluation-set uncertainty:** subject-cluster bootstrap on a fixed trained model or paired set of models.

They are reported separately. Technical reruns and latency repetitions are not biological replicates.

A hierarchical summary combining seeds and subject bootstrap may be added after the run policy is frozen, but it cannot erase either source of variability.

## 8. Q2 Pareto uncertainty

For each bootstrap replicate, all model families are evaluated on the same resampled subjects/trials. The locked Q2 vector contains direction-aware criteria such as:

- EER (minimize) and/or TAR@FAR (maximize);
- robustness loss (minimize);
- calibration loss / C_llr (minimize);
- computational cost (minimize).

The analysis reports:

- point-estimate Pareto frontier;
- pairwise probability that A dominates B;
- probability that each method is non-dominated across bootstrap replicates.

Point-estimate Pareto membership alone is not sufficient for a headline “best trade-off” claim.

## 9. Q3 rank stability

For each locked stress condition, family ranking is compared with the clean condition using:

- Kendall tau-b, including ties;
- pairwise rank reversals;
- bootstrap support for rank reversals when the metric bootstrap is available;
- whether the clean-condition Pareto-optimal set remains non-dominated.

A rank reversal is a result, not an experimental failure.

## 10. Validation before use

Before Gate 5 receives PASS, bootstrap code must pass:

1. deterministic-seed tests;
2. paired-identical-system tests yielding zero delta;
3. synthetic clustered-data tests;
4. trial/cluster schema validation;
5. one independent numerical or simulation cross-check;
6. explicit decision on one-way versus subsets/multiway bootstrap after the final trial structure is known.

**Current status:** implementation started and unit-tested; final bootstrap design remains dataset/trial-structure dependent by construction.
