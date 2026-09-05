# SOTA Search Protocol v0.1

**Study:** *Deep Learning Approaches for Multimodal Biometrics*

This protocol exists to prevent selective citation and post-hoc novelty claims. It will be executed and frozen before Gate 4 is marked PASS.

## 1. Review objective

Identify representative current work needed to answer and position Q1–Q3:

- classical vs deep multimodal biometric fusion;
- deep fusion families;
- quality-aware/robust fusion;
- missing-modality biometric learning;
- calibration/uncertainty in biometric verification;
- efficiency/cost-aware multimodal biometrics;
- controlled benchmarks comparing multiple fusion families.

## 2. Time window

Primary current-work window: **2020–2026**.

Older work is included only when it is foundational, remains a strong benchmark, or defines a metric/protocol still directly relevant (for example quality-dependent multimodal fusion and score calibration).

## 3. Target sources

Priority publisher/index sources:

- IEEE Xplore;
- ACM Digital Library;
- Elsevier/ScienceDirect;
- Springer Nature;
- Wiley;
- CVF Open Access;
- OpenReview for accepted venues;
- DBLP/Crossref/OpenAlex for metadata cross-checking;
- PubMed when a biometric/signal paper is indexed there.

Priority venues include:

- IEEE Transactions on Information Forensics and Security;
- IEEE Transactions on Biometrics, Behavior, and Identity Science;
- IEEE Transactions on Pattern Analysis and Machine Intelligence;
- IEEE Transactions on Systems, Man, and Cybernetics: Systems;
- Pattern Recognition;
- Information Fusion;
- Neurocomputing;
- Signal Processing;
- Digital Signal Processing;
- Expert Systems with Applications;
- IET Biometrics;
- International Joint Conference on Biometrics / International Conference on Biometrics;
- CVPR / ICCV / ECCV when directly relevant.

Venue rank is not the sole inclusion criterion, but weak/nonstandard venues cannot define the novelty boundary when stronger direct work exists.

## 4. Core search concepts

Search strings combine terms from four groups.

### Domain
- `multimodal biometric`
- `multi-biometric`
- `multibiometric`
- `biometric fusion`

### Deep fusion
- `deep learning`
- `CNN`
- `feature fusion`
- `score fusion`
- `intermediate fusion`
- `gated fusion`
- `quality-aware fusion`
- `attention`
- `cross-attention`
- `Transformer`
- `bilinear fusion`

### Robustness / missingness
- `missing modality`
- `incomplete modality`
- `modality dropout`
- `cross-modal generation`
- `quality degradation`
- `image quality`
- `robustness`
- `failure to acquire`

### Reliability / efficiency
- `calibration`
- `uncertainty`
- `likelihood ratio`
- `Cllr`
- `Brier`
- `ECE`
- `latency`
- `computational cost`
- `efficiency`
- `benchmark`

## 5. Inclusion criteria

A paper enters the main matrix if at least one applies:

1. proposes/evaluates a multimodal biometric fusion method;
2. provides a real paired multimodal biometric dataset/benchmark;
3. studies missing-modality or quality-dependent multimodal biometrics;
4. provides a controlled comparison of fusion families;
5. defines calibration/uncertainty methodology directly applicable to biometric verification;
6. is a high-quality systematic review that maps the above literature.

For Q2 family selection, preference is given to methods that can be faithfully instantiated under a common controlled protocol.

## 6. Exclusion criteria

Exclude from the **primary evidence matrix**:

- papers using independently assembled/chimeric modalities while claiming biological cross-modal learning without an appropriate limitation;
- purely unimodal work unless it defines an essential calibration/uncertainty/robustness method;
- non-biometric multimodal papers unless they provide methodological benchmarking precedent needed for our protocol;
- papers without enough methodological information to identify what is actually fused;
- duplicate preprint/journal versions (retain the final peer-reviewed version and link the preprint only for access when appropriate);
- withdrawn/retracted work.

## 7. Extraction fields

For every retained work record:

- citation key;
- title;
- authors;
- venue/year;
- DOI/official URL;
- publication status;
- modalities;
- subject-level pairing real/chimeric/unclear;
- task: verification/identification/both;
- dataset(s);
- number of subjects/sessions if reported;
- encoder/backbone;
- fusion level;
- fusion mechanism;
- training objective;
- classical baselines;
- unimodal baselines;
- deep baselines;
- missing-modality evaluation;
- controlled degradation/quality evaluation;
- calibration/uncertainty evaluation;
- cost/latency/parameter evaluation;
- statistical uncertainty/tests;
- code availability;
- data availability;
- principal verified result;
- key limitation relevant to Q1–Q3;
- exact claim(s) the paper can support in our manuscript.

## 8. Verification procedure

Before a reference enters the article bibliography:

1. existence and metadata checked against publisher/official proceedings/DBLP/Crossref;
2. DOI resolved or official stable record identified;
3. relevant section/abstract inspected to ensure it supports the intended sentence;
4. journal/conference status checked;
5. correction/retraction/expression-of-concern status checked before final submission;
6. no numerical result copied from a secondary source when the primary source is available.

## 9. Novelty lock test

Gate 4 can pass only after answering:

1. Is there already a biometric-specific matched benchmark of classical and representative DL fusion families?
2. Does such a benchmark jointly measure discrimination, calibration, controlled quality degradation, missing modalities, and computational cost?
3. Does it use a fair shared-data/shared-trial protocol?
4. Does it analyze whether family rankings reverse under stress?
5. Does it separate encoder strength from fusion-mechanism strength?

If a current paper answers all five, our working contribution is not novel enough and must be repositioned.

## 10. Current status

**OPEN.** The v0.2 matrix contains verified anchors but is not yet exhaustive enough to support “first”, “only”, “best”, or “state-of-the-art” wording.
