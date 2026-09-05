# Dataset Feasibility Audit v0.2

Primary evidence for Q1-Q3 requires **real subject-level multimodal correspondence**. This document records only candidates for which a credible source has been located. Access, licensing, completeness, and protocol suitability must still be checked before dataset lock.

## Dataset-selection policy

The project will **not block early scientific work on access-request workflows**. Dataset access will be handled when the experimental lock becomes necessary, while giving priority to datasets that are directly and lawfully accessible for research.

Priority order:
1. directly downloadable research datasets with verified same-subject multimodal correspondence and usable terms;
2. datasets accessible after lightweight registration or standard research agreement;
3. datasets requiring individual email approval or lengthy institutional access procedures.

Accessibility is a practical selection criterion, not a scientific substitute for validity. A directly downloadable dataset is admissible as primary evidence only if subject correspondence, sample structure, licensing, and leakage-free verification design all satisfy the dataset-lock criteria below.

## Candidate A — SDUMLA-HMT

**Source:** Shandong University Artificial Intelligence Research Center, official database page: https://www.sai.sdu.edu.cn/info/1075/1115.htm

**Verified properties from the official page:**
- 106 individuals.
- Real multimodal data.
- Modalities include face and fingerprint, as well as finger vein, gait, and iris.
- Face data include multiple view angles and acquisition variations.
- Fingerprints were acquired from six fingers using five sensors, eight impressions per finger/sensor combination (25,440 fingerprint images reported).
- Research/noncommercial use only; the official page instructs researchers to request the database by email.

**Strength for this project:** direct face-fingerprint subject correspondence is available in a recognized homologous multimodal database.

**Risks / items to verify before lock:**
- only 106 subjects may constrain end-to-end training of high-capacity fusion models;
- exact face-image count and session structure needed by our split protocol must be verified from the delivered archive/documentation;
- access is not direct and therefore this candidate is deferred until the dataset-lock stage unless no equally valid direct-access alternative is available.

**Current status:** **SCIENTIFICALLY STRONG CANDIDATE — DEFERRED ACCESS REQUEST.**

## Candidate B — BioSecure Multimodal Biometric Database (BMDB)

**Reference:** Fierrez et al., *The multiscenario multienvironment BioSecure Multimodal Database (BMDB)*. PubMed record: https://pubmed.ncbi.nlm.nih.gov/20431134/

**Verified properties from the publication abstract:**
- more than 600 individuals;
- three acquisition scenarios;
- face/audio common components;
- fingerprints acquired in desktop and mobile scenarios;
- two acquisition sessions;
- multiple sensors for some modalities;
- database designed for unimodal and multimodal evaluation.

**Strength for this project:** substantially larger subject count and realistic multisession/multienvironment design.

**Risks / items to verify before lock:**
- current access mechanism and legal terms must be verified;
- exact subset in which face and fingerprint are jointly available for the same subjects must be reconstructed from official protocol documentation;
- redistribution may be restricted.

**Current status:** **STRONG CANDIDATE — ACCESS PATH TO BE CHECKED AT DATASET-LOCK STAGE.**

## Candidate C — BiosecurID

**Reference:** Fierrez et al., *BiosecurID: a multimodal biometric database*, arXiv:2111.03472 (database paper).

**Reported properties:**
- 400 subjects;
- eight biometric traits including face and fingerprints;
- multiple acquisition characteristics and compatibility with other multimodal databases.

**Current status:** **CANDIDATE — DIRECT ACCESS, OFFICIAL LICENSING, AND EXACT FACE-FINGERPRINT PROTOCOL TO BE VERIFIED.**

## Candidate D — LUTBIO

**Dataset record:** Mendeley Data DOI: https://doi.org/10.17632/jszw485f8j.6

**Indexed description located in 2025:**
- 306 individuals;
- nine biometric modalities including face and fingerprint, plus voice, palmprint, ECG, ear and periocular information.

**Current status:** **HIGH-PRIORITY ACCESSIBILITY CHECK — appears promising for direct-access evaluation, but subject-level completeness, acquisition sessions, download terms, and benchmark literature must be verified before use.**

## Explicitly excluded as primary evidence

### Unverified face-fingerprint pair compilations
Datasets that combine publicly available face and fingerprint sources without authenticated subject correspondence are **not admissible as primary evidence** for cross-modal representation learning.

A public Kaggle dataset currently describes itself as an unofficial compilation and explicitly states that it does not contain validated face-to-fingerprint identity mappings. It is therefore unsuitable for our primary Q1-Q3 claims.

## Dataset-lock criteria

A primary dataset is locked only if all criteria pass:

1. subject correspondence across selected modalities is explicitly verified;
2. verification use is permitted by the dataset terms;
3. subject-disjoint train/validation/test splitting is possible;
4. enough within-subject samples exist to construct genuine verification comparisons without leakage;
5. impostor sampling can be frozen reproducibly;
6. quality/degradation and missing-modality experiments remain meaningful;
7. raw data can be acquired by the research team without prohibited redistribution;
8. sample count supports the planned model capacity or a justified pretraining/frozen-encoder strategy is defined;
9. among scientifically adequate candidates, practical preference is given to the least restrictive reproducible access path.

## Current decision

No dataset is locked yet. **No access request will be initiated at this stage.** The immediate priority is to complete the verified SOTA and experimental taxonomy while auditing directly accessible multimodal datasets first. Access-request datasets such as SDUMLA-HMT remain valid fallback or generalization candidates and will be handled when the experimental dataset lock becomes necessary.
