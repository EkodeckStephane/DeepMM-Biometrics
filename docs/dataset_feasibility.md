# Dataset Feasibility Audit v0.1

Primary evidence for Q1-Q3 requires **real subject-level multimodal correspondence**. This document records only candidates for which a credible source has been located. Access, licensing, completeness, and protocol suitability must still be checked before dataset lock.

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
- current response/access time is unknown until a request is made.

**Current status:** **HIGH-PRIORITY CANDIDATE — ACCESS REQUEST REQUIRED.**

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

**Current status:** **STRONG CANDIDATE — CURRENT ACCESS CONDITIONS NOT YET CONFIRMED.**

## Candidate C — BiosecurID

**Reference:** Fierrez et al., *BiosecurID: a multimodal biometric database*, arXiv:2111.03472 (database paper).

**Reported properties:**
- 400 subjects;
- eight biometric traits including face and fingerprints;
- multiple acquisition characteristics and compatibility with other multimodal databases.

**Current status:** **CANDIDATE — OFFICIAL CURRENT ACCESS/LICENSING AND exact face-fingerprint protocol still to be verified before use.**

## Candidate D — LUTBIO

**Dataset record:** Mendeley Data DOI: https://doi.org/10.17632/jszw485f8j.6

**Indexed description located in 2025:**
- 306 individuals;
- nine biometric modalities including face and fingerprint, plus voice, palmprint, ECG, ear and periocular information.

**Current status:** **PROMISING MODERN CANDIDATE — must verify subject-level completeness, acquisition sessions, download terms, and benchmark literature.**

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
8. sample count supports the planned model capacity or a justified pretraining/frozen-encoder strategy is defined.

## Current decision

No dataset is locked yet. **SDUMLA-HMT is the first face-fingerprint dataset for which an official access request should be initiated.** BioSecure/BiosecurID/LUTBIO are being evaluated as alternatives or additional generalization datasets.
