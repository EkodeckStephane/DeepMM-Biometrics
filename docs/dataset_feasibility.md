# Dataset Feasibility Audit v0.3

Primary evidence for Q1-Q3 requires **genuine subject-level multimodal correspondence** and a verification protocol that can be split without biological-subject leakage. Access convenience is relevant only after scientific admissibility is established.

## Dataset-selection policy

The project distinguishes three states:

1. **access candidate** — public documentation is sufficiently credible to justify requesting/inspecting the data;
2. **provisional experimental candidate** — the archive has been obtained and its identity/session/sample topology has passed audit;
3. **locked dataset** — split/trial/statistical design, legal terms, encoder feasibility and final-test firewall are frozen.

No dataset may move directly from a web description to `locked` status.

## P1 — NUPT-FPV (primary access candidate)

**Official project repository:** https://github.com/REN382333467/NUPT-FPV  
**Reference:** H. Ren, L. Sun, J. Guo and C. Han, *A Dataset and Benchmark for Multimodal Biometric Recognition Based on Fingerprint and Finger Vein*, IEEE Transactions on Information Forensics and Security 17 (2022), 2030–2043. DOI `10.1109/TIFS.2022.3175599`.

### Verified public properties

- 140 human volunteers;
- six fingers per volunteer (left/right index, middle and ring), yielding 840 finger instances;
- fingerprint and finger-vein modalities;
- 20 acquisitions per finger across two sessions (the public table reports 10 repetitions per session × 2 sessions);
- 16,800 fingerprint and 16,800 finger-vein images, 33,600 images total;
- reported image size 300 × 400;
- the project repository states that research use is available free of charge and provides a release-agreement PDF;
- complete-data access is by contact with the authors rather than anonymous direct download.

### Why P1 is currently preferred

NUPT-FPV offers a particularly useful structure for the present study: paired modalities, repeated acquisitions, two sessions, a nontrivial number of human subjects, modest image size, and an explicit research-access path. It supports a meaningful clean/degradation/missing-modality verification study without forcing the project back to the former face-fingerprint design.

### Mandatory biological-subject rule

The 840 fingers are **not 840 independent human subjects**. Confirmatory train/development/calibration/test partitions must be disjoint at the **volunteer/person level**. Different fingers from the same volunteer may not cross these partitions. Finger-level identity may be used inside the biometric matching protocol only after this person-level grouping constraint is enforced.

### Required archive audit before provisional lock

- signed/accepted access terms and redistribution constraints recorded;
- exact file naming and modality correspondence verified;
- session labels 1/2 and acquisition indices reconstructed;
- volunteer IDs proven identical across fingerprint/finger-vein records;
- missing/corrupt files enumerated;
- exact per-volunteer/per-finger sample counts checked rather than assumed from the paper;
- feasibility of subject-disjoint split and held-out calibration confirmed;
- trial topology selected so dependence-aware inference can be specified;
- any benchmark-specific preprocessing that would leak test information excluded.

**Current status:** **P1 ACCESS CANDIDATE — NOT YET DATASET-LOCKED.**

## G1 — SDUMLA-HMT (generalization/fallback candidate)

**Official source:** Shandong University Artificial Intelligence Research Center: https://www.sai.sdu.edu.cn/info/1075/1115.htm

### Verified public properties

- 106 individuals;
- real multimodal data;
- modalities include face, fingerprint, finger vein, gait and iris;
- fingerprints from six fingers, five sensors and eight impressions per finger/sensor combination, with 25,440 fingerprint images reported;
- research/noncommercial access is requested by email.

### Strengths and risks

It offers a recognized homologous multimodal collection and would permit an important cross-pair/generalization check. Its smaller human-subject count makes high-capacity end-to-end fusion less attractive unless upstream encoders are frozen or strongly regularized. The delivered archive/session topology must still be audited.

**Current status:** **G1 GENERALIZATION/FALLBACK CANDIDATE — ACCESS REQUEST NOT YET SENT.**

## G2 — LUTBIO (large multimodal generalization candidate)

**Dataset record:** Mendeley Data v6, DOI `10.17632/jszw485f8j.6`.  
**Associated paper:** *Information Fusion* (2025), DOI `10.1016/j.inffus.2025.102945`.

### Verified public properties

- 306 individuals;
- nine modalities including voice, face, fingerprint, contact/contactless palmprint, ECG, back-of-hand, ear and periocular data;
- current v6 description instructs researchers to download an application document, complete it, and send it to the dataset contact;
- the record states research-only/confidentiality access conditions;
- the Mendeley record displays a CC BY 4.0 licence for the record, but this must not be interpreted as unrestricted redistribution of raw biometric files.

### Important ambiguity

The v6 `Steps to reproduce` text contains the sentence that the “multimodal biometric data presented in the paper is not from the same individual”. **Je ne peux pas confirmer à partir de cette wording seule qu’il s’agit d’une absence de correspondance d’identité dans la base complète.** It may concern data displayed/presented for privacy rather than the stored database topology. Consequently, LUTBIO cannot be primary evidence until the application materials, collection protocol and obtained archive explicitly demonstrate the same-subject correspondence required by DeepMM.

**Current status:** **G2 CANDIDATE — APPLICATION REQUIRED; IDENTITY TOPOLOGY MUST BE VERIFIED BEFORE ADMISSIBILITY.**

## G3 — OU-MB (high-scale scientific candidate; access path open)

**Reference:** C. Xu et al., *OU-MB: The OU Multimodal Biometric Database and Its Performance Evaluation*, IEEE Transactions on Biometrics, Behavior, and Identity Science (2026), DOI `10.1109/TBIOM.2026.3710514`.

### Verified publication properties

- 1,099 subjects;
- eleven modalities including iris, palm vein, 2D face, signatures, gait, voice, full-body images, online-signature time series, brain signals, inertial and health data;
- the paper reports representative score-level and feature-level multimodal fusion experiments;
- OU-MB is already a critical Gate-4 comparator because its fusion baseline holds modality-specific recognition models fixed while comparing model-agnostic score/feature fusion.

### Access uncertainty

The publication describes OU-MB as publicly available, but **Je ne peux pas confirmer actuellement un stable raw-data download/request mechanism from the public sources audited for this project.** Until a canonical access route and terms are verified, it remains scientifically attractive but operationally unselected.

**Current status:** **G3 HIGH-SCALE CANDIDATE — RAW-DATA ACCESS PATH TO VERIFY.**

## G4 — BioSecure BMDB / BiosecurID

### BioSecure BMDB

Published descriptions report more than 600 individuals, multiple acquisition scenarios, face/audio, fingerprints, multiple sensors and two sessions. The current 2026 access path and exact same-subject subset required by DeepMM still need verification.

### BiosecurID

The database paper reports 400 subjects and eight traits including face and fingerprints. Historical descriptions report repeated acquisition sessions. Current official access/licensing and exact usable cross-modal subset remain to verify.

**Current status:** **VALID SCIENTIFIC CANDIDATES — CURRENT ACCESS NOT YET VERIFIED.**

## G5 — FaciaVox (secondary only at present)

**Canonical dataset record:** Zenodo DOI `10.5281/zenodo.14861092`.

### Verified public properties

- 100 participants;
- 1,800 face images and 6,000 voice recordings;
- restricted files;
- access requires a request and signed Data Usage Agreement;
- research-only/no-sharing conditions;
- Zenodo reports approximately 126 GB of data.

The pair is scientifically useful for cross-modality generalization, but the storage/access burden is high relative to the subject count and the project's no-funded-infrastructure constraint.

**Current status:** **SECONDARY/GENERALIZATION CANDIDATE — NOT PREFERRED FOR PRIMARY CAMPAIGN.**

## Explicit exclusions from primary evidence

Datasets or compilations that create multimodal identities by matching unrelated labels across independently collected unimodal databases are not admissible for a headline cross-modal/fusion claim. Chimeric data may only appear in a clearly labelled secondary sensitivity experiment if a specific scientific reason is preregistered.

## Dataset-lock criteria

A primary dataset is locked only if all criteria pass:

1. same-human correspondence across selected modalities is explicitly verified;
2. verification research use is permitted and legal terms are archived;
3. person-disjoint train/development/calibration/test splitting is feasible;
4. enough within-person repeated samples exist for genuine comparisons without leakage;
5. impostor trials can be frozen reproducibly;
6. trial dependence can be modeled with a defensible inference/resampling method;
7. controlled degradation and missing-modality experiments are meaningful;
8. selected encoders/fusion capacity are compatible with sample size and compute;
9. calibration can be fitted outside the final test set;
10. raw biometric redistribution is avoided unless explicitly permitted;
11. all final trial lists and split manifests can be hashed before confirmatory testing;
12. the dataset choice was not made because a pilot happened to favor a preferred DL family.

## Current decision

**No dataset is locked.** NUPT-FPV is promoted to **P1 primary access candidate** because its publicly documented topology is currently the strongest combination of scientific suitability and practical feasibility. The next data-dependent action is to request/obtain NUPT-FPV and audit the delivered archive before any confirmatory training or final-test evaluation. SDUMLA-HMT and LUTBIO remain the first generalization/fallback paths; OU-MB remains a high-value option pending a verified access route.
