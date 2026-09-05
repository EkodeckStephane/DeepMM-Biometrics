# FBR / MFA-ViT Architecture Audit v0.1

**Reference:** Tiong, Sigmund, Chan & Teoh, *Flexible Biometrics Recognition: Bridging the Multimodality Gap through Attention Alignment and Prompt Tuning*, CVPR 2024, DOI `10.1109/CVPR52733.2024.00033`.

**Official implementation audited:** `MIS-DevWorks/FBR` (MIT license).

## 1. Verified implementation structure

The official repository describes MFA-ViT/FBR as a system for face, periocular and soft-biometric attributes. Inspection of `models/MFA_ViT.py` confirms that the model is not a generic plug-in fusion head over arbitrary frozen embeddings:

- `MFA_ViT` defines a dedicated **face patch tokenizer**;
- a dedicated **periocular patch tokenizer**;
- an **attribute tokenizer**, with default soft-biometric attribute size 47;
- prompt parameters allocate three modality slots;
- the backbone contains `MFA_block` modules combining transformer attention and depth-wise convolution;
- `MPT` implements multimodal prompt tuning;
- the default embedding dimension is 1024;
- the default identity-classification head has 9,131 classes;
- the forward signature accepts `x_face`, `x_ocular`, and `x_attr` and produces face/periocular identity heads or feature outputs.

The repository README documents VGGFace2/MAAD training resources, pretrained-model access, and the same face/periocular/soft-attribute framing.

## 2. Consequence for the DeepMM controlled benchmark

FBR is strong **prior art for attention/flexible biometric learning**, but directly inserting the complete official MFA-ViT into Track I would violate the central confound-control contract.

Track I asks:

> what does the **fusion mechanism** add once unimodal evidence is held fixed?

The official FBR code jointly defines modality tokenizers, transformer feature extraction, prompts and identity heads. Its measured performance therefore combines representation learning and fusion effects. Comparing that full network directly with a small score/feature fusion head would not isolate fusion quality.

## 3. Permitted uses

### A. Novelty boundary — mandatory

FBR must remain in Related Work/SOTA because it establishes attention/prompt-based flexible biometric learning at a top vision venue.

### B. Track-II full-system comparator — possible, dataset dependent

If the final dataset/modalities support a scientifically faithful face/periocular/soft-biometric configuration, the official FBR implementation may be evaluated as a full-system SOTA comparator, subject to data/license/compute feasibility and a matched evaluation protocol.

It would then be reported as **full-system performance**, not as evidence that MFA is the best fusion mechanism.

### C. Track-I family representative — adaptation only, not “official FBR”

A compact D5/D6 attention/Transformer representative may adopt principles demonstrated by FBR, but to satisfy Track I it must operate on the same frozen/local unimodal evidence available to the other fusion families.

Such a model is **our controlled representative**, not an FBR reproduction, unless the architecture/input contract remains faithful enough to justify that label.

## 4. What must not be done

- Do not rename a generic two-token Transformer as “FBR”.
- Do not claim to reproduce FBR while removing its face/periocular/attribute tokenizers, MPT, training objective and identity-head structure without stating the adaptation.
- Do not compare full end-to-end FBR against frozen-encoder classical fusion and attribute the whole difference to “attention”.
- Do not select FBR as the final Transformer baseline merely because official code exists; family representativeness and modality compatibility remain primary.

## 5. Reproducibility status

**Official code availability:** VERIFIED.  
**License:** MIT, verified in repository metadata/README.  
**Direct Track-I compatibility:** NO.  
**Potential Track-II compatibility:** TO LOCK after modality/dataset selection.  
**Use as architecture/SOTA reference:** YES.

## 6. Decision

FBR is currently classified as:

- **A for code availability**;
- **C for direct Track-I numerical reuse without adaptation**;
- **A/B candidate for Track II** if the chosen modality/data setting allows faithful reproduction.

This distinction prevents “available code” from being mistaken for “fair baseline under our scientific question”.
