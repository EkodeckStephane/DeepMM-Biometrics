# V1 Development Selection Result — Public NUPT-FPV Subset

**Evidence class:** development only (`fit` + `selection`).  
**Confirmatory status:** **not a final scientific result**.  
**Calibration-role images read by this campaign:** no.  
**Final-role images read by this campaign:** no.

The locked campaign completed successfully in GitHub Actions run `34019887091` using the frozen training lock `d7a118af2bd02cdb0625602713cf3254f65a8acc06459672a72ff3a48ec22f45`. The exact JSON evidence is committed at `artifacts/locked/v1_development_training.json`.

## Deterministic classical selection scores

| System | Selection EER | Selection AUC |
|---|---:|---:|
| U-FP | 0.2000 | 0.8976 |
| U-FV | 0.3000 | 0.8132 |
| C1 equal score fusion | 0.1658 | 0.9162 |
| C2 validation-weighted score fusion | 0.2000 | 0.9207 |
| C3 logistic score fusion | 0.1553 | 0.9126 |
| C4 standardized feature concatenation | 0.1000 | 0.9517 |
| C5 quality-weighted score fusion | 0.1447 | 0.9230 |

C2 selected fingerprint/finger-vein weights `[0.6, 0.4]`. C5 selected `gamma = 0.5`. These are development choices only and are now frozen for the subsequent calibration/final pipeline.

## Neural family selection

Candidate choice follows the predeclared rule: minimum median selection EER across seeds, then mean EER, then parameter count, then candidate ID.

| Family | Selected candidate | Parameters | Median EER across seeds | Mean EER across seeds | Reporting-seed EER | Reporting-seed AUC |
|---|---|---:|---:|---:|---:|---:|
| D1 score MLP | `d1-h16` | 65 | 0.1763 | 0.1746 | 0.1763 | 0.9250 |
| D2 feature MLP | `d2-h128-z64` | 139,456 | 0.0526 | 0.0675 | 0.1000 | 0.9433 |
| D3S quality-aware score gate | `d3s-h16` | 116 | 0.1500 | 0.1500 | 0.1500 | 0.9062 |

The fixed reporting seed is `1701`; it was chosen before observing these results. The corresponding selected checkpoint hashes are frozen in `src/deepmm/training/v1_selection_lock.py`.

## What the development result does and does not establish

The selection evidence is encouraging for D2: under the development split it gives the lowest EER among the tested fusion mechanisms. This is **not** evidence that D2 is the V1 winner, because calibration, cross-session final discrimination, stress robustness, missing-modality behavior and computational cost remain unopened or unevaluated.

The development evidence is also scientifically useful because it is not uniformly favorable to deep fusion. D1 does not improve on the strongest classical development comparator, and D3S is close to rather than clearly better than C3/C5 on clean selection data. These outcomes are retained; they are not grounds for changing the locked family set.

D1 also shows seed sensitivity (`d1-h8` ranges from EER 0.1763–0.2000). D2 is stronger but still variable across seeds. The fixed reporting seed and full three-seed sensitivity evidence are therefore both retained.

## Next hard gate

Before final trials can be opened:

1. reproduce the selected reporting-seed checkpoints and verify their hashes;
2. fit the common held-out score-to-LLR calibration layer on `calibration` only;
3. freeze exact Q3 corruption levels, missing-modality policy, condition-aware calibration policy and cost measurement protocol;
4. freeze the final evaluation script and its hash.

Only after those items pass may the 4,000 final cross-session trials be evaluated.