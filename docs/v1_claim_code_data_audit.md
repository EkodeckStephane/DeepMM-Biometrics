# V1 claim–code–data audit

**Verdict:** PASS for the bounded V1 result package; NO-GO remains for
person-population inference and submission.

## Immutable chain

| Layer | Locked evidence |
|---|---|
| Final workflow | Run `34021350579`, attempt 2, commit `480c1f4e67757e4789b270b5ea12ecd0e9eac16b` |
| GitHub artifact | ID `9985665528`; downloaded ZIP SHA-256 `743ae2d30d7cf5747dce5b3cad66e789d8cc3c9923fe91f35510d0bd473180c5` |
| Data | dataset manifest `be7d83e353476e50a6193d77e47d7f176ab0a9cb805f81cb8b8a87f79368238c` |
| Selection | lock `83ecd9e4f357babc7e7e70652dc3f7f95c2cf65dacea2c558f4b6c4d656ada14` |
| Calibration | evidence `047a4a786e5d88364d44227c03a4ae471ba6ff0481e011a72a7dda3295c56eaf` |
| Final policy | `e9701015b541e9c7e4debccd01fd1f32affecc97abaf2996b8cf6c5811adbfb5` |
| Clean trial manifest | `3b60ce30d0d496c35aefe0bf0b8c48f868cb3befba0c1fdfb52986645293f324` |
| Committed result lock | `fab7227ad87dd8da0f1f9065ab377052e18b5f429fc8a9c2d1dcc79ccd036468` |

The committed bundle contains 15 conditions, 4,000 trials per condition, 148
available system-condition manifests, and 444 score arrays. The two unavailable
cases are the unimodal system whose own modality is absent. The audit recomputes
every trial-manifest hash and every raw-score-manifest hash, validates all run
manifests, and checks the artifact/file/lock hashes. Run it with:

```bash
PYTHONPATH=src python scripts/audit_v1_final_evidence.py
```

## Claim dispositions

| Proposed claim | V1 disposition | Direct evidence / boundary |
|---|---|---|
| Multimodal fusion improves over the best unimodal system | **Observed for C4 only; contradicted as a blanket claim** | C4 ROCCH-EER 0.2115 vs U-FP 0.3097; C1 is slightly worse than U-FP. Point estimates only. |
| Deep fusion improves over matched classical fusion | **Contradicted overall** | D1 worse than C2; D2 worse than C4; D3S slightly better than C5 but worse than C4. |
| One deep family is the best Q2 trade-off | **Not demonstrated** | D2 is point-Pareto, but C1/C2/C4/D1 are also non-dominated; no uncertainty-supported dominance. |
| The best approach remains the same under degradation | **Observed for the winner** | C4 ranks first in all 12 blur/contrast conditions; lower ranks reverse. No population inference. |
| A learned method handles missing modalities better | **Not demonstrated** | Frozen M0 fallback makes all fusion families tie on available unimodal evidence. |
| Calibration improves | **Method/condition dependent; unconditional claim forbidden** | Held-out clean-calibrator transfer and secondary matching-condition diagnostic are both reported. |
| Results generalize to people/full NUPT-FPV | **Not demonstrated; deferred to V2** | Public-instance subset only; person mapping unresolved. |

## Reproducibility check

`scripts/generate_v1_result_assets.py` regenerates every committed V1 CSV,
LaTeX table, and PGFPlots figure from the verified result bundle. CI tests require
the generated tree to match byte-for-byte and require the full evidence audit to
pass. No model family, seed, hyperparameter, calibrator, condition, metric weight,
or trial was changed after final-score access.

The NPZ score bundle and compressed trial JSON are committed as deterministic
700,000-byte transport parts. The loader verifies every part, recomposes the
original bytes in memory, and requires the original workflow-artifact SHA-256
before parsing. Splitting therefore changes storage only, not scientific data.
