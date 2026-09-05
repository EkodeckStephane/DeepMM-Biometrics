# V1 Neural Development Lock — Public NUPT-FPV Subset

**Status:** frozen before any calibration-role or final-role image is opened by the neural campaign.

## Scientific purpose

V1 compares fusion mechanisms while keeping the upstream unimodal representation fixed. The primary representation is `resnet18_imagenet1k_v1`, already designated as the primary frozen encoder before neural fusion results are observed. MobileNetV3-Small remains a representation-sensitivity analysis and is not allowed to replace the primary encoder because of development results.

The V1 neural families are:

- **D1** — nonlinear score fusion (`ScoreMLPFusion`);
- **D2** — nonlinear feature fusion with a shared enrollment/probe encoder and cosine verification (`FeatureFusionMLP`);
- **D3S** — quality/availability-aware learned score gate (`ScoreQualityGate`).

D3S is the V1 headline instantiation of the broader D3 family because it permits the cleanest comparison with the classical C5 quality-weighted score baseline. C5 and D3S receive exactly the same label-free quality cues. The feature-gated D3F implementation remains available but is outside the V1 confirmatory headline set.

## Locked training budget

All three neural families receive the same search budget:

| Item | Frozen value |
|---|---|
| Primary encoder | ResNet18 / `IMAGENET1K_V1`, frozen |
| Candidate architectures per family | 2 |
| Technical seeds | 1701, 2903, 4307 |
| Main reporting seed | 1701 |
| Maximum epochs | 40 |
| Early-stopping patience | 6 |
| Training objective | binary cross-entropy with logits |
| Fit sampling | deterministic class balancing: each impostor once, each genuine 19× |
| Selection data | original unmodified trial distribution |
| Selection metric | empirical EER on `selection` only |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Gradient clipping | L2 norm 5.0 |
| Batch size | 256 |
| Deterministic algorithms | enabled |

The fit role contains 120 genuine and 2,280 impostor trials. The 19× deterministic replication therefore yields 2,280 genuine and 2,280 impostor training examples without changing the validation distribution. Exactly the same balancing rule is used for D1, D2 and D3S.

The maximum number of development fits is 2 candidates × 3 seeds = **6 runs per neural family**, or 18 neural fits overall.

## Locked architecture candidates

### D1 — score MLP

- `d1-h8`: two score inputs → hidden 8 → scalar logit;
- `d1-h16`: two score inputs → hidden 16 → scalar logit.

### D2 — feature MLP

The two 512-dimensional frozen ResNet18 modality embeddings are concatenated. The same learned feature encoder is applied to enrollment and probe evidence.

- `d2-h128-z64`: input 1024 → hidden 128 → fused embedding 64;
- `d2-h256-z64`: input 1024 → hidden 256 → fused embedding 64.

Verification remains cosine similarity between the fused enrollment and probe embeddings.

### D3S — quality-aware score gate

The gate sees the two label-free quality values and two explicit availability bits. Its output is a simplex weight vector over the two unimodal scores.

- `d3s-h8`: gate hidden width 8;
- `d3s-h16`: gate hidden width 16.

For clean development trials both modalities are available. Missing-modality conditions are evaluated later using the predeclared availability policy; unavailable score slots are never represented with NaN.

## Model-selection rule

Candidate architecture selection is fixed before observing neural results:

1. smallest **median selection EER across the three locked seeds**;
2. if tied, smallest mean selection EER;
3. if tied, smallest trainable parameter count;
4. if tied, lexicographically smallest candidate identifier.

Seed 1701 is the fixed primary technical realization. Seeds 2903 and 4307 are retained as stochastic sensitivity analyses; a favorable seed is never chosen after looking at final results.

## Data firewall

The development-training workflow may read only the frozen V1 `fit` and `selection` sample roles. The quality normalizer is fitted only from `fit` images. It is forbidden for this workflow to load captures assigned to `calibration` or `final`.

Calibration is a separate held-out operation after the selected architecture IDs and checkpoint-selection rules are frozen. Final evaluation is opened only after calibration and the complete clean/stress/missingness/cost evaluation script are frozen.

## Frozen lock hash

SHA-256 of the canonical machine-readable training lock:

`d7a118af2bd02cdb0625602713cf3254f65a8acc06459672a72ff3a48ec22f45`

Changing any locked encoder, candidate, seed, optimizer value, budget, batch size, fit-sampling rule, selection rule or reporting rule invalidates this hash and must be treated as a new protocol version rather than silently replacing V1.