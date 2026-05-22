# 27 Contrastive Shared Backbone (mDeBERTa-v3-base)

Shared backbone with a rating head (EMD^2 loss + median decode) and a contrastive head. Labels are 0-4 internally.

## Quick commands

Sanity checks:

```bash
python sanity_check.py
```

Run variants:

```bash
bash run_shared_backbone_supcon_normal_lambda_003.sh
bash run_shared_backbone_supcon_normal_lambda_005.sh
bash run_shared_backbone_supcon_weighted_lambda_003.sh
bash run_shared_backbone_supcon_weighted_lambda_005.sh
```

## Notes
- Total loss: W1/EMD + lambda * SupCon.
- EMA and LLRD follow the 23_mdeberta_llrd_ema recipe.
- Default backbone: microsoft/mdeberta-v3-base.
