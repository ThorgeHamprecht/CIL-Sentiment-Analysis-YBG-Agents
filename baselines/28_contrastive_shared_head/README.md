# 28 Contrastive Shared Head (mDeBERTa-v3-base)

Shared representation head used for both SupCon and W1/EMD classification. Labels are 0-4 internally.

## Quick commands

Sanity checks:

```bash
python sanity_check.py
```

Run variants:

```bash
bash run_shared_head_supcon_normal_lambda_001.sh
bash run_shared_head_supcon_normal_lambda_003.sh
bash run_shared_head_supcon_weighted_lambda_001.sh
bash run_shared_head_supcon_weighted_lambda_003.sh
```

## Notes
- Total loss: W1/EMD + lambda * SupCon (on the shared representation).
- EMA and LLRD follow the 23_mdeberta_llrd_ema recipe.
- Default backbone: microsoft/mdeberta-v3-base.
