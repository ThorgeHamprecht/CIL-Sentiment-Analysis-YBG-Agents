# 26 Contrastive Pure (mDeBERTa-v3-base)

Pure supervised contrastive learning with mean pooling and a projection head. Labels are 0-4 internally and checkpoints only (no kNN eval).

## Quick commands

Sanity checks:

```bash
python sanity_check.py
```

Run variants:

```bash
bash run_pure_supcon_normal.sh
bash run_pure_supcon_distance_weighted.sh
```

## Notes
- Loss: SupCon only (normal or distance-weighted negatives).
- EMA and LLRD follow the 23_mdeberta_llrd_ema recipe.
- Default backbone: microsoft/mdeberta-v3-base.
