# 30 SupCon Backbone + New Heads

This experiment tests whether a pure SupCon-pretrained mDeBERTa backbone becomes a useful initialization for simple downstream heads.

## What It Does

1. Train `microsoft/mdeberta-v3-base` with normal pure SupCon for 4 epochs.
   - Distance-weighted SupCon is intentionally not part of this run.
   - Each epoch is evaluated with `medoid_distribution_median` at `tau=0.07`.
   - The best medoid-scoring checkpoint is copied to `supcon/best_backbone.pt`.
   - The projection head is discarded after pretraining.
2. Reuse the SupCon backbone for three downstream heads:
   - `frozen_coral`: frozen backbone, cached pooled features, linear CORAL ordinal-regression head.
   - `regression_linear_llrd`: scalar linear regression head with the backbone fine-tuned using LLRD + EMA.
   - `regression_mlp_llrd`: one-hidden-layer MLP scalar regression head with the backbone fine-tuned using LLRD + EMA.
3. Train each downstream head for 3 epochs and evaluate after every epoch.

Regression predictions are rounded and clipped to labels `0..4`.

## Run

```bash
sbatch /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/30_supcon_backbone_heads/submit.sh
```

The job requests an RTX 5060 Ti:

```bash
#SBATCH --gpus=5060ti:1
```

## Outputs

```text
/work/scratch/$USER/cil/artifacts/30_supcon_backbone_heads/
  supcon/epoch_001_model.pt ... epoch_004_model.pt
  supcon/best_backbone.pt
  frozen_coral_best.pt
  regression_linear_llrd_best.pt
  regression_mlp_llrd_best.pt
  analysis/supcon_pretrain.json
  analysis/head_analysis.json
  predictions/val_predictions.csv
  predictions/test_predictions.csv
  features/frozen_pooled_features.pt

/work/scratch/$USER/cil/submissions/
  30_supcon_backbone_frozen_coral_submission.csv
  30_supcon_backbone_regression_linear_llrd_submission.csv
  30_supcon_backbone_regression_mlp_llrd_submission.csv
```

## Smoke Test

```bash
cd /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/30_supcon_backbone_heads
python sanity_check.py
python train.py \
  --supcon_epochs 1 \
  --head_epochs 1 \
  --max_len 64 \
  --batch_size 10 \
  --eval_batch_size 10 \
  --retrieval_train_per_class 20 \
  --artifact_dir /tmp/30_smoke \
  --data_dir /work/scratch/$USER/cil/data \
  --output_dir /tmp/30_smoke_submissions
```

## Notes

- No LoRA and no seed averaging.
- The SupCon pretraining uses the same LLRD/EMA pattern as the earlier mDeBERTa contrastive runs.
- The frozen CORAL path caches pooled train/val/test features so the tiny ordinal head can be retrained cheaply.
