# 29 Separate EMD^2 + Contrastive Ensemble

This experiment trains one EMD^2 mDeBERTa classifier and two pure SupCon encoders, then ensembles the classifier distribution with medoid distributions from the contrastive encoder.

## What Runs

- Classifier: mDeBERTa-v3-base, EMD^2 loss, LLRD, EMA, mean pooling, multi-sample dropout with 5 samples.
- Contrastive encoders: pure SupCon `normal` and `distance_weighted`, trained separately.
- Validation ensemble sweep: 4 medoid temperatures times 6 ensemble strategies for each SupCon variant.
- Test outputs: validation-best submissions and epoch-5 submissions for every variant/tau/strategy combination.

## Main Command

```bash
sbatch /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/29_separate_ensemble/submit.sh
```

The script requests an RTX 5060 Ti:

```bash
#SBATCH --gpus=5060ti:1
```

## Outputs

```text
/work/scratch/$USER/cil/artifacts/29_separate_ensemble/
  classifier/epoch_001_model.pt ... epoch_005_model.pt
  contrastive_normal/epoch_001_model.pt ... epoch_005_model.pt
  contrastive_distance_weighted/epoch_001_model.pt ... epoch_005_model.pt
  analysis/epoch_ensemble_metrics.jsonl
  analysis/validation_summary.csv
  analysis/validation_best_by_combo.csv
  analysis/test_submission_manifest.csv
  predictions/
  embeddings/

/work/scratch/$USER/cil/submissions/
```

## Ensemble Strategies

For classifier probabilities `p_cls` and contrastive medoid probabilities `p_ret`:

- `probmix_a050`: `0.50 * p_cls + 0.50 * p_ret`
- `probmix_a075`: `0.75 * p_cls + 0.25 * p_ret`
- `probmix_a025`: `0.25 * p_cls + 0.75 * p_ret`
- `poe_symmetric`: normalize `p_cls * p_ret`
- `poe_prior_corrected`: normalize `p_cls * p_ret / p_train`
- `confidence_weighted`: entropy-based per-example weighted average

All ensemble distributions are decoded with the ordinal CDF median rule.

## Rerun Test Submission Generation

If training is finished and only the test submissions need to be regenerated:

```bash
cd /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/29_separate_ensemble
python predict.py \
  --artifact_dir /work/scratch/$USER/cil/artifacts/29_separate_ensemble \
  --data_dir /work/scratch/$USER/cil/data \
  --output_dir /work/scratch/$USER/cil/submissions \
  --retrieval_taus 0.02 0.05 0.10 0.20
```

## Smoke Test

```bash
cd /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/29_separate_ensemble
python sanity_check.py
python train.py \
  --epochs 1 \
  --max_len 64 \
  --batch_size 10 \
  --eval_batch_size 10 \
  --retrieval_train_per_class 20 \
  --artifact_dir /tmp/29_smoke \
  --data_dir /work/scratch/$USER/cil/data
```
