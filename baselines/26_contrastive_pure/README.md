# 26 Contrastive Pure (mDeBERTa-v3-base)

Pure supervised contrastive learning with mean pooling and a projection head. Labels are 0-4 internally. Validation includes SupCon loss plus retrieval metrics from kNN and class medoids.

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

Full retrieval evaluation after a checkpoint exists:

```bash
bash eval_pure_supcon_normal.sh
bash eval_pure_supcon_distance_weighted.sh
```

Create `test.csv` submissions after a checkpoint exists:

```bash
bash predict_pure_supcon_normal.sh
bash predict_pure_supcon_distance_weighted.sh
```

## Notes
- Loss: SupCon only (normal or distance-weighted negatives).
- EMA and LLRD follow the 23_mdeberta_llrd_ema recipe.
- Default backbone: microsoft/mdeberta-v3-base.
- During training, `best_model.pt` is selected by `knn_k7_weighted_median_score` unless `--checkpoint_metric supcon_val_loss` is passed.
- Epoch retrieval metrics are saved to `analysis/epoch_retrieval_metrics.jsonl`.
- Full retrieval metrics are saved to `analysis/retrieval_eval.json` with confusion matrices as CSV files.
- Test predictions are saved to `predictions/test_predictions.csv`; one Kaggle submission per retrieval method is written to `$SCRATCH/submissions`.
