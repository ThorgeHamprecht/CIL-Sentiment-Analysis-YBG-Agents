# 27 Contrastive Shared Backbone (mDeBERTa-v3-base)

Shared backbone with a rating head (EMD^2 loss + median decode) and a contrastive head. Labels are 0-4 internally.

## Quick commands

Sanity checks:

```bash
python sanity_check.py
```

Run the full matrix on the cluster:

```bash
sbatch submit.sh
```

This submits one long RTX 5060 Ti job for all 27 variants. `submit_matrix.sh` is still available if you want to submit variants separately.

## Notes
- Total loss uses explicit weights: `(W1, SupCon) = (0.5, 0.5), (0.7, 0.3), (0.3, 0.7)`.
- SupCon variants: `normal` and `distance_weighted`.
- Warmup settings: `contrastive_warmup_epochs = 0` and `2`.
- Each finished variant evaluates `best_model.pt`, `epoch_004_model.pt`, and `epoch_006_model.pt`.
- Each eval runs `eval_mixed.py`, saving classifier-only, retrieval-only, and combined validation/test predictions.
- Test submissions are written to `$SCRATCH/submissions`; wide prediction files and disagreement analysis are under the artifact directory.
- EMA and LLRD follow the 23_mdeberta_llrd_ema recipe.
- Default backbone: microsoft/mdeberta-v3-base.
