# 27 Contrastive Shared Backbone (mDeBERTa-v3-base)

Shared backbone with a rating head (EMD^2 loss + median decode) and a contrastive head. Labels are 0-4 internally.

## Quick commands

Sanity checks:

```bash
python sanity_check.py
```

Run the full 6-run matrix on the cluster:

```bash
bash submit_matrix.sh
bash submit_matrix.sh --submit 2
```

`submit_matrix.sh` prints all commands by default. With `--submit 2`, it submits only two jobs so the student-cluster queue limit is respected.

## Notes
- Total loss uses explicit weights: `(W1, SupCon) = (0.5, 0.5), (0.7, 0.3), (0.3, 0.7)`.
- SupCon variants: `normal` and `distance_weighted`.
- Each finished job runs `eval_mixed.py`, saving classifier-only, retrieval-only, and combined validation/test predictions.
- Test submissions are written to `$SCRATCH/submissions`; wide prediction files and disagreement analysis are under the artifact directory.
- EMA and LLRD follow the 23_mdeberta_llrd_ema recipe.
- Default backbone: microsoft/mdeberta-v3-base.
