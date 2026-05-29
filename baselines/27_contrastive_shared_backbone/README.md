# 27 Contrastive Shared Backbone

Joint ordinal classification and contrastive learning with one shared
mDeBERTa-v3-base backbone and two separate heads. The final supported experiment
runs the completed matrix summarized in the appendix.

See `HOW_TO_RUN.md` for the cluster command, expected data layout, and output
locations.

## Experiment Summary

- Backbone: `microsoft/mdeberta-v3-base`.
- Rating head: linear five-logit EMD^2 classifier with median decoding.
- Contrastive head: `768 -> 512 -> 128` projection head with LayerNorm, GELU,
  dropout, and L2-normalized output.
- Loss matrix: standard and distance-weighted SupCon; weights `(0.5, 0.5)` and
  `(0.3, 0.7)`; warmup-2 variants for contrastive-heavy runs.
- Training: 4 epochs per variant, batch size 32, max length 256, LLRD, EMA.
- Evaluation: classifier-only, retrieval-only, and classifier-retrieval
  combinations over kNN and medoid distributions.


