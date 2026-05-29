# 28 Contrastive Shared Head

Joint ordinal classification and contrastive learning with a shared
representation head. The pooled mDeBERTa-v3-base state is mapped to a compact
representation that is used both for SupCon and for EMD^2 classification.

See `HOW_TO_RUN.md` for the cluster command, expected data layout, and output
locations.

## Experiment Summary

- Backbone: `microsoft/mdeberta-v3-base`.
- Shared head: `768 -> 512 -> 256` with LayerNorm, GELU, and dropout.
- Rating head: linear five-logit EMD^2 classifier on the shared representation.
- Contrastive embedding: L2-normalized shared representation.
- Loss matrix: standard and distance-weighted SupCon; weights `(0.5, 0.5)` and
  `(0.3, 0.7)`; warmup-2 variants for contrastive-heavy runs.
- Training: 4 epochs per variant, batch size 32, max length 256, LLRD, EMA.
- Evaluation: classifier-only, retrieval-only, and classifier-retrieval
  combinations over kNN and medoid distributions.

Representative validation results are reported in `../../contrastive_appendix.tex`.
