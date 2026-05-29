# 30 SupCon Backbone + New Heads

SupCon pretraining followed by downstream heads. This experiment tests whether
a contrastive mDeBERTa backbone is a useful initialization for simpler ordinal
or regression heads.

See `HOW_TO_RUN.md` for the cluster command, expected data layout, and output
locations.

## Experiment Summary

- SupCon pretraining: mDeBERTa-v3-base with mean pooling and a
  `768 -> 512 -> 128` projection head.
- Checkpoint selection: best medoid-distribution validation score at
  retrieval temperature `0.07`.
- Downstream heads:
  - frozen CORAL head on cached pooled features,
  - linear scalar regression head with LLRD + EMA,
  - MLP scalar regression head `768 -> 512 -> 1` with LayerNorm, GELU, dropout.
- Training: 4 SupCon epochs followed by 3 downstream-head epochs.
- Prediction: writes one submission per downstream head.

Representative validation results are reported in `../../contrastive_appendix.tex`.
