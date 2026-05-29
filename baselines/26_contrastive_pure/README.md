# 26 Contrastive Pure

Pure supervised contrastive learning with an mDeBERTa-v3-base encoder, mean
pooling, and a projection head. The final supported experiment trains both
standard SupCon and distance-weighted SupCon, then evaluates kNN and class-medoid
retrieval decoders.

See `HOW_TO_RUN.md` for the cluster command, expected data layout, and output
locations.

## Experiment Summary

- Backbone: `microsoft/mdeberta-v3-base`.
- Head: `768 -> 512 -> 128` projection head with LayerNorm, GELU, dropout, and
  L2-normalized output.
- Losses: standard SupCon and distance-weighted SupCon.
- Training: 6 epochs per variant, batch size 32, max length 256, LLRD, EMA.
- Inference: kNN with `k in {1, 7, 101}` and medoid-distribution decoding.

Representative validation results are reported in `../../contrastive_appendix.tex`.
