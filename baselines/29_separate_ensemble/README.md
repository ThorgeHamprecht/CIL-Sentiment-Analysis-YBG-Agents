# 29 Separate EMD^2 + Contrastive Ensemble

Separate ordinal classifier and contrastive encoders, combined only at
probability level. This is the classification-SupCon ensemble reported in the
appendix.

See `HOW_TO_RUN.md` for the cluster command, expected data layout, and output
locations.

## Experiment Summary

- Classifier: mDeBERTa-v3-base, mean pooling, EMD^2 loss, median decoding,
  multi-sample dropout with 5 samples, LLRD, EMA.
- Contrastive encoders: separate pure SupCon models for standard and
  distance-weighted SupCon.
- Ensemble: mix classifier probabilities with contrastive class-medoid
  distributions; sweep `tau in {0.02, 0.05, 0.10, 0.20}` and the configured
  probability/product/confidence strategies.
- Training: 4 classifier epochs, 4 standard SupCon epochs, and 4
  distance-weighted SupCon epochs.
- Prediction: writes validation-best and epoch-4 test submissions.

Representative validation and solved-test results are reported in
`../../contrastive_appendix.tex`.
