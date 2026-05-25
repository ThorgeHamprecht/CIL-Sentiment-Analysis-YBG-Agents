#!/bin/bash
set -e

SCRATCH="${SCRATCH:-/work/scratch/$USER/cil}"

python predict_test.py \
    --artifact_dir "$SCRATCH/artifacts/26_contrastive_pure_normal" \
    --data_dir "$SCRATCH/data" \
    --output_dir "$SCRATCH/submissions" \
    --submission_prefix "26_contrastive_pure_normal" \
    --k_values 1 7 101 \
    --retrieval_tau 0.07 \
    --cache_embeddings
