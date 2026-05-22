#!/bin/bash
set -e

SCRATCH="${SCRATCH:-/work/scratch/$USER/cil}"

python eval_retrieval.py \
    --artifact_dir "$SCRATCH/artifacts/26_contrastive_pure_normal" \
    --data_dir "$SCRATCH/data" \
    --batch_size 32 \
    --k_values 1 7 101 \
    --retrieval_tau 0.07 \
    --cache_embeddings
