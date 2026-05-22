#!/bin/bash
set -e

SCRATCH="${SCRATCH:-/work/scratch/$USER/cil}"

python train.py \
    --supcon_variant normal \
    --temperature 0.07 \
    --projection_dim 128 \
    --batch_size 32 \
    --contrastive_head_lr 1e-4 \
    --artifact_dir "$SCRATCH/artifacts/26_contrastive_pure_normal" \
    --data_dir "$SCRATCH/data"
