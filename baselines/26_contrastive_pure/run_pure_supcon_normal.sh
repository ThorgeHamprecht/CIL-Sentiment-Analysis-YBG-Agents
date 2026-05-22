#!/bin/bash
set -e

python train.py \
    --supcon_variant normal \
    --temperature 0.07 \
    --projection_dim 128 \
    --contrastive_head_lr 1e-4
