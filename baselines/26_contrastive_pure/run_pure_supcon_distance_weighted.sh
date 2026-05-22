#!/bin/bash
set -e

python train.py \
    --supcon_variant distance_weighted \
    --temperature 0.07 \
    --projection_dim 128 \
    --contrastive_head_lr 1e-4
