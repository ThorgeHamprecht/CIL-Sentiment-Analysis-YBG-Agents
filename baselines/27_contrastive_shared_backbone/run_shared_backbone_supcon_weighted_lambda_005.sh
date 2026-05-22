#!/bin/bash
set -e

python train.py \
    --supcon_variant distance_weighted \
    --lambda_supcon 0.05 \
    --temperature 0.07 \
    --projection_dim 128
