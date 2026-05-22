#!/bin/bash
set -e

python train.py \
    --supcon_variant distance_weighted \
    --lambda_supcon 0.03 \
    --temperature 0.07 \
    --representation_dim 256
