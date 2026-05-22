#!/bin/bash
set -e

python train.py \
    --supcon_variant normal \
    --lambda_supcon 0.03 \
    --temperature 0.07 \
    --projection_dim 128
