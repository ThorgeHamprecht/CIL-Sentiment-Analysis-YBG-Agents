#!/bin/bash
set -e

python train.py \
    --supcon_variant normal \
    --lambda_supcon 0.01 \
    --temperature 0.07 \
    --representation_dim 256
