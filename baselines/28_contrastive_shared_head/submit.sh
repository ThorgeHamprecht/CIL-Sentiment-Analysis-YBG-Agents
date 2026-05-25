#!/bin/bash
#SBATCH --job-name=contrastive_28_default
#SBATCH --output=/work/scratch/%u/cil/logs/contrastive_28_default-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/contrastive_28_default-%j.err
#SBATCH --time=24:00:00
#SBATCH --account=cil_jobs

set -e

# Default single variant. Use submit_matrix.sh to print or submit the full matrix.
bash /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/28_contrastive_shared_head/submit_variant.sh \
    normal w050_s050 0.5 0.5
