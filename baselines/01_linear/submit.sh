#!/bin/bash
#SBATCH --job-name=01_linear
#SBATCH --output=/work/scratch/%u/cil/logs/01_linear-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/01_linear-%j.err
#SBATCH --time=01:00:00
#SBATCH --account=cil_jobs

set -e

SCRATCH="/work/scratch/$USER/cil"

mkdir -p "$SCRATCH/submissions" "$SCRATCH/logs"

. "$SCRATCH/venv/bin/activate"

cd /home/thamprecht/cil/project/baselines/01_linear

echo "=== Linear TF-IDF baseline ==="
python linear.py \
    --data-dir "$SCRATCH/data" \
    --out      "$SCRATCH/submissions/01_linear_submission.csv"

echo "Done. Sync back:"
echo "  rsync -av thamprecht@student-cluster.inf.ethz.ch:$SCRATCH/submissions/01_linear_submission.csv ./submissions/"
