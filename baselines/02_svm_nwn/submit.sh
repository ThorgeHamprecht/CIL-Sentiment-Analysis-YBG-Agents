#!/bin/bash
#SBATCH --job-name=02_svm_nwn
#SBATCH --output=/work/scratch/%u/cil/logs/02_svm_nwn-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/02_svm_nwn-%j.err
#SBATCH --time=01:00:00
#SBATCH --account=cil_jobs

set -e

SCRATCH="/work/scratch/$USER/cil"

mkdir -p "$SCRATCH/submissions" "$SCRATCH/logs"

. "$SCRATCH/venv/bin/activate"

cd /home/thamprecht/cil/project/baselines/02_svm_nwn

echo "=== TF-IDF + NWN + LinearSVC baseline ==="
python svm_nwn.py \
    --data-dir "$SCRATCH/data" \
    --out      "$SCRATCH/submissions/02_svm_nwn_submission.csv"

echo "Done. Sync back:"
echo "  rsync -av thamprecht@student-cluster.inf.ethz.ch:$SCRATCH/submissions/02_svm_nwn_submission.csv ./submissions/"
