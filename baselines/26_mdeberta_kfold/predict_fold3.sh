#!/bin/bash
#SBATCH --job-name=26_predict_fold3
#SBATCH --output=/work/scratch/%u/cil/logs/26_predict_fold3-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/26_predict_fold3-%j.err
#SBATCH --time=01:00:00
#SBATCH --account=cil_jobs

set -e

. /etc/profile.d/modules.sh
module add cuda/13.0

SCRATCH="/work/scratch/$USER/cil"
export TORCH_HOME="$SCRATCH/.cache/torch"
export HF_HOME="$SCRATCH/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

. "$SCRATCH/venv/bin/activate"

cd /home/thamprecht/cil/project/baselines/26_mdeberta_kfold

echo "=== Predict fold 3 (best val=0.9083) ==="
python predict.py \
    --artifact_dir "$SCRATCH/artifacts/26_mdeberta_kfold" \
    --data_dir     "$SCRATCH/data" \
    --output_dir   "$SCRATCH/submissions" \
    --fold 3

echo ""
echo "Done. Fetch results:"
echo "  rsync -av thamprecht@student-cluster.inf.ethz.ch:$SCRATCH/submissions/ ./submissions/"
