#!/bin/bash
#SBATCH --job-name=27_pred_s2024
#SBATCH --output=/work/scratch/%u/cil/logs/27_pred_s2024-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/27_pred_s2024-%j.err
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

cd /home/thamprecht/cil/project/baselines/27_mdeberta_seed_split

echo "=== Predict seed 2024 only (best val=0.9085) ==="
python predict.py \
    --seeds        2024 \
    --artifact_dir "$SCRATCH/artifacts/27_mdeberta_seed_split" \
    --data_dir     "$SCRATCH/data" \
    --output_dir   "$SCRATCH/submissions"

echo ""
echo "Done. Fetch results:"
echo "  rsync -av thamprecht@student-cluster.inf.ethz.ch:$SCRATCH/submissions/ ./submissions/"
