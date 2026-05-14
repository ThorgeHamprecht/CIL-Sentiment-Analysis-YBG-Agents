#!/bin/bash
#SBATCH --job-name=bilstm_emd_ensemble
#SBATCH --output=/work/scratch/%u/cil/logs/bilstm_emd_ensemble-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/bilstm_emd_ensemble-%j.err
#SBATCH --time=05:00:00
#SBATCH --account=cil_jobs

set -e

. /etc/profile.d/modules.sh
module add cuda/13.0

SCRATCH="/work/scratch/$USER/cil"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p \
    "$SCRATCH/artifacts/24_bilstm_emd_ensemble" \
    "$SCRATCH/submissions" \
    "$SCRATCH/logs"

. "$SCRATCH/venv/bin/activate"

cd /home/thamprecht/cil/project/baselines/24_bilstm_emd_ensemble

echo "=== Fine-tune (3 seeds) ==="
python train.py \
    --seeds        42 1337 2024 \
    --epochs       30  \
    --patience     6   \
    --batch_size   256 \
    --lr           1e-3 \
    --warmup_epochs 2  \
    --artifact_dir "$SCRATCH/artifacts/24_bilstm_emd_ensemble" \
    --data_dir     "$SCRATCH/data"

echo "=== Predict ==="
python predict.py \
    --artifact_dir "$SCRATCH/artifacts/24_bilstm_emd_ensemble" \
    --data_dir     "$SCRATCH/data" \
    --output_dir   "$SCRATCH/submissions"

echo ""
echo "Done. Fetch results (run on your Mac):"
echo "  rsync -av thamprecht@student-cluster.inf.ethz.ch:$SCRATCH/submissions/ ./submissions/"
