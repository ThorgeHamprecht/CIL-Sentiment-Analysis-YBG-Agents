#!/bin/bash
#SBATCH --job-name=28_mdeberta_ce
#SBATCH --output=/work/scratch/%u/cil/logs/28_mdeberta_ce-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/28_mdeberta_ce-%j.err
#SBATCH --time=06:00:00
#SBATCH --account=cil_jobs

set -e

. /etc/profile.d/modules.sh
module add cuda/13.0

SCRATCH="/work/scratch/$USER/cil"
export TORCH_HOME="$SCRATCH/.cache/torch"
export HF_HOME="$SCRATCH/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p \
    "$SCRATCH/artifacts/28_mdeberta_ce" \
    "$SCRATCH/submissions"              \
    "$SCRATCH/logs"

. "$SCRATCH/venv/bin/activate"

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

cd /home/thamprecht/cil/project/baselines/28_mdeberta_ce

echo "=== Train mDeBERTa + Cross-Entropy (ablation vs B19 W²) ==="
python train.py \
    --data_dir     "$SCRATCH/data"              \
    --artifact_dir "$SCRATCH/artifacts/28_mdeberta_ce" \
    --max_len      128  \
    --batch_size   64   \
    --encoder_lr   2e-5 \
    --head_lr      1e-4 \
    --dropout      0.1  \
    --epochs       6    \
    --patience     3

echo "=== Predict ==="
python predict.py \
    --data_dir     "$SCRATCH/data"              \
    --artifact_dir "$SCRATCH/artifacts/28_mdeberta_ce" \
    --output_dir   "$SCRATCH/submissions"       \
    --max_len      128

echo ""
echo "Done. Sync back:"
echo "  rsync -av thamprecht@student-cluster.inf.ethz.ch:$SCRATCH/submissions/28_mdeberta_ce_submission.csv ./submissions/"
