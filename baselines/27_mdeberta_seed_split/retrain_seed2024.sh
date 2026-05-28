#!/bin/bash
#SBATCH --job-name=27_probs_s2024
#SBATCH --output=/work/scratch/%u/cil/logs/27_probs_s2024-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/27_probs_s2024-%j.err
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
    "$SCRATCH/artifacts/27_probs_seed2024" \
    "$SCRATCH/submissions" \
    "$SCRATCH/logs"

. "$SCRATCH/venv/bin/activate"

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

cd /home/thamprecht/cil/project/baselines/27_mdeberta_seed_split

echo "=== Retrain seed 2024 + save raw probs ==="
python save_probs.py \
    --data_dir     "$SCRATCH/data" \
    --artifact_dir "$SCRATCH/artifacts/27_probs_seed2024" \
    --output_dir   "$SCRATCH/submissions" \
    --max_len      256  \
    --batch_size   32   \
    --encoder_lr   8e-6 \
    --head_lr      5e-5 \
    --layer_decay  0.9  \
    --dropout      0.25 \
    --weight_decay 0.01 \
    --epochs       6    \
    --patience     1

echo ""
echo "Done. Sync back:"
echo "  rsync -av thamprecht@student-cluster.inf.ethz.ch:$SCRATCH/submissions/27_seed2024_*.csv ./submissions/"
echo "  rsync -av thamprecht@student-cluster.inf.ethz.ch:$SCRATCH/submissions/seed2024_test_probs.npy ./submissions/"
