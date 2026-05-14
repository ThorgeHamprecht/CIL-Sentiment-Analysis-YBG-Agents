#!/bin/bash
#SBATCH --job-name=mdeberta_frozen
#SBATCH --output=/work/scratch/%u/cil/logs/mdeberta_frozen-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/mdeberta_frozen-%j.err
#SBATCH --time=04:00:00
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
    "$SCRATCH/artifacts/25_mdeberta_frozen" \
    "$SCRATCH/submissions" \
    "$SCRATCH/logs"

. "$SCRATCH/venv/bin/activate"

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available(): print('GPU:', torch.cuda.get_device_name(0))
"

cd /home/thamprecht/cil/project/baselines/25_mdeberta_frozen

echo "=== Fine-tune (3 seeds, frozen encoder) ==="
python train.py \
    --seeds        42 1337 2024 \
    --epochs       20   \
    --patience     5    \
    --batch_size   64   \
    --lr           1e-3 \
    --dropout      0.1  \
    --weight_decay 0.01 \
    --max_len      256  \
    --artifact_dir "$SCRATCH/artifacts/25_mdeberta_frozen" \
    --data_dir     "$SCRATCH/data"

echo "=== Predict ==="
python predict.py \
    --artifact_dir "$SCRATCH/artifacts/25_mdeberta_frozen" \
    --data_dir     "$SCRATCH/data" \
    --output_dir   "$SCRATCH/submissions"

echo "=== Eval ==="
python eval_ensemble.py \
    --artifact_dir "$SCRATCH/artifacts/25_mdeberta_frozen" \
    --data_dir     "$SCRATCH/data"

echo ""
echo "Done. Fetch results:"
echo "  rsync -av thamprecht@student-cluster.inf.ethz.ch:$SCRATCH/submissions/ ./submissions/"
echo "  rsync -av thamprecht@student-cluster.inf.ethz.ch:$SCRATCH/artifacts/25_mdeberta_frozen/analysis/ ./baselines/25_mdeberta_frozen/analysis/"
