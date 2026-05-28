#!/bin/bash
#SBATCH --job-name=mdeberta_kfold
#SBATCH --output=/work/scratch/%u/cil/logs/mdeberta_kfold-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/mdeberta_kfold-%j.err
#SBATCH --time=24:00:00
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
    "$SCRATCH/artifacts/26_mdeberta_kfold" \
    "$SCRATCH/submissions" \
    "$SCRATCH/logs"

. "$SCRATCH/venv/bin/activate"

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('BF16:', torch.cuda.is_bf16_supported())
"

cd /home/thamprecht/cil/project/baselines/26_mdeberta_kfold

echo "=== 5-fold training ==="
python train.py \
    --n_folds      5    \
    --epochs       6    \
    --patience     1    \
    --batch_size   32   \
    --max_len      256  \
    --encoder_lr   8e-6 \
    --head_lr      5e-5 \
    --layer_decay  0.9  \
    --dropout      0.25 \
    --weight_decay 0.01 \
    --artifact_dir "$SCRATCH/artifacts/26_mdeberta_kfold" \
    --data_dir     "$SCRATCH/data"

echo "=== Predict ==="
python predict.py \
    --artifact_dir "$SCRATCH/artifacts/26_mdeberta_kfold" \
    --data_dir     "$SCRATCH/data" \
    --output_dir   "$SCRATCH/submissions"

echo "=== Eval ==="
python eval_ensemble.py \
    --artifact_dir "$SCRATCH/artifacts/26_mdeberta_kfold" \
    --data_dir     "$SCRATCH/data"

echo ""
echo "Done. Fetch results:"
echo "  rsync -av thamprecht@student-cluster.inf.ethz.ch:$SCRATCH/submissions/ ./submissions/"
echo "  rsync -av thamprecht@student-cluster.inf.ethz.ch:$SCRATCH/artifacts/26_mdeberta_kfold/analysis/ ./baselines/26_mdeberta_kfold/analysis/"
