#!/bin/bash
#SBATCH --job-name=xlmr_large_emd
#SBATCH --output=/work/scratch/%u/cil/logs/xlmr_large_emd-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/xlmr_large_emd-%j.err
#SBATCH --time=12:00:00
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
    "$SCRATCH/artifacts/21_xlmr_large_emd" \
    "$SCRATCH/submissions" \
    "$SCRATCH/.cache/torch" \
    "$SCRATCH/.cache/huggingface" \
    "$SCRATCH/logs"

source "$SCRATCH/venv/bin/activate"

# ── Sanity check ──────────────────────────────────────────────────────────────
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
import transformers
print('Transformers:', transformers.__version__)
"

# ── Train ─────────────────────────────────────────────────────────────────────
cd /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/21_xlmr_large_emd

python train.py \
    --epochs       5    \
    --patience     3    \
    --batch_size   8    \
    --max_len      256  \
    --encoder_lr   1e-5 \
    --head_lr      1e-4 \
    --dropout      0.1  \
    --weight_decay 0.05 \
    --artifact_dir "$SCRATCH/artifacts/21_xlmr_large_emd" \
    --data_dir     "$SCRATCH/data"

# ── Predict + submission ──────────────────────────────────────────────────────
python predict.py \
    --checkpoint   "$SCRATCH/artifacts/21_xlmr_large_emd/best_model.pt" \
    --artifact_dir "$SCRATCH/artifacts/21_xlmr_large_emd" \
    --data_dir     "$SCRATCH/data" \
    --output_dir   "$SCRATCH/submissions"

echo ""
echo "Done. Fetch results (run locally):"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:$SCRATCH/submissions/ ./submissions/"
