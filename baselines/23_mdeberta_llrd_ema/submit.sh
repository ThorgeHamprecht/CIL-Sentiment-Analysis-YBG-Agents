#!/bin/bash
#SBATCH --job-name=mdeberta_llrd_ema
#SBATCH --output=/work/scratch/%u/cil/logs/mdeberta_llrd_ema-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/mdeberta_llrd_ema-%j.err
#SBATCH --time=20:00:00
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
    "$SCRATCH/artifacts/23_mdeberta_llrd_ema" \
    "$SCRATCH/submissions" \
    "$SCRATCH/.cache/torch" \
    "$SCRATCH/.cache/huggingface" \
    "$SCRATCH/logs"

. "$SCRATCH/venv/bin/activate"

# ── Sanity check ──────────────────────────────────────────────────────────────
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('BF16 support:', torch.cuda.is_bf16_supported())
import transformers
print('Transformers:', transformers.__version__)
"

cd /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/23_mdeberta_llrd_ema

# ── Fine-tune (3 seeds) ───────────────────────────────────────────────────────
echo "=== Fine-tune ==="
python train.py \
    --seeds        42 1337 2024 \
    --epochs       6    \
    --patience     3    \
    --batch_size   32   \
    --max_len      256  \
    --encoder_lr   8e-6 \
    --head_lr      5e-5 \
    --layer_decay  0.9  \
    --dropout      0.25 \
    --weight_decay 0.01 \
    --artifact_dir "$SCRATCH/artifacts/23_mdeberta_llrd_ema" \
    --data_dir     "$SCRATCH/data"

# ── Ensemble predict ──────────────────────────────────────────────────────────
echo "=== Predict ==="
python predict.py \
    --artifact_dir "$SCRATCH/artifacts/23_mdeberta_llrd_ema" \
    --data_dir     "$SCRATCH/data" \
    --output_dir   "$SCRATCH/submissions"

echo ""
echo "Done. Fetch results (run locally):"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:$SCRATCH/submissions/ ./submissions/"
