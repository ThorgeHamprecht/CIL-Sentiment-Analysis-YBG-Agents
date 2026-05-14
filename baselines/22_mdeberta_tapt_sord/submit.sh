#!/bin/bash
#SBATCH --job-name=mdeberta_tapt_sord
#SBATCH --output=/work/scratch/%u/cil/logs/mdeberta_tapt_sord-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/mdeberta_tapt_sord-%j.err
#SBATCH --time=18:00:00
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
    "$SCRATCH/artifacts/22_mdeberta_tapt_sord" \
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

cd /home/thamprecht/cil/project/baselines/22_mdeberta_tapt_sord

# ── Fine-tune (3 seeds) ───────────────────────────────────────────────────────
echo "=== Fine-tune ==="
python train.py \
    --seeds        42 1337 2024 \
    --backbone_dir "microsoft/mdeberta-v3-base" \
    --epochs       4    \
    --batch_size   32   \
    --max_len      256  \
    --encoder_lr   1e-5 \
    --head_lr      2e-5 \
    --layer_decay  0.9  \
    --dropout      0.3  \
    --weight_decay 0.01 \
    --artifact_dir "$SCRATCH/artifacts/22_mdeberta_tapt_sord" \
    --data_dir     "$SCRATCH/data"

# ── Phase 3: Ensemble predict ─────────────────────────────────────────────────
echo "=== Phase 3: Predict ==="
python predict.py \
    --artifact_dir "$SCRATCH/artifacts/22_mdeberta_tapt_sord" \
    --data_dir     "$SCRATCH/data" \
    --output_dir   "$SCRATCH/submissions"

echo ""
echo "Done. Fetch results (run on your Mac):"
echo "  rsync -av thamprecht@student-cluster.inf.ethz.ch:$SCRATCH/submissions/ ./submissions/"
