#!/bin/bash
#SBATCH --job-name=custom_transformer
#SBATCH --output=/work/scratch/%u/cil/logs/transformer-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/transformer-%j.err
#SBATCH --time=04:00:00
#SBATCH --account=cil_jobs

set -e

# ── Required: initialise module system, then load CUDA ────────────────────────
. /etc/profile.d/modules.sh
module add cuda/13.0

# ── Storage layout ────────────────────────────────────────────────────────────
# Code (permanent, small):   /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/
# Venv (~4 GB, recreatable): /work/scratch/$USER/cil/venv/
# Data (re-downloadable):    /work/scratch/$USER/cil/data/
# Artifacts + submissions:   /work/scratch/$USER/cil/  ← rsync locally after job
SCRATCH="/work/scratch/$USER/cil"
export TORCH_HOME="$SCRATCH/.cache/torch"

mkdir -p \
    "$SCRATCH/artifacts/07_transformer_custom" \
    "$SCRATCH/submissions" \
    "$SCRATCH/.cache/torch" \
    "$SCRATCH/logs"

# ── Activate venv ─────────────────────────────────────────────────────────────
source "$SCRATCH/venv/bin/activate"

# ── Sanity check ──────────────────────────────────────────────────────────────
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

# ── Train ─────────────────────────────────────────────────────────────────────
BASELINE_DIR="/home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/07_transformer_custom"
cd "$BASELINE_DIR"

python train.py \
    --epochs          30 \
    --patience        5  \
    --warmup_epochs   2  \
    --batch_size      256 \
    --lr              5e-4 \
    --d_model         256 \
    --nhead           4 \
    --num_layers      4 \
    --dim_feedforward 1024 \
    --max_len         256 \
    --artifact_dir "$SCRATCH/artifacts/07_transformer_custom" \
    --data_dir     "$SCRATCH/data"

# ── Predict + submission ──────────────────────────────────────────────────────
python predict.py \
    --checkpoint  "$SCRATCH/artifacts/07_transformer_custom/best_model.pt" \
    --vocab       "$SCRATCH/artifacts/07_transformer_custom/vocab.json" \
    --data_dir    "$SCRATCH/data" \
    --output_dir  "$SCRATCH/submissions"

echo ""
echo "Done. Fetch results (run locally):"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:$SCRATCH/submissions/ ./submissions/"
