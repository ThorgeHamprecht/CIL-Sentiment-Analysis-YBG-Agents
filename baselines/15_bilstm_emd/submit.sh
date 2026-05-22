#!/bin/bash
#SBATCH --job-name=bilstm_emd
#SBATCH --output=/work/scratch/%u/cil/logs/bilstm_emd-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/bilstm_emd-%j.err
#SBATCH --time=02:00:00
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
    "$SCRATCH/artifacts/15_bilstm_emd" \
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
BASELINE_DIR="/home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/15_bilstm_emd"
cd "$BASELINE_DIR"

python train.py \
    --epochs      20 \
    --patience    4  \
    --batch_size  256 \
    --lr          1e-3 \
    --max_len     256 \
    --hidden_dim  256 \
    --artifact_dir "$SCRATCH/artifacts/15_bilstm_emd" \
    --data_dir     "$SCRATCH/data"

# ── Predict + submission ──────────────────────────────────────────────────────
python predict.py \
    --checkpoint  "$SCRATCH/artifacts/15_bilstm_emd/best_model.pt" \
    --vocab       "$SCRATCH/artifacts/15_bilstm_emd/vocab.json" \
    --data_dir    "$SCRATCH/data" \
    --output_dir  "$SCRATCH/submissions"

echo ""
echo "Done. Fetch results (run locally):"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:$SCRATCH/submissions/ ./submissions/"
