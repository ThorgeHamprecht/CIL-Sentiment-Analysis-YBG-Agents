#!/bin/bash
#SBATCH --job-name=rnn_improved
#SBATCH --output=/work/scratch/%u/cil/logs/rnn_improved-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/rnn_improved-%j.err
#SBATCH --time=03:00:00
#SBATCH --account=cil_jobs

set -e

# ── Required: initialise module system, then load CUDA ────────────────────────
. /etc/profile.d/modules.sh
module add cuda/13.0

# ── Storage layout ────────────────────────────────────────────────────────────
SCRATCH="/work/scratch/$USER/cil"
export TORCH_HOME="$SCRATCH/.cache/torch"

mkdir -p \
    "$SCRATCH/artifacts/08_rnn_improved" \
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
cd /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/08_rnn_improved

python train.py \
    --epochs          30 \
    --patience        6  \
    --warmup_epochs   2  \
    --batch_size      256 \
    --lr              1e-3 \
    --embed_dim       128 \
    --hidden_dim      384 \
    --num_layers      2   \
    --dropout         0.3 \
    --label_smoothing 0.1 \
    --max_len         256 \
    --artifact_dir "$SCRATCH/artifacts/08_rnn_improved" \
    --data_dir     "$SCRATCH/data"

# ── Predict + submission ──────────────────────────────────────────────────────
python predict.py \
    --checkpoint  "$SCRATCH/artifacts/08_rnn_improved/best_model.pt" \
    --vocab       "$SCRATCH/artifacts/08_rnn_improved/vocab.json" \
    --data_dir    "$SCRATCH/data" \
    --output_dir  "$SCRATCH/submissions"

echo ""
echo "Done. Fetch results (run locally):"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:$SCRATCH/submissions/ ./submissions/"
