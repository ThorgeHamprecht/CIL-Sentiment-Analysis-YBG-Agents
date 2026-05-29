#!/bin/bash
#SBATCH --job-name=mdeberta_coral
#SBATCH --output=/work/scratch/%u/cil/logs/mdeberta_coral-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/mdeberta_coral-%j.err
#SBATCH --time=08:00:00
#SBATCH --account=cil_jobs

set -e

. /etc/profile.d/modules.sh
module add cuda/13.0

SCRATCH="/work/scratch/$USER/cil"
export TORCH_HOME="$SCRATCH/.cache/torch"
export HF_HOME="$SCRATCH/.cache/huggingface"
export TRANSFORMERS_CACHE="$SCRATCH/.cache/huggingface"
# Never attempt network calls — model must be pre-downloaded on login node
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p \
    "$SCRATCH/artifacts/09_mdeberta_coral" \
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
cd /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/09_mdeberta_coral

python train.py \
    --epochs      5   \
    --patience    3   \
    --batch_size  64  \
    --max_len     128 \
    --encoder_lr  2e-5 \
    --head_lr     1e-4 \
    --dropout     0.1 \
    --artifact_dir "$SCRATCH/artifacts/09_mdeberta_coral" \
    --data_dir     "$SCRATCH/data"

# ── Predict + submission ──────────────────────────────────────────────────────
python predict.py \
    --checkpoint  "$SCRATCH/artifacts/09_mdeberta_coral/best_model.pt" \
    --artifact_dir "$SCRATCH/artifacts/09_mdeberta_coral" \
    --data_dir    "$SCRATCH/data" \
    --output_dir  "$SCRATCH/submissions"

echo ""
echo "Done. Fetch results (run locally):"
echo "  rsync -av <user>@student-cluster.inf.ethz.ch:$SCRATCH/submissions/ <local_folder>/submissions/"
