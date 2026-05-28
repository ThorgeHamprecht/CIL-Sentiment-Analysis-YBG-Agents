#!/bin/bash
#SBATCH --job-name=oll_ablations
#SBATCH --output=/work/scratch/%u/cil/logs/oll_ablations-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/oll_ablations-%j.err
#SBATCH --time=05:00:00
#SBATCH --account=cil_jobs

set -e

. /etc/profile.d/modules.sh
module add cuda/13.0

SCRATCH="/work/scratch/$USER/cil"
export TORCH_HOME="$SCRATCH/.cache/torch"
export HF_HOME="$SCRATCH/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p \
    "$SCRATCH/artifacts/29_bilstm_oll"     \
    "$SCRATCH/artifacts/30_transformer_oll" \
    "$SCRATCH/submissions"                  \
    "$SCRATCH/logs"

. "$SCRATCH/venv/bin/activate"

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

# ── Part 1: BiLSTM + OLL (~1.5h) ─────────────────────────────────────────────
echo "======================================================"
echo "Part 1: BiLSTM + OLL (same arch as B15 W²)"
echo "======================================================"
cd /home/thamprecht/cil/project/baselines/29_bilstm_oll

python train.py \
    --data_dir     "$SCRATCH/data"                  \
    --artifact_dir "$SCRATCH/artifacts/29_bilstm_oll" \
    --max_vocab    30000 \
    --max_len      256   \
    --embed_dim    128   \
    --hidden_dim   256   \
    --num_layers   2     \
    --dropout      0.3   \
    --batch_size   256   \
    --lr           1e-3  \
    --epochs       5     \
    --patience     4

python predict.py \
    --checkpoint "$SCRATCH/artifacts/29_bilstm_oll/best_model.pt" \
    --vocab      "$SCRATCH/artifacts/29_bilstm_oll/vocab.json"    \
    --data_dir   "$SCRATCH/data"                                   \
    --output_dir "$SCRATCH/submissions"

echo "BiLSTM OLL done."

# ── Part 2: Custom Transformer + OLL (~1.5h) ──────────────────────────────────
echo "======================================================"
echo "Part 2: Custom Transformer + OLL (same arch as B16 W²)"
echo "======================================================"
cd /home/thamprecht/cil/project/baselines/30_transformer_oll

python train.py \
    --data_dir     "$SCRATCH/data"                      \
    --artifact_dir "$SCRATCH/artifacts/30_transformer_oll" \
    --max_vocab    30000 \
    --max_len      256   \
    --d_model      256   \
    --nhead        4     \
    --num_layers   4     \
    --dim_feedforward 1024 \
    --dropout      0.1   \
    --batch_size   256   \
    --lr           5e-4  \
    --warmup_epochs 2    \
    --epochs       30    \
    --patience     5

python predict.py \
    --checkpoint "$SCRATCH/artifacts/30_transformer_oll/best_model.pt" \
    --vocab      "$SCRATCH/artifacts/30_transformer_oll/vocab.json"    \
    --data_dir   "$SCRATCH/data"                                        \
    --output_dir "$SCRATCH/submissions"

echo "Transformer OLL done."

echo "======================================================"
echo "All done. Sync back:"
echo "  rsync -av thamprecht@student-cluster.inf.ethz.ch:$SCRATCH/submissions/29_bilstm_oll_submission.csv ./submissions/"
echo "  rsync -av thamprecht@student-cluster.inf.ethz.ch:$SCRATCH/submissions/30_transformer_oll_submission.csv ./submissions/"
echo "======================================================"
