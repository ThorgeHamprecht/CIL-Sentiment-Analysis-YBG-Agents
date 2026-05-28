#!/bin/bash
#SBATCH --job-name=plot3_probs
#SBATCH --output=/work/scratch/%u/cil/logs/plot3_probs-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/plot3_probs-%j.err
#SBATCH --time=08:00:00
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
    "$SCRATCH/artifacts/plot3_mdeberta" \
    "$SCRATCH/artifacts/plot3_bilstm"   \
    "$SCRATCH/submissions"               \
    "$SCRATCH/logs"

. "$SCRATCH/venv/bin/activate"

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

# ── Part 1: mDeBERTa seed 2024 (~4-5h) ───────────────────────────────────────
echo "======================================================"
echo "Part 1: mDeBERTa seed 2024 — retrain + save probs"
echo "======================================================"
cd /home/thamprecht/cil/project/baselines/27_mdeberta_seed_split

python save_probs.py \
    --data_dir     "$SCRATCH/data"               \
    --artifact_dir "$SCRATCH/artifacts/plot3_mdeberta" \
    --output_dir   "$SCRATCH/submissions"        \
    --max_len      256  \
    --batch_size   32   \
    --encoder_lr   8e-6 \
    --head_lr      5e-5 \
    --layer_decay  0.9  \
    --dropout      0.25 \
    --weight_decay 0.01 \
    --epochs       6    \
    --patience     1

echo ""
echo "mDeBERTa done."

# ── Part 2: BiLSTM seed 42 (~1.5h) ───────────────────────────────────────────
echo "======================================================"
echo "Part 2: BiLSTM seed 42 — retrain + save probs"
echo "======================================================"
cd /home/thamprecht/cil/project/baselines/24_bilstm_emd_ensemble

python save_probs_bilstm.py \
    --data_dir     "$SCRATCH/data"              \
    --artifact_dir "$SCRATCH/artifacts/plot3_bilstm" \
    --output_dir   "$SCRATCH/submissions"       \
    --max_vocab    30000 \
    --max_len_title 64   \
    --max_len_body  192  \
    --embed_dim    128   \
    --hidden_dim   384   \
    --num_layers   2     \
    --dropout      0.3   \
    --batch_size   256   \
    --lr           1e-3  \
    --epochs       30    \
    --patience     6     \
    --warmup_epochs 2

echo ""
echo "BiLSTM done."

echo "======================================================"
echo "All done. Sync back:"
echo "  rsync -av thamprecht@student-cluster.inf.ethz.ch:$SCRATCH/submissions/seed2024_test_probs.npy ./submissions/"
echo "  rsync -av thamprecht@student-cluster.inf.ethz.ch:$SCRATCH/submissions/bilstm_seed42_test_probs.npy ./submissions/"
echo "======================================================"
