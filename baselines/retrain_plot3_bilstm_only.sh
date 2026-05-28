#!/bin/bash
#SBATCH --job-name=plot3_bilstm
#SBATCH --output=/work/scratch/%u/cil/logs/plot3_bilstm-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/plot3_bilstm-%j.err
#SBATCH --time=03:00:00
#SBATCH --account=cil_jobs

set -e

. /etc/profile.d/modules.sh
module add cuda/13.0

SCRATCH="/work/scratch/$USER/cil"
export TORCH_HOME="$SCRATCH/.cache/torch"
export HF_HOME="$SCRATCH/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p \
    "$SCRATCH/artifacts/plot3_bilstm" \
    "$SCRATCH/submissions"            \
    "$SCRATCH/logs"

. "$SCRATCH/venv/bin/activate"

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

echo "======================================================"
echo "BiLSTM seed 42 — retrain + save probs"
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
echo "Done. Sync back:"
echo "  rsync -av thamprecht@student-cluster.inf.ethz.ch:$SCRATCH/submissions/bilstm_seed42_test_probs.npy ./submissions/"
