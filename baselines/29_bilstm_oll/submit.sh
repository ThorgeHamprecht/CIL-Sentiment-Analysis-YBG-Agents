#!/bin/bash
#SBATCH --job-name=29_bilstm_oll
#SBATCH --output=/work/scratch/%u/cil/logs/29_bilstm_oll-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/29_bilstm_oll-%j.err
#SBATCH --time=02:00:00
#SBATCH --account=cil_jobs

set -e

. /etc/profile.d/modules.sh
module add cuda/13.0

SCRATCH="/work/scratch/$USER/cil"
export TORCH_HOME="$SCRATCH/.cache/torch"
export HF_HOME="$SCRATCH/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$SCRATCH/artifacts/29_bilstm_oll" "$SCRATCH/submissions" "$SCRATCH/logs"

. "$SCRATCH/venv/bin/activate"

cd /home/thamprecht/cil/project/baselines/29_bilstm_oll

echo "=== Train BiLSTM + OLL (5 epochs) ==="
python train.py \
    --data_dir     "$SCRATCH/data"                    \
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

echo "=== Predict ==="
python predict.py \
    --checkpoint "$SCRATCH/artifacts/29_bilstm_oll/best_model.pt" \
    --vocab      "$SCRATCH/artifacts/29_bilstm_oll/vocab.json"    \
    --data_dir   "$SCRATCH/data"                                   \
    --output_dir "$SCRATCH/submissions"

echo "Done. Sync back:"
echo "  rsync -av thamprecht@student-cluster.inf.ethz.ch:$SCRATCH/submissions/29_bilstm_oll_submission.csv ./submissions/"
