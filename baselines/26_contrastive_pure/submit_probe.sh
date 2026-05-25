#!/bin/bash
#SBATCH --job-name=contrastive_26_probe
#SBATCH --output=/work/scratch/%u/cil/logs/contrastive_26_probe-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/contrastive_26_probe-%j.err
#SBATCH --gpus=5060ti:1
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

mkdir -p "$SCRATCH/artifacts" "$SCRATCH/submissions" "$SCRATCH/.cache/torch" "$SCRATCH/.cache/huggingface" "$SCRATCH/logs"

. "$SCRATCH/venv/bin/activate"

nvidia-smi

python -c "
import os
import torch
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('BF16 support:', torch.cuda.is_bf16_supported())
else:
    raise SystemExit('CUDA is not available in this SLURM job; refusing to train mDeBERTa on CPU.')
import transformers
print('Transformers:', transformers.__version__)
"

cd /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/26_contrastive_pure

VARIANTS=("normal" "distance_weighted")

for variant in "${VARIANTS[@]}"; do
    ARTIFACT_DIR="$SCRATCH/artifacts/26_contrastive_pure_${variant}_pretrain3_probe2"
    SUBMISSION_PREFIX="26_pure_${variant}_pretrain3_probe2"
    mkdir -p "$ARTIFACT_DIR"

    echo ""
    echo "=== 26 pure contrastive pretrain + frozen W1 probe: variant=$variant ==="

    python train.py \
        --seed 42 \
        --split_seed 42 \
        --epochs 3 \
        --patience 3 \
        --batch_size 32 \
        --max_len 256 \
        --encoder_lr 8e-6 \
        --contrastive_head_lr 1e-4 \
        --layer_decay 0.9 \
        --dropout 0.1 \
        --weight_decay 0.01 \
        --temperature 0.07 \
        --projection_dim 128 \
        --checkpoint_metric supcon_val_loss \
        --no_retrieval_eval \
        --supcon_variant "$variant" \
        --artifact_dir "$ARTIFACT_DIR" \
        --data_dir "$SCRATCH/data"

    python train_probe.py \
        --artifact_dir "$ARTIFACT_DIR" \
        --data_dir "$SCRATCH/data" \
        --output_dir "$SCRATCH/submissions" \
        --submission_prefix "$SUBMISSION_PREFIX" \
        --probe_epochs 2 \
        --probe_lr 1e-3 \
        --probe_batch_size 2048 \
        --encoder_batch_size 32 \
        --k_values 1 7 101 \
        --retrieval_tau 0.07 \
        --alphas 0.5 0.7 0.3
done

echo ""
echo "Done. Fetch results (run locally):"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:$SCRATCH/artifacts/ ./artifacts/"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:$SCRATCH/submissions/ ./submissions/"
