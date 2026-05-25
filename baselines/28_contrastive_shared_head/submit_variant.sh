#!/bin/bash
#SBATCH --job-name=contrastive_28
#SBATCH --output=/work/scratch/%u/cil/logs/contrastive_28-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/contrastive_28-%j.err
#SBATCH --time=24:00:00
#SBATCH --account=cil_jobs

set -e

SUPCON_VARIANT="${1:-normal}"
WEIGHT_TAG="${2:-w050_s050}"
W1_WEIGHT="${3:-0.5}"
SUPCON_WEIGHT="${4:-0.5}"

. /etc/profile.d/modules.sh
module add cuda/13.0

SCRATCH="/work/scratch/$USER/cil"
ARTIFACT_DIR="$SCRATCH/artifacts/28_contrastive_shared_head_${SUPCON_VARIANT}_${WEIGHT_TAG}"
SUBMISSION_PREFIX="28_shared_head_${SUPCON_VARIANT}_${WEIGHT_TAG}"

export TORCH_HOME="$SCRATCH/.cache/torch"
export HF_HOME="$SCRATCH/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$ARTIFACT_DIR" "$SCRATCH/submissions" "$SCRATCH/.cache/torch" "$SCRATCH/.cache/huggingface" "$SCRATCH/logs"

. "$SCRATCH/venv/bin/activate"

nvidia-smi

python -c "
import os
import torch
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print('GPU:', gpu_name)
    print('BF16 support:', torch.cuda.is_bf16_supported())
    if 'RTX 5060 Ti' not in gpu_name:
        raise SystemExit(f'Need RTX 5060 Ti for this job to avoid OOM; got {gpu_name}. Resubmit later.')
else:
    raise SystemExit('CUDA is not available in this SLURM job; refusing to train mDeBERTa on CPU.')
import transformers
print('Transformers:', transformers.__version__)
"

cd /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/28_contrastive_shared_head

echo "Running 28 shared-head variant=$SUPCON_VARIANT weights=$WEIGHT_TAG w1=$W1_WEIGHT supcon=$SUPCON_WEIGHT"

python train.py \
    --seed 42 \
    --epochs 6 \
    --patience 3 \
    --batch_size 32 \
    --max_len 256 \
    --encoder_lr 8e-6 \
    --shared_head_lr 1e-4 \
    --rating_head_lr 5e-5 \
    --layer_decay 0.9 \
    --dropout 0.1 \
    --weight_decay 0.01 \
    --temperature 0.07 \
    --representation_dim 256 \
    --w1_loss_weight "$W1_WEIGHT" \
    --supcon_loss_weight "$SUPCON_WEIGHT" \
    --contrastive_warmup_epochs 2 \
    --supcon_variant "$SUPCON_VARIANT" \
    --artifact_dir "$ARTIFACT_DIR" \
    --data_dir "$SCRATCH/data"

python eval_mixed.py \
    --artifact_dir "$ARTIFACT_DIR" \
    --data_dir "$SCRATCH/data" \
    --output_dir "$SCRATCH/submissions" \
    --submission_prefix "$SUBMISSION_PREFIX" \
    --batch_size 32 \
    --k_values 1 7 101 \
    --retrieval_tau 0.07 \
    --alphas 0.5 0.7 0.3 \
    --cache_embeddings

echo ""
echo "Done. Fetch results (run locally):"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:$ARTIFACT_DIR/ ./artifacts/"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:$SCRATCH/submissions/ ./submissions/"
