#!/bin/bash
#SBATCH --job-name=contrastive_shared_backbone
#SBATCH --output=/work/scratch/%u/cil/logs/contrastive_shared_backbone-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/contrastive_shared_backbone-%j.err
#SBATCH --time=20:00:00
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
    "$SCRATCH/artifacts/27_contrastive_shared_backbone_normal" \
    "$SCRATCH/artifacts/27_contrastive_shared_backbone_distance_weighted" \
    "$SCRATCH/submissions" \
    "$SCRATCH/.cache/torch" \
    "$SCRATCH/.cache/huggingface" \
    "$SCRATCH/logs"

. "$SCRATCH/venv/bin/activate"

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('BF16 support:', torch.cuda.is_bf16_supported())
import transformers
print('Transformers:', transformers.__version__)
"

cd /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/27_contrastive_shared_backbone

python train.py \
    --seed 42 \
    --epochs 6 \
    --patience 3 \
    --batch_size 32 \
    --max_len 256 \
    --encoder_lr 8e-6 \
    --rating_head_lr 5e-5 \
    --contrastive_head_lr 1e-4 \
    --layer_decay 0.9 \
    --dropout 0.2 \
    --weight_decay 0.01 \
    --temperature 0.07 \
    --projection_dim 128 \
    --lambda_supcon 0.05 \
    --contrastive_warmup_epochs 2 \
    --supcon_variant normal \
    --artifact_dir "$SCRATCH/artifacts/27_contrastive_shared_backbone_normal" \
    --data_dir "$SCRATCH/data"

python train.py \
    --seed 42 \
    --epochs 6 \
    --patience 3 \
    --batch_size 32 \
    --max_len 256 \
    --encoder_lr 8e-6 \
    --rating_head_lr 5e-5 \
    --contrastive_head_lr 1e-4 \
    --layer_decay 0.9 \
    --dropout 0.2 \
    --weight_decay 0.01 \
    --temperature 0.07 \
    --projection_dim 128 \
    --lambda_supcon 0.05 \
    --contrastive_warmup_epochs 2 \
    --supcon_variant distance_weighted \
    --artifact_dir "$SCRATCH/artifacts/27_contrastive_shared_backbone_distance_weighted" \
    --data_dir "$SCRATCH/data"

echo ""
echo "Done. Fetch results (run locally):"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:$SCRATCH/artifacts/27_contrastive_shared_backbone_normal/ ./artifacts/"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:$SCRATCH/artifacts/27_contrastive_shared_backbone_distance_weighted/ ./artifacts/"
