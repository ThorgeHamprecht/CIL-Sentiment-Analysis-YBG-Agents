#!/bin/bash
#SBATCH --job-name=contrastive_29
#SBATCH --output=/work/scratch/%u/cil/logs/contrastive_29-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/contrastive_29-%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --time=7-00:00:00
#SBATCH --account=cil_jobs

set -e

. /etc/profile.d/modules.sh
module add cuda/13.0

SCRATCH="/work/scratch/$USER/cil"
ARTIFACT_DIR="$SCRATCH/artifacts/29_separate_ensemble"

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

cd /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/29_separate_ensemble

echo "=== Sanity checks ==="
python sanity_check.py

echo "=== Train folder 29 separate ensemble ==="
python train.py \
    --seed 42 \
    --split_seed 42 \
    --epochs 5 \
    --batch_size 32 \
    --eval_batch_size 64 \
    --max_len 256 \
    --classifier_encoder_lr 8e-6 \
    --classifier_head_lr 5e-5 \
    --contrastive_encoder_lr 8e-6 \
    --contrastive_head_lr 1e-4 \
    --layer_decay 0.9 \
    --weight_decay 0.01 \
    --classifier_dropout 0.25 \
    --contrastive_dropout 0.1 \
    --msd_samples 5 \
    --projection_dim 128 \
    --temperature 0.07 \
    --ema_decay 0.999 \
    --warmup_fraction 0.06 \
    --retrieval_train_per_class 1000 \
    --retrieval_taus 0.02 0.05 0.10 0.20 \
    --artifact_dir "$ARTIFACT_DIR" \
    --data_dir "$SCRATCH/data"

echo "=== Write validation-best and epoch-5 test submissions ==="
python predict.py \
    --artifact_dir "$ARTIFACT_DIR" \
    --data_dir "$SCRATCH/data" \
    --output_dir "$SCRATCH/submissions" \
    --max_len 256 \
    --batch_size 32 \
    --retrieval_taus 0.02 0.05 0.10 0.20 \
    --final_epoch 5

echo ""
echo "Done. Fetch results (run locally):"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:$ARTIFACT_DIR/ ./artifacts/29_separate_ensemble/"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:$SCRATCH/submissions/ ./submissions/"
