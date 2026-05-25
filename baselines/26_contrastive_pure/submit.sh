#!/bin/bash
#SBATCH --job-name=contrastive_pure
#SBATCH --output=/work/scratch/%u/cil/logs/contrastive_pure-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/contrastive_pure-%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --time=24:00:00
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
    "$SCRATCH/artifacts/26_contrastive_pure_normal" \
    "$SCRATCH/artifacts/26_contrastive_pure_distance_weighted" \
    "$SCRATCH/submissions" \
    "$SCRATCH/.cache/torch" \
    "$SCRATCH/.cache/huggingface" \
    "$SCRATCH/logs"

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

cd /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/26_contrastive_pure

python train.py \
    --seed 42 \
    --split_seed 42 \
    --epochs 6 \
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
    --retrieval_train_per_class 1000 \
    --checkpoint_metric knn_k7_weighted_median_score \
    --supcon_variant normal \
    --artifact_dir "$SCRATCH/artifacts/26_contrastive_pure_normal" \
    --data_dir "$SCRATCH/data"

python eval_retrieval.py \
    --artifact_dir "$SCRATCH/artifacts/26_contrastive_pure_normal" \
    --data_dir "$SCRATCH/data" \
    --k_values 1 7 101 \
    --retrieval_tau 0.07 \
    --cache_embeddings

python predict_test.py \
    --artifact_dir "$SCRATCH/artifacts/26_contrastive_pure_normal" \
    --data_dir "$SCRATCH/data" \
    --output_dir "$SCRATCH/submissions" \
    --submission_prefix "26_contrastive_pure_normal" \
    --k_values 1 7 101 \
    --retrieval_tau 0.07 \
    --cache_embeddings

python train.py \
    --seed 42 \
    --split_seed 42 \
    --epochs 6 \
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
    --retrieval_train_per_class 1000 \
    --checkpoint_metric knn_k7_weighted_median_score \
    --supcon_variant distance_weighted \
    --artifact_dir "$SCRATCH/artifacts/26_contrastive_pure_distance_weighted" \
    --data_dir "$SCRATCH/data"

python eval_retrieval.py \
    --artifact_dir "$SCRATCH/artifacts/26_contrastive_pure_distance_weighted" \
    --data_dir "$SCRATCH/data" \
    --k_values 1 7 101 \
    --retrieval_tau 0.07 \
    --cache_embeddings

python predict_test.py \
    --artifact_dir "$SCRATCH/artifacts/26_contrastive_pure_distance_weighted" \
    --data_dir "$SCRATCH/data" \
    --output_dir "$SCRATCH/submissions" \
    --submission_prefix "26_contrastive_pure_distance_weighted" \
    --k_values 1 7 101 \
    --retrieval_tau 0.07 \
    --cache_embeddings

echo ""
echo "Done. Fetch results (run locally):"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:$SCRATCH/artifacts/26_contrastive_pure_normal/ ./artifacts/"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:$SCRATCH/artifacts/26_contrastive_pure_distance_weighted/ ./artifacts/"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:$SCRATCH/submissions/ ./submissions/"
