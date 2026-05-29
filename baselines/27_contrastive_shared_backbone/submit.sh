#!/bin/bash
#SBATCH --job-name=contrastive_27_all
#SBATCH --output=/work/scratch/%u/cil/logs/contrastive_27_all-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/contrastive_27_all-%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --time=7-00:00:00
#SBATCH --account=cil_jobs

set -euo pipefail

BASE="/home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/27_contrastive_shared_backbone"
SCRATCH="/work/scratch/$USER/cil"

. /etc/profile.d/modules.sh
module add cuda/13.0

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

cd "$BASE"

run_variant() {
    local supcon_variant="$1"
    local weight_tag="$2"
    local w1_weight="$3"
    local supcon_weight="$4"
    local warmup_epochs="$5"
    local warmup_tag="warmup${warmup_epochs}"
    local artifact_dir="$SCRATCH/artifacts/27_contrastive_shared_backbone_${supcon_variant}_${weight_tag}_${warmup_tag}"
    local submission_prefix="27_shared_backbone_${supcon_variant}_${weight_tag}_${warmup_tag}"

    mkdir -p "$artifact_dir"

    echo ""
    echo "=== 27 shared-backbone: variant=$supcon_variant weights=$weight_tag warmup=$warmup_epochs ==="
    echo "Running 27 shared-backbone variant=$supcon_variant weights=$weight_tag warmup=$warmup_epochs w1=$w1_weight supcon=$supcon_weight"

    python train.py \
        --seed 42 \
        --epochs 4 \
        --patience 4 \
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
        --w1_loss_weight "$w1_weight" \
        --supcon_loss_weight "$supcon_weight" \
        --contrastive_warmup_epochs "$warmup_epochs" \
        --supcon_variant "$supcon_variant" \
        --save_epoch_checkpoints 4 \
        --artifact_dir "$artifact_dir" \
        --data_dir "$SCRATCH/data"

    for checkpoint in best_model.pt epoch_004_model.pt; do
        if [ -f "$artifact_dir/$checkpoint" ]; then
            local checkpoint_tag="${checkpoint%.pt}"
            echo "Evaluating checkpoint $checkpoint"
            python eval_mixed.py \
                --artifact_dir "$artifact_dir" \
                --checkpoint_name "$checkpoint" \
                --data_dir "$SCRATCH/data" \
                --output_dir "$SCRATCH/submissions" \
                --submission_prefix "${submission_prefix}_${checkpoint_tag}" \
                --batch_size 32 \
                --k_values 1 7 101 \
                --retrieval_tau 0.07 \
                --alphas 0.5 0.7 0.3 \
                --cache_embeddings
        else
            echo "Skipping missing checkpoint $artifact_dir/$checkpoint"
        fi
    done
}

JOBS=(
    "normal w050_s050 0.5 0.5 0"
    "normal w030_s070 0.3 0.7 0"
    "distance_weighted w050_s050 0.5 0.5 0"
    "distance_weighted w030_s070 0.3 0.7 0"
    "normal w030_s070 0.3 0.7 2"
    "distance_weighted w030_s070 0.3 0.7 2"
)

for job in "${JOBS[@]}"; do
    read -r variant tag w1 supcon warmup <<< "$job"
    run_variant "$variant" "$tag" "$w1" "$supcon" "$warmup"
done

echo ""
echo "Done. Fetch results (run locally):"
echo "  rsync -av <user>@student-cluster.inf.ethz.ch:$SCRATCH/artifacts/ <local_folder>/artifacts/"
echo "  rsync -av <user>@student-cluster.inf.ethz.ch:$SCRATCH/submissions/ <local_folder>/submissions/"
