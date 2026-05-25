#!/bin/bash
#SBATCH --job-name=contrastive_27_all
#SBATCH --output=/work/scratch/%u/cil/logs/contrastive_27_all-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/contrastive_27_all-%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --time=7-00:00:00
#SBATCH --account=cil_jobs

set -e

BASE="/home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/27_contrastive_shared_backbone"

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
    echo ""
    echo "=== 27 shared-backbone: variant=$variant weights=$tag warmup=$warmup ==="
    bash "$BASE/submit_variant.sh" "$variant" "$tag" "$w1" "$supcon" "$warmup"
done

echo ""
echo "Done. Fetch results (run locally):"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:/work/scratch/roliveir/cil/artifacts/ ./artifacts/"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:/work/scratch/roliveir/cil/submissions/ ./submissions/"
