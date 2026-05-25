#!/bin/bash
#SBATCH --job-name=contrastive_28_all
#SBATCH --output=/work/scratch/%u/cil/logs/contrastive_28_all-%j.out
#SBATCH --error=/work/scratch/%u/cil/logs/contrastive_28_all-%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --time=7-00:00:00
#SBATCH --account=cil_jobs

set -e

BASE="/home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/28_contrastive_shared_head"

JOBS=(
    "normal w100_s000 1.0 0.0 0"
    "normal w050_s050 0.5 0.5 0"
    "normal w070_s030 0.7 0.3 0"
    "normal w030_s070 0.3 0.7 0"
    "distance_weighted w050_s050 0.5 0.5 0"
    "distance_weighted w070_s030 0.7 0.3 0"
    "distance_weighted w030_s070 0.3 0.7 0"
    "normal w050_s050 0.5 0.5 2"
    "normal w070_s030 0.7 0.3 2"
    "normal w030_s070 0.3 0.7 2"
    "distance_weighted w050_s050 0.5 0.5 2"
    "distance_weighted w070_s030 0.7 0.3 2"
    "distance_weighted w030_s070 0.3 0.7 2"
)

for job in "${JOBS[@]}"; do
    read -r variant tag w1 supcon warmup <<< "$job"
    echo ""
    echo "=== 28 shared-head: variant=$variant weights=$tag warmup=$warmup ==="
    bash "$BASE/submit_variant.sh" "$variant" "$tag" "$w1" "$supcon" "$warmup"
done

echo ""
echo "Done. Fetch results (run locally):"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:/work/scratch/roliveir/cil/artifacts/ ./artifacts/"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:/work/scratch/roliveir/cil/submissions/ ./submissions/"
