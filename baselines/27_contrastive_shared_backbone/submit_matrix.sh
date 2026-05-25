#!/bin/bash
set -e

BASE="/home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/27_contrastive_shared_backbone"
SUBMIT=0
LIMIT=2

if [[ "${1:-}" == "--submit" ]]; then
    SUBMIT=1
    LIMIT="${2:-2}"
fi

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

count=0
echo "27 shared-backbone matrix:"
for job in "${JOBS[@]}"; do
    read -r variant tag w1 supcon warmup <<< "$job"
    cmd="sbatch $BASE/submit_variant.sh $variant $tag $w1 $supcon $warmup"
    if [[ "$SUBMIT" -eq 1 && "$count" -lt "$LIMIT" ]]; then
        echo "+ $cmd"
        $cmd
        count=$((count + 1))
    else
        echo "$cmd"
    fi
done

if [[ "$SUBMIT" -eq 1 ]]; then
    echo "Submitted $count job(s). Remaining commands are printed above for later, respecting the cluster queue limit."
fi
