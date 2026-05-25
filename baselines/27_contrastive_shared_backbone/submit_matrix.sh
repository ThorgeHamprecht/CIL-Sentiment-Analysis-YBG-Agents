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
    "normal w050_s050 0.5 0.5"
    "normal w070_s030 0.7 0.3"
    "normal w030_s070 0.3 0.7"
    "distance_weighted w050_s050 0.5 0.5"
    "distance_weighted w070_s030 0.7 0.3"
    "distance_weighted w030_s070 0.3 0.7"
)

count=0
echo "27 shared-backbone matrix:"
for job in "${JOBS[@]}"; do
    read -r variant tag w1 supcon <<< "$job"
    cmd="sbatch $BASE/submit_variant.sh $variant $tag $w1 $supcon"
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
