#!/bin/bash
set -e

BASE="/home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines"

echo "Submitting one all-combinations job for 27 and one for 28."
echo "Each job requests an RTX 5060 Ti via --gpus=5060ti:1 and runs:"
echo "  2 SupCon variants x 3 W1/SupCon loss splits x 2 warmup settings."
echo ""

sbatch "$BASE/27_contrastive_shared_backbone/submit.sh"
sbatch "$BASE/28_contrastive_shared_head/submit.sh"

echo ""
echo "Monitor:"
echo "  squeue -u $USER -o '%i %j %T %M %l %R'"
echo "  tail -f /work/scratch/$USER/cil/logs/contrastive_27_all-<jobid>.out"
echo "  tail -f /work/scratch/$USER/cil/logs/contrastive_28_all-<jobid>.out"
