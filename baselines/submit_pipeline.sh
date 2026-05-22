#!/bin/bash
# Submit the full distillation pipeline with automatic SLURM dependency chaining.
# Run this once on the cluster login node:
#   bash /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/submit_pipeline.sh
#
# Execution order (only 1 job runs at a time per account):
#   [10 BiLSTM ordinal] ─┐
#                        ├─ run in parallel (both queue)
#   [11 XLM-R train]   ─┘
#       └─> [11 soft label generation]
#               └─> [12 distilled BiLSTM]

BASE=/home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines

# Step 1 and Teacher training — independent, queue together
JOB10=$(sbatch --parsable "${BASE}/10_bilstm_ordinal/submit.sh")
JOB11=$(sbatch --parsable "${BASE}/11_xlmr_lora/submit_train.sh")
echo "Submitted 10_bilstm_ordinal     -> job ${JOB10}"
echo "Submitted 11_xlmr_lora (train)  -> job ${JOB11}"

# Soft label generation — must wait for teacher training
JOB11S=$(sbatch --parsable --dependency=afterok:${JOB11} "${BASE}/11_xlmr_lora/submit_soft.sh")
echo "Submitted 11_xlmr_lora (soft)   -> job ${JOB11S}  [after ${JOB11}]"

# Distillation — submit only if queue allows, otherwise print manual command
JOB12=$(sbatch --parsable --dependency=afterok:${JOB11S} "${BASE}/12_distilled_bilstm/submit.sh" 2>/dev/null)
if [ -n "${JOB12}" ]; then
    echo "Submitted 12_distilled_bilstm   -> job ${JOB12}  [after ${JOB11S}]"
else
    echo ""
    echo "*** Queue limit reached — submit job 12 manually once job ${JOB11S} finishes: ***"
    echo "  sbatch --dependency=afterok:${JOB11S} ${BASE}/12_distilled_bilstm/submit.sh"
fi

echo ""
echo "Monitor: squeue -u $USER -o '%i %j %T %M %l %R'"
echo ""
echo "Fetch submissions when done:"
echo "  rsync -av roliveir@student-cluster.inf.ethz.ch:/work/scratch/$USER/cil/submissions/ \\"
echo "    \"/Users/thorge/Documents/ETH/MS Semester 2/CIL/CIL Sentiment Analysis/submissions/\""
