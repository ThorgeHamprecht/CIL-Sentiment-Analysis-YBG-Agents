#!/bin/bash
#SBATCH --job-name=distilled_bilstm
#SBATCH --time=04:00:00
#SBATCH --account=cil_jobs
#SBATCH --output=/work/scratch/thamprecht/cil/logs/distilled_bilstm-%j.out

. /etc/profile.d/modules.sh
module add cuda/13.0
source /work/scratch/thamprecht/cil/venv/bin/activate

ARTIFACT_DIR=/work/scratch/thamprecht/cil/artifacts/12_distilled_bilstm
DATA_DIR=/work/scratch/thamprecht/cil/data
SOFT_DIR=/work/scratch/thamprecht/cil/artifacts/11_xlmr_lora

python /home/thamprecht/cil/project/baselines/12_distilled_bilstm/train.py \
    --data_dir "$DATA_DIR" \
    --artifact_dir "$ARTIFACT_DIR" \
    --soft_dir "$SOFT_DIR"

python /home/thamprecht/cil/project/baselines/12_distilled_bilstm/predict.py \
    --data_dir "$DATA_DIR" \
    --artifact_dir "$ARTIFACT_DIR" \
    --submission_dir /work/scratch/thamprecht/cil/submissions
