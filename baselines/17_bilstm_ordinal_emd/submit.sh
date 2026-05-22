#!/bin/bash
#SBATCH --job-name=bilstm_ordinal_emd
#SBATCH --time=03:00:00
#SBATCH --account=cil_jobs
#SBATCH --output=/work/scratch/%u/cil/logs/bilstm_ordinal_emd-%j.out

. /etc/profile.d/modules.sh
module add cuda/13.0
source /work/scratch/$USER/cil/venv/bin/activate

ARTIFACT_DIR=/work/scratch/$USER/cil/artifacts/17_bilstm_ordinal_emd
DATA_DIR=/work/scratch/$USER/cil/data

python /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/17_bilstm_ordinal_emd/train.py \
    --data_dir "$DATA_DIR" \
    --artifact_dir "$ARTIFACT_DIR"

python /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/17_bilstm_ordinal_emd/predict.py \
    --data_dir "$DATA_DIR" \
    --artifact_dir "$ARTIFACT_DIR" \
    --submission_dir /work/scratch/$USER/cil/submissions
