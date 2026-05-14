#!/bin/bash
#SBATCH --job-name=xlmr_lora_soft
#SBATCH --time=01:00:00
#SBATCH --account=cil
#SBATCH --output=/work/scratch/thamprecht/cil/logs/xlmr_lora_soft-%j.out

. /etc/profile.d/modules.sh
module add cuda/13.0
source /work/scratch/thamprecht/cil/venv/bin/activate

export HF_HOME=/work/scratch/thamprecht/cil/.cache/huggingface

pip install --quiet --no-cache-dir peft

python /home/thamprecht/cil/project/baselines/11_xlmr_lora/predict_soft.py \
    --data_dir /work/scratch/thamprecht/cil/data \
    --artifact_dir /work/scratch/thamprecht/cil/artifacts/11_xlmr_lora
