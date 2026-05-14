#!/bin/bash
#SBATCH --job-name=xlmr_lora_train
#SBATCH --time=08:00:00
#SBATCH --account=cil_jobs
#SBATCH --output=/work/scratch/thamprecht/cil/logs/xlmr_lora_train-%j.out

. /etc/profile.d/modules.sh
module add cuda/13.0
source /work/scratch/thamprecht/cil/venv/bin/activate

export HF_HOME=/work/scratch/thamprecht/cil/.cache/huggingface

pip install --quiet --no-cache-dir peft

python /home/thamprecht/cil/project/baselines/11_xlmr_lora/train.py \
    --data_dir /work/scratch/thamprecht/cil/data \
    --artifact_dir /work/scratch/thamprecht/cil/artifacts/11_xlmr_lora
