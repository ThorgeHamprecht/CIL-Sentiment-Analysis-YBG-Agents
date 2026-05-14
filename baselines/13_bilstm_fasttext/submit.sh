#!/bin/bash
#SBATCH --job-name=bilstm_fasttext
#SBATCH --time=04:00:00
#SBATCH --account=cil_jobs
#SBATCH --output=/work/scratch/thamprecht/cil/logs/bilstm_fasttext-%j.out

. /etc/profile.d/modules.sh
module add cuda/13.0
source /work/scratch/thamprecht/cil/venv/bin/activate

ARTIFACT_DIR=/work/scratch/thamprecht/cil/artifacts/13_bilstm_fasttext
DATA_DIR=/work/scratch/thamprecht/cil/data
FT_DIR=/work/scratch/thamprecht/cil/fasttext

# Fail fast if FastText vectors haven't been downloaded yet
if [ ! -f "$FT_DIR/cc.en.300.vec.gz" ] || [ ! -f "$FT_DIR/cc.de.300.vec.gz" ]; then
    echo "ERROR: FastText vectors not found in $FT_DIR"
    echo "Run this first on the login node:"
    echo "  bash /home/thamprecht/cil/project/baselines/13_bilstm_fasttext/download_fasttext.sh"
    exit 1
fi

# Build vocab + embedding matrix (skipped if already done)
if [ ! -f "$ARTIFACT_DIR/embeddings.npy" ]; then
    echo "Building embedding matrix..."
    python /home/thamprecht/cil/project/baselines/13_bilstm_fasttext/build_embeddings.py \
        --data_dir "$DATA_DIR" \
        --artifact_dir "$ARTIFACT_DIR" \
        --fasttext_dir "$FT_DIR"
fi

python /home/thamprecht/cil/project/baselines/13_bilstm_fasttext/train.py \
    --data_dir "$DATA_DIR" \
    --artifact_dir "$ARTIFACT_DIR"

python /home/thamprecht/cil/project/baselines/13_bilstm_fasttext/predict.py \
    --data_dir "$DATA_DIR" \
    --artifact_dir "$ARTIFACT_DIR" \
    --submission_dir /work/scratch/thamprecht/cil/submissions
