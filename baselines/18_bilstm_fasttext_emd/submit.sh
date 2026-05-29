#!/bin/bash
#SBATCH --job-name=bilstm_fasttext_emd
#SBATCH --time=04:00:00
#SBATCH --account=cil_jobs
#SBATCH --output=/work/scratch/%u/cil/logs/bilstm_fasttext_emd-%j.out

. /etc/profile.d/modules.sh
module add cuda/13.0
source /work/scratch/$USER/cil/venv/bin/activate

ARTIFACT_DIR=/work/scratch/$USER/cil/artifacts/18_bilstm_fasttext_emd
DATA_DIR=/work/scratch/$USER/cil/data
FT_DIR=/work/scratch/$USER/cil/fasttext

# Fail fast if FastText vectors haven't been downloaded yet
if [ ! -f "$FT_DIR/cc.en.300.vec.gz" ] || [ ! -f "$FT_DIR/cc.de.300.vec.gz" ]; then
    echo "ERROR: FastText vectors not found in $FT_DIR"
    echo "Run this first on the login node:"
    echo "  bash /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/13_bilstm_fasttext/download_fasttext.sh"
    exit 1
fi

# Build vocab + embedding matrix (skipped if already done)
if [ ! -f "$ARTIFACT_DIR/embeddings.npy" ]; then
    echo "Building embedding matrix..."
    python /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/18_bilstm_fasttext_emd/build_embeddings.py \
        --data_dir "$DATA_DIR" \
        --artifact_dir "$ARTIFACT_DIR" \
        --fasttext_dir "$FT_DIR"
fi

python /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/18_bilstm_fasttext_emd/train.py \
    --data_dir "$DATA_DIR" \
    --artifact_dir "$ARTIFACT_DIR"

python /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/18_bilstm_fasttext_emd/predict.py \
    --data_dir "$DATA_DIR" \
    --artifact_dir "$ARTIFACT_DIR" \
    --submission_dir /work/scratch/$USER/cil/submissions
