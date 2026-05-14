#!/bin/bash
#SBATCH --job-name=transformer_fasttext
#SBATCH --time=06:00:00
#SBATCH --account=cil_jobs
#SBATCH --output=/work/scratch/thamprecht/cil/logs/transformer_fasttext-%j.out

. /etc/profile.d/modules.sh
module add cuda/13.0
source /work/scratch/thamprecht/cil/venv/bin/activate

ARTIFACT_DIR=/work/scratch/thamprecht/cil/artifacts/14_transformer_fasttext
DATA_DIR=/work/scratch/thamprecht/cil/data
FT_DIR=/work/scratch/thamprecht/cil/fasttext
SRC_EMBED=/work/scratch/thamprecht/cil/artifacts/13_bilstm_fasttext

if [ ! -f "$FT_DIR/cc.en.300.vec.gz" ] || [ ! -f "$FT_DIR/cc.de.300.vec.gz" ]; then
    echo "ERROR: FastText vectors not found in $FT_DIR"
    exit 1
fi

# Reuse vocab + embeddings from step 13 if already built (same 50k vocab)
if [ ! -f "$ARTIFACT_DIR/embeddings.npy" ]; then
    if [ -f "$SRC_EMBED/embeddings.npy" ] && [ -f "$SRC_EMBED/vocab.json" ]; then
        echo "Reusing embedding matrix from step 13..."
        mkdir -p "$ARTIFACT_DIR"
        cp "$SRC_EMBED/embeddings.npy" "$ARTIFACT_DIR/embeddings.npy"
        cp "$SRC_EMBED/vocab.json"     "$ARTIFACT_DIR/vocab.json"
    else
        echo "Building embedding matrix..."
        python /home/thamprecht/cil/project/baselines/13_bilstm_fasttext/build_embeddings.py \
            --data_dir "$DATA_DIR" \
            --artifact_dir "$ARTIFACT_DIR" \
            --fasttext_dir "$FT_DIR"
    fi
fi

python /home/thamprecht/cil/project/baselines/14_transformer_fasttext/train.py \
    --data_dir "$DATA_DIR" \
    --artifact_dir "$ARTIFACT_DIR"

python /home/thamprecht/cil/project/baselines/14_transformer_fasttext/predict.py \
    --data_dir "$DATA_DIR" \
    --artifact_dir "$ARTIFACT_DIR" \
    --submission_dir /work/scratch/thamprecht/cil/submissions
