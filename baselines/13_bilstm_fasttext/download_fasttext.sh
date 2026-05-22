#!/bin/bash
# Download FastText CC vectors for English and German.
# Run this ONCE interactively on the cluster login node (not via sbatch).
#   bash /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/13_bilstm_fasttext/download_fasttext.sh
#
# Downloads ~8.4 GB total to /work/scratch/$USER/cil/fasttext/
# Takes ~10-20 min depending on network speed.

DEST=/work/scratch/$USER/cil/fasttext
mkdir -p "$DEST"

echo "Downloading cc.en.300.vec.gz (~4.2 GB)..."
wget -c -P "$DEST" https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz

echo "Downloading cc.de.300.vec.gz (~4.2 GB)..."
wget -c -P "$DEST" https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz

echo "Done. Files:"
ls -lh "$DEST"
