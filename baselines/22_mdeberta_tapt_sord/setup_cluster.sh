#!/bin/bash
# Run this ONCE on the login node (has internet access) before submitting the job.
# Compute nodes are likely firewalled — model must be cached here first.
set -e

SCRATCH="/work/scratch/$USER/cil"
export HF_HOME="$SCRATCH/.cache/huggingface"

source "$SCRATCH/venv/bin/activate"

echo "=== Installing transformers, sentencepiece, protobuf ==="
pip install --no-cache-dir transformers==4.40.0 sentencepiece protobuf accelerate

echo ""
echo "=== Pre-downloading microsoft/mdeberta-v3-base to $HF_HOME ==="
python - <<'EOF'
import os
from transformers import AutoModel, AutoTokenizer
model_name = "microsoft/mdeberta-v3-base"
print(f"Tokenizer...")
AutoTokenizer.from_pretrained(model_name)
print(f"Model weights...")
AutoModel.from_pretrained(model_name)
print("Download complete.")
EOF

echo ""
echo "=== All done. You can now submit the job: ==="
echo "  sbatch /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/09_mdeberta_coral/submit.sh"
