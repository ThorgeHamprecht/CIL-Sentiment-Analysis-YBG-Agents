#!/bin/bash
# Run ONCE on the ETH student cluster LOGIN NODE (not as a SLURM job).
#
# Usage:
#   ssh thamprecht@student-cluster.inf.ethz.ch
#   cd /home/$USER/cil/project
#   bash scripts/setup_cluster_env.sh

set -e

SCRATCH="/work/scratch/$USER/cil"
VENV="$SCRATCH/venv"

# ── 1. Create scratch directories ─────────────────────────────────────────────
mkdir -p \
    "$SCRATCH/data" \
    "$SCRATCH/artifacts/06_rnn_bilstm" \
    "$SCRATCH/artifacts/07_transformer_custom" \
    "$SCRATCH/submissions" \
    "$SCRATCH/logs" \
    "$SCRATCH/.cache/torch"

# ── 2. Load module system + CUDA ──────────────────────────────────────────────
# The cluster uses modules, not conda. `. /etc/profile.d/modules.sh` is required
# before any `module` command (both here and inside SBATCH scripts).
. /etc/profile.d/modules.sh
module add cuda/13.0

echo "CUDA module loaded."

# ── 3. Create Python venv on scratch (keeps ~20 GB home quota free) ───────────
if [ -d "$VENV" ]; then
    echo "Venv already exists at $VENV — skipping creation."
else
    echo "Creating venv at $VENV ..."
    python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"
pip install --no-cache-dir --upgrade pip

# ── 4. Install PyTorch — cu130 wheel matches cuda/13.0 module ─────────────────
echo "Installing PyTorch (cu130)..."
pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

echo "Installing project dependencies..."
pip install --no-cache-dir pandas scikit-learn

# ── 5. Persist env vars in ~/.bashrc ──────────────────────────────────────────
if ! grep -q "SCRATCH_CIL" ~/.bashrc; then
    cat >> ~/.bashrc <<'EOF'

# CIL project
export SCRATCH_CIL="/work/scratch/$USER/cil"
export TORCH_HOME="$SCRATCH_CIL/.cache/torch"
EOF
fi

# ── 6. Verify ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Installed ==="
pip show torch | grep -E "^(Name|Version)"
echo ""
echo "=== Setup complete ==="
echo "Venv:  $VENV"
echo "CUDA shows False on login node (no GPU) — normal."
python -c "import torch; print('PyTorch:', torch.__version__)"
echo ""
echo "Next — copy Kaggle data to scratch (run on your Mac):"
echo "  rsync -av /path/to/train.csv /path/to/test.csv /path/to/test_solved.csv \\"
echo "    thamprecht@student-cluster.inf.ethz.ch:$SCRATCH/data/"
