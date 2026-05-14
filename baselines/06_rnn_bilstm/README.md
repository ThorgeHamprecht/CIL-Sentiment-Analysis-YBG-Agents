# Baseline 06 — BiLSTM (RNN)

**Approach**: 2-layer bidirectional LSTM over word embeddings trained from scratch.
**Role**: Establishes what a sequential model (no attention, no pretraining) can achieve.
**Expected val score**: ~0.85–0.87
**Estimated training time**: ~45–90 min on a single GPU

## Architecture

```
Embedding(30k vocab, 128d, padding_idx=0)
  → Dropout(0.3)
  → BiLSTM(hidden=256, 2 layers, bidirectional)
  → concat [h_fwd, h_bwd] of last layer  →  512d
  → Dropout(0.3)
  → Linear(512 → 5 classes)
```

~8M parameters (dominated by embedding table).

- Packed sequences (no wasted compute on padding)
- CrossEntropyLoss, Adam lr=1e-3
- ReduceLROnPlateau on val loss (patience=2, factor=0.5)
- Early stopping (patience=4)
- Stratified 90/10 train/val split (random_state=42)

## Quick start (local)

```bash
cd baselines/06_rnn_bilstm
python train.py
python predict.py   # writes submissions/06_rnn_bilstm_submission.csv
```

## Cluster (ETH student cluster)

### One-time environment setup (run on login node)

```bash
ssh <nethz>@student-cluster.inf.ethz.ch

# Create env (skip if already done)
conda create -n cil python=3.11 -y
conda activate cil
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas scikit-learn

# Put cache on scratch to save home quota
export SCRATCH="/work/scratch/$USER/cil"
export TORCH_HOME="$SCRATCH/.cache/torch"
mkdir -p "$TORCH_HOME"
# Add the two exports above to ~/.bashrc
```

### Data placement

Place the Kaggle CSVs in the repo's `data/` folder:
```
data/train.csv
data/test.csv
data/test_solved.csv   # optional — enables local scoring
```

### Submit job

```bash
# From repo root
mkdir -p baselines/06_rnn_bilstm/logs
sbatch baselines/06_rnn_bilstm/submit.sh

# Monitor
squeue -u $USER -o '%i %j %T %M %l %R'
tail -f baselines/06_rnn_bilstm/logs/<jobid>.out
```

### Expected outputs

| File | Description |
|------|-------------|
| `baselines/06_rnn_bilstm/artifacts/vocab.json` | Vocabulary (30k tokens) |
| `baselines/06_rnn_bilstm/artifacts/best_model.pt` | Best checkpoint |
| `submissions/06_rnn_bilstm_submission.csv` | Kaggle submission (id, label) |

## Hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| `--max_vocab` | 30000 | Rare tokens → `[UNK]` |
| `--max_len` | 256 | Tokens per review |
| `--embed_dim` | 128 | Embedding dimension |
| `--hidden_dim` | 256 | Per-direction LSTM hidden size (512 total) |
| `--num_layers` | 2 | LSTM depth |
| `--dropout` | 0.3 | Embedding + final hidden |
| `--batch_size` | 256 | Fits any cluster GPU |
| `--lr` | 1e-3 | Adam learning rate |
| `--epochs` | 20 | Max (early stopping usually ~10–15) |
| `--patience` | 4 | Early stopping patience |
