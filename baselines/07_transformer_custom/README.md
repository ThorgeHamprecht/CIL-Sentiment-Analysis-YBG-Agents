# Baseline 07 — Custom Transformer (from scratch)

**Approach**: 4-layer Transformer encoder with [CLS] pooling, trained entirely from scratch.
**Role**: Answers "does attention help over sequential processing (BiLSTM)?" and "how much does pretraining matter (vs RoBERTa)?"
**Expected val score**: ~0.88–0.90
**Estimated training time**: ~1.5–3h on a single GPU

## Architecture

```
[CLS] prepended to each sequence
Embedding(~30k vocab, 256d, padding_idx=0)
  × sqrt(256)
  + LearnedPositionalEncoding(256d, max_len=256)
  → TransformerEncoder × 4 layers
      each: Pre-LN, MultiHeadAttention(4 heads, 64d/head), FFN(1024d), Dropout(0.1)
  → LayerNorm
  → [CLS] token repr at position 0  →  256d
  → Linear(256 → 5 classes)
```

~11M parameters.

Key design choices:
- **Same tokenizer + vocab size as BiLSTM** (baseline 06) — isolates the architectural difference
- **Pre-LN** (`norm_first=True`) for stable training from random init
- **Learned positional encoding** (simpler than sinusoidal for classification)
- **[CLS] pooling** — standard for encoder-only classification
- **Warmup + cosine annealing**: 2 epochs linear warmup → cosine decay
- AdamW (weight_decay=1e-2), early stopping (patience=5)
- Stratified 90/10 train/val split (random_state=42)

## Quick start (local)

```bash
cd baselines/07_transformer_custom
python train.py
python predict.py   # writes submissions/07_transformer_custom_submission.csv
```

## Cluster (ETH student cluster)

### One-time environment setup (run on login node)

```bash
ssh <nethz>@student-cluster.inf.ethz.ch

# Create env (skip if already done for baseline 06)
conda create -n cil python=3.11 -y
conda activate cil
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas scikit-learn

export SCRATCH="/work/scratch/$USER/cil"
export TORCH_HOME="$SCRATCH/.cache/torch"
mkdir -p "$TORCH_HOME"
# Add the two exports above to ~/.bashrc
```

### Data placement

```
data/train.csv
data/test.csv
data/test_solved.csv   # optional — enables local scoring
```

### Submit job

```bash
# From repo root
mkdir -p baselines/07_transformer_custom/logs
sbatch baselines/07_transformer_custom/submit.sh

# Monitor
squeue -u $USER -o '%i %j %T %M %l %R'
tail -f baselines/07_transformer_custom/logs/<jobid>.out
```

### Expected outputs

| File | Description |
|------|-------------|
| `baselines/07_transformer_custom/artifacts/vocab.json` | Vocabulary (~30k tokens, includes [CLS]) |
| `baselines/07_transformer_custom/artifacts/best_model.pt` | Best checkpoint |
| `submissions/07_transformer_custom_submission.csv` | Kaggle submission (id, label) |

## Hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| `--max_vocab` | 30000 | Rare tokens → `[UNK]` |
| `--max_len` | 256 | Includes [CLS]; content = 255 tokens |
| `--d_model` | 256 | Embedding + hidden dimension |
| `--nhead` | 4 | Attention heads (64d per head) |
| `--num_layers` | 4 | Encoder depth |
| `--dim_feedforward` | 1024 | FFN inner dim (4× d_model) |
| `--dropout` | 0.1 | Applied throughout |
| `--batch_size` | 256 | Fits any cluster GPU |
| `--lr` | 5e-4 | Peak LR after warmup |
| `--warmup_epochs` | 2 | Linear warmup before cosine decay |
| `--epochs` | 30 | Max (from scratch needs more epochs) |
| `--patience` | 5 | Early stopping patience |

## Ablation narrative

| Model | Pretrained? | Architecture | Expected score |
|-------|------------|--------------|----------------|
| BiLSTM (06) | No | Sequential (RNN) | ~0.85–0.87 |
| **This model (07)** | **No** | **Self-attention** | **~0.88–0.90** |
| RoBERTa (fine-tuned) | Yes | Self-attention | ~0.92+ |
