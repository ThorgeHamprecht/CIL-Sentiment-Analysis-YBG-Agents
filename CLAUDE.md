# ETHZ CIL Text Classification 2026

## Task
5-class ordinal sentiment classification of English and German product reviews.

- **Labels**: integers in `{0, 1, 2, 3, 4}` (0 = most negative, 4 = most positive)
- **Languages**: English and German (mixed)
- **Metric**: `L = 1 - MAE/4` (higher is better; ordinal scale — large errors hurt more)

## Data (`data/`)
| File | Rows | Columns | Notes |
|---|---|---|---|
| `train.csv` | ~252,000 | `id, sentence, label` | Perfectly balanced (50,400 per class) |
| `test.csv` | ~168,000 | `id, sentence` | No labels |
| `example_submission.csv` | ~168,000 | `id, label` | Format reference |

**Important**: `test_solved.csv` does NOT exist — Kaggle does not release test labels. The baselines skip local scoring automatically if the file is absent.

**`id` column**: is NOT shared between train and test — the same id value refers to different sentences in each file. Ensure submission ids come from `test.csv` only, written as plain integers (no decimals).

**Public vs private leaderboard**: public LB scores on 45% of test data; your grade is based on the private LB (55%), which is only revealed after the competition ends. Do not overfit to the public LB.

## Review length stats
- Median: 47 tokens, p99: 247 tokens, max: 527 tokens
- Only 0.9% of reviews exceed 256 tokens; none exceed 512
- **max_len=256 is optimal** — 512 wastes compute with no meaningful coverage gain

## Scoring
```
L(ŷ, y) = 1 - (1/4n) * sum(|y_i - ŷ_i|) = 1 - MAE/4
```
- 1.0 = perfect (MAE=0)
- 0.75 = off by 1 on every example (MAE=1)
- 0.5 = off by 2 on average (MAE=2)
- 0.0 = maximally wrong (MAE=4)

## Submission Format
```
id,label
0,3
1,1
...
```
- Must contain exactly the `id` values from `test.csv`
- Labels must be integers in `{0,1,2,3,4}`

---

## Baselines

### Ablation narrative

| Question | Comparison |
|----------|-----------|
| Does attention help over RNNs? | 06 (BiLSTM) vs 07 (Transformer) |
| How much does pretraining matter? | 07 vs 19 (mDeBERTa) |
| Does ordinal loss/decode help? | CrossEntropy+argmax vs EMD²+median |

### "Approach A" — EMD² loss + median decode
Applied to every baseline from 15 onward. Each is a copy of its original with only the loss/decode changed.

- **EMD² loss**: `Σ_{k=0}^{K-2} (CDF_P(k) - 1[k >= y])²` — Wasserstein-1 squared, closed-form, differentiable
- **Median decode**: `min{k : CDF_P(k) ≥ 0.5}` — Bayes-optimal under MAE (vs argmax which is optimal under 0-1 loss)

**Rule**: never modify original baselines. Always copy to a new numbered folder.

### Baseline table

| # | Folder | Model | Architecture | Loss / Decode | Val score |
|---|--------|-------|-------------|---------------|-----------|
| 06 | rnn_bilstm | BiLSTM scratch | 2-layer BiLSTM, 128d embed | CE + argmax | pending |
| 07 | transformer_custom | Transformer scratch | 4-layer enc, d=256, [CLS] | CE + argmax | pending |
| 09 | mdeberta_coral | mDeBERTa-v3-base | CORAL head (K-1 binary) | CORAL | pending |
| 10 | bilstm_ordinal | TwoStream BiLSTM | OLL loss | OLL + EV | pending |
| 13 | bilstm_fasttext | FastText TwoStream | OLL + SWA | OLL + EV + SWA | pending |
| 15 | bilstm_emd | BiLSTM scratch | same as 06 | **EMD² + median** | pending |
| 16 | transformer_emd | Transformer scratch | same as 07 | **EMD² + median** | pending |
| 17 | bilstm_ordinal_emd | TwoStream BiLSTM | same as 10 | **EMD² + median** | **0.8911** |
| 18 | bilstm_fasttext_emd | FastText TwoStream | same as 13 | **EMD² + median** | **0.8935** |
| 19 | mdeberta_emd | mDeBERTa-v3-base 270M | [CLS] pool, standard head | **EMD² + median** | **0.9048** (peaked ep3, overfitting) |
| 20 | mdeberta_emd_v2 | mDeBERTa-v3-base 270M | **mean pool + K=5 dropout** | **EMD² + median** | pending (job 64764) |
| 21 | mdeberta_large_emd | mDeBERTa-v3-large 680M | mean pool + K=5 dropout | **EMD² + median** | pending |

### mDeBERTa architecture decisions (from baseline 20 onward)
- **Mean pooling** over non-padding tokens (better than [CLS] — all tokens contribute)
- **Multi-sample dropout (K=5)**: 5 independent dropout masks on pooled vector, logits averaged → implicit ensemble, reduces variance
- **Layerwise LR**: encoder at `encoder_lr` (1e-5), classifier head at `head_lr` (1e-4)
- **Weight decay**: encoder 0.05, head 0.01
- **AMP**: autocast + GradScaler for mixed precision
- **Cosine LR schedule** with 6% linear warmup

### Hyperparameter reference

| Param | Base-v2 (20) | Large (21) | Rationale |
|-------|-------------|-----------|-----------|
| model | mdeberta-v3-base | mdeberta-v3-large | size |
| batch_size | 32 | 16 | VRAM (16GB) |
| max_len | 256 | 256 | p99=247 tokens |
| encoder_lr | 1e-5 | 1e-5 | standard |
| head_lr | 1e-4 | 1e-4 | standard |
| dropout | 0.2 | 0.1 | fewer epochs → less overfitting risk |
| weight_decay | 0.05 | 0.05 | standard |
| epochs | 10 | 5 | large model is slower |
| patience | 4 | 3 | matches epoch budget |
| time limit | 8h | 12h | ~2–2.5h/epoch for large |

---

## Project Structure
```
data/                         # raw data (train.csv, test.csv)
submissions/                  # generated Kaggle submission CSVs
baselines/
  06_rnn_bilstm/              # BiLSTM baseline (train from scratch)
  07_transformer_custom/      # Custom Transformer baseline (train from scratch)
  09_mdeberta_coral/          # mDeBERTa + CORAL head
  10_bilstm_ordinal/          # TwoStream BiLSTM + OLL
  13_bilstm_fasttext/         # FastText BiLSTM + OLL + SWA
  15_bilstm_emd/              # Approach A copy of 06
  16_transformer_emd/         # Approach A copy of 07
  17_bilstm_ordinal_emd/      # Approach A copy of 10 — val 0.8911
  18_bilstm_fasttext_emd/     # Approach A copy of 13 — val 0.8935
  19_mdeberta_emd/            # mDeBERTa-base + EMD — val 0.9048
  20_mdeberta_emd_v2/         # mDeBERTa-base + mean pool + multi-dropout
  21_mdeberta_large_emd/      # mDeBERTa-large + mean pool + multi-dropout
scripts/                      # cluster setup and utility scripts
src/                          # reusable Python modules
```

---

## ETH Student Cluster — Complete Guide

### Access
- **SSH**: `ssh thamprecht@student-cluster.inf.ethz.ch`
- **Login nodes**: `student-cluster1.inf.ethz.ch`, `student-cluster2.inf.ethz.ch`
- Login banner shows remaining GPU budget and home quota — check it each session.

### Compute budget (thamprecht)
| Account tag | Hours | Max runtime/job |
|-------------|-------|-----------------|
| `cil`       | 100h  | 60 min          |
| `cil_jobs`  | 200h  | 24h             |

Use `cil_jobs` for training runs. Use `cil` only for quick interactive tests.

### GPUs (priority order, no explicit request)
| Priority | Type | Count | VRAM |
|----------|------|-------|------|
| 1 | RTX 5060 Ti | 32 | 16 GB |
| 2 | RTX 2080 Ti | 32 | 11 GB |
| 3 | GTX 1080 Ti | 192 | 11 GB |

### Storage
| Path | Quota | Retention | Use for |
|------|-------|-----------|---------|
| `/home/thamprecht/` | 20 GB | Permanent (deleted at course end) | Code only |
| `/work/scratch/thamprecht/` | 100 GB | Age-based (see below) | Venv, data, artifacts |

**Scratch retention** (cleaning runs at 23:00 daily — do NOT touch mtimes):
- Used < 10 GB → max 7 days
- Used 10–50 GB → max 2 days
- Used > 50 GB → max 1 day

**Rule**: always rsync submission CSVs back to your Mac immediately after a job finishes.

### Directory layout on cluster
```
/home/thamprecht/cil/project/          ← rsync of repo (code only, permanent)
/work/scratch/thamprecht/cil/
  venv/                                ← Python venv with PyTorch (~4 GB)
  data/                                ← train.csv, test.csv
  artifacts/
    19_mdeberta_emd/                   ← best_model.pt, tokenizer/
    20_mdeberta_emd_v2/
    21_mdeberta_large_emd/
  submissions/                         ← output CSVs — rsync back to Mac after job
  .cache/huggingface/                  ← HF model weights (needed for OFFLINE mode)
  .cache/torch/
  logs/                                ← SLURM .out/.err files
```

### SBATCH rules — critical
Only these headers work:
```bash
#SBATCH --time=HH:MM:SS
#SBATCH --account=cil_jobs
```
Do NOT add `--gpus`, `--gres`, `--cpus-per-task`, or `--mem` — all give "Specifying TRES not allowed". GPU (RTX 5060 Ti), 2 CPUs, 24 GB RAM allocated automatically.

**Always hardcode the baseline directory — never use `dirname $0`:**
```bash
cd /home/thamprecht/cil/project/baselines/20_mdeberta_emd_v2
```
SLURM copies scripts to its spool directory (`/var/spool/slurm/d/jobXXX/`), so `dirname $0` resolves there, causing `python: can't open file 'train.py': No such file or directory`.

**Only 1 job runs at a time** (`QOSMaxJobsPerUserLimit`). Submit both jobs at once — second queues and auto-starts.

### Pre-downloading HuggingFace model weights (REQUIRED)
All submit scripts set `TRANSFORMERS_OFFLINE=1`. Weights must be pre-downloaded on the login node before submitting:
```bash
ssh thamprecht@student-cluster.inf.ethz.ch
source /work/scratch/thamprecht/cil/venv/bin/activate
export HF_HOME=/work/scratch/thamprecht/cil/.cache/huggingface

# For mDeBERTa-base (baselines 19, 20):
python -c "
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained('microsoft/mdeberta-v3-base')
AutoModel.from_pretrained('microsoft/mdeberta-v3-base')
"

# For mDeBERTa-large (baseline 21):
python -c "
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained('microsoft/mdeberta-v3-large')
AutoModel.from_pretrained('microsoft/mdeberta-v3-large')
"
```
If the HF cache is cleared by scratch retention, re-download on the login node before re-submitting.

### Syncing code to cluster (run on Mac)
```bash
rsync -av --exclude='__pycache__' --exclude='*.pyc' --exclude='artifacts/' \
  "/Users/thorge/Documents/ETH/MS Semester 2/CIL/CIL Sentiment Analysis/" \
  thamprecht@student-cluster.inf.ethz.ch:/home/thamprecht/cil/project/
```

### Submitting jobs
```bash
# Use absolute paths — relative paths break in SLURM spool
sbatch /home/thamprecht/cil/project/baselines/20_mdeberta_emd_v2/submit.sh
sbatch /home/thamprecht/cil/project/baselines/21_mdeberta_large_emd/submit.sh
```

### Monitoring
```bash
squeue -u $USER -o '%i %j %T %M %l %R'              # queue status
tail -f /work/scratch/thamprecht/cil/logs/<name>-<id>.out  # live log
sacct -j <id> --format=JobID,State,Elapsed,ExitCode -P     # finished job
scancel <id>                                               # cancel job
```

### Fetching results (run on Mac after job finishes)
```bash
rsync -av \
  thamprecht@student-cluster.inf.ethz.ch:/work/scratch/thamprecht/cil/submissions/ \
  "/Users/thorge/Documents/ETH/MS Semester 2/CIL/CIL Sentiment Analysis/submissions/"
```

### Environment setup (module system, no conda)
```bash
. /etc/profile.d/modules.sh   # must source before any module command
module add cuda/13.0          # use 'add', not 'load'
source /work/scratch/$USER/cil/venv/bin/activate
```

### Known issues and fixes
- **`--gpus` and `--gres` rejected**: do not use — GPU is auto-allocated by account.
- **`train.py not found`**: use hardcoded `cd /home/thamprecht/...` in submit scripts, never `dirname $0`.
- **`module` not found in job**: source `. /etc/profile.d/modules.sh` first.
- **pip cache filling home**: always use `--no-cache-dir`.
- **Venv gone after scratch cleaning**: re-run `bash scripts/setup_cluster_env.sh`.
- **SSH key auth**: requires `chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys` on cluster.
- **SWA collapsed to 0.700 in baseline 18**: known bug — use the non-SWA best checkpoint only.
- **Access expires**: last Monday morning of semester holidays.

### Confirmed working environment
- **PyTorch**: 2.11.0+cu130
- **GPU**: NVIDIA GeForce RTX 5060 Ti (16 GB)
- **CUDA**: 13.0

### Typical runtimes (RTX 5060 Ti)
- mDeBERTa-v3-base, batch=32, max_len=256: ~1h/epoch
- mDeBERTa-v3-large, batch=16, max_len=256: ~2–2.5h/epoch
