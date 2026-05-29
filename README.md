# CIL 2026 — Ordinal Sentiment Classification

Reproduction instructions for all submission predictions.
Neural baselines (B06–B30) run on the **ETH D-INFK student cluster** (NVIDIA RTX 5060 Ti, 16 GB VRAM).
Linear baselines (B01, B02) are CPU-only and can be run locally or on the cluster.

---

## 1. Cluster environment setup (one-time)

```bash
ssh thamprecht@student-cluster.inf.ethz.ch

# Create venv
python3 -m venv /work/scratch/$USER/cil/venv
source /work/scratch/$USER/cil/venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install transformers==4.40.0 pandas numpy scikit-learn

# Pre-download model weights (required — jobs run offline)
export HF_HOME=/work/scratch/$USER/cil/.cache/huggingface
python -c "
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained('microsoft/mdeberta-v3-base')
AutoModel.from_pretrained('microsoft/mdeberta-v3-base')
"
```

## 2. Data

```bash
source /work/scratch/$USER/cil/venv/bin/activate
pip install kaggle
# Place ~/.kaggle/kaggle.json (chmod 600) first
kaggle competitions download -c ethz-cil-text-class-2026 -p /work/scratch/$USER/cil/data/
cd /work/scratch/$USER/cil/data/ && unzip ethz-cil-text-class-2026.zip
```

## 3. Copy project to cluster

```bash
rsync -av --exclude='submissions/' --exclude='data/' \
  "/path/to/project/" \
  thamprecht@student-cluster.inf.ethz.ch:/home/thamprecht/cil/project/
```

## 4. Run a baseline

Each baseline has a `submit.sh`. Submit with:

```bash
sbatch /home/thamprecht/cil/project/baselines/<XX_baseline>/submit.sh
```

Monitor with `squeue -u thamprecht`. Sync predictions back:

```bash
rsync -av thamprecht@student-cluster.inf.ethz.ch:/work/scratch/thamprecht/cil/submissions/ ./submissions/
```

## 5. Baselines

**B01 and B02 can also be run locally** (no GPU, scikit-learn only):
```bash
pip install scikit-learn pandas numpy scipy
python baselines/01_linear/linear.py --data-dir /path/to/data
python baselines/02_svm_nwn/svm_nwn.py --data-dir /path/to/data
```

| ID | Directory | Test score |
|---|---|---|
| B01 | `baselines/01_linear/` | 0.8828 |
| B02 | `baselines/02_svm_nwn/` | 0.8604 |
| B06 | `baselines/06_rnn_bilstm/` | 0.8876 |
| B07 | `baselines/07_transformer_custom/` | 0.8733 |
| B15 | `baselines/15_bilstm_emd/` | 0.8913 |
| B16 | `baselines/16_transformer_emd/` | 0.8749 |
| B17 | `baselines/17_bilstm_ordinal_emd/` | 0.8917 |
| B18 | `baselines/18_bilstm_fasttext_emd/` | 0.8930 |
| B19 | `baselines/19_mdeberta_emd/` | 0.9044 |
| B20 | `baselines/20_mdeberta_emd_v2/` | 0.9060 |
| B21 | `baselines/21_xlmr_large_emd/` | 0.9065 |
| B23 | `baselines/23_mdeberta_llrd_ema/` | 0.9077 |
| B24 | `baselines/24_bilstm_emd_ensemble/` | 0.8960 |
| B26 | `baselines/26_mdeberta_kfold/` | 0.9076 |
| B27 | `baselines/27_mdeberta_seed_split/` | **0.9078** |
| B28 | `baselines/28_mdeberta_ce/` | 0.9050 |
| B29 | `baselines/29_bilstm_oll/` | 0.8900 |
| B30 | `baselines/30_transformer_oll/` | 0.8750 |

**B27 is the final submission** (~12 h runtime, 3 seeds).
