# Cluster Setup

This repo is configured for an ETH student cluster account. Replace `<user>` with your cluster username. On the cluster, `$USER` should expand to that username, so cluster-side commands use `$USER` for paths.

## Paths

Use home only for code:

```bash
/home/$USER/CIL-Sentiment-Analysis-YBG-Agents
```

Use scratch for everything that is large or reproducible:

```bash
/work/scratch/$USER/cil/
  data/
  venv/
  artifacts/
  submissions/
  logs/
  .cache/huggingface/
  .cache/torch/
```

Scratch can be cleaned by age, so copy important submissions and analysis files back after runs.

## First Setup On The Cluster

```bash
ssh <user>@student-cluster.inf.ethz.ch
cd /home/$USER/CIL-Sentiment-Analysis-YBG-Agents
bash scripts/setup_cluster_env.sh
```
```bash
source /work/scratch/$USER/cil/venv/bin/activate
```
The setup script creates `/work/scratch/$USER/cil/venv`, installs PyTorch, and creates the scratch folders.

## Copy Data From Windows

Local data folder:

```text
<local_folder>\data
```

From PowerShell:

```powershell
$clusterUser = "<user>"
scp -r "<local_folder>\data\*" ${clusterUser}@student-cluster.inf.ethz.ch:/work/scratch/${clusterUser}/cil/data/
```

If using Git Bash or WSL with `rsync`:

```bash
CLUSTER_USER=<user>
rsync -av <local_folder>/data/ \
  ${CLUSTER_USER}@student-cluster.inf.ethz.ch:/work/scratch/${CLUSTER_USER}/cil/data/
```

Expected files in scratch:

```bash
/work/scratch/$USER/cil/data/train.csv
/work/scratch/$USER/cil/data/test.csv
/work/scratch/$USER/cil/data/example_submission.csv
```

Verify on the cluster:

```bash
ls -lh /work/scratch/$USER/cil/data
```

## Transformer Setup

Run this once on the login node before submitting mDeBERTa/XLM-R jobs, because compute jobs use offline HuggingFace cache mode.

```bash
source /work/scratch/$USER/cil/venv/bin/activate
export HF_HOME=/work/scratch/$USER/cil/.cache/huggingface

pip install --no-cache-dir transformers==4.40.0 sentencepiece protobuf accelerate

python - <<'PY'
from transformers import AutoModel, AutoTokenizer

for model_name in [
    "microsoft/mdeberta-v3-base",
]:
    AutoTokenizer.from_pretrained(model_name)
    AutoModel.from_pretrained(model_name)
PY
```

If scratch is cleaned, rerun this block.

## Submit A Smoke Test

Start with the smaller BiLSTM baseline to verify data, env, GPU allocation, logs, and submissions.

```bash
sbatch /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/06_rnn_bilstm/submit.sh
```

Monitor:

```bash
squeue -u $USER -o '%i %j %T %M %l %R'
tail -f /work/scratch/$USER/cil/logs/bilstm-<jobid>.out
```
