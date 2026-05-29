# How To Run B29

Run from the ETH cluster. The script assumes this layout:

```text
/home/$USER/CIL-Sentiment-Analysis-YBG-Agents/
/work/scratch/$USER/cil/data/train.csv
/work/scratch/$USER/cil/data/test.csv
/work/scratch/$USER/cil/venv/
```

The CSV reader expects review text in one of `sentence`, `text`, or
`title` + `paragraph`; labels in `label`, `rating`, or `stars`; and optional
`id` values.

The scratch venv must already contain the project dependencies and the
mDeBERTa-v3-base Hugging Face cache must be available for offline loading. The
submit script loads CUDA 13.0, activates `/work/scratch/$USER/cil/venv`, trains
the classifier and both contrastive encoders, runs the
validation ensemble sweep, then writes test submissions.

```bash
cd /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/29_separate_ensemble
sbatch submit.sh
```

Outputs:

```text
/work/scratch/$USER/cil/artifacts/29_separate_ensemble/
/work/scratch/$USER/cil/submissions/
```
