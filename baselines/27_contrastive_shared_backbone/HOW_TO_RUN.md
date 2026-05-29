# How To Run B27

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
submit script loads CUDA 13.0, activates `/work/scratch/$USER/cil/venv`, runs
all shared-backbone variants, evaluates `best_model.pt` and
`epoch_004_model.pt`, and writes validation/test prediction files plus Kaggle
submissions.

```bash
cd /home/$USER/CIL-Sentiment-Analysis-YBG-Agents/baselines/27_contrastive_shared_backbone
sbatch submit.sh
```

Outputs:

```text
/work/scratch/$USER/cil/artifacts/27_contrastive_shared_backbone_*/
/work/scratch/$USER/cil/submissions/
```
