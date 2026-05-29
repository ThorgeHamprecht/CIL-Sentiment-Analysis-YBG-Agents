"""
Compute final Kaggle test scores (1 - MAE/4) for all submissions
against the ground-truth test labels.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
LABELS_PATH = ROOT / "testlabels" / "test_solved.csv"
SUBMISSIONS_DIR = ROOT / "submissions"

# Load ground truth
gt = pd.read_csv(LABELS_PATH).sort_values("id").reset_index(drop=True)
gt_labels = gt["label"].values

def kaggle_score(preds, labels):
    return 1.0 - np.abs(preds - labels).mean() / 4.0

# Priority display order (specific models of interest first)
PRIORITY = [
    "26_mdeberta_kfold",           # 5-fold ensemble (baseline 26)
    "23_mdeberta_llrd_ema_submission",      # 3-seed ensemble (baseline 23)
    "23_mdeberta_llrd_ema_seed42",          # best single seed baseline 23
    "23_mdeberta_llrd_ema_seed1337",
    "23_mdeberta_llrd_ema_seed2024",
]

results = []
for csv_path in sorted(SUBMISSIONS_DIR.glob("*.csv")):
    sub = pd.read_csv(csv_path).sort_values("id").reset_index(drop=True)
    if list(sub.columns) != ["id", "label"]:
        print(f"Skipping {csv_path.name} — unexpected columns: {sub.columns.tolist()}")
        continue
    if len(sub) != len(gt):
        print(f"Skipping {csv_path.name} — length mismatch ({len(sub)} vs {len(gt)})")
        continue
    preds = sub["label"].values
    score = kaggle_score(preds, gt_labels)
    results.append({"name": csv_path.stem, "score": score, "path": csv_path})

# Sort: priority names first, then by score descending
def sort_key(r):
    stem = r["name"]
    for i, p in enumerate(PRIORITY):
        if stem.startswith(p):
            return (0, i, -r["score"])
    return (1, 999, -r["score"])

results.sort(key=sort_key)

print(f"\n{'='*60}")
print(f"  TEST SET SCORES  (Kaggle: 1 - MAE/4)")
print(f"{'='*60}")
print(f"  {'Submission':<45} {'Score':>7}")
print(f"  {'-'*45} {'-'*7}")

for r in results:
    marker = " ◄" if any(r["name"].startswith(p) for p in PRIORITY[:3]) else ""
    print(f"  {r['name']:<45} {r['score']:.4f}{marker}")

print(f"{'='*60}")
best = max(results, key=lambda r: r["score"])
print(f"  Best: {best['name']}")
print(f"        Score = {best['score']:.4f}")
print(f"{'='*60}\n")
