"""OOF evaluation + full analysis for the k-fold ensemble."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from dataset import ReviewDataset, read_csv
from model import mDeBERTaEMD
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
NUM_CLASSES = 5


def kaggle_score(preds, labels): return 1.0 - np.abs(preds - labels).mean() / 4.0
def median_decode_np(probs):
    cdf = np.cumsum(probs, axis=1)[:, :-1]
    return (cdf < 0.5).sum(axis=1).clip(0, probs.shape[1] - 1)

@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    all_probs = []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        all_probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(all_probs)

def confusion_matrix_np(preds, labels, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(labels, preds): cm[t][p] += 1
    return cm

def error_distribution(preds, labels):
    errors = np.abs(preds - labels)
    return {int(k): int((errors == k).sum()) for k in range(NUM_CLASSES)}

def per_class_stats(preds, labels, n):
    return [{"class": c, "n": int((labels==c).sum()),
             "accuracy": round(float((preds[labels==c]==c).mean()), 4),
             "mae": round(float(np.abs(preds[labels==c]-c).mean()), 4)}
            for c in range(n) if (labels==c).sum() > 0]

def avg_prob_per_class(probs, labels, n):
    return [{"true_class": c, **{f"p{k}": round(float(probs[labels==c].mean(0)[k]), 4) for k in range(n)}}
            for c in range(n) if (labels==c).sum() > 0]

def calibration_data(probs, labels, n_bins=10):
    preds = median_decode_np(probs)
    conf  = probs[np.arange(len(probs)), preds]
    correct = (preds == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i+1])
        if m.sum(): bin_data.append({"acc": float(correct[m].mean()), "conf": float(conf[m].mean()), "n": int(m.sum())})
    ece = sum(abs(b["acc"]-b["conf"])*b["n"] for b in bin_data) / len(labels)
    return {"ece": round(ece, 4), "bins": bin_data}


def main(args):
    data_dir     = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    out_dir      = artifact_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir / "tokenizer"))
    texts, labels, _ = read_csv(data_dir / "train.csv")
    labels_arr = np.array(labels)

    first_ckpt = torch.load(sorted(artifact_dir.glob("best_model_fold*.pt"))[0],
                            map_location="cpu", weights_only=False)
    max_len  = first_ckpt["args"]["max_len"]
    backbone = first_ckpt.get("backbone_dir", "microsoft/mdeberta-v3-base")
    n_folds  = first_ckpt["args"]["n_folds"]

    full_dataset = ReviewDataset(texts, tokenizer, max_len=max_len, labels=labels)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_indices = [(tr, va) for tr, va in skf.split(texts, labels)]

    results = {}
    oof_probs  = np.zeros((len(texts), NUM_CLASSES), dtype=np.float32)
    oof_labels = labels_arr.copy()

    # Per-fold stats
    for fold_idx, (_, val_idx) in enumerate(fold_indices):
        ckpt_path = artifact_dir / f"best_model_fold{fold_idx}.pt"
        if not ckpt_path.exists():
            print(f"Fold {fold_idx}: checkpoint not found, skipping.")
            continue

        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = mDeBERTaEMD(model_name=backbone, dropout=0.0).to(device)
        model.load_state_dict(ckpt["model"])

        val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=128, shuffle=False, num_workers=2)
        probs = predict_probs(model, val_loader, device)
        preds = median_decode_np(probs)
        score = kaggle_score(preds, labels_arr[val_idx])
        print(f"Fold {fold_idx}: {score:.4f}")

        oof_probs[val_idx] = probs
        results[f"fold_{fold_idx}"] = {
            "val_score": round(score, 4),
            "val_size": int(len(val_idx)),
            "confusion_matrix": confusion_matrix_np(preds, labels_arr[val_idx], NUM_CLASSES).tolist(),
            "error_distribution": error_distribution(preds, labels_arr[val_idx]),
            "per_class_stats": per_class_stats(preds, labels_arr[val_idx], NUM_CLASSES),
            "calibration": calibration_data(probs, labels_arr[val_idx]),
        }
        del model
        torch.cuda.empty_cache()

    # OOF ensemble stats
    oof_preds = median_decode_np(oof_probs)
    oof_score = kaggle_score(oof_preds, oof_labels)
    print(f"OOF ensemble score: {oof_score:.4f}  ({len(texts):,} examples)")

    results["oof_ensemble"] = {
        "oof_score": round(oof_score, 4),
        "n_examples": len(texts),
        "confusion_matrix": confusion_matrix_np(oof_preds, oof_labels, NUM_CLASSES).tolist(),
        "error_distribution": error_distribution(oof_preds, oof_labels),
        "per_class_stats": per_class_stats(oof_preds, oof_labels, NUM_CLASSES),
        "avg_prob_per_class": avg_prob_per_class(oof_probs, oof_labels, NUM_CLASSES),
        "calibration": calibration_data(oof_probs, oof_labels),
    }

    print(f"\n── Val scores per fold ─────────────────")
    for fold_idx in range(n_folds):
        key = f"fold_{fold_idx}"
        if key in results:
            print(f"  Fold {fold_idx}: {results[key]['val_score']:.4f}")
    print(f"  OOF ensemble: {oof_score:.4f}")

    print(f"\n── Error distribution (OOF) ────────────")
    for k, v in results["oof_ensemble"]["error_distribution"].items():
        print(f"  |error|={k}: {v:6d}  ({100*v/len(texts):.1f}%)")

    print(f"\n── Per-class stats (OOF) ───────────────")
    print(f"  {'Class':<8} {'N':>6} {'Acc':>8} {'MAE':>8}")
    for s in results["oof_ensemble"]["per_class_stats"]:
        print(f"  {s['class']:<8} {s['n']:>6} {s['accuracy']:>8.4f} {s['mae']:>8.4f}")

    print(f"\n── Avg predicted probs per true class ──")
    print(f"  {'True':>6}  " + "  ".join(f"  p{k}" for k in range(NUM_CLASSES)))
    for row in results["oof_ensemble"]["avg_prob_per_class"]:
        print(f"  {row['true_class']:>6}   " + "  ".join(f"{row[f'p{k}']:.3f}" for k in range(NUM_CLASSES)))

    print(f"\n── Calibration ─────────────────────────")
    print(f"  ECE OOF ensemble: {results['oof_ensemble']['calibration']['ece']:.4f}")
    for fold_idx in range(n_folds):
        key = f"fold_{fold_idx}"
        if key in results:
            print(f"  ECE fold {fold_idx}:       {results[key]['calibration']['ece']:.4f}")

    json_path = out_dir / "eval_results.json"
    with open(json_path, "w") as f: json.dump(results, f, indent=2)
    print(f"\nFull results saved to {json_path}")

    for key in [f"fold_{i}" for i in range(n_folds)] + ["oof_ensemble"]:
        if key in results:
            cm = np.array(results[key]["confusion_matrix"])
            pd.DataFrame(cm, index=range(NUM_CLASSES), columns=range(NUM_CLASSES)).to_csv(
                out_dir / f"confusion_matrix_{key}.csv")
    print(f"Confusion matrix CSVs saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--data_dir",     default=str(_DEFAULT_DATA_DIR))
    main(parser.parse_args())
