"""Eval and analysis for the 3-seed BiLSTM ensemble."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from dataset import TwoStreamDataset, load_vocab, read_csv
from model import TwoStreamBiLSTM

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
SEEDS = [42, 1337, 2024]
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
        x_t, l_t, x_b, l_b = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        main_logits, _ = model(x_t, l_t, x_b, l_b)
        all_probs.append(F.softmax(main_logits.float(), dim=1).cpu().numpy())
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

def seed_agreement(preds_list, labels):
    p = np.stack(preds_list, axis=1)
    all_agree = (p[:,0]==p[:,1]) & (p[:,1]==p[:,2])
    return {"all_agree_pct": round(float(all_agree.mean()), 4), "n_all_agree": int(all_agree.sum()),
            "two_agree_pct": round(float((~all_agree).mean()), 4), "n_two_agree": int((~all_agree).sum())}


def main(args):
    data_dir     = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    out_dir      = artifact_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab = load_vocab(artifact_dir / "vocab.json")
    _, titles, bodies, labels, _ = read_csv(data_dir / "train.csv")
    labels_arr = np.array(labels)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    _, val_idx = next(sss.split(titles, labels))
    va_t = [titles[i] for i in val_idx]
    va_b = [bodies[i] for i in val_idx]
    val_labels = labels_arr[val_idx]

    first_ckpt = torch.load(artifact_dir / f"best_model_seed{SEEDS[0]}.pt", map_location="cpu", weights_only=False)
    ckpt_args  = first_ckpt["args"]

    val_ds     = TwoStreamDataset(va_t, va_b, vocab, ckpt_args["max_len_title"], ckpt_args["max_len_body"])
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=2)

    results = {}
    all_probs, preds_per_seed = [], []

    for seed in SEEDS:
        ckpt_path = artifact_dir / f"best_model_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"Seed {seed}: not found, skipping.")
            continue
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = TwoStreamBiLSTM(
            vocab_size=ckpt["vocab_size"],
            embed_dim=ckpt_args["embed_dim"],
            hidden_dim=ckpt_args["hidden_dim"],
            num_layers=ckpt_args["num_layers"],
            dropout=0.0,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        probs = predict_probs(model, val_loader, device)
        preds = median_decode_np(probs)
        score = kaggle_score(preds, val_labels)
        print(f"Seed {seed}: {score:.4f}")
        results[f"seed_{seed}"] = {
            "val_score": round(score, 4),
            "confusion_matrix": confusion_matrix_np(preds, val_labels, NUM_CLASSES).tolist(),
            "error_distribution": error_distribution(preds, val_labels),
            "per_class_stats": per_class_stats(preds, val_labels, NUM_CLASSES),
            "avg_prob_per_class": avg_prob_per_class(probs, val_labels, NUM_CLASSES),
            "calibration": calibration_data(probs, val_labels),
        }
        all_probs.append(probs)
        preds_per_seed.append(preds)
        del model

    avg_probs = np.mean(all_probs, axis=0)
    ens_preds = median_decode_np(avg_probs)
    ens_score = kaggle_score(ens_preds, val_labels)
    print(f"Ensemble: {ens_score:.4f}")

    results["ensemble"] = {
        "val_score": round(ens_score, 4),
        "confusion_matrix": confusion_matrix_np(ens_preds, val_labels, NUM_CLASSES).tolist(),
        "error_distribution": error_distribution(ens_preds, val_labels),
        "per_class_stats": per_class_stats(ens_preds, val_labels, NUM_CLASSES),
        "avg_prob_per_class": avg_prob_per_class(avg_probs, val_labels, NUM_CLASSES),
        "calibration": calibration_data(avg_probs, val_labels),
        "seed_agreement": seed_agreement(preds_per_seed, val_labels),
    }

    print(f"\n── Val scores ──────────────────────────")
    for s in SEEDS: print(f"  Seed {s:<6}: {results[f'seed_{s}']['val_score']:.4f}")
    print(f"  Ensemble: {ens_score:.4f}")

    print(f"\n── Error distribution (ensemble) ───────")
    for k, v in results["ensemble"]["error_distribution"].items():
        print(f"  |error|={k}: {v:6d}  ({100*v/len(val_labels):.1f}%)")

    print(f"\n── Per-class stats (ensemble) ──────────")
    print(f"  {'Class':<8} {'N':>6} {'Acc':>8} {'MAE':>8}")
    for s in results["ensemble"]["per_class_stats"]:
        print(f"  {s['class']:<8} {s['n']:>6} {s['accuracy']:>8.4f} {s['mae']:>8.4f}")

    print(f"\n── Avg predicted probs per true class ──")
    print(f"  {'True':>6}  " + "  ".join(f"  p{k}" for k in range(NUM_CLASSES)))
    for row in results["ensemble"]["avg_prob_per_class"]:
        print(f"  {row['true_class']:>6}   " + "  ".join(f"{row[f'p{k}']:.3f}" for k in range(NUM_CLASSES)))

    print(f"\n── Calibration ─────────────────────────")
    print(f"  ECE ensemble: {results['ensemble']['calibration']['ece']:.4f}")
    for s in SEEDS: print(f"  ECE seed {s}:  {results[f'seed_{s}']['calibration']['ece']:.4f}")

    print(f"\n── Seed agreement ──────────────────────")
    ag = results["ensemble"]["seed_agreement"]
    print(f"  All 3 agree:  {ag['all_agree_pct']*100:.1f}%  ({ag['n_all_agree']:,})")
    print(f"  Only 2 agree: {ag['two_agree_pct']*100:.1f}%  ({ag['n_two_agree']:,})")

    json_path = out_dir / "eval_results.json"
    with open(json_path, "w") as f: json.dump(results, f, indent=2)
    print(f"\nFull results saved to {json_path}")

    for key in [f"seed_{s}" for s in SEEDS] + ["ensemble"]:
        cm = np.array(results[key]["confusion_matrix"])
        pd.DataFrame(cm, index=range(NUM_CLASSES), columns=range(NUM_CLASSES)).to_csv(
            out_dir / f"confusion_matrix_{key}.csv")
    print(f"Confusion matrix CSVs saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--data_dir",     default=str(_DEFAULT_DATA_DIR))
    main(parser.parse_args())
