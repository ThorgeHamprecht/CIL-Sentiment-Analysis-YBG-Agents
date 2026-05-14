"""
Evaluation and analysis for the 3-seed ensemble.
Produces val scores, confusion matrices, error distributions, calibration,
per-class stats, and seed agreement — all saved for report use.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import ReviewDataset, read_csv
from model import mDeBERTaEMD

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
SEEDS = [42, 1337, 2024]
NUM_CLASSES = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def kaggle_score(preds: np.ndarray, labels: np.ndarray) -> float:
    return 1.0 - np.abs(preds - labels).mean() / 4.0


def median_decode_np(probs: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(probs, axis=1)[:, :-1]
    return (cdf < 0.5).sum(axis=1).clip(0, probs.shape[1] - 1)


@torch.no_grad()
def predict_probs(model, loader, device) -> np.ndarray:
    model.eval()
    all_probs = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask)
        all_probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(all_probs)


def confusion_matrix_np(preds: np.ndarray, labels: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(labels, preds):
        cm[t][p] += 1
    return cm


def error_distribution(preds: np.ndarray, labels: np.ndarray) -> dict:
    errors = np.abs(preds - labels)
    return {int(k): int((errors == k).sum()) for k in range(NUM_CLASSES)}


def per_class_stats(preds: np.ndarray, labels: np.ndarray, n: int) -> list:
    stats = []
    for c in range(n):
        mask = labels == c
        if mask.sum() == 0:
            continue
        mae_c = np.abs(preds[mask] - labels[mask]).mean()
        acc_c = (preds[mask] == labels[mask]).mean()
        stats.append({"class": c, "n": int(mask.sum()), "accuracy": round(float(acc_c), 4),
                      "mae": round(float(mae_c), 4)})
    return stats


def avg_prob_per_class(probs: np.ndarray, labels: np.ndarray, n: int) -> list:
    """Mean predicted probability vector for each true class (shows ordinal structure)."""
    rows = []
    for c in range(n):
        mask = labels == c
        if mask.sum() == 0:
            continue
        mean_probs = probs[mask].mean(axis=0)
        rows.append({"true_class": c, **{f"p{k}": round(float(mean_probs[k]), 4) for k in range(n)}})
    return rows


def calibration_data(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    """ECE and per-bin calibration for the predicted class."""
    preds = median_decode_np(probs)
    confidences = probs[np.arange(len(probs)), preds]  # confidence in predicted class
    correct = (preds == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc, bin_conf, bin_count = [], [], []
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc.append(float(correct[mask].mean()))
        bin_conf.append(float(confidences[mask].mean()))
        bin_count.append(int(mask.sum()))

    ece = sum(abs(a - c) * n for a, c, n in zip(bin_acc, bin_conf, bin_count)) / len(labels)
    return {"ece": round(ece, 4), "bins": [{"acc": a, "conf": c, "n": n}
                                            for a, c, n in zip(bin_acc, bin_conf, bin_count)]}


def seed_agreement(preds_per_seed: list, labels: np.ndarray) -> dict:
    """How often seeds agree, and whether ensemble benefits from disagreement."""
    preds = np.stack(preds_per_seed, axis=1)  # (N, 3)
    all_agree = (preds[:, 0] == preds[:, 1]) & (preds[:, 1] == preds[:, 2])
    two_agree = ~all_agree
    return {
        "all_agree_pct": round(float(all_agree.mean()), 4),
        "two_agree_pct": round(float(two_agree.mean()), 4),
        "n_all_agree": int(all_agree.sum()),
        "n_two_agree": int(two_agree.sum()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    out_dir = artifact_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir / "tokenizer"))
    texts, labels, _ = read_csv(data_dir / "train.csv")
    labels_arr = np.array(labels)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    _, val_idx = next(sss.split(texts, labels))
    val_texts = [texts[i] for i in val_idx]
    val_labels = labels_arr[val_idx]

    first_ckpt = torch.load(
        artifact_dir / f"best_model_seed{SEEDS[0]}.pt", map_location="cpu", weights_only=False
    )
    max_len = first_ckpt["args"]["max_len"]
    val_dataset = ReviewDataset(val_texts, tokenizer, max_len=max_len)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    results = {}
    all_probs = []
    preds_per_seed = []

    # ── Per-seed analysis ─────────────────────────────────────────────────────
    for seed in SEEDS:
        ckpt_path = artifact_dir / f"best_model_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"Seed {seed}: checkpoint not found, skipping.")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        backbone_dir = ckpt.get("backbone_dir", "microsoft/mdeberta-v3-base")
        model = mDeBERTaEMD(model_name=str(backbone_dir), dropout=0.0).to(device)
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
        torch.cuda.empty_cache()

    # ── Ensemble analysis ─────────────────────────────────────────────────────
    avg_probs = np.mean(all_probs, axis=0)
    ens_preds = median_decode_np(avg_probs)
    ens_score = kaggle_score(ens_preds, val_labels)
    print(f"Ensemble:  {ens_score:.4f}")

    results["ensemble"] = {
        "val_score": round(ens_score, 4),
        "confusion_matrix": confusion_matrix_np(ens_preds, val_labels, NUM_CLASSES).tolist(),
        "error_distribution": error_distribution(ens_preds, val_labels),
        "per_class_stats": per_class_stats(ens_preds, val_labels, NUM_CLASSES),
        "avg_prob_per_class": avg_prob_per_class(avg_probs, val_labels, NUM_CLASSES),
        "calibration": calibration_data(avg_probs, val_labels),
        "seed_agreement": seed_agreement(preds_per_seed, val_labels),
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n── Val scores ──────────────────────────")
    for seed in SEEDS:
        print(f"  Seed {seed:<6}: {results[f'seed_{seed}']['val_score']:.4f}")
    print(f"  Ensemble: {ens_score:.4f}")

    print("\n── Error distribution (ensemble) ───────")
    for k, v in results["ensemble"]["error_distribution"].items():
        pct = 100 * v / len(val_labels)
        print(f"  |error|={k}: {v:6d}  ({pct:.1f}%)")

    print("\n── Per-class stats (ensemble) ──────────")
    print(f"  {'Class':<8} {'N':>6} {'Acc':>8} {'MAE':>8}")
    for s in results["ensemble"]["per_class_stats"]:
        print(f"  {s['class']:<8} {s['n']:>6} {s['accuracy']:>8.4f} {s['mae']:>8.4f}")

    print("\n── Avg predicted probs per true class (ensemble) ──")
    print(f"  {'True':>6}  " + "  ".join(f"  p{k}" for k in range(NUM_CLASSES)))
    for row in results["ensemble"]["avg_prob_per_class"]:
        probs_str = "  ".join(f"{row[f'p{k}']:.3f}" for k in range(NUM_CLASSES))
        print(f"  {row['true_class']:>6}   {probs_str}")

    print(f"\n── Calibration ─────────────────────────")
    print(f"  ECE (ensemble): {results['ensemble']['calibration']['ece']:.4f}")
    for s in SEEDS:
        print(f"  ECE seed {s}:  {results[f'seed_{s}']['calibration']['ece']:.4f}")

    print(f"\n── Seed agreement ──────────────────────")
    ag = results["ensemble"]["seed_agreement"]
    print(f"  All 3 agree:  {ag['all_agree_pct']*100:.1f}%  ({ag['n_all_agree']:,})")
    print(f"  Only 2 agree: {ag['two_agree_pct']*100:.1f}%  ({ag['n_two_agree']:,})")

    # ── Save ──────────────────────────────────────────────────────────────────
    json_path = out_dir / "eval_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {json_path}")

    # Confusion matrices as CSVs
    classes = list(range(NUM_CLASSES))
    for key in [f"seed_{s}" for s in SEEDS] + ["ensemble"]:
        cm = np.array(results[key]["confusion_matrix"])
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(
            out_dir / f"confusion_matrix_{key}.csv"
        )
    print(f"Confusion matrix CSVs saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--data_dir",     default=str(_DEFAULT_DATA_DIR))
    main(parser.parse_args())
