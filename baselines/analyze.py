"""
Error analysis for trained sentiment models.

Loads a checkpoint, reconstructs the exact val split used during training
(StratifiedShuffleSplit random_state=42, test_size=0.1), runs inference,
prints summary statistics, and saves val_preds.csv to the artifact dir.

Usage:
    python analyze.py --model_type bilstm      --artifact_dir 06_rnn_bilstm/artifacts
    python analyze.py --model_type bilstm_attn --artifact_dir 08_rnn_improved/artifacts
    python analyze.py --model_type transformer  --artifact_dir 07_transformer_custom/artifacts
    python analyze.py --model_type mdeberta    --artifact_dir 09_mdeberta_coral/artifacts

Output:
    <artifact_dir>/val_preds.csv  — columns: text, label, pred, error
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = ROOT / "data"

MODEL_DIRS = {
    "bilstm":      "06_rnn_bilstm",
    "transformer": "07_transformer_custom",
    "bilstm_attn": "08_rnn_improved",
    "mdeberta":    "09_mdeberta_coral",
}


def kaggle_score(preds: np.ndarray, labels: np.ndarray) -> float:
    return 1.0 - np.abs(preds - labels).mean() / 4.0


# ---------------------------------------------------------------------------
# Model-specific loaders
# ---------------------------------------------------------------------------

def _load_bilstm(artifact_dir: Path, device):
    model_dir = Path(__file__).parent / MODEL_DIRS["bilstm"]
    sys.path.insert(0, str(model_dir))
    from model import BiLSTMClassifier
    from dataset import ReviewDataset, load_vocab, read_csv
    sys.path.pop(0)

    ckpt = torch.load(artifact_dir / "best_model.pt", map_location=device)
    saved_args = ckpt["args"]
    vocab = load_vocab(artifact_dir / "vocab.json")

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=saved_args["embed_dim"],
        hidden_dim=saved_args["hidden_dim"],
        num_layers=saved_args["num_layers"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    def make_loader(texts, labels, data_dir):
        ds = ReviewDataset(texts, vocab, max_len=saved_args["max_len"], labels=labels)
        return DataLoader(ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    @torch.no_grad()
    def infer(loader):
        model.eval()
        all_preds = []
        for x, lengths, _ in loader:
            x, lengths = x.to(device), lengths.to(device)
            logits = model(x, lengths)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(all_preds)

    return make_loader, infer, read_csv, saved_args


def _load_bilstm_attn(artifact_dir: Path, device):
    model_dir = Path(__file__).parent / MODEL_DIRS["bilstm_attn"]
    sys.path.insert(0, str(model_dir))
    from model import AttentionBiLSTM
    from dataset import ReviewDataset, load_vocab, read_csv
    sys.path.pop(0)

    ckpt = torch.load(artifact_dir / "best_model.pt", map_location=device)
    saved_args = ckpt["args"]
    vocab = load_vocab(artifact_dir / "vocab.json")

    model = AttentionBiLSTM(
        vocab_size=len(vocab),
        embed_dim=saved_args["embed_dim"],
        hidden_dim=saved_args["hidden_dim"],
        num_layers=saved_args["num_layers"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    def make_loader(texts, labels, data_dir):
        ds = ReviewDataset(texts, vocab, max_len=saved_args["max_len"], labels=labels)
        return DataLoader(ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    @torch.no_grad()
    def infer(loader):
        model.eval()
        all_preds = []
        for x, lengths, _ in loader:
            x, lengths = x.to(device), lengths.to(device)
            logits = model(x, lengths)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(all_preds)

    return make_loader, infer, read_csv, saved_args


def _load_transformer(artifact_dir: Path, device):
    model_dir = Path(__file__).parent / MODEL_DIRS["transformer"]
    sys.path.insert(0, str(model_dir))
    from model import CustomTransformerClassifier
    from dataset import ReviewDataset, load_vocab, read_csv
    sys.path.pop(0)

    ckpt = torch.load(artifact_dir / "best_model.pt", map_location=device)
    saved_args = ckpt["args"]
    vocab = load_vocab(artifact_dir / "vocab.json")

    model = CustomTransformerClassifier(
        vocab_size=len(vocab),
        d_model=saved_args["d_model"],
        nhead=saved_args["nhead"],
        num_layers=saved_args["num_layers"],
        dim_feedforward=saved_args["dim_feedforward"],
        dropout=0.0,
        max_len=saved_args["max_len"],
    ).to(device)
    model.load_state_dict(ckpt["model"])

    def make_loader(texts, labels, data_dir):
        ds = ReviewDataset(texts, vocab, max_len=saved_args["max_len"], labels=labels)
        return DataLoader(ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    @torch.no_grad()
    def infer(loader):
        model.eval()
        all_preds = []
        for x, mask, _ in loader:
            x, mask = x.to(device), mask.to(device)
            logits = model(x, mask)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(all_preds)

    return make_loader, infer, read_csv, saved_args


def _load_mdeberta(artifact_dir: Path, device):
    model_dir = Path(__file__).parent / MODEL_DIRS["mdeberta"]
    sys.path.insert(0, str(model_dir))
    from model import mDeBERTaCORAL
    from dataset import ReviewDataset, read_csv
    from transformers import AutoTokenizer
    sys.path.pop(0)

    ckpt = torch.load(artifact_dir / "best_model.pt", map_location=device)
    saved_args = ckpt["args"]
    model_name = ckpt.get("model_name", "microsoft/mdeberta-v3-base")

    tokenizer_path = artifact_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path) if tokenizer_path.exists() else model_name)

    model = mDeBERTaCORAL(model_name=model_name, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model"])

    def make_loader(texts, labels, data_dir):
        ds = ReviewDataset(texts, tokenizer, max_len=saved_args["max_len"], labels=labels)
        return DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    @torch.no_grad()
    def infer(loader):
        model.eval()
        all_preds = []
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            all_preds.append(model.predict(logits).cpu().numpy())
        return np.concatenate(all_preds)

    return make_loader, infer, read_csv, saved_args


LOADERS = {
    "bilstm":      _load_bilstm,
    "bilstm_attn": _load_bilstm_attn,
    "transformer": _load_transformer,
    "mdeberta":    _load_mdeberta,
}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(preds: np.ndarray, labels: np.ndarray, model_type: str):
    errors = np.abs(preds - labels)

    print(f"\n{'='*60}")
    print(f"  Error analysis — {model_type}")
    print(f"{'='*60}")

    print(f"\nKaggle score (1 - MAE/4):  {kaggle_score(preds, labels):.4f}")
    print(f"MAE:                       {errors.mean():.4f}")
    print(f"Exact accuracy:            {(errors == 0).mean():.4f}")
    print(f"Off-by-1 rate:             {(errors == 1).mean():.4f}")
    print(f"Off-by->=2 rate:           {(errors >= 2).mean():.4f}")
    print(f"Off-by->=3 rate:           {(errors >= 3).mean():.4f}")

    print("\nPer-true-class MAE:")
    for cls in range(5):
        mask = labels == cls
        if mask.sum() == 0:
            continue
        cls_mae = errors[mask].mean()
        cls_n = mask.sum()
        print(f"  Class {cls}: MAE={cls_mae:.3f}  (n={cls_n})")

    print("\nPrediction distribution (predicted class counts):")
    for cls in range(5):
        count = (preds == cls).sum()
        pct = 100 * count / len(preds)
        print(f"  Class {cls}: {count:6d}  ({pct:.1f}%)")

    print("\nTrue label distribution:")
    for cls in range(5):
        count = (labels == cls).sum()
        pct = 100 * count / len(labels)
        print(f"  Class {cls}: {count:6d}  ({pct:.1f}%)")

    print("\nConfusion matrix (rows=true, cols=pred):")
    header = "      " + "".join(f"  P{c}" for c in range(5))
    print(header)
    for true_cls in range(5):
        row = f"  T{true_cls}  "
        for pred_cls in range(5):
            count = ((labels == true_cls) & (preds == pred_cls)).sum()
            row += f"  {count:4d}"
        print(row)

    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    artifact_dir = Path(args.artifact_dir)
    data_dir = Path(args.data_dir)

    make_loader, infer, read_csv, saved_args = LOADERS[args.model_type](artifact_dir, device)

    texts, labels, _ = read_csv(data_dir / "train.csv")
    labels_arr = np.array(labels)
    print(f"Loaded {len(texts)} examples from train.csv")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    _, val_idx = next(sss.split(texts, labels))

    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    print(f"Val split: {len(val_texts)} examples")

    loader = make_loader(val_texts, val_labels, data_dir)
    print("Running inference...")
    preds = infer(loader)
    labels_np = np.array(val_labels)

    print_stats(preds, labels_np, args.model_type)

    out_csv = artifact_dir / "val_preds.csv"
    df = pd.DataFrame({
        "text": val_texts,
        "label": labels_np,
        "pred": preds,
        "error": np.abs(preds - labels_np),
    })
    df.to_csv(out_csv, index=False)
    print(f"Saved predictions to: {out_csv}")
    print(f"  ({len(df)} rows, {out_csv.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", required=True,
        choices=["bilstm", "bilstm_attn", "transformer", "mdeberta"],
    )
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--data_dir", default=str(_DEFAULT_DATA_DIR))
    main(parser.parse_args())
