"""Fine-tune mDeBERTa-v3-base with EMD² loss and median decode."""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import ReviewDataset, read_csv
from model import emd_loss, median_decode, mDeBERTaEMD

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_NAME = "microsoft/mdeberta-v3-base"
# Resolved at runtime so TRANSFORMERS_OFFLINE=1 works when HF_HOME is set
_HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def kaggle_score(preds: np.ndarray, labels: np.ndarray) -> float:
    return 1.0 - np.abs(preds - labels).mean() / 4.0


def train_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with autocast():
            logits = model(input_ids, attention_mask)
            loss = emd_loss(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast():
            logits = model(input_ids, attention_mask)
            loss = emd_loss(logits, labels)

        total_loss += loss.item() * len(labels)
        all_preds.append(median_decode(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return total_loss / len(loader.dataset), kaggle_score(preds, labels)


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    texts, labels, _ = read_csv(data_dir / "train.csv")
    print(f"Loaded {len(texts)} examples")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(sss.split(texts, labels))

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    print("Tokenizing train set...")
    train_dataset = ReviewDataset(train_texts, tokenizer, max_len=args.max_len, labels=train_labels)
    print("Tokenizing val set...")
    val_dataset = ReviewDataset(val_texts, tokenizer, max_len=args.max_len, labels=val_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    model = mDeBERTaEMD(model_name=MODEL_NAME, dropout=args.dropout).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr, "weight_decay": args.weight_decay},
            {"params": model.classifier.parameters(), "lr": args.head_lr, "weight_decay": 0.01},
        ],
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.06 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler()

    best_score, patience_counter = 0.0, 0
    checkpoint_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device)
        val_loss, val_score = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:2d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"| val_score={val_score:.4f} | {time.time()-t0:.1f}s"
        )

        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "model_name": MODEL_NAME,
                },
                checkpoint_path,
            )
            print(f"  -> New best: {best_score:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    print(f"\nBest val score: {best_score:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    tokenizer.save_pretrained(str(out_dir / "tokenizer"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len",      type=int,   default=256)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--encoder_lr",   type=float, default=1e-5)
    parser.add_argument("--head_lr",      type=float, default=1e-4)
    parser.add_argument("--dropout",      type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--patience",     type=int,   default=4)
    parser.add_argument("--data_dir",    default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    main(parser.parse_args())
