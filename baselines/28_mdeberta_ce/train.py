"""
mDeBERTa-v3-base + Cross-Entropy loss + argmax decode.
Controlled ablation against baseline 19 (same arch, same HPs) — only the
loss function changes (CE here vs W² there) to isolate the loss effect.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import ReviewDataset, read_csv
from model import mDeBERTaEMD

ROOT       = Path(__file__).resolve().parents[2]
MODEL_NAME = "microsoft/mdeberta-v3-base"


def kaggle_score(preds, labels):
    return 1.0 - np.abs(np.array(preds) - np.array(labels)).mean() / 4.0


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask)
            loss   = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(batch["labels"].numpy())
    return kaggle_score(np.concatenate(all_preds), np.concatenate(all_labels))


def main(args):
    seed         = args.seed
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  seed={seed}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    texts, labels, _ = read_csv(Path(args.data_dir) / "train.csv")
    print(f"Loaded {len(texts):,} examples")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    train_idx, val_idx = next(sss.split(texts, labels))
    train_texts  = [texts[i]  for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts    = [texts[i]  for i in val_idx]
    val_labels   = [labels[i] for i in val_idx]

    train_ds = ReviewDataset(train_texts, tokenizer, max_len=args.max_len, labels=train_labels)
    val_ds   = ReviewDataset(val_texts,   tokenizer, max_len=args.max_len, labels=val_labels)

    torch.manual_seed(seed); np.random.seed(seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = mDeBERTaEMD(model_name=MODEL_NAME, dropout=args.dropout).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(),    "lr": args.encoder_lr},
            {"params": model.classifier.parameters(), "lr": args.head_lr},
        ],
        weight_decay=0.01,
    )
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = int(0.06 * total_steps)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_score, patience_counter = 0.0, 0
    ckpt_path = artifact_dir / "best_model_ce.pt"

    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_score  = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch:2d} | train_loss={train_loss:.4f} "
              f"| val_score={val_score:.4f} | {time.time()-t0:.1f}s")
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save({"model": model.state_dict(), "backbone": MODEL_NAME}, ckpt_path)
            print(f"    -> New best: {best_score:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"    Early stop at epoch {epoch}")
                break

    print(f"Best val score: {best_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     default=str(ROOT / "data"))
    parser.add_argument("--artifact_dir", default=str(Path(__file__).parent / "artifacts"))
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--max_len",      type=int,   default=128)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--encoder_lr",   type=float, default=2e-5)
    parser.add_argument("--head_lr",      type=float, default=1e-4)
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--epochs",       type=int,   default=6)
    parser.add_argument("--patience",     type=int,   default=3)
    main(parser.parse_args())
