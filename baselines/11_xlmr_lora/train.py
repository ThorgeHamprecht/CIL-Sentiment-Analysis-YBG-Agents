"""Fine-tune XLM-R-base with LoRA + OLL loss."""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import ReviewDataset, read_csv
from model import XLMRLoRA, ev_decode, oll_loss

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_NAME = "xlm-roberta-base"
SCRATCH = Path("/work/scratch") / os.environ.get("USER", "<user>") / "cil"


def kaggle_score(preds, labels):
    return 1.0 - np.abs(preds - labels).mean() / 4.0


def train_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids     = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels        = batch["labels"].to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(input_ids, attention_mask)
            loss   = oll_loss(logits, labels)
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
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids     = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask)
        all_preds.append(ev_decode(logits).cpu().numpy())
        all_labels.append(batch["labels"].numpy())
    return kaggle_score(np.concatenate(all_preds), np.concatenate(all_labels))


def main(args):
    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.environ.setdefault("HF_HOME", str(SCRATCH / ".cache/huggingface"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    texts, labels, _ = read_csv(Path(args.data_dir) / "train.csv")
    print(f"Loaded {len(texts)} examples")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(sss.split(texts, labels))

    tr_texts = [texts[i] for i in train_idx]
    tr_labels = [labels[i] for i in train_idx]
    va_texts = [texts[i] for i in val_idx]
    va_labels = [labels[i] for i in val_idx]

    print("Tokenising train...")
    train_ds = ReviewDataset(tr_texts, tokenizer, args.max_len, tr_labels)
    print("Tokenising val...")
    val_ds   = ReviewDataset(va_texts, tokenizer, args.max_len, va_labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)

    model = XLMRLoRA(MODEL_NAME, dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if "lora" not in n], "lr": args.encoder_lr},
            {"params": [p for n, p in model.named_parameters() if "lora" in n],     "lr": args.lora_lr},
            {"params": model.classifier.parameters(),                                "lr": args.head_lr},
        ],
        weight_decay=0.01,
    )
    total_steps   = args.epochs * len(train_loader)
    warmup_steps  = int(0.06 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler    = GradScaler()

    best_score, patience_counter = 0.0, 0
    checkpoint_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device)
        val_score  = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:2d} | train_loss={train_loss:.4f} | val_score={val_score:.4f} | {time.time()-t0:.1f}s")

        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            model.encoder.save_pretrained(str(out_dir / "lora_adapter"))
            torch.save({"classifier": model.classifier.state_dict(), "args": vars(args)}, checkpoint_path)
            tokenizer.save_pretrained(str(out_dir / "tokenizer"))
            print(f"  -> New best: {best_score:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest val score: {best_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len",     type=int,   default=128)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--encoder_lr",  type=float, default=1e-5)
    parser.add_argument("--lora_lr",     type=float, default=3e-4)
    parser.add_argument("--head_lr",     type=float, default=1e-4)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--patience",    type=int,   default=3)
    parser.add_argument("--data_dir",    default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    main(parser.parse_args())
