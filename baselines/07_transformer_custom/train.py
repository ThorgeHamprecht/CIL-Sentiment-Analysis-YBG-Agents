"""Train the custom Transformer sentiment classifier with warmup + cosine annealing."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

from dataset import ReviewDataset, build_vocab, load_vocab, read_csv, save_vocab
from model import CustomTransformerClassifier

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"


def kaggle_score(preds: np.ndarray, labels: np.ndarray) -> float:
    return 1.0 - np.abs(preds - labels).mean() / 4.0


def get_warmup_lr(step: int, d_model: int, warmup_steps: int) -> float:
    if step == 0:
        return 0.0
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)


def train_epoch(model, loader, optimizer, scheduler, criterion, device, warmup_steps, step):
    model.train()
    total_loss = 0.0
    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        step += 1
        if step <= warmup_steps:
            lr = get_warmup_lr(step, model.embedding.embedding_dim, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        optimizer.zero_grad()
        logits = model(x, mask)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)

    if step > warmup_steps:
        scheduler.step()

    return total_loss / len(loader.dataset), step


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        logits = model(x, mask)
        total_loss += criterion(logits, y).item() * len(y)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return total_loss / len(loader.dataset), kaggle_score(preds, labels)


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    texts, labels, _ = read_csv(data_dir / "train.csv")
    print(f"Loaded {len(texts)} examples")

    vocab_path = out_dir / "vocab.json"
    if vocab_path.exists():
        print("Loading existing vocab...")
        vocab = load_vocab(vocab_path)
    else:
        print("Building vocab...")
        vocab = build_vocab(texts, max_vocab=args.max_vocab)
        save_vocab(vocab, vocab_path)
    print(f"Vocab size: {len(vocab)}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(sss.split(texts, labels))

    dataset = ReviewDataset(texts, vocab, max_len=args.max_len, labels=labels)
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=args.batch_size * 2,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    model = CustomTransformerClassifier(
        vocab_size=len(vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.max_len,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    best_score, patience_counter, step = 0.0, 0, 0
    checkpoint_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, step = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, warmup_steps, step
        )
        val_loss, val_score = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:2d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"| val_score={val_score:.4f} | lr={current_lr:.2e} | {time.time()-t0:.1f}s"
        )

        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "vocab_size": len(vocab)},
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_vocab", type=int, default=30_000)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--data_dir",     default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    main(parser.parse_args())
