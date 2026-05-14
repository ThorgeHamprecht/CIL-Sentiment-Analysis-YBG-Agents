"""Train the BiLSTM sentiment classifier."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

from dataset import ReviewDataset, build_vocab, load_vocab, read_csv, save_vocab
from model import BiLSTMClassifier

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"


def kaggle_score(preds: np.ndarray, labels: np.ndarray) -> float:
    return 1.0 - np.abs(preds - labels).mean() / 4.0


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        logits = model(x, lengths)
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

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_score, patience_counter = 0.0, 0
    checkpoint_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_score = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:2d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"| val_score={val_score:.4f} | {time.time()-t0:.1f}s"
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
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--data_dir", default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    main(parser.parse_args())
