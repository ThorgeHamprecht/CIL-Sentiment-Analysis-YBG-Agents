"""Train the FastText-initialised two-stream BiLSTM."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader

from dataset import TwoStreamDataset, load_vocab, read_csv
from model import TwoStreamBiLSTM, ev_decode, oll_loss

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"

AUX_WEIGHT = 0.3


def kaggle_score(preds, labels):
    return 1.0 - np.abs(preds - labels).mean() / 4.0


def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for x_t, l_t, x_b, l_b, y in loader:
        x_t, l_t, x_b, l_b, y = x_t.to(device), l_t.to(device), x_b.to(device), l_b.to(device), y.to(device)
        optimizer.zero_grad()
        main_logits, title_logits = model(x_t, l_t, x_b, l_b)
        loss = oll_loss(main_logits, y) + AUX_WEIGHT * oll_loss(title_logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for x_t, l_t, x_b, l_b, y in loader:
        x_t, l_t, x_b, l_b = x_t.to(device), l_t.to(device), x_b.to(device), l_b.to(device)
        main_logits, _ = model(x_t, l_t, x_b, l_b)
        all_preds.append(ev_decode(main_logits).cpu().numpy())
        all_labels.append(y.numpy())
    return kaggle_score(np.concatenate(all_preds), np.concatenate(all_labels))


def main(args):
    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    embeddings_path = out_dir / "embeddings.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"embeddings.npy not found at {embeddings_path}\n"
            "Run build_embeddings.py first (or let submit.sh do it)."
        )

    vocab = load_vocab(out_dir / "vocab.json")
    print(f"Vocab size: {len(vocab)}")

    _, titles, bodies, labels, _ = read_csv(Path(args.data_dir) / "train.csv")
    print(f"Loaded {len(titles)} examples")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(sss.split(titles, labels))

    def subset(idx):
        return [titles[i] for i in idx], [bodies[i] for i in idx], [labels[i] for i in idx]

    tr_t, tr_b, tr_l = subset(train_idx)
    va_t, va_b, va_l = subset(val_idx)

    train_ds = TwoStreamDataset(tr_t, tr_b, vocab, args.max_len_title, args.max_len_body, tr_l)
    val_ds   = TwoStreamDataset(va_t, va_b, vocab, args.max_len_title, args.max_len_body, va_l)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

    model = TwoStreamBiLSTM(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        embeddings_path=str(embeddings_path),
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.freeze_epochs > 0:
        model.embedding.weight.requires_grad_(False)
        print(f"Embeddings frozen for first {args.freeze_epochs} epochs")

    optimizer = torch.optim.AdamW([
        {"params": model.embedding.parameters(),                                      "lr": args.embedding_lr, "weight_decay": 1e-2},
        {"params": [p for n, p in model.named_parameters() if "embedding" not in n], "lr": args.lr,           "weight_decay": 3e-2},
    ])
    warmup_steps = args.warmup_epochs * len(train_loader)
    total_steps  = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps)

    best_score, patience_counter = 0.0, 0
    checkpoint_path = out_dir / "best_model.pt"
    swa_model = AveragedModel(model)
    swa_start = args.epochs - args.swa_epochs

    for epoch in range(1, args.epochs + 1):
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            model.embedding.weight.requires_grad_(True)
            print(f"Embeddings unfrozen at epoch {epoch}")

        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_score  = evaluate(model, val_loader, device)
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:2d} | train_loss={train_loss:.4f} | val_score={val_score:.4f} | lr={lr:.2e} | {time.time()-t0:.1f}s")

        if epoch >= swa_start:
            swa_model.update_parameters(model)

        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "vocab_size": len(vocab)}, checkpoint_path)
            print(f"  -> New best: {best_score:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nUpdating BN for SWA model...")
    update_bn(train_loader, swa_model, device=device)
    swa_score = evaluate(swa_model, val_loader, device)
    print(f"SWA val score: {swa_score:.4f}  (best single: {best_score:.4f})")
    if swa_score >= best_score:
        torch.save({"model": swa_model.module.state_dict(), "args": vars(args), "vocab_size": len(vocab)}, checkpoint_path)
        print("SWA model is better — saved as final checkpoint.")

    print(f"\nFinal val score: {max(best_score, swa_score):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim",      type=int,   default=300)
    parser.add_argument("--max_len_title",  type=int,   default=64)
    parser.add_argument("--max_len_body",   type=int,   default=192)
    parser.add_argument("--hidden_dim",     type=int,   default=384)
    parser.add_argument("--num_layers",     type=int,   default=2)
    parser.add_argument("--dropout",        type=float, default=0.4)
    parser.add_argument("--batch_size",     type=int,   default=512)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--embedding_lr",  type=float, default=1e-4)
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--patience",       type=int,   default=6)
    parser.add_argument("--warmup_epochs",  type=int,   default=3)
    parser.add_argument("--freeze_epochs",  type=int,   default=5)
    parser.add_argument("--swa_epochs",     type=int,   default=5)
    parser.add_argument("--data_dir",       default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--artifact_dir",   default=str(_DEFAULT_ARTIFACT_DIR))
    main(parser.parse_args())
