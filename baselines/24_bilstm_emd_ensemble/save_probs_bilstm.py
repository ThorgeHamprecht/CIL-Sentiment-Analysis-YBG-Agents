"""
Retrain BiLSTM seed 42 and save raw softmax probs on the test set.
For Plot 3: median vs argmax decode comparison.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from dataset import TwoStreamDataset, build_vocab, save_vocab, read_csv
from model import EMA, TwoStreamBiLSTM, emd_loss, median_decode

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
AUX_WEIGHT = 0.3


def kaggle_score(preds, labels):
    return 1.0 - np.abs(np.array(preds) - np.array(labels)).mean() / 4.0


def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, scheduler, ema, device):
    model.train()
    total_loss = 0.0
    for x_t, l_t, x_b, l_b, y in loader:
        x_t, l_t, x_b, l_b, y = (x_t.to(device), l_t.to(device),
                                   x_b.to(device), l_b.to(device), y.to(device))
        optimizer.zero_grad()
        main_logits, title_logits = model(x_t, l_t, x_b, l_b)
        loss = emd_loss(main_logits, y) + AUX_WEIGHT * emd_loss(title_logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        ema.update()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for x_t, l_t, x_b, l_b, y in loader:
        x_t, l_t, x_b, l_b = (x_t.to(device), l_t.to(device),
                                x_b.to(device), l_b.to(device))
        main_logits, _ = model(x_t, l_t, x_b, l_b)
        all_preds.append(median_decode(main_logits).cpu().numpy())
        all_labels.append(y.numpy())
    return kaggle_score(np.concatenate(all_preds), np.concatenate(all_labels))


@torch.no_grad()
def get_probs(model, loader, device):
    model.eval()
    all_probs = []
    for batch in loader:
        x_t, l_t, x_b, l_b = (batch[0].to(device), batch[1].to(device),
                                batch[2].to(device), batch[3].to(device))
        main_logits, _ = model(x_t, l_t, x_b, l_b)
        all_probs.append(F.softmax(main_logits.float(), dim=1).cpu().numpy())
    return np.concatenate(all_probs)


def main(args):
    artifact_dir = Path(args.artifact_dir)
    output_dir   = Path(args.output_dir)
    data_dir     = Path(args.data_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load training data ────────────────────────────────────────────────────
    _, titles, bodies, labels, _ = read_csv(data_dir / "train.csv")
    print(f"Loaded {len(titles):,} training examples")

    vocab = build_vocab([t + " " + b for t, b in zip(titles, bodies)],
                        max_vocab=args.max_vocab)
    save_vocab(vocab, artifact_dir / "vocab.json")
    print(f"Built vocab: {len(vocab):,}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    train_idx, val_idx = next(sss.split(titles, labels))

    def subset(idx):
        return ([titles[i] for i in idx],
                [bodies[i] for i in idx],
                [labels[i] for i in idx])

    tr_t, tr_b, tr_l = subset(train_idx)
    va_t, va_b, va_l = subset(val_idx)

    train_ds = TwoStreamDataset(tr_t, tr_b, vocab, args.max_len_title, args.max_len_body, tr_l)
    val_ds   = TwoStreamDataset(va_t, va_b, vocab, args.max_len_title, args.max_len_body, va_l)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TwoStreamBiLSTM(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    warmup_steps = args.warmup_epochs * len(train_loader)
    total_steps  = args.epochs * len(train_loader)
    scheduler    = get_cosine_schedule(optimizer, warmup_steps, total_steps)
    ema          = EMA(model, decay=0.999)

    best_score, patience_counter = 0.0, 0
    ckpt_path = artifact_dir / f"best_model_seed{SEED}_probs.pt"

    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, ema, device)
        ema.apply_shadow()
        val_score = evaluate(model, val_loader, device)
        ema.restore()
        print(f"  Epoch {epoch:2d} | train_loss={train_loss:.4f} "
              f"| val_score={val_score:.4f} | {time.time()-t0:.1f}s")
        if val_score > best_score:
            best_score       = val_score
            patience_counter = 0
            ema.apply_shadow()
            torch.save({
                "model": model.state_dict(),
                "vocab_size": len(vocab),
                "args": vars(args),
            }, ckpt_path)
            ema.restore()
            print(f"    -> New best: {best_score:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"    Early stop at epoch {epoch}")
                break

    print(f"Best val score: {best_score:.4f}")

    # ── Inference on test set — save raw probs ───────────────────────────────
    print("\nRunning inference on test set...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    _, test_titles, test_bodies, _, test_ids = read_csv(data_dir / "test.csv")
    test_ds     = TwoStreamDataset(test_titles, test_bodies, vocab,
                                   args.max_len_title, args.max_len_body)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    probs = get_probs(model, test_loader, device)

    # Save raw probs
    probs_path = output_dir / f"bilstm_seed{SEED}_test_probs.npy"
    np.save(probs_path, probs)
    print(f"Raw probs saved to {probs_path}  shape={probs.shape}")

    # ── Median vs argmax ─────────────────────────────────────────────────────
    cdf          = np.cumsum(probs, axis=1)[:, :-1]
    median_preds = (cdf < 0.5).sum(axis=1).clip(0, 4)
    argmax_preds = probs.argmax(axis=1)

    pd.DataFrame({"id": test_ids, "label": median_preds}).to_csv(
        output_dir / f"24_bilstm_seed{SEED}_median_submission.csv", index=False)
    pd.DataFrame({"id": test_ids, "label": argmax_preds}).to_csv(
        output_dir / f"24_bilstm_seed{SEED}_argmax_submission.csv", index=False)

    agreement = (median_preds == argmax_preds).mean()
    print(f"\nDecoder agreement (median vs argmax): {agreement:.4f} "
          f"({(1-agreement)*len(median_preds):.0f} examples differ)")
    print(f"Submissions saved: 24_bilstm_seed{SEED}_median/argmax_submission.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",      default=str(ROOT / "data"))
    parser.add_argument("--artifact_dir",  default=str(Path(__file__).parent / "artifacts_probs"))
    parser.add_argument("--output_dir",    default=str(ROOT / "submissions"))
    parser.add_argument("--max_vocab",     type=int,   default=30_000)
    parser.add_argument("--max_len_title", type=int,   default=64)
    parser.add_argument("--max_len_body",  type=int,   default=192)
    parser.add_argument("--embed_dim",     type=int,   default=128)
    parser.add_argument("--hidden_dim",    type=int,   default=384)
    parser.add_argument("--num_layers",    type=int,   default=2)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--batch_size",    type=int,   default=256)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--patience",      type=int,   default=6)
    parser.add_argument("--warmup_epochs", type=int,   default=2)
    main(parser.parse_args())
