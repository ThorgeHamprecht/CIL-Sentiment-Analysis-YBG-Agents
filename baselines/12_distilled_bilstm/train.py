"""Transductive distillation: BiLSTM student trained on train (OLL+KL) and test (KL only).

Loss per batch:
  labeled   (train): alpha * OLL(logits, true_label) + (1-alpha) * distillation_loss(logits, soft)
  unlabeled (test):  distillation_loss(logits, soft)

SWA is applied over the last swa_epochs epochs to smooth the final checkpoint.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import ConcatDataset, DataLoader

from dataset import (
    DistillDataset,
    build_vocab,
    load_soft_labels,
    load_vocab,
    read_csv,
    save_vocab,
)
from model import TwoStreamBiLSTM, distillation_loss, ev_decode, oll_loss

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"

ALPHA = 0.5      # weight on OLL vs KL
TEMPERATURE = 2.0
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
    for x_t, l_t, x_b, l_b, labels, soft, has_label in loader:
        x_t, l_t, x_b, l_b = x_t.to(device), l_t.to(device), x_b.to(device), l_b.to(device)
        labels, soft, has_label = labels.to(device), soft.to(device), has_label.to(device)

        optimizer.zero_grad()
        main_logits, title_logits = model(x_t, l_t, x_b, l_b)

        kl = distillation_loss(main_logits, soft, TEMPERATURE)

        labeled = has_label.bool()
        if labeled.any():
            oll = oll_loss(main_logits[labeled], labels[labeled])
            title_oll = oll_loss(title_logits[labeled], labels[labeled])
            loss = ALPHA * (oll + AUX_WEIGHT * title_oll) + (1 - ALPHA) * kl
        else:
            loss = kl

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate on the labeled val split only."""
    model.eval()
    all_preds, all_labels = [], []
    for x_t, l_t, x_b, l_b, labels, soft, has_label in loader:
        x_t, l_t, x_b, l_b = x_t.to(device), l_t.to(device), x_b.to(device), l_b.to(device)
        main_logits, _ = model(x_t, l_t, x_b, l_b)
        all_preds.append(ev_decode(main_logits).cpu().numpy())
        all_labels.append(labels.numpy())
    return kaggle_score(np.concatenate(all_preds), np.concatenate(all_labels))


def main(args):
    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    soft_dir = Path(args.soft_dir)

    # --- load texts ---
    _, tr_titles, tr_bodies, tr_labels, _ = read_csv(data_dir / "train.csv")
    _, te_titles, te_bodies, _, _ = read_csv(data_dir / "test.csv")
    print(f"Train: {len(tr_titles)}  Test: {len(te_titles)}")

    # --- load soft labels (must match row order of the CSVs) ---
    _, train_soft = load_soft_labels(soft_dir / "train_soft.csv")
    _, test_soft  = load_soft_labels(soft_dir / "test_soft.csv")

    # --- vocab ---
    vocab_path = out_dir / "vocab.json"
    if vocab_path.exists():
        vocab = load_vocab(vocab_path)
        print(f"Loaded vocab: {len(vocab)}")
    else:
        all_texts = [t + " " + b for t, b in zip(tr_titles + te_titles, tr_bodies + te_bodies)]
        vocab = build_vocab(all_texts, max_vocab=args.max_vocab)
        save_vocab(vocab, vocab_path)
        print(f"Built vocab: {len(vocab)}")

    # --- val split from train (same seed as all other baselines) ---
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(sss.split(tr_titles, tr_labels))

    def sub(idx, titles, bodies, labels, soft):
        return ([titles[i] for i in idx], [bodies[i] for i in idx],
                [labels[i] for i in idx], soft[idx])

    s_tr_titles, s_tr_bodies, s_tr_labels, s_tr_soft = sub(train_idx, tr_titles, tr_bodies, tr_labels, train_soft)
    s_va_titles, s_va_bodies, s_va_labels, s_va_soft  = sub(val_idx,   tr_titles, tr_bodies, tr_labels, train_soft)

    labeled_ds   = DistillDataset(s_tr_titles, s_tr_bodies, vocab, s_tr_soft,
                                  args.max_len_title, args.max_len_body, s_tr_labels)
    val_ds       = DistillDataset(s_va_titles, s_va_bodies, vocab, s_va_soft,
                                  args.max_len_title, args.max_len_body, s_va_labels)
    unlabeled_ds = DistillDataset(te_titles, te_bodies, vocab, test_soft,
                                  args.max_len_title, args.max_len_body)

    combined_ds = ConcatDataset([labeled_ds, unlabeled_ds])
    train_loader = DataLoader(combined_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,      batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

    model = TwoStreamBiLSTM(
        vocab_size=len(vocab), embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    warmup_steps = args.warmup_epochs * len(train_loader)
    total_steps  = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps)

    swa_model = AveragedModel(model)
    swa_start = args.epochs - args.swa_epochs

    best_score, patience_counter = 0.0, 0
    checkpoint_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_score  = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:2d} | loss={train_loss:.4f} | val={val_score:.4f} | {time.time()-t0:.1f}s")

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

    # Update BN statistics for SWA model then save
    print("Updating BN for SWA model...")
    update_bn(train_loader, swa_model, device=device)
    swa_score = evaluate(swa_model, val_loader, device)
    print(f"SWA val score: {swa_score:.4f}  (base best: {best_score:.4f})")

    if swa_score >= best_score:
        torch.save({"model": swa_model.module.state_dict(), "args": vars(args), "vocab_size": len(vocab)},
                   checkpoint_path)
        print("SWA model is better — saved as final checkpoint.")

    print(f"\nFinal checkpoint val score: {max(best_score, swa_score):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--swa_epochs",    type=int,   default=5)
    parser.add_argument("--data_dir",      default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--artifact_dir",  default=str(_DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--soft_dir",      default="/work/scratch/thamprecht/cil/artifacts/11_xlmr_lora")
    main(parser.parse_args())
