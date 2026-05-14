"""Two-stream BiLSTM + EMD² + EMA, 3-seed ensemble with live ensemble val score."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from dataset import TwoStreamDataset, build_vocab, load_vocab, read_csv, save_vocab
from model import EMA, TwoStreamBiLSTM, emd_loss, median_decode

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"

AUX_WEIGHT = 0.3


def kaggle_score(preds: np.ndarray, labels: np.ndarray) -> float:
    return 1.0 - np.abs(preds - labels).mean() / 4.0


def get_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
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
        x_t, l_t, x_b, l_b, y = x_t.to(device), l_t.to(device), x_b.to(device), l_b.to(device), y.to(device)
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
        x_t, l_t, x_b, l_b = x_t.to(device), l_t.to(device), x_b.to(device), l_b.to(device)
        main_logits, _ = model(x_t, l_t, x_b, l_b)
        all_preds.append(median_decode(main_logits).cpu().numpy())
        all_labels.append(y.numpy())
    return kaggle_score(np.concatenate(all_preds), np.concatenate(all_labels))


@torch.no_grad()
def predict_probs(model, loader, device) -> np.ndarray:
    model.eval()
    all_probs = []
    for batch in loader:
        x_t, l_t, x_b, l_b = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        main_logits, _ = model(x_t, l_t, x_b, l_b)
        all_probs.append(F.softmax(main_logits.float(), dim=1).cpu().numpy())
    return np.concatenate(all_probs)


def eval_ensemble(completed_seeds, out_dir, vocab, args, val_loader, val_labels, device):
    """Load best checkpoints of completed seeds and report ensemble val score."""
    all_probs = []
    for seed in completed_seeds:
        ckpt = torch.load(out_dir / f"best_model_seed{seed}.pt", map_location=device, weights_only=False)
        model = TwoStreamBiLSTM(
            vocab_size=ckpt["vocab_size"],
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=0.0,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        all_probs.append(predict_probs(model, val_loader, device))
        del model
    avg_probs = np.mean(all_probs, axis=0)
    cdf = np.cumsum(avg_probs, axis=1)[:, :-1]
    preds = (cdf < 0.5).sum(axis=1).clip(0, avg_probs.shape[1] - 1)
    return kaggle_score(preds, val_labels)


def run_seed(seed, vocab, train_ds, val_ds, val_labels, args, out_dir, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

    model = TwoStreamBiLSTM(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    print(f"  [seed={seed}] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    warmup_steps = args.warmup_epochs * len(train_loader)
    total_steps  = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps)
    ema = EMA(model, decay=0.999)

    best_score, patience_counter = 0.0, 0
    checkpoint_path = out_dir / f"best_model_seed{seed}.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, ema, device)

        ema.apply_shadow()
        val_score = evaluate(model, val_loader, device)
        ema.restore()

        lr = scheduler.get_last_lr()[0]
        print(
            f"  [seed={seed}] Epoch {epoch:2d} | train_loss={train_loss:.4f} "
            f"| val_score={val_score:.4f} | lr={lr:.2e} | {time.time()-t0:.1f}s"
        )

        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            ema.apply_shadow()
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "vocab_size": len(vocab),
                "seed": seed,
            }, checkpoint_path)
            ema.restore()
            print(f"    -> New best: {best_score:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"    Early stop at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    print(f"  [seed={seed}] Best val score: {best_score:.4f}")
    return best_score


def main(args):
    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, titles, bodies, labels, _ = read_csv(Path(args.data_dir) / "train.csv")
    print(f"Loaded {len(titles):,} examples")

    vocab_path = out_dir / "vocab.json"
    if vocab_path.exists():
        vocab = load_vocab(vocab_path)
        print(f"Loaded vocab: {len(vocab):,}")
    else:
        vocab = build_vocab([t + " " + b for t, b in zip(titles, bodies)], max_vocab=args.max_vocab)
        save_vocab(vocab, vocab_path)
        print(f"Built vocab: {len(vocab):,}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(sss.split(titles, labels))

    def subset(idx):
        return [titles[i] for i in idx], [bodies[i] for i in idx], [labels[i] for i in idx]

    tr_t, tr_b, tr_l = subset(train_idx)
    va_t, va_b, va_l = subset(val_idx)
    val_labels = np.array(va_l)

    train_ds = TwoStreamDataset(tr_t, tr_b, vocab, args.max_len_title, args.max_len_body, tr_l)
    val_ds   = TwoStreamDataset(va_t, va_b, vocab, args.max_len_title, args.max_len_body, va_l)

    # Val loader without labels for ensemble prob prediction
    val_ds_nolabel = TwoStreamDataset(va_t, va_b, vocab, args.max_len_title, args.max_len_body)
    val_loader_nolabel = DataLoader(val_ds_nolabel, batch_size=args.batch_size * 2, shuffle=False, num_workers=2)

    seeds = [int(s) for s in args.seeds]
    completed_seeds = []

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        run_seed(seed, vocab, train_ds, val_ds, val_labels, args, out_dir, device)
        completed_seeds.append(seed)

        ens_score = eval_ensemble(completed_seeds, out_dir, vocab, args, val_loader_nolabel, val_labels, device)
        if len(completed_seeds) == 1:
            print(f"  [ensemble={completed_seeds}] Val score: {ens_score:.4f}  (single seed, no averaging yet)")
        else:
            print(f"  [ensemble={completed_seeds}] Val score: {ens_score:.4f}")

    print("\nAll seeds done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",         nargs="+", default=["42", "1337", "2024"])
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
    parser.add_argument("--data_dir",      default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--artifact_dir",  default=str(_DEFAULT_ARTIFACT_DIR))
    main(parser.parse_args())
