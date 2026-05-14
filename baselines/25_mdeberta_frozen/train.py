"""Frozen mDeBERTa encoder: train head only, EMA, 3-seed ensemble with live scores."""
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
from model import EMA, emd_loss, median_decode, mDeBERTaFrozen

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
BACKBONE = "microsoft/mdeberta-v3-base"


def kaggle_score(preds: np.ndarray, labels: np.ndarray) -> float:
    return 1.0 - np.abs(preds - labels).mean() / 4.0


def train_epoch(model, loader, optimizer, scheduler, ema, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = emd_loss(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        ema.update()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
        all_preds.append(median_decode(logits).cpu().numpy())
        all_labels.append(batch["labels"].numpy())
    return kaggle_score(np.concatenate(all_preds), np.concatenate(all_labels))


@torch.no_grad()
def predict_probs(model, loader, device) -> np.ndarray:
    model.eval()
    all_probs = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
        all_probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(all_probs)


def eval_ensemble(completed_seeds, out_dir, val_loader_nolabel, val_labels, device):
    all_probs = []
    for seed in completed_seeds:
        ckpt = torch.load(out_dir / f"best_model_seed{seed}.pt", map_location=device, weights_only=False)
        model = mDeBERTaFrozen(model_name=BACKBONE, dropout=0.0).to(device)
        model.load_state_dict(ckpt["model"])
        all_probs.append(predict_probs(model, val_loader_nolabel, device))
        del model
    avg = np.mean(all_probs, axis=0)
    cdf = np.cumsum(avg, axis=1)[:, :-1]
    preds = (cdf < 0.5).sum(axis=1).clip(0, avg.shape[1] - 1)
    return kaggle_score(preds, val_labels)


def run_seed(seed, train_dataset, val_dataset, args, out_dir, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

    model = mDeBERTaFrozen(model_name=BACKBONE, dropout=args.dropout).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  [seed={seed}] Trainable: {trainable:,} / {total:,} params (encoder frozen)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = int(0.06 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    ema = EMA(model, decay=0.999)

    best_score, patience_counter = 0.0, 0
    checkpoint_path = out_dir / f"best_model_seed{seed}.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, ema, device)

        ema.apply_shadow()
        val_score = evaluate(model, val_loader, device)
        ema.restore()

        print(f"  [seed={seed}] Epoch {epoch:2d} | train_loss={train_loss:.4f} | val_score={val_score:.4f} | {time.time()-t0:.1f}s")

        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            ema.apply_shadow()
            torch.save({"model": model.state_dict(), "args": vars(args), "backbone": BACKBONE, "seed": seed}, checkpoint_path)
            ema.restore()
            print(f"    -> New best: {best_score:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"    Early stop at epoch {epoch}")
                break

    print(f"  [seed={seed}] Best val score: {best_score:.4f}")
    return best_score


def main(args):
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    texts, labels, _ = read_csv(data_dir / "train.csv")
    print(f"Loaded {len(texts):,} examples")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(sss.split(texts, labels))
    tr_texts = [texts[i] for i in train_idx];  tr_labels = [labels[i] for i in train_idx]
    va_texts = [texts[i] for i in val_idx];    va_labels = [labels[i] for i in val_idx]
    val_labels_arr = np.array(va_labels)

    print("Tokenizing...")
    train_dataset     = ReviewDataset(tr_texts, tokenizer, max_len=args.max_len, labels=tr_labels)
    val_dataset       = ReviewDataset(va_texts, tokenizer, max_len=args.max_len, labels=va_labels)
    val_dataset_nolbl = ReviewDataset(va_texts, tokenizer, max_len=args.max_len)

    tokenizer.save_pretrained(str(out_dir / "tokenizer"))

    val_loader_nolabel = DataLoader(val_dataset_nolbl, batch_size=args.batch_size * 2, shuffle=False, num_workers=2)

    seeds = [int(s) for s in args.seeds]
    completed_seeds = []

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        run_seed(seed, train_dataset, val_dataset, args, out_dir, device)
        completed_seeds.append(seed)

        ens_score = eval_ensemble(completed_seeds, out_dir, val_loader_nolabel, val_labels_arr, device)
        if len(completed_seeds) == 1:
            print(f"  [ensemble={completed_seeds}] Val score: {ens_score:.4f}  (single seed)")
        else:
            print(f"  [ensemble={completed_seeds}] Val score: {ens_score:.4f}")

    print("\nAll seeds done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",        nargs="+", default=["42", "1337", "2024"])
    parser.add_argument("--max_len",      type=int,   default=256)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--patience",     type=int,   default=5)
    parser.add_argument("--data_dir",     default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    main(parser.parse_args())
