"""5-fold cross-validation ensemble: mDeBERTa + LLRD + EMA, one seed per fold.
Reports OOF score after each fold completes.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import ReviewDataset, read_csv
from model import EMA, emd_loss, median_decode, mDeBERTaEMD

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
BACKBONE = "microsoft/mdeberta-v3-base"

# One distinct seed per fold
FOLD_SEEDS = [42, 1337, 2024, 99, 777]


def kaggle_score(preds: np.ndarray, labels: np.ndarray) -> float:
    return 1.0 - np.abs(preds - labels).mean() / 4.0


def build_optimizer(model, head_lr, encoder_top_lr, layer_decay, weight_decay):
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

    head_params = list(model.classifier.named_parameters())
    groups = [
        {"params": [p for n, p in head_params if not any(nd in n for nd in no_decay)],
         "lr": head_lr, "weight_decay": weight_decay},
        {"params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
         "lr": head_lr, "weight_decay": 0.0},
    ]

    num_layers = model.encoder.config.num_hidden_layers
    for i, layer_idx in enumerate(range(num_layers - 1, -1, -1)):
        lr = encoder_top_lr * (layer_decay ** i)
        params = list(model.encoder.encoder.layer[layer_idx].named_parameters())
        groups.extend([
            {"params": [p for n, p in params if not any(nd in n for nd in no_decay)],
             "lr": lr, "weight_decay": weight_decay},
            {"params": [p for n, p in params if any(nd in n for nd in no_decay)],
             "lr": lr, "weight_decay": 0.0},
        ])

    emb_lr = encoder_top_lr * (layer_decay ** num_layers)
    emb_params = list(model.encoder.embeddings.named_parameters())
    groups.extend([
        {"params": [p for n, p in emb_params if not any(nd in n for nd in no_decay)],
         "lr": emb_lr, "weight_decay": weight_decay},
        {"params": [p for n, p in emb_params if any(nd in n for nd in no_decay)],
         "lr": emb_lr, "weight_decay": 0.0},
    ])
    return torch.optim.AdamW(groups)


def train_epoch(model, loader, optimizer, scheduler, ema, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask)
        all_probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(all_probs)


def run_fold(fold_idx, seed, train_dataset, val_dataset, args, out_dir, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    model = mDeBERTaEMD(model_name=BACKBONE, dropout=args.dropout).to(device)
    print(f"  [fold={fold_idx}] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = build_optimizer(
        model,
        head_lr=args.head_lr,
        encoder_top_lr=args.encoder_lr,
        layer_decay=args.layer_decay,
        weight_decay=args.weight_decay,
    )
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = int(0.06 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    ema = EMA(model, decay=0.999)

    best_score, patience_counter = 0.0, 0
    checkpoint_path = out_dir / f"best_model_fold{fold_idx}.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, ema, device)

        ema.apply_shadow()
        val_score = evaluate(model, val_loader, device)
        ema.restore()

        print(
            f"  [fold={fold_idx}] Epoch {epoch:2d} | train_loss={train_loss:.4f} "
            f"| val_score={val_score:.4f} | {time.time()-t0:.1f}s"
        )

        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            ema.apply_shadow()
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "backbone_dir": BACKBONE,
                "fold": fold_idx,
                "seed": seed,
            }, checkpoint_path)
            ema.restore()
            print(f"    -> New best: {best_score:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"    Early stop at epoch {epoch}")
                break

    print(f"  [fold={fold_idx}] Best val score: {best_score:.4f}")
    return best_score


def main(args):
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    texts, labels, _ = read_csv(data_dir / "train.csv")
    labels_arr = np.array(labels)
    print(f"Loaded {len(texts):,} examples")

    print("Tokenizing all examples...")
    full_dataset = ReviewDataset(texts, tokenizer, max_len=args.max_len, labels=labels)
    tokenizer.save_pretrained(str(out_dir / "tokenizer"))

    # Save fold indices for eval script to reconstruct OOF
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_indices = [(train_idx, val_idx) for train_idx, val_idx in skf.split(texts, labels)]
    np.save(out_dir / "fold_indices.npy", np.array(fold_indices, dtype=object), allow_pickle=True)

    # OOF predictions array (filled in as folds complete)
    oof_preds  = np.full(len(texts), -1, dtype=int)
    oof_labels = labels_arr.copy()

    for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
        seed = FOLD_SEEDS[fold_idx]
        print(f"\n=== Fold {fold_idx} (seed={seed}, train={len(train_idx):,}, val={len(val_idx):,}) ===")

        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset   = Subset(full_dataset, val_idx)

        run_fold(fold_idx, seed, train_dataset, val_dataset, args, out_dir, device)

        # Collect OOF predictions for this fold
        ckpt  = torch.load(out_dir / f"best_model_fold{fold_idx}.pt", map_location=device, weights_only=False)
        model = mDeBERTaEMD(model_name=BACKBONE, dropout=0.0).to(device)
        model.load_state_dict(ckpt["model"])

        val_loader_nolbl = DataLoader(
            Subset(full_dataset, val_idx),
            batch_size=args.batch_size * 2, shuffle=False, num_workers=2,
        )
        probs = predict_probs(model, val_loader_nolbl, device)
        cdf   = np.cumsum(probs, axis=1)[:, :-1]
        oof_preds[val_idx] = (cdf < 0.5).sum(axis=1).clip(0, 4)
        del model
        torch.cuda.empty_cache()

        # OOF score so far (only over completed folds)
        completed_mask = oof_preds >= 0
        oof_score = kaggle_score(oof_preds[completed_mask], oof_labels[completed_mask])
        print(f"  OOF score after fold {fold_idx}: {oof_score:.4f}  ({completed_mask.sum():,} examples)")

    final_oof = kaggle_score(oof_preds, oof_labels)
    print(f"\nFinal OOF score (all {len(texts):,} examples): {final_oof:.4f}")
    np.save(out_dir / "oof_preds.npy", oof_preds)
    print("All folds done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_folds",      type=int,   default=5)
    parser.add_argument("--max_len",      type=int,   default=256)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--encoder_lr",   type=float, default=8e-6)
    parser.add_argument("--head_lr",      type=float, default=5e-5)
    parser.add_argument("--layer_decay",  type=float, default=0.9)
    parser.add_argument("--dropout",      type=float, default=0.25)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs",       type=int,   default=6)
    parser.add_argument("--patience",     type=int,   default=1)
    parser.add_argument("--data_dir",     default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    main(parser.parse_args())
