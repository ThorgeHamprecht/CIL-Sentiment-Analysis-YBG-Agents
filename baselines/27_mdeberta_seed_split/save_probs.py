"""
Retrain seed 2024 then save raw softmax probs on the test set.
Computes both median decode and argmax decode scores for Plot 3.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import time

from dataset import ReviewDataset, read_csv
from model import EMA, emd_loss, median_decode, mDeBERTaEMD

ROOT     = Path(__file__).resolve().parents[2]
BACKBONE = "microsoft/mdeberta-v3-base"
SEED     = 2024


def kaggle_score(preds, labels):
    return 1.0 - np.abs(np.array(preds) - np.array(labels)).mean() / 4.0


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
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask)
            loss   = emd_loss(logits, labels)
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
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask)
        all_preds.append(median_decode(logits).cpu().numpy())
        all_labels.append(batch["labels"].numpy())
    return kaggle_score(np.concatenate(all_preds), np.concatenate(all_labels))


@torch.no_grad()
def get_probs(model, loader, device):
    model.eval()
    all_probs = []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask)
        all_probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(all_probs)


def main(args):
    data_dir     = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    output_dir   = Path(args.output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    tokenizer.save_pretrained(str(artifact_dir / "tokenizer"))

    # ── Training data (seed-specific split) ──────────────────────────────────
    texts, labels, _ = read_csv(data_dir / "train.csv")
    print(f"Loaded {len(texts):,} training examples")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    train_idx, val_idx = next(sss.split(texts, labels))
    train_texts  = [texts[i]  for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts    = [texts[i]  for i in val_idx]
    val_labels   = [labels[i] for i in val_idx]
    print(f"train={len(train_texts):,}  val={len(val_texts):,}")

    train_dataset = ReviewDataset(train_texts, tokenizer, max_len=args.max_len, labels=train_labels)
    val_dataset   = ReviewDataset(val_texts,   tokenizer, max_len=args.max_len, labels=val_labels)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Model + optimizer ────────────────────────────────────────────────────
    model = mDeBERTaEMD(model_name=BACKBONE, dropout=args.dropout).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer    = build_optimizer(model, args.head_lr, args.encoder_lr,
                                   args.layer_decay, args.weight_decay)
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = int(0.06 * total_steps)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
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
        if val_score >= best_score:
            best_score      = val_score
            patience_counter = 0
            ema.apply_shadow()
            torch.save({"model": model.state_dict(), "backbone_dir": BACKBONE}, ckpt_path)
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

    test_texts, _, test_ids = read_csv(data_dir / "test.csv")
    test_dataset = ReviewDataset(test_texts, tokenizer, max_len=args.max_len)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    probs = get_probs(model, test_loader, device)

    # Save raw probs
    probs_path = output_dir / f"seed{SEED}_test_probs.npy"
    np.save(probs_path, probs)
    print(f"Raw probs saved to {probs_path}  shape={probs.shape}")

    # ── Median decode ────────────────────────────────────────────────────────
    cdf           = np.cumsum(probs, axis=1)[:, :-1]
    median_preds  = (cdf < 0.5).sum(axis=1).clip(0, 4)

    # ── Argmax decode ────────────────────────────────────────────────────────
    argmax_preds  = probs.argmax(axis=1)

    # Save both submissions
    pd.DataFrame({"id": test_ids, "label": median_preds}).to_csv(
        output_dir / f"27_seed{SEED}_median_submission.csv", index=False)
    pd.DataFrame({"id": test_ids, "label": argmax_preds}).to_csv(
        output_dir / f"27_seed{SEED}_argmax_submission.csv", index=False)

    # Agreement between decoders
    agreement = (median_preds == argmax_preds).mean()
    print(f"\nDecoder agreement (median vs argmax): {agreement:.4f} "
          f"({(1-agreement)*len(median_preds):.0f} examples differ)")

    # Cases where they differ
    diff_mask = median_preds != argmax_preds
    print(f"Cases where decoders disagree: {diff_mask.sum():,} / {len(median_preds):,}")

    print("\nSubmissions saved:")
    print(f"  median: 27_seed{SEED}_median_submission.csv")
    print(f"  argmax: 27_seed{SEED}_argmax_submission.csv")
    print(f"  probs:  seed{SEED}_test_probs.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     default=str(ROOT / "data"))
    parser.add_argument("--artifact_dir", default=str(Path(__file__).parent / "artifacts_probs"))
    parser.add_argument("--output_dir",   default=str(ROOT / "submissions"))
    parser.add_argument("--max_len",      type=int,   default=256)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--encoder_lr",   type=float, default=8e-6)
    parser.add_argument("--head_lr",      type=float, default=5e-5)
    parser.add_argument("--layer_decay",  type=float, default=0.9)
    parser.add_argument("--dropout",      type=float, default=0.25)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs",       type=int,   default=6)
    parser.add_argument("--patience",     type=int,   default=1)
    main(parser.parse_args())
