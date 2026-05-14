"""Fine-tune mDeBERTa-v3-base with LLRD + EMA, EMD² loss, 3-seed ensemble."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import ReviewDataset, read_csv
from model import EMA, emd_loss, median_decode, mDeBERTaEMD

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
BACKBONE = "microsoft/mdeberta-v3-base"


def kaggle_score(preds: np.ndarray, labels: np.ndarray) -> float:
    return 1.0 - np.abs(preds - labels).mean() / 4.0


def build_optimizer(model, head_lr, encoder_top_lr, layer_decay, weight_decay):
    """LLRD: head > top encoder layer > ... > embeddings."""
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
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask)
            loss = emd_loss(logits, labels)
        total_loss += loss.item() * len(labels)
        all_preds.append(median_decode(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    return total_loss / len(loader.dataset), kaggle_score(preds, labels_arr)


def run_seed(seed, train_dataset, val_dataset, args, out_dir, device):
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
    print(f"  [seed={seed}] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = build_optimizer(
        model,
        head_lr=args.head_lr,
        encoder_top_lr=args.encoder_lr,
        layer_decay=args.layer_decay,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.06 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    ema = EMA(model, decay=0.999)

    best_score, patience_counter = 0.0, 0
    checkpoint_path = out_dir / f"best_model_seed{seed}.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, ema, device)

        ema.apply_shadow()
        val_loss, val_score = evaluate(model, val_loader, device)
        ema.restore()

        print(
            f"  [seed={seed}] Epoch {epoch:2d} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | val_score={val_score:.4f} | {time.time()-t0:.1f}s"
        )

        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            ema.apply_shadow()
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "backbone_dir": BACKBONE,
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
    data_dir = Path(args.data_dir)
    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)

    texts, labels, _ = read_csv(data_dir / "train.csv")
    print(f"Loaded {len(texts):,} examples")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(sss.split(texts, labels))
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    print("Tokenizing...")
    train_dataset = ReviewDataset(train_texts, tokenizer, max_len=args.max_len, labels=train_labels)
    val_dataset = ReviewDataset(val_texts, tokenizer, max_len=args.max_len, labels=val_labels)

    tokenizer.save_pretrained(str(out_dir / "tokenizer"))

    seeds = [int(s) for s in args.seeds]
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        run_seed(seed, train_dataset, val_dataset, args, out_dir, device)

    print("\nAll seeds done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",        nargs="+", default=["42", "1337", "2024"])
    parser.add_argument("--max_len",      type=int,   default=256)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--encoder_lr",   type=float, default=8e-6)
    parser.add_argument("--head_lr",      type=float, default=5e-5)
    parser.add_argument("--layer_decay",  type=float, default=0.9)
    parser.add_argument("--dropout",      type=float, default=0.25)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs",       type=int,   default=6)
    parser.add_argument("--patience",     type=int,   default=3)
    parser.add_argument("--data_dir",     default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    main(parser.parse_args())
