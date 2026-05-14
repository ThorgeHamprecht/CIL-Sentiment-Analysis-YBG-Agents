"""Phase 2: Fine-tune with SORD, LLRD, FGM (epoch 2+), EMA, 3 seeds."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import ReviewDataset, read_csv
from model import EMA, FGM, ev_decode, mDeBERTaAdvanced, sord_loss

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"


def kaggle_score(preds: np.ndarray, labels: np.ndarray) -> float:
    return 1.0 - np.abs(preds - labels).mean() / 4.0


def build_optimizer(model, head_lr, encoder_top_lr, layer_decay, weight_decay):
    """Layer-wise LR decay: head > top encoder layer > ... > embeddings."""
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

    head_params = (
        list(model.dense.named_parameters())
        + list(model.norm.named_parameters())
        + list(model.classifier.named_parameters())
    )
    groups = [
        {"params": [p for n, p in head_params if not any(nd in n for nd in no_decay)],
         "lr": head_lr, "weight_decay": weight_decay},
        {"params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
         "lr": head_lr, "weight_decay": 0.0},
    ]

    num_layers = model.encoder.config.num_hidden_layers  # 12
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


def train_epoch(model, loader, optimizer, scheduler, ema, fgm, device, use_fgm):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask)
            loss = sord_loss(logits, labels)
        loss.backward()

        if use_fgm:
            fgm.attack()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_adv = sord_loss(model(input_ids, attention_mask), labels)
            loss_adv.backward()
            fgm.restore()

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
            loss = sord_loss(logits, labels)
        total_loss += loss.item() * len(labels)
        all_preds.append(ev_decode(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    return total_loss / len(loader.dataset), kaggle_score(preds, labels_arr)


def run_seed(seed, backbone_dir, train_dataset, val_dataset, args, out_dir, device):
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

    model = mDeBERTaAdvanced(model_name=str(backbone_dir), dropout=args.dropout).to(device)
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
    fgm = FGM(model, eps=1.0)

    best_score = 0.0
    checkpoint_path = out_dir / f"best_model_seed{seed}.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        use_fgm = (epoch >= 2)
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, ema, fgm, device, use_fgm)

        ema.apply_shadow()
        val_loss, val_score = evaluate(model, val_loader, device)
        ema.restore()

        print(
            f"  [seed={seed}] Epoch {epoch:2d} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | val_score={val_score:.4f} | {time.time()-t0:.1f}s"
        )

        if val_score > best_score:
            best_score = val_score
            ema.apply_shadow()
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "backbone_dir": str(backbone_dir),
                "seed": seed,
            }, checkpoint_path)
            ema.restore()
            print(f"    -> New best: {best_score:.4f} (saved)")

    print(f"  [seed={seed}] Best val score: {best_score:.4f}")
    return best_score


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    backbone_dir = Path(args.backbone_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(backbone_dir))

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
        run_seed(seed, backbone_dir, train_dataset, val_dataset, args, out_dir, device)

    print("\nAll seeds done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",        nargs="+", default=["42", "1337", "2024"])
    parser.add_argument("--backbone_dir", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--max_len",      type=int,   default=256)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--encoder_lr",   type=float, default=1e-5)
    parser.add_argument("--head_lr",      type=float, default=2e-5)
    parser.add_argument("--layer_decay",  type=float, default=0.9)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs",       type=int,   default=4)
    parser.add_argument("--data_dir",     default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    main(parser.parse_args())
