"""Train mDeBERTa-v3-base with pure supervised contrastive learning."""
import argparse
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import ReviewDataset, read_csv
from model import EMA, PureContrastiveMDeBERTa, SupervisedContrastiveLoss

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

ROOT = Path(__file__).resolve().parents[2]
_SCRATCH_DATA_DIR = Path("/work/scratch") / os.environ.get("USER", "") / "cil" / "data"
_DEFAULT_DATA_DIR = _SCRATCH_DATA_DIR if _SCRATCH_DATA_DIR.exists() else ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
BACKBONE = "microsoft/mdeberta-v3-base"


def build_optimizer(model, head_lr, encoder_top_lr, layer_decay, weight_decay):
    """LLRD: contrastive head > top encoder layer > ... > embeddings."""
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

    head_params = list(model.contrastive_head.named_parameters())
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


def _progress_bar(total, enabled, desc):
    if not enabled or tqdm is None:
        return None
    return tqdm(total=total, desc=desc)


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    ema,
    loss_fn,
    device,
    accum_steps,
    show_progress,
    epoch,
    total_epochs,
):
    model.train()
    total_loss = 0.0
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    optimizer.zero_grad()
    desc = f"Train {epoch}/{total_epochs}"
    total = len(loader)
    progress = _progress_bar(total, show_progress, desc)
    last_grad_norm = None
    window_loss = 0.0
    window_count = 0
    update_every = max(1, math.ceil(total * 0.02))
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast_ctx:
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs["embeddings"], labels)
            scaled_loss = loss / accum_steps

        scaled_loss.backward()
        if step % accum_steps == 0 or step == len(loader):
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            last_grad_norm = float(grad_norm)
            optimizer.step()
            scheduler.step()
            ema.update()
            optimizer.zero_grad()
        total_loss += loss.item() * len(labels)
        window_loss += loss.item() * len(labels)
        window_count += len(labels)
        if show_progress and progress is not None:
            if (step % update_every == 0) or (step == total):
                if hasattr(progress, "set_postfix"):
                    grad_display = f"{last_grad_norm:.2f}" if last_grad_norm is not None else "n/a"
                    avg_window = window_loss / max(1, window_count)
                    progress.set_postfix(loss=f"{avg_window:.4f}", grad=grad_display)
                increment = update_every if step % update_every == 0 else (total % update_every or update_every)
                progress.update(increment)
                window_loss = 0.0
                window_count = 0
    if progress is not None:
        progress.close()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast_ctx:
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs["embeddings"], labels)
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, use_fast=False)

    texts, labels, _ = read_csv(data_dir / "train.csv")
    print(f"Loaded {len(texts):,} examples")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=args.seed)
    train_idx, val_idx = next(sss.split(texts, labels))
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    print("Tokenizing...")
    train_dataset = ReviewDataset(
        train_texts,
        tokenizer,
        max_len=args.max_len,
        labels=train_labels,
        show_progress=not args.no_progress,
        batch_size=args.tokenize_batch_size,
    )
    val_dataset = ReviewDataset(val_texts, tokenizer, max_len=args.max_len, labels=val_labels)

    tokenizer.save_pretrained(str(out_dir / "tokenizer"))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    model = PureContrastiveMDeBERTa(
        model_name=BACKBONE,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
    ).to(device)
    if args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
        model.encoder.config.use_cache = False
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = build_optimizer(
        model,
        head_lr=args.contrastive_head_lr,
        encoder_top_lr=args.encoder_lr,
        layer_decay=args.layer_decay,
        weight_decay=args.weight_decay,
    )

    accum_steps = max(1, args.grad_accum_steps)
    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(0.06 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    ema = EMA(model, decay=args.ema_decay)

    loss_fn = SupervisedContrastiveLoss(
        temperature=args.temperature,
        variant=args.supcon_variant,
    )

    best_loss = float("inf")
    patience_counter = 0
    checkpoint_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            ema,
            loss_fn,
            device,
            accum_steps,
            show_progress=not args.no_progress,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        ema.apply_shadow()
        val_loss = evaluate(model, val_loader, loss_fn, device)
        ema.restore()

        print(
            f"Epoch {epoch:2d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"| supcon_variant={args.supcon_variant} | temp={args.temperature} | {time.time()-t0:.1f}s"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            ema.apply_shadow()
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "backbone_dir": BACKBONE,
                },
                checkpoint_path,
            )
            ema.restore()
            print(f"  -> New best: {best_loss:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    print(f"\nBest val loss: {best_loss:.4f}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--max_len",          type=int,   default=256)
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--encoder_lr",       type=float, default=8e-6)
    parser.add_argument("--contrastive_head_lr", type=float, default=1e-4)
    parser.add_argument("--layer_decay",      type=float, default=0.9)
    parser.add_argument("--dropout",          type=float, default=0.1)
    parser.add_argument("--weight_decay",     type=float, default=0.01)
    parser.add_argument("--epochs",           type=int,   default=6)
    parser.add_argument("--patience",         type=int,   default=3)
    parser.add_argument("--temperature",      type=float, default=0.07)
    parser.add_argument("--projection_dim",   type=int,   default=128)
    parser.add_argument("--supcon_variant",   type=str,   default="normal", choices=["normal", "distance_weighted"])
    parser.add_argument("--ema_decay",        type=float, default=0.999)
    parser.add_argument("--grad_accum_steps", type=int,   default=1)
    parser.add_argument("--tokenize_batch_size", type=int, default=1024)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--data_dir",         default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--artifact_dir",     default=str(_DEFAULT_ARTIFACT_DIR))
    main(parser.parse_args())
