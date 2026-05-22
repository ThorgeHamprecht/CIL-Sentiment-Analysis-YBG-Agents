"""Train mDeBERTa-v3-base with pure supervised contrastive learning."""
import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import ReviewDataset, read_csv
from eval_retrieval import (
    DEFAULT_K_VALUES,
    encode_embeddings,
    evaluate_retrieval_from_embeddings,
    flatten_scores,
    sample_per_class_indices,
)
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


@torch.no_grad()
def evaluate_retrieval_subset(model, train_loader, val_loader, device, args):
    z_train, y_train = encode_embeddings(model, train_loader, device)
    z_val, y_val = encode_embeddings(model, val_loader, device)
    return evaluate_retrieval_from_embeddings(
        z_train=z_train,
        y_train=y_train,
        z_val=z_val,
        y_val=y_val,
        k_values=args.retrieval_k_values,
        tau=args.retrieval_tau,
        chunk_size=args.similarity_chunk_size,
        device=device,
    )


def _metric_value(checkpoint_metric, val_loss, retrieval_scores):
    if checkpoint_metric == "supcon_val_loss":
        return -float(val_loss)
    return float(retrieval_scores[checkpoint_metric])


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.no_retrieval_eval and args.checkpoint_metric != "supcon_val_loss":
        raise ValueError("--checkpoint_metric must be supcon_val_loss when --no_retrieval_eval is used")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, use_fast=False)

    texts, labels, _ = read_csv(data_dir / "train.csv")
    print(f"Loaded {len(texts):,} examples")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=args.split_seed)
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
    retrieval_train_loader = None
    if not args.no_retrieval_eval:
        selected_train = sample_per_class_indices(
            train_labels,
            max_per_class=args.retrieval_train_per_class,
            seed=args.split_seed,
        )
        retrieval_train_dataset = Subset(train_dataset, selected_train)
        retrieval_train_loader = DataLoader(
            retrieval_train_dataset,
            batch_size=args.retrieval_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        print(
            f"Epoch retrieval eval: {len(retrieval_train_dataset):,} train examples "
            f"({args.retrieval_train_per_class}/class cap), {len(val_dataset):,} val examples"
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

    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    epoch_metrics_path = analysis_dir / "epoch_retrieval_metrics.jsonl"
    if epoch_metrics_path.exists():
        epoch_metrics_path.unlink()

    best_metric_value = -float("inf")
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
        retrieval_scores = {}
        retrieval_metrics = {}
        if retrieval_train_loader is not None:
            retrieval_metrics = evaluate_retrieval_subset(
                model,
                retrieval_train_loader,
                val_loader,
                device,
                args,
            )
            retrieval_scores = flatten_scores(retrieval_metrics)
        ema.restore()

        print(
            f"Epoch {epoch:2d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"| supcon_variant={args.supcon_variant} | temp={args.temperature} | {time.time()-t0:.1f}s"
        )
        if retrieval_scores:
            key_scores = [
                "knn_k1_weighted_median_score",
                "knn_k7_weighted_median_score",
                "knn_k101_weighted_median_score",
                "medoid_distribution_median_score",
            ]
            print("  retrieval " + " | ".join(
                f"{name.replace('_score', '')}={retrieval_scores[name]:.4f}"
                for name in key_scores
                if name in retrieval_scores
            ))
            with open(epoch_metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "supcon_val_loss": val_loss,
                    "retrieval_scores": retrieval_scores,
                    "retrieval_metrics": retrieval_metrics,
                }) + "\n")

        metric_value = _metric_value(args.checkpoint_metric, val_loss, retrieval_scores)
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            patience_counter = 0
            ema.apply_shadow()
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "backbone_dir": BACKBONE,
                    "best_checkpoint_metric": args.checkpoint_metric,
                    "best_checkpoint_metric_value": metric_value,
                },
                checkpoint_path,
            )
            ema.restore()
            display_value = -metric_value if args.checkpoint_metric == "supcon_val_loss" else metric_value
            print(f"  -> New best {args.checkpoint_metric}: {display_value:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    best_display = -best_metric_value if args.checkpoint_metric == "supcon_val_loss" else best_metric_value
    print(f"\nBest {args.checkpoint_metric}: {best_display:.4f}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--split_seed",       type=int,   default=42)
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
    parser.add_argument("--retrieval_train_per_class", type=int, default=1000)
    parser.add_argument("--retrieval_batch_size", type=int, default=64)
    parser.add_argument("--retrieval_tau", type=float, default=0.07)
    parser.add_argument("--retrieval_k_values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--similarity_chunk_size", type=int, default=512)
    parser.add_argument("--checkpoint_metric", type=str, default="knn_k7_weighted_median_score")
    parser.add_argument("--no_retrieval_eval", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--data_dir",         default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--artifact_dir",     default=str(_DEFAULT_ARTIFACT_DIR))
    main(parser.parse_args())
