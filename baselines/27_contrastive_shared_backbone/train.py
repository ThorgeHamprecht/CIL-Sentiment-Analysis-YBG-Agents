"""Train mDeBERTa-v3-base with W1 loss + supervised contrastive loss (shared backbone)."""
import argparse
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import ReviewDataset, read_csv
from model import EMA, SupervisedContrastiveLoss, SharedBackboneContrastiveMDeBERTa, emd_loss, median_decode

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

ROOT = Path(__file__).resolve().parents[2]
_SCRATCH_DATA_DIR = Path("/work/scratch") / os.environ.get("USER", "") / "cil" / "data"
_DEFAULT_DATA_DIR = _SCRATCH_DATA_DIR if _SCRATCH_DATA_DIR.exists() else ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
BACKBONE = "microsoft/mdeberta-v3-base"


def kaggle_score(preds: np.ndarray, labels: np.ndarray) -> float:
    return 1.0 - np.abs(preds - labels).mean() / 4.0


def get_contrastive_lambda(epoch: int, lambda_supcon: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return lambda_supcon
    if epoch <= warmup_epochs:
        return lambda_supcon * (epoch - 1) / max(1, warmup_epochs)
    return lambda_supcon


def _add_param_groups(groups, named_params, lr, weight_decay, no_decay):
    groups.append({
        "params": [p for n, p in named_params if not any(nd in n for nd in no_decay)],
        "lr": lr,
        "weight_decay": weight_decay,
    })
    groups.append({
        "params": [p for n, p in named_params if any(nd in n for nd in no_decay)],
        "lr": lr,
        "weight_decay": 0.0,
    })


def build_optimizer(model, rating_head_lr, contrastive_head_lr, encoder_top_lr, layer_decay, weight_decay):
    """LLRD: heads > top encoder layer > ... > embeddings."""
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    groups = []

    _add_param_groups(groups, list(model.rating_head.named_parameters()), rating_head_lr, weight_decay, no_decay)
    _add_param_groups(groups, list(model.contrastive_head.named_parameters()), contrastive_head_lr, weight_decay, no_decay)

    num_layers = model.encoder.config.num_hidden_layers
    for i, layer_idx in enumerate(range(num_layers - 1, -1, -1)):
        lr = encoder_top_lr * (layer_decay ** i)
        params = list(model.encoder.encoder.layer[layer_idx].named_parameters())
        _add_param_groups(groups, params, lr, weight_decay, no_decay)

    emb_lr = encoder_top_lr * (layer_decay ** num_layers)
    emb_params = list(model.encoder.embeddings.named_parameters())
    _add_param_groups(groups, emb_params, emb_lr, weight_decay, no_decay)

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
    supcon_loss_fn,
    device,
    w1_loss_weight,
    supcon_loss_weight,
    accum_steps,
    show_progress,
    epoch,
    total_epochs,
):
    model.train()
    total_loss = 0.0
    total_w1 = 0.0
    total_supcon = 0.0
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
            w1 = emd_loss(outputs["logits"], labels)
            supcon = supcon_loss_fn(outputs["embeddings"], labels)
            loss = w1_loss_weight * w1 + supcon_loss_weight * supcon
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
        total_w1 += w1.item() * len(labels)
        total_supcon += supcon.item() * len(labels)
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

    n = len(loader.dataset)
    return total_loss / n, total_w1 / n, total_supcon / n


@torch.no_grad()
def evaluate(model, loader, supcon_loss_fn, device, w1_loss_weight, supcon_loss_weight):
    model.eval()
    total_loss = 0.0
    total_w1 = 0.0
    total_supcon = 0.0
    all_preds = []
    all_labels = []
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast_ctx:
            outputs = model(input_ids, attention_mask)
            w1 = emd_loss(outputs["logits"], labels)
            supcon = supcon_loss_fn(outputs["embeddings"], labels)
            loss = w1_loss_weight * w1 + supcon_loss_weight * supcon

        total_loss += loss.item() * len(labels)
        total_w1 += w1.item() * len(labels)
        total_supcon += supcon.item() * len(labels)

        preds = median_decode(outputs["logits"]).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "mae": mean_absolute_error(labels, preds),
        "qwk": cohen_kappa_score(labels, preds, weights="quadratic"),
        "confusion": confusion_matrix(labels, preds, labels=[0, 1, 2, 3, 4]),
        "kaggle_score": kaggle_score(preds, labels),
    }

    n = len(loader.dataset)
    return total_loss / n, total_w1 / n, total_supcon / n, metrics


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for mDeBERTa training; refusing to run on CPU.")

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

    model = SharedBackboneContrastiveMDeBERTa(
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
        rating_head_lr=args.rating_head_lr,
        contrastive_head_lr=args.contrastive_head_lr,
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

    supcon_loss_fn = SupervisedContrastiveLoss(
        temperature=args.temperature,
        variant=args.supcon_variant,
    )
    target_supcon_weight = args.supcon_loss_weight
    if target_supcon_weight is None:
        target_supcon_weight = args.lambda_supcon

    best_score = -float("inf")
    patience_counter = 0
    checkpoint_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        supcon_loss_weight = get_contrastive_lambda(
            epoch,
            target_supcon_weight,
            args.contrastive_warmup_epochs,
        )

        train_loss, train_w1, train_supcon = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            ema,
            supcon_loss_fn,
            device,
            args.w1_loss_weight,
            supcon_loss_weight,
            accum_steps,
            show_progress=not args.no_progress,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        ema.apply_shadow()
        val_loss, val_w1, val_supcon, metrics = evaluate(
            model,
            val_loader,
            supcon_loss_fn,
            device,
            args.w1_loss_weight,
            supcon_loss_weight,
        )
        ema.restore()

        print(
            f"Epoch {epoch:2d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"| w1={val_w1:.4f} | supcon={val_supcon:.4f} "
            f"| w1_weight={args.w1_loss_weight:.3f} | supcon_weight={supcon_loss_weight:.3f} "
            f"| score={metrics['kaggle_score']:.4f} | {time.time()-t0:.1f}s"
        )
        print(
            f"  acc={metrics['accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f} "
            f"| mae={metrics['mae']:.4f} | qwk={metrics['qwk']:.4f}"
        )
        print(f"  confusion=\n{metrics['confusion']}")

        if metrics["kaggle_score"] > best_score:
            best_score = metrics["kaggle_score"]
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
            print(f"  -> New best: {best_score:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    print(f"\nBest val score: {best_score:.4f}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--max_len",          type=int,   default=256)
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--encoder_lr",       type=float, default=8e-6)
    parser.add_argument("--rating_head_lr",   type=float, default=5e-5)
    parser.add_argument("--contrastive_head_lr", type=float, default=1e-4)
    parser.add_argument("--layer_decay",      type=float, default=0.9)
    parser.add_argument("--dropout",          type=float, default=0.2)
    parser.add_argument("--weight_decay",     type=float, default=0.01)
    parser.add_argument("--epochs",           type=int,   default=6)
    parser.add_argument("--patience",         type=int,   default=3)
    parser.add_argument("--temperature",      type=float, default=0.07)
    parser.add_argument("--projection_dim",   type=int,   default=128)
    parser.add_argument("--lambda_supcon",    type=float, default=0.05)
    parser.add_argument("--w1_loss_weight",   type=float, default=1.0)
    parser.add_argument("--supcon_loss_weight", type=float, default=None)
    parser.add_argument("--contrastive_warmup_epochs", type=int, default=2)
    parser.add_argument("--supcon_variant",   type=str,   default="normal", choices=["normal", "distance_weighted"])
    parser.add_argument("--ema_decay",        type=float, default=0.999)
    parser.add_argument("--grad_accum_steps", type=int,   default=1)
    parser.add_argument("--tokenize_batch_size", type=int, default=1024)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--require_cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data_dir",         default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--artifact_dir",     default=str(_DEFAULT_ARTIFACT_DIR))
    main(parser.parse_args())
