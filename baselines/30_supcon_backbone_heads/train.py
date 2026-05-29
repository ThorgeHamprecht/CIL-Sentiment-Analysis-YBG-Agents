"""SupCon-pretrain a backbone, then compare three downstream heads."""
import argparse
import gc
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset, TensorDataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import ReviewDataset, read_csv
from model import (
    CORALHead,
    EMA,
    MDeBERTaRegressor,
    PureContrastiveMDeBERTa,
    SupervisedContrastiveLoss,
    coral_decode,
    coral_loss,
    regression_decode,
    regression_loss,
)

ROOT = Path(__file__).resolve().parents[2]
BACKBONE = "microsoft/mdeberta-v3-base"
_SCRATCH = Path("/work/scratch") / os.environ.get("USER", "") / "cil"
_DEFAULT_DATA_DIR = _SCRATCH / "data" if (_SCRATCH / "data").exists() else ROOT / "data"
_DEFAULT_ARTIFACT_DIR = _SCRATCH / "artifacts" / "30_supcon_backbone_heads" if _SCRATCH.exists() else Path(__file__).parent / "artifacts"
_DEFAULT_OUTPUT_DIR = _SCRATCH / "submissions" if _SCRATCH.exists() else ROOT / "submissions"


def set_seed(seed: int) -> None:
    """Seed numpy and torch for a single-run experiment."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device):
    """Use bf16 autocast on CUDA and no autocast on CPU."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def kaggle_score(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute the repo ordinal score, 1 - MAE / 4."""
    return 1.0 - np.abs(preds - labels).mean() / 4.0


def metrics_for_predictions(preds: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    """Return score, accuracy, macro-F1, MAE, QWK, and confusion matrix."""
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    return {
        "score": float(kaggle_score(preds, labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "mae": float(mean_absolute_error(labels, preds)),
        "qwk": float(cohen_kappa_score(labels, preds, weights="quadratic")),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(5))).tolist(),
    }


def sample_per_class_indices(labels: List[int], max_per_class: int, seed: int) -> List[int]:
    """Select a deterministic balanced support subset for retrieval validation."""
    labels_arr = np.asarray(labels)
    if max_per_class is None or max_per_class <= 0:
        return list(range(len(labels_arr)))
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for cls in sorted(np.unique(labels_arr)):
        cls_idx = np.flatnonzero(labels_arr == cls)
        if len(cls_idx) > max_per_class:
            cls_idx = rng.choice(cls_idx, size=max_per_class, replace=False)
        selected.extend(int(i) for i in cls_idx)
    selected.sort()
    return selected


def median_decode_probs_torch(probs: torch.Tensor) -> torch.Tensor:
    """Decode probabilities with the ordinal CDF median rule."""
    cdf = torch.cumsum(probs, dim=1)
    return (cdf < 0.5).sum(dim=1).clamp(0, probs.shape[1] - 1).long()


def _progress_marks(total_batches: int) -> set[int]:
    """Return batch indices where a 10% progress update should be printed."""
    if total_batches <= 0:
        return set()
    return {max(1, math.ceil(total_batches * step / 10)) for step in range(1, 11)}


def _print_progress(progress_name: str | None, batch_idx: int, total_batches: int, marks: set[int]) -> None:
    """Print a compact heartbeat for long validation and encoding loops."""
    if progress_name is None or batch_idx not in marks:
        return
    pct = min(100, int(round(100.0 * batch_idx / max(1, total_batches))))
    print(f"{progress_name}: {pct}% ({batch_idx}/{total_batches} batches)", flush=True)


def json_default(obj):
    """Convert numpy values to JSON-friendly values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(path: Path, payload: Dict[str, object]) -> None:
    """Save a JSON payload with numpy-aware conversion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)


def write_submission(path: Path, ids, preds: np.ndarray) -> None:
    """Write one Kaggle-ready id,label submission CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(preds, dtype=np.int64).clip(0, 4)
    pd.DataFrame({"id": list(ids), "label": labels}).to_csv(path, index=False)


def _add_param_groups(groups, named_params, lr, weight_decay, no_decay) -> None:
    """Append decay and no-decay parameter groups for AdamW."""
    groups.extend([
        {
            "params": [p for n, p in named_params if not any(nd in n for nd in no_decay)],
            "lr": lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in named_params if any(nd in n for nd in no_decay)],
            "lr": lr,
            "weight_decay": 0.0,
        },
    ])


def build_llrd_optimizer(model, head_module, head_lr, encoder_top_lr, layer_decay, weight_decay):
    """Create LLRD optimizer groups for a task head plus all encoder layers."""
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    groups = []
    _add_param_groups(groups, list(head_module.named_parameters()), head_lr, weight_decay, no_decay)
    num_layers = model.encoder.config.num_hidden_layers
    for i, layer_idx in enumerate(range(num_layers - 1, -1, -1)):
        lr = encoder_top_lr * (layer_decay ** i)
        params = list(model.encoder.encoder.layer[layer_idx].named_parameters())
        _add_param_groups(groups, params, lr, weight_decay, no_decay)
    emb_lr = encoder_top_lr * (layer_decay ** num_layers)
    _add_param_groups(groups, list(model.encoder.embeddings.named_parameters()), emb_lr, weight_decay, no_decay)
    return torch.optim.AdamW(groups)


def save_ema_checkpoint(model, ema: EMA, path: Path, args, extra: Dict[str, object]) -> None:
    """Save a checkpoint with EMA weights temporarily applied."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ema.apply_shadow()
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "backbone_dir": BACKBONE,
            **extra,
        },
        path,
    )
    ema.restore()


def extract_encoder_state(model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract encoder-only weights from a full SupCon model state dict."""
    return {key.replace("encoder.", "", 1): value for key, value in model_state.items() if key.startswith("encoder.")}


def load_supcon_encoder(model, checkpoint_path: Path, device: torch.device) -> None:
    """Load a SupCon-pretrained encoder into a downstream model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder_state = extract_encoder_state(ckpt["model"])
    model.encoder.load_state_dict(encoder_state)


def train_supcon_backbone(args, train_loader, support_loader, val_loader, device: torch.device, artifact_dir: Path) -> Path:
    """Train the SupCon encoder and return the medoid-selected best checkpoint."""
    model = PureContrastiveMDeBERTa(
        model_name=BACKBONE,
        projection_dim=args.projection_dim,
        dropout=args.contrastive_dropout,
    ).to(device)
    print(f"SupCon parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = build_llrd_optimizer(
        model,
        model.contrastive_head,
        args.contrastive_head_lr,
        args.encoder_lr,
        args.layer_decay,
        args.weight_decay,
    )
    total_steps = args.supcon_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_fraction * total_steps), total_steps)
    ema = EMA(model, decay=args.ema_decay)
    loss_fn = SupervisedContrastiveLoss(temperature=args.temperature, variant=args.supcon_variant)
    analysis_rows = []
    best_score = -float("inf")
    best_path = artifact_dir / "supcon" / "best_backbone.pt"
    best_epoch = None

    for epoch in range(1, args.supcon_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            with autocast_context(device):
                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs["embeddings"], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ema.update()
            total_loss += loss.item() * len(labels)
        train_loss = total_loss / len(train_loader.dataset)

        ema.apply_shadow()
        val_loss = evaluate_supcon_loss(
            model,
            val_loader,
            loss_fn,
            device,
            progress_name=f"SupCon epoch {epoch} val loss",
        )
        medoid_metrics = evaluate_supcon_medoid(model, support_loader, val_loader, args.retrieval_tau, device)
        ema.restore()
        ckpt_path = artifact_dir / "supcon" / f"epoch_{epoch:03d}_model.pt"
        save_ema_checkpoint(
            model,
            ema,
            ckpt_path,
            args,
            {"epoch": epoch, "kind": "supcon", "validation": {"supcon_loss": val_loss, **medoid_metrics}},
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "medoid_score": medoid_metrics["score"],
            "medoid_mae": medoid_metrics["mae"],
            "medoid_qwk": medoid_metrics["qwk"],
        }
        analysis_rows.append(row)
        print(
            f"SupCon epoch {epoch:2d}/{args.supcon_epochs} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | medoid_score={medoid_metrics['score']:.4f} "
            f"| medoid_mae={medoid_metrics['mae']:.4f} | {time.time() - t0:.1f}s"
        )
        if medoid_metrics["score"] > best_score:
            best_score = medoid_metrics["score"]
            best_epoch = epoch
            save_ema_checkpoint(
                model,
                ema,
                best_path,
                args,
                {
                    "epoch": epoch,
                    "kind": "supcon_best_backbone",
                    "selection_metric": "medoid_distribution_median_score",
                    "validation": {"supcon_loss": val_loss, **medoid_metrics},
                },
            )
            print(f"  -> New best SupCon backbone: epoch {epoch} score={best_score:.4f}")

    save_json(
        artifact_dir / "analysis" / "supcon_pretrain.json",
        {
            "variant": args.supcon_variant,
            "selection_metric": "medoid_distribution_median_score",
            "retrieval_tau": args.retrieval_tau,
            "retrieval_train_per_class": args.retrieval_train_per_class,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_checkpoint": str(best_path),
            "rows": analysis_rows,
        },
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_path


@torch.no_grad()
def evaluate_supcon_loss(model, loader, loss_fn, device: torch.device, progress_name: str | None = None) -> float:
    """Evaluate SupCon loss on a labeled loader."""
    model.eval()
    total_loss = 0.0
    total_batches = len(loader)
    marks = _progress_marks(total_batches)
    for batch_idx, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with autocast_context(device):
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs["embeddings"], labels)
        total_loss += loss.item() * len(labels)
        _print_progress(progress_name, batch_idx, total_batches, marks)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def encode_contrastive_embeddings(
    model,
    loader,
    device: torch.device,
    progress_name: str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode normalized SupCon embeddings and labels for retrieval validation."""
    model.eval()
    embeddings_out: List[torch.Tensor] = []
    labels_out: List[torch.Tensor] = []
    total_batches = len(loader)
    marks = _progress_marks(total_batches)
    for batch_idx, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast_context(device):
            outputs = model(input_ids, attention_mask)
        embeddings_out.append(outputs["embeddings"].float().cpu())
        if "labels" in batch:
            labels_out.append(batch["labels"].long().cpu())
        _print_progress(progress_name, batch_idx, total_batches, marks)
    embeddings = F.normalize(torch.cat(embeddings_out), p=2, dim=-1)
    labels = torch.cat(labels_out) if labels_out else torch.empty(0, dtype=torch.long)
    return embeddings, labels


def compute_class_medoids(z_train: torch.Tensor, y_train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Choose one actual support embedding nearest to each class centroid."""
    medoids: List[torch.Tensor] = []
    labels: List[int] = []
    z_train = F.normalize(z_train.float(), p=2, dim=-1)
    y_train = y_train.long()
    for cls in range(5):
        idx = torch.nonzero(y_train == cls, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        cls_embeddings = z_train[idx]
        centroid = F.normalize(cls_embeddings.mean(dim=0, keepdim=True), p=2, dim=-1)
        best_local = torch.argmax((cls_embeddings @ centroid.T).squeeze(1))
        medoids.append(cls_embeddings[best_local])
        labels.append(cls)
    if not medoids:
        raise ValueError("Cannot compute medoids without labeled support examples.")
    return torch.stack(medoids), torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def medoid_distribution_predictions(
    z_support: torch.Tensor,
    y_support: torch.Tensor,
    z_query: torch.Tensor,
    tau: float,
    device: torch.device,
) -> np.ndarray:
    """Predict with medoid softmax probabilities decoded by ordinal median."""
    medoids, medoid_labels = compute_class_medoids(z_support, y_support)
    medoids = medoids.to(device)
    medoid_labels = medoid_labels.to(device)
    z_query = F.normalize(z_query.float(), p=2, dim=-1)
    pred_parts: List[torch.Tensor] = []
    for start in range(0, z_query.size(0), 2048):
        chunk = z_query[start:start + 2048].to(device)
        sims = chunk @ medoids.T
        sparse_probs = torch.softmax(sims / tau, dim=1)
        probs = torch.zeros((chunk.size(0), 5), device=device)
        probs[:, medoid_labels] = sparse_probs
        pred_parts.append(median_decode_probs_torch(probs).cpu())
    return torch.cat(pred_parts).numpy().astype(np.int64)


@torch.no_grad()
def evaluate_supcon_medoid(model, support_loader, val_loader, tau: float, device: torch.device) -> Dict[str, object]:
    """Evaluate a SupCon checkpoint with medoid-distribution median decoding."""
    z_support, y_support = encode_contrastive_embeddings(
        model,
        support_loader,
        device,
        progress_name="SupCon medoid eval support embeddings",
    )
    z_val, y_val = encode_contrastive_embeddings(
        model,
        val_loader,
        device,
        progress_name="SupCon medoid eval val embeddings",
    )
    preds = medoid_distribution_predictions(z_support, y_support, z_val, tau, device)
    return metrics_for_predictions(preds, y_val.numpy())


@torch.no_grad()
def encode_pooled_features(
    model,
    loader,
    device: torch.device,
    progress_name: str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode pooled backbone features and optional labels onto CPU."""
    model.eval()
    features: List[torch.Tensor] = []
    labels_out: List[torch.Tensor] = []
    total_batches = len(loader)
    marks = _progress_marks(total_batches)
    for batch_idx, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast_context(device):
            outputs = model(input_ids, attention_mask)
        features.append(outputs["pooled"].float().cpu())
        if "labels" in batch:
            labels_out.append(batch["labels"].long().cpu())
        _print_progress(progress_name, batch_idx, total_batches, marks)
    labels = torch.cat(labels_out) if labels_out else torch.empty(0, dtype=torch.long)
    return torch.cat(features), labels


def train_frozen_coral_head(args, supcon_ckpt: Path, loaders, labels_and_ids, device: torch.device, artifact_dir: Path):
    """Train a CORAL ordinal head on cached frozen SupCon backbone features."""
    feature_cache = artifact_dir / "features" / "frozen_pooled_features.pt"
    if feature_cache.exists() and not args.recompute_features:
        payload = torch.load(feature_cache, map_location="cpu", weights_only=False)
        train_features = payload["train_features"]
        train_labels = payload["train_labels"]
        val_features = payload["val_features"]
        val_labels = payload["val_labels"]
        test_features = payload["test_features"]
    else:
        encoder = PureContrastiveMDeBERTa(
            model_name=BACKBONE,
            projection_dim=args.projection_dim,
            dropout=args.contrastive_dropout,
        ).to(device)
        encoder.load_state_dict(torch.load(supcon_ckpt, map_location=device, weights_only=False)["model"])
        train_features, train_labels = encode_pooled_features(
            encoder,
            loaders["train_eval"],
            device,
            progress_name="Frozen-head feature cache train embeddings",
        )
        val_features, val_labels = encode_pooled_features(
            encoder,
            loaders["val"],
            device,
            progress_name="Frozen-head feature cache val embeddings",
        )
        test_features, _ = encode_pooled_features(
            encoder,
            loaders["test"],
            device,
            progress_name="Frozen-head feature cache test embeddings",
        )
        feature_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "train_features": train_features,
                "train_labels": train_labels,
                "val_features": val_features,
                "val_labels": val_labels,
                "test_features": test_features,
            },
            feature_cache,
        )
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    head = CORALHead(hidden_size=train_features.size(1)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.frozen_head_lr, weight_decay=args.weight_decay)
    train_ds = TensorDataset(train_features, train_labels)
    val_ds = TensorDataset(val_features, val_labels)
    train_loader = DataLoader(train_ds, batch_size=args.feature_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.feature_batch_size, shuffle=False)

    best_score = -float("inf")
    best_state = None
    rows = []
    for epoch in range(1, args.head_epochs + 1):
        head.train()
        total_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = head(features)
            loss = coral_loss(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        val_loss, preds, labels_np = evaluate_coral_head(head, val_loader, device)
        metrics = metrics_for_predictions(preds, labels_np)
        rows.append({"epoch": epoch, "train_loss": total_loss / len(train_ds), "val_loss": val_loss, **metrics})
        if metrics["score"] > best_score:
            best_score = metrics["score"]
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
        print(f"Frozen CORAL epoch {epoch:2d}/{args.head_epochs} | score={metrics['score']:.4f} | mae={metrics['mae']:.4f}")

    head.load_state_dict(best_state)
    test_preds = predict_coral_features(head, test_features, device)
    val_preds = predict_coral_features(head, val_features, device)
    torch.save({"head": best_state, "args": vars(args), "best_score": best_score}, artifact_dir / "frozen_coral_best.pt")
    return "frozen_coral", rows, val_preds, test_preds


@torch.no_grad()
def evaluate_coral_head(head, loader, device: torch.device):
    """Evaluate a CORAL head on pooled-feature tensors."""
    head.eval()
    total_loss = 0.0
    preds_all = []
    labels_all = []
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = head(features)
        loss = coral_loss(logits, labels)
        total_loss += loss.item() * len(labels)
        preds_all.append(coral_decode(logits).cpu().numpy())
        labels_all.append(labels.cpu().numpy())
    return total_loss / len(loader.dataset), np.concatenate(preds_all), np.concatenate(labels_all)


@torch.no_grad()
def predict_coral_features(head, features: torch.Tensor, device: torch.device) -> np.ndarray:
    """Predict labels from cached pooled features using a CORAL head."""
    head.eval()
    preds = []
    loader = DataLoader(TensorDataset(features), batch_size=2048, shuffle=False)
    for (batch_features,) in loader:
        logits = head(batch_features.to(device))
        preds.append(coral_decode(logits).cpu().numpy())
    return np.concatenate(preds)


def train_regression_head(head_name: str, head_type: str, args, supcon_ckpt: Path, loaders, device: torch.device, artifact_dir: Path):
    """Fine-tune a scalar regression head and SupCon-initialized backbone."""
    model = MDeBERTaRegressor(
        model_name=BACKBONE,
        head_type=head_type,
        hidden_dim=args.mlp_hidden_dim,
        dropout=args.regression_dropout,
    ).to(device)
    load_supcon_encoder(model, supcon_ckpt, device)
    optimizer = build_llrd_optimizer(
        model,
        model.regression_head,
        args.regression_head_lr,
        args.encoder_lr,
        args.layer_decay,
        args.weight_decay,
    )
    total_steps = args.head_epochs * len(loaders["train"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_fraction * total_steps), total_steps)
    ema = EMA(model, decay=args.ema_decay)
    best_score = -float("inf")
    best_path = artifact_dir / f"{head_name}_best.pt"
    rows = []

    for epoch in range(1, args.head_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for batch in loaders["train"]:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            with autocast_context(device):
                preds = model(input_ids, attention_mask)
                loss = regression_loss(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ema.update()
            total_loss += loss.item() * len(labels)

        ema.apply_shadow()
        val_loss, val_preds, val_labels = evaluate_regression_model(model, loaders["val"], device)
        metrics = metrics_for_predictions(val_preds, val_labels)
        ema.restore()
        rows.append({"epoch": epoch, "train_loss": total_loss / len(loaders["train"].dataset), "val_loss": val_loss, **metrics})
        print(
            f"{head_name} epoch {epoch:2d}/{args.head_epochs} | score={metrics['score']:.4f} "
            f"| mae={metrics['mae']:.4f} | {time.time() - t0:.1f}s"
        )
        if metrics["score"] > best_score:
            best_score = metrics["score"]
            save_ema_checkpoint(model, ema, best_path, args, {"kind": head_name, "epoch": epoch, "validation": metrics})

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    _, val_preds, _ = evaluate_regression_model(model, loaders["val"], device)
    test_preds = predict_regression_model(model, loaders["test"], device)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return head_name, rows, val_preds, test_preds


@torch.no_grad()
def evaluate_regression_model(model, loader, device: torch.device):
    """Evaluate a scalar regression model and rounded/clamped predictions."""
    model.eval()
    total_loss = 0.0
    preds_all = []
    labels_all = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with autocast_context(device):
            scalar = model(input_ids, attention_mask)
            loss = regression_loss(scalar, labels)
        total_loss += loss.item() * len(labels)
        preds_all.append(regression_decode(scalar).cpu().numpy())
        labels_all.append(labels.cpu().numpy())
    return total_loss / len(loader.dataset), np.concatenate(preds_all), np.concatenate(labels_all)


@torch.no_grad()
def predict_regression_model(model, loader, device: torch.device) -> np.ndarray:
    """Predict labels for an unlabeled loader with a scalar regression model."""
    model.eval()
    preds_all = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast_context(device):
            scalar = model(input_ids, attention_mask)
        preds_all.append(regression_decode(scalar).cpu().numpy())
    return np.concatenate(preds_all)


def build_datasets(args, tokenizer):
    """Create stratified train/val datasets and the unlabeled test dataset."""
    texts, labels, ids = read_csv(Path(args.data_dir) / "train.csv")
    test_texts, _test_labels, test_ids = read_csv(Path(args.data_dir) / "test.csv")
    print(f"Loaded {len(texts):,} train examples and {len(test_texts):,} test examples")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=args.split_seed)
    train_idx, val_idx = next(sss.split(texts, labels))
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    val_ids = [ids[i] for i in val_idx]

    train_dataset = ReviewDataset(
        train_texts,
        tokenizer,
        max_len=args.max_len,
        labels=train_labels,
        show_progress=not args.no_progress,
        batch_size=args.tokenize_batch_size,
    )
    val_dataset = ReviewDataset(val_texts, tokenizer, max_len=args.max_len, labels=val_labels)
    test_dataset = ReviewDataset(
        test_texts,
        tokenizer,
        max_len=args.max_len,
        show_progress=not args.no_progress,
        batch_size=args.tokenize_batch_size,
    )
    return train_dataset, train_labels, val_dataset, val_labels, val_ids, test_dataset, test_ids


def main(args):
    """Run SupCon pretraining, train three heads, and write analysis/submissions."""
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (artifact_dir / "predictions").mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this mDeBERTa experiment.")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, use_fast=False)
    tokenizer.save_pretrained(str(artifact_dir / "tokenizer"))
    train_dataset, train_labels, val_dataset, val_labels, val_ids, test_dataset, test_ids = build_datasets(args, tokenizer)
    support_idx = sample_per_class_indices(train_labels, args.retrieval_train_per_class, args.split_seed)
    support_dataset = Subset(train_dataset, support_idx)
    print(
        f"SupCon medoid validation support: {len(support_dataset):,} train examples "
        f"({args.retrieval_train_per_class}/class cap; <=0 means full split)"
    )
    loaders = {
        "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True),
        "train_eval": DataLoader(train_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=2, pin_memory=True),
        "support": DataLoader(support_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=2, pin_memory=True),
        "val": DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=2, pin_memory=True),
        "test": DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=2, pin_memory=True),
    }

    supcon_ckpt = Path(args.supcon_checkpoint) if args.supcon_checkpoint else None
    if supcon_ckpt is None:
        supcon_ckpt = train_supcon_backbone(args, loaders["train"], loaders["support"], loaders["val"], device, artifact_dir)
    print(f"Using SupCon checkpoint: {supcon_ckpt}")

    results = []
    val_prediction_payload = {"id": list(val_ids), "label_true": list(val_labels)}
    test_prediction_payload = {"id": list(test_ids)}

    for name, rows, val_preds, test_preds in [
        train_frozen_coral_head(args, supcon_ckpt, loaders, {"val_labels": val_labels}, device, artifact_dir),
        train_regression_head("regression_linear_llrd", "linear", args, supcon_ckpt, loaders, device, artifact_dir),
        train_regression_head("regression_mlp_llrd", "mlp", args, supcon_ckpt, loaders, device, artifact_dir),
    ]:
        metrics = metrics_for_predictions(val_preds, np.asarray(val_labels))
        results.append({"head": name, "best_metrics": metrics, "epochs": rows})
        val_prediction_payload[name] = np.asarray(val_preds, dtype=np.int64)
        test_prediction_payload[name] = np.asarray(test_preds, dtype=np.int64)
        write_submission(output_dir / f"30_supcon_backbone_{name}_submission.csv", test_ids, test_preds)
        print(f"{name}: score={metrics['score']:.4f} mae={metrics['mae']:.4f} qwk={metrics['qwk']:.4f}")

    pd.DataFrame(val_prediction_payload).to_csv(artifact_dir / "predictions" / "val_predictions.csv", index=False)
    pd.DataFrame(test_prediction_payload).to_csv(artifact_dir / "predictions" / "test_predictions.csv", index=False)
    save_json(
        artifact_dir / "analysis" / "head_analysis.json",
        {
            "supcon_checkpoint": str(supcon_ckpt),
            "supcon_epochs": args.supcon_epochs,
            "supcon_variant": args.supcon_variant,
            "results": results,
        },
    )
    print(f"Done. Analysis saved to {artifact_dir / 'analysis' / 'head_analysis.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--supcon_epochs", type=int, default=4)
    parser.add_argument("--head_epochs", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--feature_batch_size", type=int, default=1024)
    parser.add_argument("--tokenize_batch_size", type=int, default=1024)
    parser.add_argument("--encoder_lr", type=float, default=8e-6)
    parser.add_argument("--contrastive_head_lr", type=float, default=1e-4)
    parser.add_argument("--regression_head_lr", type=float, default=5e-5)
    parser.add_argument("--frozen_head_lr", type=float, default=1e-3)
    parser.add_argument("--layer_decay", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--contrastive_dropout", type=float, default=0.1)
    parser.add_argument("--regression_dropout", type=float, default=0.1)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--supcon_variant", choices=["normal"], default="normal")
    parser.add_argument("--retrieval_train_per_class", type=int, default=1000)
    parser.add_argument("--retrieval_tau", type=float, default=0.07)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--warmup_fraction", type=float, default=0.06)
    parser.add_argument("--supcon_checkpoint", default=None)
    parser.add_argument("--recompute_features", action="store_true")
    parser.add_argument("--require_cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--data_dir", default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--output_dir", default=str(_DEFAULT_OUTPUT_DIR))
    main(parser.parse_args())
