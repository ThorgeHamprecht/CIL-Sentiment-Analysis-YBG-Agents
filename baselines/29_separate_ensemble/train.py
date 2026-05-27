"""Train separate EMD^2 and pure SupCon models, then validate their ensemble."""
import argparse
import gc
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import ReviewDataset, read_csv
from ensemble import (
    DEFAULT_TAUS,
    append_jsonl,
    class_prior,
    encode_classifier_probs,
    encode_contrastive_embeddings,
    evaluate_ensemble_grid,
    metrics_for_predictions,
    sample_per_class_indices,
    save_json,
    write_prediction_wide_csv,
)
from model import EMA, MDeBERTaEMD, PureContrastiveMDeBERTa, SupervisedContrastiveLoss, emd_loss, median_decode_logits

ROOT = Path(__file__).resolve().parents[2]
BACKBONE = "microsoft/mdeberta-v3-base"
_SCRATCH = Path("/work/scratch") / os.environ.get("USER", "") / "cil"
_DEFAULT_DATA_DIR = _SCRATCH / "data" if (_SCRATCH / "data").exists() else ROOT / "data"
_DEFAULT_ARTIFACT_DIR = _SCRATCH / "artifacts" / "29_separate_ensemble" if _SCRATCH.exists() else Path(__file__).parent / "artifacts"


def set_seed(seed: int) -> None:
    """Seed numpy and torch RNGs for a single-run experiment."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _add_param_groups(groups, named_params, lr, weight_decay, no_decay) -> None:
    """Append decay and no-decay AdamW parameter groups."""
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


def build_llrd_optimizer(model, head_specs, encoder_top_lr, layer_decay, weight_decay):
    """Build AdamW groups with task heads above layer-wise-decayed encoder layers."""
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    groups = []
    for module, lr in head_specs:
        _add_param_groups(groups, list(module.named_parameters()), lr, weight_decay, no_decay)

    num_layers = model.encoder.config.num_hidden_layers
    for i, layer_idx in enumerate(range(num_layers - 1, -1, -1)):
        lr = encoder_top_lr * (layer_decay ** i)
        params = list(model.encoder.encoder.layer[layer_idx].named_parameters())
        _add_param_groups(groups, params, lr, weight_decay, no_decay)

    emb_lr = encoder_top_lr * (layer_decay ** num_layers)
    emb_params = list(model.encoder.embeddings.named_parameters())
    _add_param_groups(groups, emb_params, emb_lr, weight_decay, no_decay)
    return torch.optim.AdamW(groups)


def move_optimizer_to_device(optimizer, device: torch.device) -> None:
    """Move optimizer state tensors when a model is offloaded or reactivated."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def move_ema_to_device(ema: EMA, device: torch.device) -> None:
    """Move EMA shadow and backup tensors to the target device."""
    ema.shadow = {name: value.to(device) for name, value in ema.shadow.items()}
    ema.backup = {name: value.to(device) for name, value in ema.backup.items()}


def activate_training_state(model, optimizer, ema: EMA, device: torch.device) -> None:
    """Move a trainable model, optimizer state, and EMA state onto the GPU."""
    model.to(device)
    move_optimizer_to_device(optimizer, device)
    move_ema_to_device(ema, device)


def offload_training_state(model, optimizer, ema: EMA) -> None:
    """Move a trainable model state back to CPU and free CUDA cache."""
    cpu = torch.device("cpu")
    model.to(cpu)
    move_optimizer_to_device(optimizer, cpu)
    move_ema_to_device(ema, cpu)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_ema_checkpoint(model, ema: EMA, path: Path, args, epoch: int, kind: str, extra: Dict[str, object]) -> None:
    """Save an EMA-weight checkpoint for an epoch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ema.apply_shadow()
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "backbone_dir": BACKBONE,
            "epoch": epoch,
            "kind": kind,
            **extra,
        },
        path,
    )
    ema.restore()


def autocast_context(device: torch.device):
    """Return bf16 autocast on CUDA and a no-op context on CPU."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train_classifier_epoch(model, loader, optimizer, scheduler, ema: EMA, device: torch.device) -> float:
    """Train the EMD^2 classifier for one epoch."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with autocast_context(device):
            logits = model(input_ids, attention_mask)
            loss = emd_loss(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        ema.update()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def train_contrastive_epoch(model, loader, optimizer, scheduler, ema: EMA, loss_fn, device: torch.device) -> float:
    """Train one pure SupCon encoder for one epoch."""
    model.train()
    total_loss = 0.0
    for batch in loader:
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
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_classifier(model, loader, device: torch.device, return_probs: bool = False) -> Dict[str, object]:
    """Compute classifier validation loss, metrics, and optionally softmax probabilities."""
    model.eval()
    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with autocast_context(device):
            logits = model(input_ids, attention_mask)
            loss = emd_loss(logits, labels)
        total_loss += loss.item() * len(labels)
        all_preds.append(median_decode_logits(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        if return_probs:
            all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    preds = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)
    metrics = metrics_for_predictions(preds, labels_np)
    metrics["loss"] = total_loss / len(loader.dataset)
    if return_probs:
        metrics["probabilities"] = np.concatenate(all_probs, axis=0)
    return metrics


def load_classifier_checkpoint(path: Path, args, device: torch.device) -> MDeBERTaEMD:
    """Load an EMA classifier checkpoint onto the target device."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model = MDeBERTaEMD(
        model_name=str(ckpt.get("backbone_dir", BACKBONE)),
        dropout=float(ckpt_args.get("classifier_dropout", ckpt_args.get("dropout", args.classifier_dropout))),
        dropout_samples=int(ckpt_args.get("msd_samples", args.msd_samples)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def load_contrastive_checkpoint(path: Path, args, device: torch.device) -> PureContrastiveMDeBERTa:
    """Load an EMA contrastive checkpoint onto the target device."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model = PureContrastiveMDeBERTa(
        model_name=str(ckpt.get("backbone_dir", BACKBONE)),
        projection_dim=int(ckpt_args.get("projection_dim", args.projection_dim)),
        dropout=float(ckpt_args.get("contrastive_dropout", ckpt_args.get("dropout", args.contrastive_dropout))),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def evaluate_epoch_from_checkpoints(
    variant: str,
    epoch: int,
    classifier_ckpt: Path,
    contrastive_ckpt: Path,
    loaders: Dict[str, DataLoader],
    labels_and_ids: Dict[str, object],
    args,
    artifact_dir: Path,
    device: torch.device,
) -> List[Dict[str, object]]:
    """Run validation ensemble sweep for one classifier/SupCon epoch pair."""
    embeddings_dir = artifact_dir / "embeddings"
    predictions_dir = artifact_dir / "predictions"
    analysis_jsonl = artifact_dir / "analysis" / "epoch_ensemble_metrics.jsonl"

    shared_val_probs_path = embeddings_dir / f"classifier_epoch_{epoch:03d}_val_probs.pt"
    if args.cache_embeddings and shared_val_probs_path.exists():
        val_probs = torch.load(shared_val_probs_path, map_location="cpu", weights_only=False)["classifier_probs"]
    else:
        classifier = load_classifier_checkpoint(classifier_ckpt, args, device)
        val_probs, _ = encode_classifier_probs(
            classifier,
            loaders["val"],
            device,
            progress_name=f"Eval classifier epoch {epoch} val probabilities",
        )
        del classifier
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.cache_embeddings:
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"classifier_probs": val_probs}, shared_val_probs_path)

    contrastive = load_contrastive_checkpoint(contrastive_ckpt, args, device)
    z_support, y_support = encode_contrastive_embeddings(
        contrastive,
        loaders["support"],
        device,
        progress_name=f"Eval {variant} epoch {epoch} support embeddings",
    )
    z_val, y_val = encode_contrastive_embeddings(
        contrastive,
        loaders["val"],
        device,
        progress_name=f"Eval {variant} epoch {epoch} val embeddings",
    )
    del contrastive
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.cache_embeddings:
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"embeddings": z_support, "labels": y_support},
            embeddings_dir / f"{variant}_epoch_{epoch:03d}_support_embeddings.pt",
        )
        torch.save(
            {"embeddings": z_val, "labels": y_val},
            embeddings_dir / f"{variant}_epoch_{epoch:03d}_val_embeddings.pt",
        )

    rows, predictions = evaluate_ensemble_grid(
        class_probs=val_probs,
        z_support=z_support,
        y_support=y_support,
        z_query=z_val,
        labels=y_val.numpy(),
        taus=args.retrieval_taus,
        prior=labels_and_ids["train_prior"],
        device=device,
    )

    for row in rows:
        payload = {
            "variant": variant,
            "epoch": epoch,
            **row,
        }
        append_jsonl(analysis_jsonl, payload)

    write_prediction_wide_csv(
        predictions_dir / f"{variant}_epoch_{epoch:03d}_val_predictions.csv",
        labels_and_ids["val_ids"],
        predictions,
        labels=labels_and_ids["val_labels"],
    )

    result_rows = [{"variant": variant, "epoch": epoch, **row} for row in rows]
    top = sorted(
        [r for r in result_rows if r["strategy"] in set(args.ensemble_strategies)],
        key=lambda r: r["score"],
        reverse=True,
    )[:5]
    print(f"  Ensemble top validation methods for {variant} epoch {epoch}:")
    for row in top:
        print(
            f"    {row['strategy']:22s} tau={row['tau']} "
            f"score={row['score']:.4f} mae={row['mae']:.4f} qwk={row['qwk']:.4f}"
        )
    return result_rows


def build_datasets(args, tokenizer):
    """Read data, create the fixed split, and tokenize train/val datasets."""
    texts, labels, ids = read_csv(Path(args.data_dir) / "train.csv")
    print(f"Loaded {len(texts):,} labeled examples")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=args.split_seed)
    train_idx, val_idx = next(sss.split(texts, labels))
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    train_ids = [ids[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    val_ids = [ids[i] for i in val_idx]

    print("Tokenizing train/val split...")
    train_dataset = ReviewDataset(
        train_texts,
        tokenizer,
        max_len=args.max_len,
        labels=train_labels,
        show_progress=not args.no_progress,
        batch_size=args.tokenize_batch_size,
    )
    val_dataset = ReviewDataset(
        val_texts,
        tokenizer,
        max_len=args.max_len,
        labels=val_labels,
        show_progress=not args.no_progress,
        batch_size=args.tokenize_batch_size,
    )
    return train_dataset, train_labels, train_ids, val_dataset, val_labels, val_ids


def write_summary_tables(artifact_dir: Path) -> None:
    """Convert the epoch JSONL metrics into sorted CSV/JSON summaries."""
    jsonl_path = artifact_dir / "analysis" / "epoch_ensemble_metrics.jsonl"
    if not jsonl_path.exists():
        return
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    df = pd.DataFrame(rows)
    csv_path = artifact_dir / "analysis" / "validation_summary.csv"
    df.to_csv(csv_path, index=False)

    ensemble_df = df[df["strategy"].isin(ENSEMBLE_STRATEGIES_FOR_SUMMARY)].copy()
    best_rows = []
    for (variant, tau, strategy), group in ensemble_df.groupby(["variant", "tau", "strategy"], dropna=False):
        best_rows.append(group.sort_values("score", ascending=False).iloc[0].to_dict())
    best_df = pd.DataFrame(best_rows).sort_values(["variant", "score"], ascending=[True, False])
    best_df.to_csv(artifact_dir / "analysis" / "validation_best_by_combo.csv", index=False)
    save_json(
        artifact_dir / "analysis" / "validation_summary.json",
        {
            "summary_csv": str(csv_path),
            "best_by_combo_csv": str(artifact_dir / "analysis" / "validation_best_by_combo.csv"),
            "best_rows": best_rows,
        },
    )


ENSEMBLE_STRATEGIES_FOR_SUMMARY = {
    "probmix_a050",
    "probmix_a075",
    "probmix_a025",
    "poe_symmetric",
    "poe_prior_corrected",
    "confidence_weighted",
}


def main(args):
    """Train classifier and SupCon models, evaluating the separate ensemble each epoch."""
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (artifact_dir / "classifier").mkdir(parents=True, exist_ok=True)
    for variant in args.supcon_variants:
        (artifact_dir / f"contrastive_{variant}").mkdir(parents=True, exist_ok=True)

    metrics_path = artifact_dir / "analysis" / "epoch_ensemble_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this mDeBERTa experiment.")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, use_fast=False)
    tokenizer.save_pretrained(str(artifact_dir / "tokenizer"))
    train_dataset, train_labels, _train_ids, val_dataset, val_labels, val_ids = build_datasets(args, tokenizer)

    selected_support = sample_per_class_indices(train_labels, args.retrieval_train_per_class, args.split_seed)
    support_dataset = Subset(train_dataset, selected_support)
    print(
        f"Validation medoid support: {len(support_dataset):,} train examples "
        f"({args.retrieval_train_per_class}/class cap; <=0 means full split)"
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    support_loader = DataLoader(support_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)
    loaders = {"val": val_loader, "support": support_loader}
    labels_and_ids = {
        "val_labels": val_labels,
        "val_ids": val_ids,
        "train_prior": class_prior(train_labels),
    }

    classifier = MDeBERTaEMD(
        model_name=BACKBONE,
        dropout=args.classifier_dropout,
        dropout_samples=args.msd_samples,
    ).to(device)
    print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    classifier_optimizer = build_llrd_optimizer(
        classifier,
        head_specs=[(classifier.classifier, args.classifier_head_lr)],
        encoder_top_lr=args.classifier_encoder_lr,
        layer_decay=args.layer_decay,
        weight_decay=args.weight_decay,
    )
    class_total_steps = args.epochs * len(train_loader)
    classifier_scheduler = get_cosine_schedule_with_warmup(
        classifier_optimizer,
        int(args.warmup_fraction * class_total_steps),
        class_total_steps,
    )
    classifier_ema = EMA(classifier, decay=args.ema_decay)
    offload_training_state(classifier, classifier_optimizer, classifier_ema)

    contrastive_states = {}
    for variant in args.supcon_variants:
        model = PureContrastiveMDeBERTa(
            model_name=BACKBONE,
            projection_dim=args.projection_dim,
            dropout=args.contrastive_dropout,
        ).to(device)
        optimizer = build_llrd_optimizer(
            model,
            head_specs=[(model.contrastive_head, args.contrastive_head_lr)],
            encoder_top_lr=args.contrastive_encoder_lr,
            layer_decay=args.layer_decay,
            weight_decay=args.weight_decay,
        )
        total_steps = args.epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            int(args.warmup_fraction * total_steps),
            total_steps,
        )
        ema = EMA(model, decay=args.ema_decay)
        loss_fn = SupervisedContrastiveLoss(temperature=args.temperature, variant=variant)
        contrastive_states[variant] = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "ema": ema,
            "loss_fn": loss_fn,
        }
        offload_training_state(model, optimizer, ema)

    all_rows: List[Dict[str, object]] = []
    normal_variant = args.supcon_variants[0]
    normal_state = contrastive_states[normal_variant]

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        activate_training_state(classifier, classifier_optimizer, classifier_ema, device)
        class_loss = train_classifier_epoch(
            classifier,
            train_loader,
            classifier_optimizer,
            classifier_scheduler,
            classifier_ema,
            device,
        )
        classifier_ema.apply_shadow()
        class_metrics = evaluate_classifier(classifier, val_loader, device, return_probs=True)
        class_val_probs = class_metrics.pop("probabilities")
        if args.cache_embeddings:
            embeddings_dir = artifact_dir / "embeddings"
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"classifier_probs": class_val_probs},
                embeddings_dir / f"classifier_epoch_{epoch:03d}_val_probs.pt",
            )
        classifier_ema.restore()
        class_ckpt = artifact_dir / "classifier" / f"epoch_{epoch:03d}_model.pt"
        save_ema_checkpoint(
            classifier,
            classifier_ema,
            class_ckpt,
            args,
            epoch,
            "classifier",
            {"validation": class_metrics},
        )
        offload_training_state(classifier, classifier_optimizer, classifier_ema)

        activate_training_state(normal_state["model"], normal_state["optimizer"], normal_state["ema"], device)
        contrastive_loss = train_contrastive_epoch(
            normal_state["model"],
            train_loader,
            normal_state["optimizer"],
            normal_state["scheduler"],
            normal_state["ema"],
            normal_state["loss_fn"],
            device,
        )
        contrastive_ckpt = artifact_dir / f"contrastive_{normal_variant}" / f"epoch_{epoch:03d}_model.pt"
        save_ema_checkpoint(
            normal_state["model"],
            normal_state["ema"],
            contrastive_ckpt,
            args,
            epoch,
            "contrastive",
            {"supcon_variant": normal_variant},
        )
        offload_training_state(normal_state["model"], normal_state["optimizer"], normal_state["ema"])

        print(
            f"Epoch {epoch:2d}/{args.epochs} normal | classifier_loss={class_loss:.4f} "
            f"| classifier_score={class_metrics['score']:.4f} | supcon_loss={contrastive_loss:.4f} "
            f"| {time.time() - t0:.1f}s"
        )
        rows = evaluate_epoch_from_checkpoints(
            normal_variant,
            epoch,
            class_ckpt,
            contrastive_ckpt,
            loaders,
            labels_and_ids,
            args,
            artifact_dir,
            device,
        )
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(artifact_dir / "analysis" / "validation_summary_live.csv", index=False)

    del classifier, classifier_optimizer, classifier_scheduler, classifier_ema
    del contrastive_states[normal_variant]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for variant in args.supcon_variants[1:]:
        state = contrastive_states[variant]
        print(f"\n=== Training SupCon variant: {variant} ===")
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            activate_training_state(state["model"], state["optimizer"], state["ema"], device)
            contrastive_loss = train_contrastive_epoch(
                state["model"],
                train_loader,
                state["optimizer"],
                state["scheduler"],
                state["ema"],
                state["loss_fn"],
                device,
            )
            contrastive_ckpt = artifact_dir / f"contrastive_{variant}" / f"epoch_{epoch:03d}_model.pt"
            save_ema_checkpoint(
                state["model"],
                state["ema"],
                contrastive_ckpt,
                args,
                epoch,
                "contrastive",
                {"supcon_variant": variant},
            )
            offload_training_state(state["model"], state["optimizer"], state["ema"])

            class_ckpt = artifact_dir / "classifier" / f"epoch_{epoch:03d}_model.pt"
            print(
                f"Epoch {epoch:2d}/{args.epochs} {variant} | supcon_loss={contrastive_loss:.4f} "
                f"| {time.time() - t0:.1f}s"
            )
            rows = evaluate_epoch_from_checkpoints(
                variant,
                epoch,
                class_ckpt,
                contrastive_ckpt,
                loaders,
                labels_and_ids,
                args,
                artifact_dir,
                device,
            )
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(artifact_dir / "analysis" / "validation_summary_live.csv", index=False)

    write_summary_tables(artifact_dir)
    save_json(
        artifact_dir / "analysis" / "run_config.json",
        {
            "args": vars(args),
            "backbone": BACKBONE,
            "train_examples": len(train_dataset),
            "val_examples": len(val_dataset),
            "support_examples": len(support_dataset),
        },
    )
    print(f"\nTraining and validation sweep finished: {artifact_dir}")
    print(f"Validation summary: {artifact_dir / 'analysis' / 'validation_summary.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--tokenize_batch_size", type=int, default=1024)
    parser.add_argument("--classifier_encoder_lr", type=float, default=8e-6)
    parser.add_argument("--classifier_head_lr", type=float, default=5e-5)
    parser.add_argument("--contrastive_encoder_lr", type=float, default=8e-6)
    parser.add_argument("--contrastive_head_lr", type=float, default=1e-4)
    parser.add_argument("--layer_decay", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--classifier_dropout", type=float, default=0.25)
    parser.add_argument("--contrastive_dropout", type=float, default=0.1)
    parser.add_argument("--msd_samples", type=int, default=5)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--warmup_fraction", type=float, default=0.06)
    parser.add_argument("--supcon_variants", nargs="+", default=["normal", "distance_weighted"])
    parser.add_argument("--retrieval_taus", type=float, nargs="+", default=list(DEFAULT_TAUS))
    parser.add_argument("--retrieval_train_per_class", type=int, default=1000)
    parser.add_argument("--ensemble_strategies", nargs="+", default=list(ENSEMBLE_STRATEGIES_FOR_SUMMARY))
    parser.add_argument("--cache_embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require_cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--data_dir", default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    main(parser.parse_args())
