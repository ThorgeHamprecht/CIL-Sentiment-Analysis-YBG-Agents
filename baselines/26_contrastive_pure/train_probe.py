"""Train a lightweight W1 classifier on frozen pure-contrastive embeddings."""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baselines.contrastive_eval_utils import (  # noqa: E402
    DEFAULT_ALPHAS,
    DEFAULT_K_VALUES,
    classification_predictions,
    combined_predictions,
    encode_contrastive_embeddings,
    evaluate_prediction_dict,
    retrieval_predictions_and_distributions,
    save_json,
    write_confusion_csvs,
    write_disagreement_analysis,
    write_prediction_wide_csv,
    write_submission_files,
)
from dataset import ReviewDataset, read_csv  # noqa: E402
from eval_retrieval import load_model_from_checkpoint  # noqa: E402

NUM_CLASSES = 5
_SCRATCH = Path("/work/scratch") / os.environ.get("USER", "") / "cil"
_DEFAULT_DATA_DIR = _SCRATCH / "data" if (_SCRATCH / "data").exists() else ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
_DEFAULT_OUTPUT_DIR = _SCRATCH / "submissions" if _SCRATCH.exists() else ROOT / "submissions"


class EmbeddingW1Probe(nn.Module):
    """Small classifier trained on frozen normalized contrastive embeddings."""

    def __init__(self, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, NUM_CLASSES)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(embeddings))


def emd_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """EMD-squared/W1-style ordinal loss used by the mDeBERTa classifiers."""
    probs = F.softmax(logits, dim=1)
    cdf = torch.cumsum(probs, dim=1)[:, :-1]
    k_vals = torch.arange(NUM_CLASSES - 1, device=labels.device)
    targets = (labels.unsqueeze(1) <= k_vals).float()
    return ((cdf - targets) ** 2).sum(dim=1).mean()


def median_decode_logits(logits: torch.Tensor) -> torch.Tensor:
    """Decode logits with the ordinal CDF median rule."""
    cdf = torch.cumsum(F.softmax(logits, dim=1), dim=1)
    return (cdf < 0.5).sum(dim=1).clamp(0, logits.shape[1] - 1).long()


def build_split_and_test_datasets(data_dir: Path, artifact_dir: Path, split_seed: int, max_len: int, batch_size: int):
    """Rebuild the pure-contrastive split and the test dataset."""
    tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir / "tokenizer"), use_fast=False)
    texts, labels, ids = read_csv(data_dir / "train.csv")
    test_texts, _, test_ids = read_csv(data_dir / "test.csv")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=split_seed)
    train_idx, val_idx = next(sss.split(texts, labels))

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    train_ids = [ids[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    val_ids = [ids[i] for i in val_idx]

    train_dataset = ReviewDataset(train_texts, tokenizer, max_len=max_len, labels=train_labels, batch_size=batch_size)
    val_dataset = ReviewDataset(val_texts, tokenizer, max_len=max_len, labels=val_labels, batch_size=batch_size)
    test_dataset = ReviewDataset(test_texts, tokenizer, max_len=max_len, batch_size=batch_size)
    return train_dataset, train_ids, val_dataset, val_ids, test_dataset, test_ids


def train_probe_epoch(probe, loader, optimizer, device) -> float:
    """Train one epoch of the frozen-embedding classifier."""
    probe.train()
    total = 0.0
    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        logits = probe(embeddings)
        loss = emd_loss(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(labels)
    return total / len(loader.dataset)


@torch.no_grad()
def predict_probe_probs(probe, embeddings: torch.Tensor, batch_size: int, device) -> np.ndarray:
    """Run the frozen-embedding classifier and return softmax probabilities."""
    probe.eval()
    parts = []
    for start in range(0, embeddings.size(0), batch_size):
        batch = embeddings[start:start + batch_size].to(device)
        parts.append(F.softmax(probe(batch).float(), dim=1).cpu().numpy())
    return np.concatenate(parts, axis=0)


def evaluate_probe(probe, embeddings: torch.Tensor, labels: torch.Tensor, batch_size: int, device) -> Dict[str, object]:
    """Compute classifier-only validation metrics for the probe."""
    probs = predict_probe_probs(probe, embeddings, batch_size, device)
    preds = classification_predictions(probs)["classification_median"]
    return evaluate_prediction_dict({"classification_median": preds}, labels.numpy())["classification_median"]


def build_all_predictions(
    class_probs: np.ndarray,
    z_support: torch.Tensor,
    y_support: torch.Tensor,
    z_query: torch.Tensor,
    args,
    device,
) -> Dict[str, np.ndarray]:
    """Create probe-classifier, retrieval, and combo predictions."""
    class_preds = classification_predictions(class_probs)
    retrieval_preds, retrieval_dists = retrieval_predictions_and_distributions(
        z_train=z_support,
        y_train=y_support,
        z_query=z_query,
        k_values=args.k_values,
        tau=args.retrieval_tau,
        chunk_size=args.similarity_chunk_size,
        device=device,
    )
    combo_preds = combined_predictions(
        class_probs=class_probs,
        class_preds=class_preds,
        retrieval_preds=retrieval_preds,
        retrieval_distributions=retrieval_dists,
        alphas=args.alphas,
    )
    predictions: Dict[str, np.ndarray] = {}
    predictions.update({f"probe_{name}": preds for name, preds in class_preds.items()})
    predictions.update(retrieval_preds)
    predictions.update({f"probe_{name}": preds for name, preds in combo_preds.items()})
    return predictions


def main(args):
    """Train a two-epoch W1 probe on frozen pure-contrastive embeddings."""
    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    analysis_dir = artifact_dir / "analysis"
    predictions_dir = artifact_dir / "predictions"
    embeddings_dir = artifact_dir / "embeddings"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for pure contrastive probe training; refusing to run on CPU.")
    print(f"Device: {device}")

    encoder, ckpt_args = load_model_from_checkpoint(artifact_dir, device)
    max_len = args.max_len or int(ckpt_args.get("max_len", 256))
    split_seed = args.split_seed if args.split_seed is not None else int(ckpt_args.get("split_seed", ckpt_args.get("seed", 42)))

    train_dataset, train_ids, val_dataset, val_ids, test_dataset, test_ids = build_split_and_test_datasets(
        data_dir,
        artifact_dir,
        split_seed,
        max_len,
        args.tokenize_batch_size,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.encoder_batch_size, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.encoder_batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.encoder_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print("Encoding frozen contrastive embeddings...")
    z_train, y_train = encode_contrastive_embeddings(encoder, train_loader, device)
    z_val, y_val = encode_contrastive_embeddings(encoder, val_loader, device)
    z_test, _ = encode_contrastive_embeddings(encoder, test_loader, device)

    torch.save({"embeddings": z_train, "labels": y_train}, embeddings_dir / "probe_train.pt")
    torch.save({"embeddings": z_val, "labels": y_val}, embeddings_dir / "probe_val.pt")
    torch.save({"embeddings": z_test}, embeddings_dir / "probe_test.pt")

    probe = EmbeddingW1Probe(z_train.size(1), dropout=args.probe_dropout).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=args.probe_weight_decay)
    probe_loader = DataLoader(
        TensorDataset(z_train, y_train),
        batch_size=args.probe_batch_size,
        shuffle=True,
        num_workers=0,
    )

    best_score = -float("inf")
    best_path = artifact_dir / "probe_best.pt"
    for epoch in range(1, args.probe_epochs + 1):
        train_loss = train_probe_epoch(probe, probe_loader, optimizer, device)
        metrics = evaluate_probe(probe, z_val, y_val, args.probe_batch_size, device)
        print(
            f"Probe epoch {epoch:2d} | loss={train_loss:.4f} | "
            f"score={metrics['score']:.4f} | mae={metrics['mae']:.4f} | qwk={metrics['qwk']:.4f}"
        )
        epoch_path = artifact_dir / f"probe_epoch_{epoch:03d}.pt"
        torch.save({"probe": probe.state_dict(), "args": vars(args), "epoch": epoch, "metrics": metrics}, epoch_path)
        if metrics["score"] > best_score:
            best_score = metrics["score"]
            torch.save({"probe": probe.state_dict(), "args": vars(args), "epoch": epoch, "metrics": metrics}, best_path)
            print(f"  -> New best probe: {best_score:.4f} (saved)")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    probe.load_state_dict(ckpt["probe"])
    val_probs = predict_probe_probs(probe, z_val, args.probe_batch_size, device)
    test_probs = predict_probe_probs(probe, z_test, args.probe_batch_size, device)

    val_predictions = build_all_predictions(val_probs, z_train, y_train, z_val, args, device)
    val_metrics = evaluate_prediction_dict(val_predictions, y_val.numpy())
    write_prediction_wide_csv(predictions_dir / "probe_val_predictions.csv", val_ids, val_predictions, labels=y_val.numpy())
    write_confusion_csvs(val_metrics, analysis_dir / "probe_confusions")
    val_disagreement_path = write_disagreement_analysis(
        val_predictions,
        analysis_dir,
        labels=y_val.numpy(),
        prefix="probe_val",
        metrics=val_metrics,
    )

    test_support_z = torch.cat([z_train, z_val], dim=0)
    test_support_y = torch.cat([y_train, y_val], dim=0)
    test_predictions = build_all_predictions(test_probs, test_support_z, test_support_y, z_test, args, device)
    write_prediction_wide_csv(predictions_dir / "probe_test_predictions.csv", test_ids, test_predictions)
    test_disagreement_path = write_disagreement_analysis(test_predictions, analysis_dir, prefix="probe_test")

    prefix = args.submission_prefix or f"{artifact_dir.name}_probe"
    submission_paths = write_submission_files(test_ids, test_predictions, output_dir, prefix)
    payload = {
        "artifact_dir": str(artifact_dir),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "max_len": max_len,
        "split_seed": split_seed,
        "probe_epochs": args.probe_epochs,
        "probe_lr": args.probe_lr,
        "probe_batch_size": args.probe_batch_size,
        "val_metrics": val_metrics,
        "val_disagreement_csv": str(val_disagreement_path),
        "test_disagreement_csv": str(test_disagreement_path),
        "submissions": [str(path) for path in submission_paths],
    }
    save_json(analysis_dir / "probe_eval.json", payload)

    print("\nTop probe validation methods:")
    for name, values in sorted(val_metrics.items(), key=lambda item: item[1]["score"], reverse=True)[:20]:
        print(f"  {name:<76} score={values['score']:.4f} mae={values['mae']:.4f} qwk={values['qwk']:.4f}")
    print(f"\nSaved probe predictions under {predictions_dir}")
    print(f"Saved {len(submission_paths)} probe submissions under {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--data_dir", default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--output_dir", default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--submission_prefix", default=None)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--encoder_batch_size", type=int, default=32)
    parser.add_argument("--tokenize_batch_size", type=int, default=1024)
    parser.add_argument("--probe_epochs", type=int, default=2)
    parser.add_argument("--probe_batch_size", type=int, default=2048)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_weight_decay", type=float, default=0.01)
    parser.add_argument("--probe_dropout", type=float, default=0.1)
    parser.add_argument("--retrieval_tau", type=float, default=0.07)
    parser.add_argument("--k_values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--similarity_chunk_size", type=int, default=512)
    parser.add_argument("--require_cuda", action=argparse.BooleanOptionalAction, default=True)
    main(parser.parse_args())
