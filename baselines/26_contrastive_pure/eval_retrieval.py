"""Retrieval evaluation for pure contrastive mDeBERTa checkpoints."""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from dataset import ReviewDataset, read_csv
from model import PureContrastiveMDeBERTa

ROOT = Path(__file__).resolve().parents[2]
_SCRATCH_DATA_DIR = Path("/work/scratch") / os.environ.get("USER", "") / "cil" / "data"
_DEFAULT_DATA_DIR = _SCRATCH_DATA_DIR if _SCRATCH_DATA_DIR.exists() else ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
BACKBONE = "microsoft/mdeberta-v3-base"
NUM_CLASSES = 5
DEFAULT_K_VALUES = (1, 7, 101)


def kaggle_score(preds: np.ndarray, labels: np.ndarray) -> float:
    """Return the competition ordinal score from integer predictions.

    The score is 1 - MAE / 4 for labels on the 0..4 ordinal scale.
    """
    return 1.0 - np.abs(preds - labels).mean() / 4.0


def median_decode_probs(probs: torch.Tensor) -> torch.Tensor:
    """Decode class probabilities by the ordinal median/CDF rule.

    This chooses the first class whose cumulative probability reaches 0.5,
    matching the repo's median decode used for ordinal/MAE-style evaluation.
    """
    cdf = torch.cumsum(probs, dim=1)
    return (cdf < 0.5).sum(dim=1).clamp(0, probs.shape[1] - 1).long()


def metrics_for_predictions(preds: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    """Compute common validation metrics for one prediction vector.

    Returns the repo score, accuracy, macro-F1, MAE, quadratic weighted kappa,
    and a 5x5 confusion matrix using the repo-native 0..4 labels.
    """
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    return {
        "score": float(kaggle_score(preds, labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "mae": float(mean_absolute_error(labels, preds)),
        "qwk": float(cohen_kappa_score(labels, preds, weights="quadratic")),
        "confusion_matrix": cm.tolist(),
    }


def _json_default(obj):
    """JSON fallback for numpy values used in metric payloads.

    This keeps saved analysis files readable without manually converting every
    numpy scalar or array before calling json.dump.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def sample_per_class_indices(labels: Sequence[int], max_per_class: Optional[int], seed: int) -> List[int]:
    """Select up to max_per_class examples per label with a deterministic RNG.

    Used for cheap epoch-level retrieval validation. Passing 0 or None returns
    the full training split, which is useful for final post-training eval.
    """
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


@torch.no_grad()
def encode_embeddings(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the contrastive encoder over a loader.

    Returns L2-normalized embeddings on CPU together with labels, so downstream
    kNN and medoid evaluation can run without another model forward pass.
    """
    model.eval()
    all_embeddings: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad()

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast_ctx:
            outputs = model(input_ids, attention_mask)
        all_embeddings.append(outputs["embeddings"].float().cpu())
        if "labels" in batch:
            all_labels.append(batch["labels"].long().cpu())

    embeddings = F.normalize(torch.cat(all_embeddings, dim=0), p=2, dim=-1)
    labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
    return embeddings, labels


def knn_predict_all(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_val: torch.Tensor,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    tau: float = 0.07,
    chunk_size: int = 512,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """Predict labels with all requested kNN decoders.

    For each k, this computes majority vote plus a similarity-weighted class
    distribution over neighbors. The weighted distribution is decoded both by
    argmax and by the ordinal median/CDF rule.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_train = F.normalize(z_train.float(), p=2, dim=-1).to(device)
    y_train = y_train.long().to(device)
    z_val = F.normalize(z_val.float(), p=2, dim=-1)
    max_k = min(max(k_values), z_train.size(0))

    preds: Dict[str, List[torch.Tensor]] = {}
    for k in k_values:
        k_eff = min(k, z_train.size(0))
        for suffix in ("majority", "weighted_argmax", "weighted_median"):
            preds[f"knn_k{k_eff}_{suffix}"] = []

    for start in range(0, z_val.size(0), chunk_size):
        chunk = z_val[start:start + chunk_size].to(device)
        sims = chunk @ z_train.T
        top_sim, top_idx = sims.topk(k=max_k, dim=1)
        top_labels = y_train[top_idx]

        for k in k_values:
            k_eff = min(k, z_train.size(0))
            labels_k = top_labels[:, :k_eff]
            sims_k = top_sim[:, :k_eff]

            one_hot = F.one_hot(labels_k, num_classes=NUM_CLASSES).float()
            counts = one_hot.sum(dim=1)
            preds[f"knn_k{k_eff}_majority"].append(counts.argmax(dim=1).cpu())

            weights = torch.softmax(sims_k / tau, dim=1).unsqueeze(-1)
            probs = (weights * one_hot).sum(dim=1)
            preds[f"knn_k{k_eff}_weighted_argmax"].append(probs.argmax(dim=1).cpu())
            preds[f"knn_k{k_eff}_weighted_median"].append(median_decode_probs(probs).cpu())

    return {name: torch.cat(parts).numpy() for name, parts in preds.items()}


def compute_class_medoids(z_train: torch.Tensor, y_train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick one actual training embedding per class as the class medoid.

    For each class, first compute the normalized class centroid, then choose
    the training embedding with highest cosine similarity to that centroid.
    """
    medoids: List[torch.Tensor] = []
    labels: List[int] = []
    z_train = F.normalize(z_train.float(), p=2, dim=-1)
    y_train = y_train.long()
    for cls in range(NUM_CLASSES):
        idx = torch.nonzero(y_train == cls, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        cls_embeddings = z_train[idx]
        centroid = F.normalize(cls_embeddings.mean(dim=0, keepdim=True), p=2, dim=-1)
        best_local = torch.argmax((cls_embeddings @ centroid.T).squeeze(1))
        medoids.append(cls_embeddings[best_local])
        labels.append(cls)
    return torch.stack(medoids, dim=0), torch.tensor(labels, dtype=torch.long)


def medoid_predict_all(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_val: torch.Tensor,
    tau: float = 0.07,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """Predict labels from class medoids.

    The nearest-medoid decoder chooses the class of the closest medoid. The
    distribution decoders apply softmax(similarity / tau) across the class
    medoids, treating those normalized similarities as a 5-class probability
    distribution, then decode it with argmax or ordinal median/CDF.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    medoids, medoid_labels = compute_class_medoids(z_train, y_train)
    medoids = medoids.to(device)
    medoid_labels = medoid_labels.to(device)
    z_val = F.normalize(z_val.float(), p=2, dim=-1)

    nearest_parts: List[torch.Tensor] = []
    argmax_parts: List[torch.Tensor] = []
    median_parts: List[torch.Tensor] = []
    for start in range(0, z_val.size(0), 2048):
        chunk = z_val[start:start + 2048].to(device)
        sims = chunk @ medoids.T
        nearest_parts.append(medoid_labels[sims.argmax(dim=1)].cpu())

        probs_sparse = torch.softmax(sims / tau, dim=1)
        probs = torch.zeros((chunk.size(0), NUM_CLASSES), device=device)
        probs[:, medoid_labels] = probs_sparse
        argmax_parts.append(probs.argmax(dim=1).cpu())
        median_parts.append(median_decode_probs(probs).cpu())

    return {
        "medoid_nearest": torch.cat(nearest_parts).numpy(),
        "medoid_distribution_argmax": torch.cat(argmax_parts).numpy(),
        "medoid_distribution_median": torch.cat(median_parts).numpy(),
    }


def evaluate_retrieval_from_embeddings(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    tau: float = 0.07,
    chunk_size: int = 512,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, object]]:
    """Run every retrieval evaluator and attach metrics to each method.

    This is the shared entry point used both during lightweight epoch eval and
    by the full checkpoint evaluation CLI.
    """
    labels_np = y_val.cpu().numpy()
    predictions = {}
    predictions.update(knn_predict_all(z_train, y_train, z_val, k_values, tau, chunk_size, device))
    predictions.update(medoid_predict_all(z_train, y_train, z_val, tau, device))
    return {name: metrics_for_predictions(preds, labels_np) for name, preds in predictions.items()}


def flatten_scores(metrics: Dict[str, Dict[str, object]]) -> Dict[str, float]:
    """Extract only score values for compact logging and checkpoint selection.

    The resulting keys are named like knn_k7_weighted_median_score.
    """
    return {f"{name}_score": float(values["score"]) for name, values in metrics.items()}


def write_confusion_csvs(metrics: Dict[str, Dict[str, object]], out_dir: Path) -> None:
    """Write one confusion-matrix CSV per retrieval prediction method.

    These CSVs mirror the JSON confusion matrices but are easier to inspect in
    spreadsheet tools or paste into reports.
    """
    classes = list(range(NUM_CLASSES))
    for name, values in metrics.items():
        cm = np.array(values["confusion_matrix"])
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(out_dir / f"confusion_{name}.csv")


def load_model_from_checkpoint(artifact_dir: Path, device: torch.device) -> Tuple[PureContrastiveMDeBERTa, dict]:
    """Recreate the contrastive model from best_model.pt.

    Uses the saved projection dimension, dropout, and backbone path when
    available, then loads the checkpoint weights and returns saved args.
    """
    ckpt_path = artifact_dir / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = PureContrastiveMDeBERTa(
        model_name=str(ckpt.get("backbone_dir", BACKBONE)),
        projection_dim=int(args.get("projection_dim", 128)),
        dropout=float(args.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model, args


def build_split_datasets(data_dir: Path, artifact_dir: Path, split_seed: int, max_len: int, tokenize_batch_size: int):
    """Rebuild the fixed train/validation split and tokenized datasets.

    The split must match training so retrieval eval compares validation points
    against exactly the training pool seen by the contrastive model.
    """
    tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir / "tokenizer"), use_fast=False)
    texts, labels, _ = read_csv(data_dir / "train.csv")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=split_seed)
    train_idx, val_idx = next(sss.split(texts, labels))
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_dataset = ReviewDataset(train_texts, tokenizer, max_len=max_len, labels=train_labels, batch_size=tokenize_batch_size)
    val_dataset = ReviewDataset(val_texts, tokenizer, max_len=max_len, labels=val_labels, batch_size=tokenize_batch_size)
    return train_dataset, train_labels, val_dataset, val_labels


def main(args):
    """CLI entry point for full checkpoint retrieval evaluation.

    Loads a saved checkpoint, recomputes train/validation embeddings, caches
    them if requested, evaluates all retrieval decoders, and writes analysis
    JSON plus confusion-matrix CSVs.
    """
    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    analysis_dir = artifact_dir / "analysis"
    embeddings_dir = artifact_dir / "embeddings"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, ckpt_args = load_model_from_checkpoint(artifact_dir, device)
    max_len = args.max_len or int(ckpt_args.get("max_len", 256))
    split_seed = args.split_seed if args.split_seed is not None else int(ckpt_args.get("split_seed", ckpt_args.get("seed", 42)))

    train_dataset, train_labels, val_dataset, _ = build_split_datasets(
        data_dir, artifact_dir, split_seed, max_len, args.tokenize_batch_size
    )
    selected_train = sample_per_class_indices(train_labels, args.max_eval_train_per_class, split_seed)
    train_eval_dataset = Subset(train_dataset, selected_train)
    print(f"Retrieval train pool: {len(train_eval_dataset):,} examples")
    print(f"Validation set: {len(val_dataset):,} examples")

    train_loader = DataLoader(train_eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    z_train, y_train = encode_embeddings(model, train_loader, device)
    z_val, y_val = encode_embeddings(model, val_loader, device)

    if args.cache_embeddings:
        tag = "full" if args.max_eval_train_per_class is None or args.max_eval_train_per_class <= 0 else f"per_class_{args.max_eval_train_per_class}"
        torch.save({"embeddings": z_train, "labels": y_train}, embeddings_dir / f"train_{tag}.pt")
        torch.save({"embeddings": z_val, "labels": y_val}, embeddings_dir / "val.pt")

    metrics = evaluate_retrieval_from_embeddings(
        z_train=z_train,
        y_train=y_train,
        z_val=z_val,
        y_val=y_val,
        k_values=args.k_values,
        tau=args.retrieval_tau,
        chunk_size=args.similarity_chunk_size,
        device=device,
    )

    payload = {
        "artifact_dir": str(artifact_dir),
        "data_dir": str(data_dir),
        "split_seed": split_seed,
        "max_len": max_len,
        "retrieval_tau": args.retrieval_tau,
        "k_values": args.k_values,
        "max_eval_train_per_class": args.max_eval_train_per_class,
        "n_train_pool": len(train_eval_dataset),
        "n_val": len(val_dataset),
        "metrics": metrics,
    }
    out_path = analysis_dir / "retrieval_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    write_confusion_csvs(metrics, analysis_dir)

    print("\nRetrieval scores:")
    for name, values in sorted(metrics.items()):
        print(f"  {name:<34} score={values['score']:.4f} mae={values['mae']:.4f} qwk={values['qwk']:.4f}")
    print(f"\nSaved retrieval eval to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--data_dir", default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tokenize_batch_size", type=int, default=1024)
    parser.add_argument("--retrieval_tau", type=float, default=0.07)
    parser.add_argument("--k_values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--similarity_chunk_size", type=int, default=512)
    parser.add_argument("--max_eval_train_per_class", type=int, default=0)
    parser.add_argument("--cache_embeddings", action="store_true", default=True)
    main(parser.parse_args())
