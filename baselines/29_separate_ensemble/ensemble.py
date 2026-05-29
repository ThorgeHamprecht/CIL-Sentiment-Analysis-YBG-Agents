"""Evaluation and ensembling helpers for separate classifier/SupCon models."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error

NUM_CLASSES = 5
DEFAULT_TAUS = (0.02, 0.05, 0.10, 0.20)
ENSEMBLE_STRATEGIES = (
    "probmix_a050",
    "probmix_a075",
    "probmix_a025",
    "poe_symmetric",
    "poe_prior_corrected",
    "confidence_weighted",
)


def json_default(obj):
    """Convert numpy values to JSON-friendly Python objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def sanitize_name(name: str) -> str:
    """Make a method name safe for use in filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def tau_tag(tau: float) -> str:
    """Format a retrieval temperature as a stable short method tag."""
    return f"tau{int(round(float(tau) * 100)):03d}"


def kaggle_score(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute the repo score, 1 - MAE / 4, for labels on 0..4."""
    return 1.0 - np.abs(preds - labels).mean() / 4.0


def median_decode_probs_np(probs: np.ndarray) -> np.ndarray:
    """Decode class probabilities with the ordinal CDF median rule."""
    cdf = np.cumsum(probs, axis=1)
    return np.sum(cdf < 0.5, axis=1).clip(0, probs.shape[1] - 1).astype(np.int64)


def median_decode_probs_torch(probs: torch.Tensor) -> torch.Tensor:
    """Torch version of ordinal CDF median decoding."""
    cdf = torch.cumsum(probs, dim=1)
    return (cdf < 0.5).sum(dim=1).clamp(0, probs.shape[1] - 1).long()


def normalize_probs(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Clamp and row-normalize a probability matrix."""
    probs = np.asarray(probs, dtype=np.float32)
    probs = np.maximum(probs, eps)
    return probs / np.maximum(probs.sum(axis=1, keepdims=True), eps)


def class_prior(labels: Sequence[int], eps: float = 1e-8) -> np.ndarray:
    """Return empirical class probabilities for labels on the 0..4 scale."""
    labels_arr = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_arr, minlength=NUM_CLASSES).astype(np.float32)
    counts = np.maximum(counts, eps)
    return counts / counts.sum()


def metrics_for_predictions(preds: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    """Compute validation metrics and a 5x5 confusion matrix."""
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    return {
        "score": float(kaggle_score(preds, labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "mae": float(mean_absolute_error(labels, preds)),
        "qwk": float(cohen_kappa_score(labels, preds, weights="quadratic")),
        "confusion_matrix": cm.tolist(),
    }


def sample_per_class_indices(labels: Sequence[int], max_per_class: Optional[int], seed: int) -> List[int]:
    """Select a deterministic class-balanced subset, or all indices if disabled."""
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


def _normalized_entropy_confidence(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return 1 - normalized entropy as a per-example confidence score."""
    probs = normalize_probs(probs, eps)
    entropy = -(probs * np.log(probs + eps)).sum(axis=1) / math.log(probs.shape[1])
    return np.clip(1.0 - entropy, 0.0, 1.0)


def ensemble_distributions(
    class_probs: np.ndarray,
    retrieval_probs: np.ndarray,
    prior: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """Build the six configured classifier/retrieval ensemble distributions."""
    class_probs = normalize_probs(class_probs, eps)
    retrieval_probs = normalize_probs(retrieval_probs, eps)
    prior = np.maximum(np.asarray(prior, dtype=np.float32), eps)
    prior = prior / prior.sum()

    c_conf = _normalized_entropy_confidence(class_probs, eps)
    r_conf = _normalized_entropy_confidence(retrieval_probs, eps)
    denom = c_conf + r_conf
    c_weight = np.where(denom > eps, c_conf / np.maximum(denom, eps), 0.5).astype(np.float32)

    outputs = {
        "probmix_a050": normalize_probs(0.50 * class_probs + 0.50 * retrieval_probs, eps),
        "probmix_a075": normalize_probs(0.75 * class_probs + 0.25 * retrieval_probs, eps),
        "probmix_a025": normalize_probs(0.25 * class_probs + 0.75 * retrieval_probs, eps),
        "poe_symmetric": normalize_probs(class_probs * retrieval_probs, eps),
        "poe_prior_corrected": normalize_probs(class_probs * retrieval_probs / prior.reshape(1, -1), eps),
        "confidence_weighted": normalize_probs(
            c_weight.reshape(-1, 1) * class_probs + (1.0 - c_weight).reshape(-1, 1) * retrieval_probs,
            eps,
        ),
    }
    return outputs


def _progress_marks(total_batches: int) -> set[int]:
    """Return batch indices where a 10% progress update should be printed."""
    if total_batches <= 0:
        return set()
    return {max(1, math.ceil(total_batches * step / 10)) for step in range(1, 11)}


def _print_progress(progress_name: Optional[str], batch_idx: int, total_batches: int, marks: set[int]) -> None:
    """Print a compact eval progress heartbeat for long mDeBERTa encoding loops."""
    if progress_name is None or batch_idx not in marks:
        return
    pct = min(100, int(round(100.0 * batch_idx / max(1, total_batches))))
    print(f"{progress_name}: {pct}% ({batch_idx}/{total_batches} batches)", flush=True)


@torch.no_grad()
def encode_classifier_probs(model, loader, device, progress_name: Optional[str] = None) -> Tuple[np.ndarray, torch.Tensor]:
    """Encode a loader into classifier probabilities and optional labels."""
    model.eval()
    all_probs: List[np.ndarray] = []
    all_labels: List[torch.Tensor] = []
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad()
    total_batches = len(loader)
    marks = _progress_marks(total_batches)
    for batch_idx, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast_ctx:
            logits = model(input_ids, attention_mask)
        all_probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        if "labels" in batch:
            all_labels.append(batch["labels"].long().cpu())
        _print_progress(progress_name, batch_idx, total_batches, marks)
    labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
    return np.concatenate(all_probs, axis=0), labels


@torch.no_grad()
def encode_contrastive_embeddings(model, loader, device, progress_name: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a loader into normalized contrastive embeddings and labels."""
    model.eval()
    all_embeddings: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad()
    total_batches = len(loader)
    marks = _progress_marks(total_batches)
    for batch_idx, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast_ctx:
            outputs = model(input_ids, attention_mask)
        all_embeddings.append(outputs["embeddings"].float().cpu())
        if "labels" in batch:
            all_labels.append(batch["labels"].long().cpu())
        _print_progress(progress_name, batch_idx, total_batches, marks)
    embeddings = F.normalize(torch.cat(all_embeddings, dim=0), p=2, dim=-1)
    labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
    return embeddings, labels


def compute_class_medoids(z_train: torch.Tensor, y_train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick the training embedding closest to each normalized class centroid."""
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
    if not medoids:
        raise ValueError("Cannot compute medoids without labeled support embeddings.")
    return torch.stack(medoids, dim=0), torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def medoid_distribution_from_medoids(
    medoids: torch.Tensor,
    medoid_labels: torch.Tensor,
    z_query: torch.Tensor,
    tau: float,
    chunk_size: int = 2048,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Convert query similarities to precomputed medoids into probabilities."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    medoids = medoids.to(device)
    medoid_labels = medoid_labels.to(device)
    z_query = F.normalize(z_query.float(), p=2, dim=-1)
    parts: List[np.ndarray] = []
    for start in range(0, z_query.size(0), chunk_size):
        chunk = z_query[start:start + chunk_size].to(device)
        sims = chunk @ medoids.T
        sparse_probs = torch.softmax(sims / tau, dim=1)
        probs = torch.zeros((chunk.size(0), NUM_CLASSES), device=device)
        probs[:, medoid_labels] = sparse_probs
        parts.append(probs.cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


@torch.no_grad()
def medoid_distribution(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_query: torch.Tensor,
    tau: float,
    chunk_size: int = 2048,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Compute medoids once and return the probability matrix for one tau."""
    medoids, medoid_labels = compute_class_medoids(z_train, y_train)
    return medoid_distribution_from_medoids(medoids, medoid_labels, z_query, tau, chunk_size, device)


@torch.no_grad()
def medoid_distributions_for_taus(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_query: torch.Tensor,
    taus: Iterable[float],
    chunk_size: int = 2048,
    device: Optional[torch.device] = None,
) -> Dict[float, np.ndarray]:
    """Compute class medoids once and evaluate all requested temperatures."""
    medoids, medoid_labels = compute_class_medoids(z_train, y_train)
    return {
        float(tau): medoid_distribution_from_medoids(medoids, medoid_labels, z_query, tau, chunk_size, device)
        for tau in taus
    }


def evaluate_ensemble_grid(
    class_probs: np.ndarray,
    z_support: torch.Tensor,
    y_support: torch.Tensor,
    z_query: torch.Tensor,
    labels: np.ndarray,
    taus: Iterable[float],
    prior: np.ndarray,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray]]:
    """Evaluate all tau/strategy ensemble predictions on labeled data."""
    labels = np.asarray(labels, dtype=np.int64)
    rows: List[Dict[str, object]] = []
    predictions: Dict[str, np.ndarray] = {}

    cls_pred = median_decode_probs_np(class_probs)
    cls_metrics = metrics_for_predictions(cls_pred, labels)
    rows.append({"tau": "", "strategy": "classification_median", **cls_metrics})
    predictions["classification_median"] = cls_pred

    ret_probs_by_tau = medoid_distributions_for_taus(z_support, y_support, z_query, taus=taus, device=device)
    for tau, ret_probs in ret_probs_by_tau.items():
        ret_pred = median_decode_probs_np(ret_probs)
        ret_metrics = metrics_for_predictions(ret_pred, labels)
        ret_name = f"{tau_tag(tau)}__medoid_distribution_median"
        rows.append({"tau": float(tau), "strategy": "medoid_distribution_median", **ret_metrics})
        predictions[ret_name] = ret_pred

        for strategy, probs in ensemble_distributions(class_probs, ret_probs, prior).items():
            pred = median_decode_probs_np(probs)
            method = f"{tau_tag(tau)}__{strategy}"
            rows.append({"tau": float(tau), "strategy": strategy, **metrics_for_predictions(pred, labels)})
            predictions[method] = pred
    return rows, predictions


def build_test_predictions_for_taus(
    class_probs: np.ndarray,
    z_support: torch.Tensor,
    y_support: torch.Tensor,
    z_test: torch.Tensor,
    taus: Iterable[float],
    prior: np.ndarray,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Create ensemble predictions for every tau/strategy on unlabeled test data."""
    predictions: Dict[str, np.ndarray] = {
        "classification_median": median_decode_probs_np(class_probs),
    }
    ret_probs_by_tau = medoid_distributions_for_taus(z_support, y_support, z_test, taus=taus, device=device)
    for tau, ret_probs in ret_probs_by_tau.items():
        predictions[f"{tau_tag(tau)}__medoid_distribution_median"] = median_decode_probs_np(ret_probs)
        for strategy, probs in ensemble_distributions(class_probs, ret_probs, prior).items():
            predictions[f"{tau_tag(tau)}__{strategy}"] = median_decode_probs_np(probs)
    return predictions


def write_prediction_wide_csv(
    path: Path,
    ids: Sequence[int],
    predictions: Dict[str, np.ndarray],
    labels: Optional[Sequence[int]] = None,
) -> None:
    """Write one wide CSV with ids, optional truth, and prediction columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"id": list(ids)}
    if labels is not None:
        payload["label_true"] = list(labels)
    for name, preds in sorted(predictions.items()):
        payload[name] = np.asarray(preds, dtype=np.int64).clip(0, NUM_CLASSES - 1)
    pd.DataFrame(payload).to_csv(path, index=False)


def write_submission_file(path: Path, ids: Sequence[int], preds: np.ndarray) -> None:
    """Write one Kaggle-ready id,label submission file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(preds, dtype=np.int64).clip(0, NUM_CLASSES - 1)
    pd.DataFrame({"id": list(ids), "label": labels}).to_csv(path, index=False)


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    """Append one JSON payload to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=json_default) + "\n")


def save_json(path: Path, payload: Dict[str, object]) -> None:
    """Save an indented JSON file with numpy-aware conversion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)
