"""Shared retrieval, submission, and disagreement utilities for contrastive runs."""
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
DEFAULT_K_VALUES = (1, 7, 101)
DEFAULT_ALPHAS = (0.5, 0.7, 0.3)


def json_default(obj):
    """Convert numpy values to JSON-friendly Python objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def sanitize_name(name: str) -> str:
    """Make a metric or method name safe for filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def kaggle_score(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute the repository score, 1 - MAE / 4, on labels 0..4."""
    return 1.0 - np.abs(preds - labels).mean() / 4.0


def median_decode_probs_torch(probs: torch.Tensor) -> torch.Tensor:
    """Decode probabilities with the ordinal CDF median rule."""
    cdf = torch.cumsum(probs, dim=1)
    return (cdf < 0.5).sum(dim=1).clamp(0, probs.shape[1] - 1).long()


def median_decode_probs_np(probs: np.ndarray) -> np.ndarray:
    """Numpy version of ordinal CDF median decoding."""
    cdf = np.cumsum(probs, axis=1)
    return np.sum(cdf < 0.5, axis=1).clip(0, probs.shape[1] - 1).astype(np.int64)


def metrics_for_predictions(preds: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    """Return validation metrics for one prediction vector."""
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
    """Select a deterministic class-balanced subset or return all indices."""
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


def evaluate_prediction_dict(
    predictions: Dict[str, np.ndarray],
    labels: np.ndarray,
) -> Dict[str, Dict[str, object]]:
    """Compute validation metrics for every named prediction vector."""
    labels = np.asarray(labels, dtype=np.int64)
    return {
        name: metrics_for_predictions(np.asarray(preds, dtype=np.int64), labels)
        for name, preds in predictions.items()
    }


def write_confusion_csvs(metrics: Dict[str, Dict[str, object]], out_dir: Path) -> None:
    """Write one confusion matrix CSV for each evaluated prediction method."""
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = list(range(NUM_CLASSES))
    for name, values in metrics.items():
        cm = np.asarray(values["confusion_matrix"], dtype=np.int64)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(
            out_dir / f"confusion_{sanitize_name(name)}.csv"
        )


@torch.no_grad()
def encode_contrastive_embeddings(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a loader into normalized contrastive embeddings and optional labels."""
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


@torch.no_grad()
def encode_mixed_outputs(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Encode a mixed model into embeddings, optional labels, and classifier probabilities."""
    model.eval()
    all_embeddings: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_probs: List[np.ndarray] = []
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad()

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast_ctx:
            outputs = model(input_ids, attention_mask)
        all_embeddings.append(outputs["embeddings"].float().cpu())
        all_probs.append(F.softmax(outputs["logits"].float(), dim=1).cpu().numpy())
        if "labels" in batch:
            all_labels.append(batch["labels"].long().cpu())

    embeddings = F.normalize(torch.cat(all_embeddings, dim=0), p=2, dim=-1)
    labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
    probs = np.concatenate(all_probs, axis=0)
    return embeddings, labels, probs


def knn_predictions_and_distributions(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_query: torch.Tensor,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    tau: float = 0.07,
    chunk_size: int = 512,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Run all kNN decoders and return both labels and probability distributions."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_values = tuple(int(k) for k in k_values)
    z_train = F.normalize(z_train.float(), p=2, dim=-1).to(device)
    y_train = y_train.long().to(device)
    z_query = F.normalize(z_query.float(), p=2, dim=-1)
    max_k = min(max(k_values), z_train.size(0))

    pred_parts: Dict[str, List[torch.Tensor]] = {}
    dist_parts: Dict[str, List[np.ndarray]] = {}
    for k in k_values:
        k_eff = min(k, z_train.size(0))
        for suffix in ("majority", "weighted_argmax", "weighted_median", "label_median"):
            pred_parts[f"knn_k{k_eff}_{suffix}"] = []
        dist_parts[f"knn_k{k_eff}_weighted"] = []

    for start in range(0, z_query.size(0), chunk_size):
        chunk = z_query[start:start + chunk_size].to(device)
        sims = chunk @ z_train.T
        top_sim, top_idx = sims.topk(k=max_k, dim=1)
        top_labels = y_train[top_idx]

        for k in k_values:
            k_eff = min(k, z_train.size(0))
            labels_k = top_labels[:, :k_eff]
            sims_k = top_sim[:, :k_eff]
            one_hot = F.one_hot(labels_k, num_classes=NUM_CLASSES).float()

            counts = one_hot.sum(dim=1)
            pred_parts[f"knn_k{k_eff}_majority"].append(counts.argmax(dim=1).cpu())

            label_median = labels_k.float().median(dim=1).values.long()
            pred_parts[f"knn_k{k_eff}_label_median"].append(label_median.cpu())

            weights = torch.softmax(sims_k / tau, dim=1).unsqueeze(-1)
            probs = (weights * one_hot).sum(dim=1)
            pred_parts[f"knn_k{k_eff}_weighted_argmax"].append(probs.argmax(dim=1).cpu())
            pred_parts[f"knn_k{k_eff}_weighted_median"].append(median_decode_probs_torch(probs).cpu())
            dist_parts[f"knn_k{k_eff}_weighted"].append(probs.cpu().numpy())

    predictions = {name: torch.cat(parts).numpy().astype(np.int64) for name, parts in pred_parts.items()}
    distributions = {name: np.concatenate(parts, axis=0).astype(np.float32) for name, parts in dist_parts.items()}
    return predictions, distributions


def compute_class_medoids(z_train: torch.Tensor, y_train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Choose the training point closest to each normalized class centroid."""
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
        raise ValueError("Cannot compute medoids without at least one labeled example.")
    return torch.stack(medoids, dim=0), torch.tensor(labels, dtype=torch.long)


def medoid_predictions_and_distributions(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_query: torch.Tensor,
    tau: float = 0.07,
    chunk_size: int = 2048,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Run nearest-medoid and medoid-probability decoders."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    medoids, medoid_labels = compute_class_medoids(z_train, y_train)
    medoids = medoids.to(device)
    medoid_labels = medoid_labels.to(device)
    z_query = F.normalize(z_query.float(), p=2, dim=-1)

    nearest_parts: List[torch.Tensor] = []
    argmax_parts: List[torch.Tensor] = []
    median_parts: List[torch.Tensor] = []
    prob_parts: List[np.ndarray] = []

    for start in range(0, z_query.size(0), chunk_size):
        chunk = z_query[start:start + chunk_size].to(device)
        sims = chunk @ medoids.T
        nearest_parts.append(medoid_labels[sims.argmax(dim=1)].cpu())

        sparse_probs = torch.softmax(sims / tau, dim=1)
        probs = torch.zeros((chunk.size(0), NUM_CLASSES), device=device)
        probs[:, medoid_labels] = sparse_probs
        argmax_parts.append(probs.argmax(dim=1).cpu())
        median_parts.append(median_decode_probs_torch(probs).cpu())
        prob_parts.append(probs.cpu().numpy())

    predictions = {
        "medoid_nearest": torch.cat(nearest_parts).numpy().astype(np.int64),
        "medoid_distribution_argmax": torch.cat(argmax_parts).numpy().astype(np.int64),
        "medoid_distribution_median": torch.cat(median_parts).numpy().astype(np.int64),
    }
    distributions = {
        "medoid_distribution": np.concatenate(prob_parts, axis=0).astype(np.float32),
    }
    return predictions, distributions


def retrieval_predictions_and_distributions(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_query: torch.Tensor,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    tau: float = 0.07,
    chunk_size: int = 512,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Run all retrieval methods and return predictions plus reusable distributions."""
    predictions: Dict[str, np.ndarray] = {}
    distributions: Dict[str, np.ndarray] = {}
    knn_preds, knn_dists = knn_predictions_and_distributions(
        z_train, y_train, z_query, k_values, tau, chunk_size, device
    )
    medoid_preds, medoid_dists = medoid_predictions_and_distributions(
        z_train, y_train, z_query, tau, device=device
    )
    predictions.update(knn_preds)
    predictions.update(medoid_preds)
    distributions.update(knn_dists)
    distributions.update(medoid_dists)
    return predictions, distributions


def classification_predictions(class_probs: np.ndarray) -> Dict[str, np.ndarray]:
    """Decode classifier probabilities with argmax and ordinal median rules."""
    return {
        "classification_argmax": np.asarray(class_probs).argmax(axis=1).astype(np.int64),
        "classification_median": median_decode_probs_np(np.asarray(class_probs)),
    }


def _round_half_up(values: np.ndarray) -> np.ndarray:
    """Round .5 upward for deterministic ordinal midpoint combinations."""
    return np.floor(values + 0.5).astype(np.int64)


def combined_predictions(
    class_probs: np.ndarray,
    class_preds: Dict[str, np.ndarray],
    retrieval_preds: Dict[str, np.ndarray],
    retrieval_distributions: Dict[str, np.ndarray],
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """Combine classifier and retrieval outputs with simple ordinal rules."""
    combined: Dict[str, np.ndarray] = {}
    cls_median = class_preds["classification_median"].astype(np.float32)

    for name, preds in retrieval_preds.items():
        midpoint = _round_half_up((cls_median + preds.astype(np.float32)) / 2.0)
        combined[f"combo_midpoint_cls_median__{name}"] = midpoint.clip(0, NUM_CLASSES - 1)

    class_probs = np.asarray(class_probs, dtype=np.float32)
    for dist_name, ret_probs in retrieval_distributions.items():
        ret_probs = np.asarray(ret_probs, dtype=np.float32)
        ret_decode = f"{dist_name}_median"
        for alpha in alphas:
            alpha_tag = f"a{int(round(alpha * 100)):03d}"
            mix = alpha * class_probs + (1.0 - alpha) * ret_probs
            mix = mix / np.maximum(mix.sum(axis=1, keepdims=True), eps)
            combined[f"combo_probmix_{alpha_tag}__{ret_decode}"] = median_decode_probs_np(mix)

            poe = np.power(class_probs + eps, alpha) * np.power(ret_probs + eps, 1.0 - alpha)
            poe = poe / np.maximum(poe.sum(axis=1, keepdims=True), eps)
            combined[f"combo_poe_{alpha_tag}__{ret_decode}"] = median_decode_probs_np(poe)

    return combined


def write_prediction_wide_csv(
    path: Path,
    ids: Sequence[int],
    predictions: Dict[str, np.ndarray],
    labels: Optional[Sequence[int]] = None,
) -> None:
    """Write one wide CSV containing ids, optional labels, and all predictions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"id": list(ids)}
    if labels is not None:
        payload["label_true"] = list(labels)
    for name, preds in sorted(predictions.items()):
        payload[name] = np.asarray(preds, dtype=np.int64)
    pd.DataFrame(payload).to_csv(path, index=False)


def write_submission_files(
    ids: Sequence[int],
    predictions: Dict[str, np.ndarray],
    output_dir: Path,
    prefix: str,
    selected_methods: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Write Kaggle submission CSVs for selected prediction methods."""
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = selected_methods or sorted(predictions.keys())
    paths: List[Path] = []
    for method in methods:
        if method not in predictions:
            continue
        out_path = output_dir / f"{sanitize_name(prefix)}__{sanitize_name(method)}_submission.csv"
        labels = np.asarray(predictions[method], dtype=np.int64).clip(0, NUM_CLASSES - 1)
        pd.DataFrame({"id": list(ids), "label": labels}).to_csv(out_path, index=False)
        paths.append(out_path)
    return paths


def pairwise_disagreement(
    predictions: Dict[str, np.ndarray],
    labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Measure pairwise disagreement and absolute prediction differences."""
    names = sorted(predictions.keys())
    rows: List[Dict[str, object]] = []
    labels_arr = None if labels is None else np.asarray(labels, dtype=np.int64)
    for i, left_name in enumerate(names):
        left = np.asarray(predictions[left_name], dtype=np.int64)
        for right_name in names[i + 1:]:
            right = np.asarray(predictions[right_name], dtype=np.int64)
            diff = np.abs(left - right)
            disagree = diff > 0
            row: Dict[str, object] = {
                "method_a": left_name,
                "method_b": right_name,
                "n": int(len(left)),
                "disagreement_rate": float(disagree.mean()),
                "mean_abs_diff": float(diff.mean()),
            }
            for d in range(NUM_CLASSES):
                row[f"abs_diff_{d}"] = int(np.sum(diff == d))
            if labels_arr is not None:
                agree = ~disagree
                if agree.any():
                    row["agree_accuracy"] = float((left[agree] == labels_arr[agree]).mean())
                if disagree.any():
                    row["method_a_disagree_accuracy"] = float((left[disagree] == labels_arr[disagree]).mean())
                    row["method_b_disagree_accuracy"] = float((right[disagree] == labels_arr[disagree]).mean())
                    row["either_correct_when_disagree"] = float(
                        ((left[disagree] == labels_arr[disagree]) | (right[disagree] == labels_arr[disagree])).mean()
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def write_disagreement_analysis(
    predictions: Dict[str, np.ndarray],
    out_dir: Path,
    labels: Optional[np.ndarray] = None,
    prefix: str = "val",
    metrics: Optional[Dict[str, Dict[str, object]]] = None,
) -> Path:
    """Save pairwise disagreement CSV and optional paper-friendly plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    disagreement = pairwise_disagreement(predictions, labels)
    csv_path = out_dir / f"{prefix}_pairwise_disagreement.csv"
    disagreement.to_csv(csv_path, index=False)

    plots_dir = out_dir / "plots"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return csv_path

    names = sorted(predictions.keys())
    index = {name: i for i, name in enumerate(names)}
    for value_col, title, filename in [
        ("disagreement_rate", "Pairwise Disagreement Rate", f"{prefix}_disagreement_rate.png"),
        ("mean_abs_diff", "Pairwise Mean Absolute Difference", f"{prefix}_mean_abs_diff.png"),
    ]:
        matrix = np.zeros((len(names), len(names)), dtype=np.float32)
        for _, row in disagreement.iterrows():
            i = index[row["method_a"]]
            j = index[row["method_b"]]
            matrix[i, j] = matrix[j, i] = float(row[value_col])
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.25), max(6, len(names) * 0.25)))
        im = ax.imshow(matrix, cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_yticklabels(names, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=180)
        plt.close(fig)

    if metrics:
        score_rows = [
            {"method": name, "score": values["score"], "mae": values["mae"], "qwk": values["qwk"]}
            for name, values in sorted(metrics.items())
        ]
        if score_rows:
            score_df = pd.DataFrame(score_rows).sort_values("score", ascending=False)
            fig_height = max(5, min(20, 0.22 * len(score_df)))
            fig, ax = plt.subplots(figsize=(10, fig_height))
            ax.barh(score_df["method"], score_df["score"])
            ax.invert_yaxis()
            ax.set_xlabel("Validation score")
            ax.set_title("Validation Scores By Prediction Method")
            ax.tick_params(axis="y", labelsize=7)
            fig.tight_layout()
            fig.savefig(plots_dir / f"{prefix}_method_scores.png", dpi=180)
            plt.close(fig)

    return csv_path


def save_json(path: Path, payload: Dict[str, object]) -> None:
    """Save a JSON payload with numpy-aware conversion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)
