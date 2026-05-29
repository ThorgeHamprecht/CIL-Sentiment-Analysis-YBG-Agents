"""Evaluate shared-head checkpoints with classifier, retrieval, and blends."""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baselines.contrastive_eval_utils import (  # noqa: E402
    DEFAULT_ALPHAS,
    DEFAULT_K_VALUES,
    classification_predictions,
    combined_predictions,
    encode_mixed_outputs,
    evaluate_prediction_dict,
    retrieval_predictions_and_distributions,
    sample_per_class_indices,
    save_json,
    write_confusion_csvs,
    write_disagreement_analysis,
    write_prediction_wide_csv,
    write_submission_files,
)
from dataset import ReviewDataset, read_csv  # noqa: E402
from model import SharedHeadContrastiveMDeBERTa  # noqa: E402

BACKBONE = "microsoft/mdeberta-v3-base"
_SCRATCH = Path("/work/scratch") / os.environ.get("USER", "") / "cil"
_DEFAULT_DATA_DIR = _SCRATCH / "data" if (_SCRATCH / "data").exists() else ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
_DEFAULT_OUTPUT_DIR = _SCRATCH / "submissions" if _SCRATCH.exists() else ROOT / "submissions"


def load_model_from_checkpoint(
    artifact_dir: Path,
    device: torch.device,
    checkpoint_name: str = "best_model.pt",
) -> Tuple[SharedHeadContrastiveMDeBERTa, dict]:
    """Recreate the shared-head model from best_model.pt."""
    ckpt = torch.load(artifact_dir / checkpoint_name, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model = SharedHeadContrastiveMDeBERTa(
        model_name=str(ckpt.get("backbone_dir", BACKBONE)),
        representation_dim=int(ckpt_args.get("representation_dim", 256)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt_args


def build_split_and_test_datasets(data_dir: Path, artifact_dir: Path, split_seed: int, max_len: int, batch_size: int):
    """Rebuild train/val split plus test dataset using the checkpoint tokenizer."""
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
    return train_dataset, train_labels, train_ids, val_dataset, val_labels, val_ids, test_dataset, test_ids


def build_all_predictions(
    class_probs: np.ndarray,
    z_support: torch.Tensor,
    y_support: torch.Tensor,
    z_query: torch.Tensor,
    args,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Create classifier-only, retrieval-only, and mixed prediction methods."""
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
    predictions.update(class_preds)
    predictions.update(retrieval_preds)
    predictions.update(combo_preds)
    return predictions


def main(args):
    """Run full validation analysis and test submission generation."""
    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    analysis_dir = artifact_dir / "analysis"
    predictions_dir = artifact_dir / "predictions"
    embeddings_dir = artifact_dir / "embeddings"
    checkpoint_tag = Path(args.checkpoint_name).stem
    analysis_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for full mDeBERTa mixed evaluation; refusing to run on CPU.")
    print(f"Device: {device}")

    model, ckpt_args = load_model_from_checkpoint(artifact_dir, device, args.checkpoint_name)
    max_len = args.max_len or int(ckpt_args.get("max_len", 256))
    split_seed = args.split_seed if args.split_seed is not None else int(ckpt_args.get("split_seed", ckpt_args.get("seed", 42)))

    (
        train_dataset,
        train_labels,
        _train_ids,
        val_dataset,
        val_labels,
        val_ids,
        test_dataset,
        test_ids,
    ) = build_split_and_test_datasets(data_dir, artifact_dir, split_seed, max_len, args.tokenize_batch_size)

    selected_train = sample_per_class_indices(train_labels, args.max_eval_train_per_class, split_seed)
    retrieval_train_dataset = Subset(train_dataset, selected_train)
    print(f"Retrieval train pool for validation: {len(retrieval_train_dataset):,}")
    print(f"Validation set: {len(val_dataset):,}")
    print(f"Test set: {len(test_dataset):,}")

    train_loader = DataLoader(retrieval_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    z_train, y_train, _ = encode_mixed_outputs(model, train_loader, device)
    z_val, y_val, val_probs = encode_mixed_outputs(model, val_loader, device)
    z_test, _, test_probs = encode_mixed_outputs(model, test_loader, device)

    if args.cache_embeddings:
        torch.save({"embeddings": z_train, "labels": y_train}, embeddings_dir / f"{checkpoint_tag}_train_eval.pt")
        torch.save(
            {"embeddings": z_val, "labels": y_val, "classifier_probs": val_probs},
            embeddings_dir / f"{checkpoint_tag}_val.pt",
        )
        torch.save(
            {"embeddings": z_test, "classifier_probs": test_probs},
            embeddings_dir / f"{checkpoint_tag}_test.pt",
        )

    val_predictions = build_all_predictions(val_probs, z_train, y_train, z_val, args, device)
    val_labels_np = y_val.numpy()
    val_metrics = evaluate_prediction_dict(val_predictions, val_labels_np)
    write_prediction_wide_csv(predictions_dir / f"{checkpoint_tag}_val_predictions.csv", val_ids, val_predictions, labels=val_labels)
    write_confusion_csvs(val_metrics, analysis_dir / f"{checkpoint_tag}_confusions")
    val_disagreement_path = write_disagreement_analysis(
        val_predictions,
        analysis_dir,
        labels=val_labels_np,
        prefix=f"{checkpoint_tag}_val",
        metrics=val_metrics,
    )

    test_support_z = torch.cat([z_train, z_val], dim=0)
    test_support_y = torch.cat([y_train, y_val], dim=0)
    test_predictions = build_all_predictions(test_probs, test_support_z, test_support_y, z_test, args, device)
    write_prediction_wide_csv(predictions_dir / f"{checkpoint_tag}_test_predictions.csv", test_ids, test_predictions)
    test_disagreement_path = write_disagreement_analysis(test_predictions, analysis_dir, prefix=f"{checkpoint_tag}_test")

    prefix = args.submission_prefix or f"{artifact_dir.name}_{checkpoint_tag}"
    selected_methods = args.submission_methods if args.submission_methods else None
    submission_paths = write_submission_files(test_ids, test_predictions, output_dir, prefix, selected_methods)

    payload = {
        "artifact_dir": str(artifact_dir),
        "checkpoint_name": args.checkpoint_name,
        "checkpoint_tag": checkpoint_tag,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "split_seed": split_seed,
        "max_len": max_len,
        "retrieval_tau": args.retrieval_tau,
        "k_values": args.k_values,
        "alphas": args.alphas,
        "max_eval_train_per_class": args.max_eval_train_per_class,
        "n_retrieval_train": len(retrieval_train_dataset),
        "n_val": len(val_dataset),
        "n_test": len(test_dataset),
        "val_metrics": val_metrics,
        "val_disagreement_csv": str(val_disagreement_path),
        "test_disagreement_csv": str(test_disagreement_path),
        "submissions": [str(path) for path in submission_paths],
    }
    save_json(analysis_dir / f"mixed_eval_{checkpoint_tag}.json", payload)

    if "classification_median" in val_metrics:
        cls = val_metrics["classification_median"]
        print(
            f"\nCheckpoint {args.checkpoint_name} classification_median: "
            f"score={cls['score']:.4f} mae={cls['mae']:.4f} qwk={cls['qwk']:.4f}"
        )
    print("Top validation methods:")
    for name, values in sorted(val_metrics.items(), key=lambda item: item[1]["score"], reverse=True)[:20]:
        print(f"  {name:<70} score={values['score']:.4f} mae={values['mae']:.4f} qwk={values['qwk']:.4f}")
    print(f"\nSaved validation/test predictions under {predictions_dir}")
    print(f"Saved {len(submission_paths)} submissions under {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--checkpoint_name", default="best_model.pt")
    parser.add_argument("--data_dir", default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--output_dir", default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--submission_prefix", default=None)
    parser.add_argument("--submission_methods", nargs="+", default=None)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tokenize_batch_size", type=int, default=1024)
    parser.add_argument("--retrieval_tau", type=float, default=0.07)
    parser.add_argument("--k_values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--similarity_chunk_size", type=int, default=512)
    parser.add_argument("--max_eval_train_per_class", type=int, default=0)
    parser.add_argument("--cache_embeddings", action="store_true", default=True)
    parser.add_argument("--require_cuda", action=argparse.BooleanOptionalAction, default=True)
    main(parser.parse_args())
