"""Write test submissions for the folder 29 separate-model ensemble."""
import argparse
import gc
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import ReviewDataset, read_csv
from ensemble import (
    DEFAULT_TAUS,
    ENSEMBLE_STRATEGIES,
    build_test_predictions_for_taus,
    class_prior,
    encode_classifier_probs,
    encode_contrastive_embeddings,
    sanitize_name,
    save_json,
    tau_tag,
    write_prediction_wide_csv,
    write_submission_file,
)
from model import MDeBERTaEMD, PureContrastiveMDeBERTa

ROOT = Path(__file__).resolve().parents[2]
BACKBONE = "microsoft/mdeberta-v3-base"
_SCRATCH = Path("/work/scratch") / os.environ.get("USER", "") / "cil"
_DEFAULT_DATA_DIR = _SCRATCH / "data" if (_SCRATCH / "data").exists() else ROOT / "data"
_DEFAULT_ARTIFACT_DIR = _SCRATCH / "artifacts" / "29_separate_ensemble" if _SCRATCH.exists() else Path(__file__).parent / "artifacts"
_DEFAULT_OUTPUT_DIR = _SCRATCH / "submissions" if _SCRATCH.exists() else ROOT / "submissions"


def load_classifier(path: Path, device: torch.device) -> MDeBERTaEMD:
    """Load one saved EMA classifier epoch checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model = MDeBERTaEMD(
        model_name=str(ckpt.get("backbone_dir", BACKBONE)),
        dropout=float(ckpt_args.get("classifier_dropout", 0.25)),
        dropout_samples=int(ckpt_args.get("msd_samples", 5)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def load_contrastive(path: Path, device: torch.device) -> PureContrastiveMDeBERTa:
    """Load one saved EMA contrastive epoch checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model = PureContrastiveMDeBERTa(
        model_name=str(ckpt.get("backbone_dir", BACKBONE)),
        projection_dim=int(ckpt_args.get("projection_dim", 128)),
        dropout=float(ckpt_args.get("contrastive_dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def load_validation_specs(summary_path: Path, final_epoch: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return validation-best and final-epoch ensemble specs."""
    df = pd.read_csv(summary_path)
    ensemble_df = df[df["strategy"].isin(ENSEMBLE_STRATEGIES)].copy()
    if ensemble_df.empty:
        raise ValueError(f"No ensemble rows found in {summary_path}")

    best_rows = []
    for (_variant, _tau, _strategy), group in ensemble_df.groupby(["variant", "tau", "strategy"], dropna=False):
        best_rows.append(group.sort_values("score", ascending=False).iloc[0].to_dict())
    best_df = pd.DataFrame(best_rows)
    best_df["selection"] = "best"

    final_df = ensemble_df[ensemble_df["epoch"] == final_epoch].copy()
    final_df["selection"] = f"epoch{final_epoch:03d}"
    return best_df, final_df


def build_datasets(args, tokenizer):
    """Tokenize full train.csv as medoid support and test.csv for submissions."""
    data_dir = Path(args.data_dir)
    train_texts, train_labels, _train_ids = read_csv(data_dir / "train.csv")
    test_texts, _test_labels, test_ids = read_csv(data_dir / "test.csv")
    print(f"Full medoid support: {len(train_texts):,} labeled examples")
    print(f"Test set: {len(test_texts):,} examples")

    support_dataset = ReviewDataset(
        train_texts,
        tokenizer,
        max_len=args.max_len,
        labels=train_labels,
        show_progress=not args.no_progress,
        batch_size=args.tokenize_batch_size,
    )
    test_dataset = ReviewDataset(
        test_texts,
        tokenizer,
        max_len=args.max_len,
        show_progress=not args.no_progress,
        batch_size=args.tokenize_batch_size,
    )
    return support_dataset, train_labels, test_dataset, test_ids


def classifier_probs_for_epoch(
    epoch: int,
    artifact_dir: Path,
    test_loader: DataLoader,
    args,
    device: torch.device,
    cache: Dict[int, np.ndarray],
) -> np.ndarray:
    """Load or compute test classifier probabilities for one epoch."""
    if epoch in cache:
        return cache[epoch]
    cache_path = artifact_dir / "embeddings" / f"classifier_epoch_{epoch:03d}_test_probs.pt"
    if cache_path.exists() and not args.recompute:
        cache[epoch] = torch.load(cache_path, map_location="cpu", weights_only=False)["classifier_probs"]
        return cache[epoch]

    model = load_classifier(artifact_dir / "classifier" / f"epoch_{epoch:03d}_model.pt", device)
    probs, _ = encode_classifier_probs(model, test_loader, device)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"classifier_probs": probs}, cache_path)
    cache[epoch] = probs
    return probs


def contrastive_predictions_for_epoch(
    variant: str,
    epoch: int,
    artifact_dir: Path,
    support_loader: DataLoader,
    test_loader: DataLoader,
    train_labels: List[int],
    args,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load or compute full-support and test embeddings for one contrastive checkpoint."""
    cache_path = artifact_dir / "embeddings" / f"{variant}_epoch_{epoch:03d}_full_train_test_embeddings.pt"
    if cache_path.exists() and not args.recompute:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return payload["support_embeddings"], payload["support_labels"], payload["test_embeddings"]

    model = load_contrastive(artifact_dir / f"contrastive_{variant}" / f"epoch_{epoch:03d}_model.pt", device)
    support_embeddings, support_labels = encode_contrastive_embeddings(model, support_loader, device)
    test_embeddings, _ = encode_contrastive_embeddings(model, test_loader, device)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if support_labels.numel() == 0:
        support_labels = torch.tensor(train_labels, dtype=torch.long)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "support_embeddings": support_embeddings,
            "support_labels": support_labels,
            "test_embeddings": test_embeddings,
        },
        cache_path,
    )
    return support_embeddings, support_labels, test_embeddings


def write_group_outputs(
    selection: str,
    variant: str,
    epoch: int,
    group_specs: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    test_ids,
    artifact_dir: Path,
    output_dir: Path,
) -> List[Dict[str, object]]:
    """Write selected submission files and one wide test prediction file."""
    manifest_rows: List[Dict[str, object]] = []
    prediction_dir = artifact_dir / "predictions"
    write_prediction_wide_csv(
        prediction_dir / f"{variant}_{selection}_epoch_{epoch:03d}_test_predictions.csv",
        test_ids,
        predictions,
    )

    for _, row in group_specs.iterrows():
        method = f"{tau_tag(float(row['tau']))}__{row['strategy']}"
        if method not in predictions:
            continue
        prefix = f"29_separate_ensemble_{variant}_{selection}_{method}"
        out_path = output_dir / f"{sanitize_name(prefix)}_submission.csv"
        write_submission_file(out_path, test_ids, predictions[method])
        manifest_rows.append(
            {
                "selection": selection,
                "variant": variant,
                "epoch": int(epoch),
                "tau": float(row["tau"]),
                "strategy": row["strategy"],
                "validation_score": float(row["score"]),
                "submission": str(out_path),
            }
        )
    return manifest_rows


def main(args):
    """Generate validation-best and final-epoch test submissions."""
    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for full mDeBERTa prediction.")

    tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir / "tokenizer"), use_fast=False)
    support_dataset, train_labels, test_dataset, test_ids = build_datasets(args, tokenizer)
    support_loader = DataLoader(support_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    prior = class_prior(train_labels)

    best_df, final_df = load_validation_specs(artifact_dir / "analysis" / "validation_summary.csv", args.final_epoch)
    specs = pd.concat([best_df, final_df], ignore_index=True)
    specs.to_csv(artifact_dir / "analysis" / "test_submission_specs.csv", index=False)

    classifier_cache: Dict[int, np.ndarray] = {}
    manifest_rows: List[Dict[str, object]] = []

    for (selection, variant, epoch), group in specs.groupby(["selection", "variant", "epoch"], sort=False):
        epoch = int(epoch)
        print(f"Writing test predictions for {variant} {selection} epoch {epoch}")
        class_probs = classifier_probs_for_epoch(epoch, artifact_dir, test_loader, args, device, classifier_cache)
        z_support, y_support, z_test = contrastive_predictions_for_epoch(
            variant,
            epoch,
            artifact_dir,
            support_loader,
            test_loader,
            train_labels,
            args,
            device,
        )
        predictions = build_test_predictions_for_taus(
            class_probs=class_probs,
            z_support=z_support,
            y_support=y_support,
            z_test=z_test,
            taus=args.retrieval_taus,
            prior=prior,
            device=device,
        )
        manifest_rows.extend(
            write_group_outputs(
                selection=str(selection),
                variant=str(variant),
                epoch=epoch,
                group_specs=group,
                predictions=predictions,
                test_ids=test_ids,
                artifact_dir=artifact_dir,
                output_dir=output_dir,
            )
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = artifact_dir / "analysis" / "test_submission_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    save_json(
        artifact_dir / "analysis" / "test_submission_manifest.json",
        {
            "n_submissions": len(manifest_rows),
            "manifest_csv": str(manifest_path),
            "output_dir": str(output_dir),
            "retrieval_taus": args.retrieval_taus,
        },
    )
    print(f"Saved {len(manifest_rows)} submissions to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--data_dir", default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--output_dir", default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tokenize_batch_size", type=int, default=1024)
    parser.add_argument("--retrieval_taus", type=float, nargs="+", default=list(DEFAULT_TAUS))
    parser.add_argument("--final_epoch", type=int, default=4)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--require_cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no_progress", action="store_true")
    main(parser.parse_args())
