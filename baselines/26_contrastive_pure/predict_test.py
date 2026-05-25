"""Create test-set retrieval submissions for a pure contrastive checkpoint."""
import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baselines.contrastive_eval_utils import (  # noqa: E402
    DEFAULT_K_VALUES,
    encode_contrastive_embeddings,
    retrieval_predictions_and_distributions,
    save_json,
    write_disagreement_analysis,
    write_prediction_wide_csv,
    write_submission_files,
)
from dataset import ReviewDataset, read_csv  # noqa: E402
from eval_retrieval import load_model_from_checkpoint  # noqa: E402

_SCRATCH = Path("/work/scratch") / os.environ.get("USER", "") / "cil"
_DEFAULT_DATA_DIR = _SCRATCH / "data" if (_SCRATCH / "data").exists() else ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
_DEFAULT_OUTPUT_DIR = _SCRATCH / "submissions" if _SCRATCH.exists() else ROOT / "submissions"


def main(args):
    """Encode train/test reviews and write one submission per retrieval decoder."""
    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    predictions_dir = artifact_dir / "predictions"
    analysis_dir = artifact_dir / "analysis"
    embeddings_dir = artifact_dir / "embeddings"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for full mDeBERTa test prediction; refusing to run on CPU.")
    print(f"Device: {device}")

    model, ckpt_args = load_model_from_checkpoint(artifact_dir, device)
    max_len = args.max_len or int(ckpt_args.get("max_len", 256))
    tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir / "tokenizer"), use_fast=False)

    train_texts, train_labels, _ = read_csv(data_dir / "train.csv")
    test_texts, _, test_ids = read_csv(data_dir / "test.csv")
    print(f"Retrieval train pool for test: {len(train_texts):,} labeled examples")
    print(f"Test set: {len(test_texts):,} examples")

    train_dataset = ReviewDataset(
        train_texts,
        tokenizer,
        max_len=max_len,
        labels=train_labels,
        show_progress=not args.no_progress,
        batch_size=args.tokenize_batch_size,
    )
    test_dataset = ReviewDataset(
        test_texts,
        tokenizer,
        max_len=max_len,
        show_progress=not args.no_progress,
        batch_size=args.tokenize_batch_size,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    z_train, y_train = encode_contrastive_embeddings(model, train_loader, device)
    z_test, _ = encode_contrastive_embeddings(model, test_loader, device)

    if args.cache_embeddings:
        torch.save({"embeddings": z_train, "labels": y_train}, embeddings_dir / "train_for_test.pt")
        torch.save({"embeddings": z_test}, embeddings_dir / "test.pt")

    predictions, _ = retrieval_predictions_and_distributions(
        z_train=z_train,
        y_train=y_train,
        z_query=z_test,
        k_values=args.k_values,
        tau=args.retrieval_tau,
        chunk_size=args.similarity_chunk_size,
        device=device,
    )

    prefix = args.submission_prefix or artifact_dir.name
    wide_path = predictions_dir / "test_predictions.csv"
    write_prediction_wide_csv(wide_path, test_ids, predictions)
    submission_paths = write_submission_files(test_ids, predictions, output_dir, prefix)
    disagreement_path = write_disagreement_analysis(predictions, analysis_dir, prefix="test")

    payload = {
        "artifact_dir": str(artifact_dir),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "max_len": max_len,
        "retrieval_tau": args.retrieval_tau,
        "k_values": args.k_values,
        "n_train_pool": len(train_dataset),
        "n_test": len(test_dataset),
        "wide_predictions": str(wide_path),
        "submissions": [str(path) for path in submission_paths],
        "disagreement_csv": str(disagreement_path),
    }
    out_path = analysis_dir / "test_retrieval_predictions.json"
    save_json(out_path, payload)

    print(f"Saved wide test predictions to {wide_path}")
    print(f"Saved {len(submission_paths)} submission files to {output_dir}")
    print(f"Saved test prediction metadata to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--data_dir", default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--output_dir", default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--submission_prefix", default=None)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tokenize_batch_size", type=int, default=1024)
    parser.add_argument("--retrieval_tau", type=float, default=0.07)
    parser.add_argument("--k_values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--similarity_chunk_size", type=int, default=512)
    parser.add_argument("--cache_embeddings", action="store_true", default=True)
    parser.add_argument("--require_cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no_progress", action="store_true")
    main(parser.parse_args())
