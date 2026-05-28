"""Ensemble all fold checkpoints, average softmax probs, median decode."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import ReviewDataset, read_csv
from model import mDeBERTaEMD

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
_DEFAULT_OUTPUT_DIR = ROOT / "submissions"


@torch.no_grad()
def predict_probs(model, loader, device) -> np.ndarray:
    model.eval()
    all_probs = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask)
        all_probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(all_probs)


def median_decode_np(avg_probs: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(avg_probs, axis=1)[:, :-1]
    return (cdf < 0.5).sum(axis=1).clip(0, avg_probs.shape[1] - 1)


def main(args):
    data_dir     = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    output_dir   = Path(args.output_dir)
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir / "tokenizer"))
    texts, _, ids = read_csv(data_dir / "test.csv")

    # Discover fold checkpoints (optionally restricted to a single fold)
    ckpt_paths = sorted(artifact_dir.glob("best_model_fold*.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No fold checkpoints found in {artifact_dir}")
    if args.fold is not None:
        ckpt_paths = [p for p in ckpt_paths if p.stem == f"best_model_fold{args.fold}"]
        if not ckpt_paths:
            raise FileNotFoundError(f"Checkpoint for fold {args.fold} not found in {artifact_dir}")
    print(f"Found {len(ckpt_paths)} fold checkpoint(s)")

    first_ckpt = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
    max_len    = first_ckpt["args"]["max_len"]
    backbone   = first_ckpt.get("backbone_dir", "microsoft/mdeberta-v3-base")

    test_dataset = ReviewDataset(texts, tokenizer, max_len=max_len)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    all_probs = []
    for ckpt_path in ckpt_paths:
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = mDeBERTaEMD(model_name=backbone, dropout=0.0).to(device)
        model.load_state_dict(ckpt["model"])
        fold  = ckpt.get("fold", ckpt_path.stem)
        print(f"Loaded {ckpt_path.name} (fold={fold})")
        all_probs.append(predict_probs(model, test_loader, device))
        del model
        torch.cuda.empty_cache()

    avg_probs = np.mean(all_probs, axis=0)
    preds     = median_decode_np(avg_probs)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"26_mdeberta_kfold_{len(ckpt_paths)}fold_submission.csv"
    pd.DataFrame({"id": ids, "label": preds}).to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--data_dir",     default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--output_dir",   default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fold",         type=int, default=None,
                        help="If set, run only this fold's checkpoint (e.g. --fold 3)")
    main(parser.parse_args())
