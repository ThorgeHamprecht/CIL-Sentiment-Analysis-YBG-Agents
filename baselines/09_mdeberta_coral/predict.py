"""Run trained mDeBERTa-CORAL on test.csv and write a Kaggle submission CSV."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import ReviewDataset, read_csv
from model import mDeBERTaCORAL

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
_DEFAULT_OUTPUT_DIR = ROOT / "submissions"


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    all_preds = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask)
        all_preds.append(model.predict(logits).cpu().numpy())
    return np.concatenate(all_preds)


def local_score(preds: np.ndarray, truth_path: Path) -> float:
    truth = pd.read_csv(truth_path)
    mae = np.abs(preds - truth["label"].values).mean()
    return 1.0 - mae / 4.0


def main(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint["args"]
    model_name = checkpoint.get("model_name", "microsoft/mdeberta-v3-base")

    tokenizer_path = Path(args.artifact_dir) / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path) if tokenizer_path.exists() else model_name)

    model = mDeBERTaCORAL(model_name=model_name, dropout=0.0).to(device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded model from {args.checkpoint}")

    texts, _, ids = read_csv(data_dir / "test.csv")
    dataset = ReviewDataset(texts, tokenizer, max_len=ckpt_args["max_len"])
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    preds = predict(model, loader, device)

    solved_path = data_dir / "test_solved.csv"
    if solved_path.exists():
        score = local_score(preds, solved_path)
        print(f"Local test score (vs test_solved.csv): {score:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "09_mdeberta_coral_submission.csv"
    pd.DataFrame({"id": ids, "label": preds}).to_csv(out_path, index=False)
    print(f"Kaggle submission saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   default=str(_DEFAULT_ARTIFACT_DIR / "best_model.pt"))
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--data_dir",     default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--output_dir",   default=str(_DEFAULT_OUTPUT_DIR))
    main(parser.parse_args())
