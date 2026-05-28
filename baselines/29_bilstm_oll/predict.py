"""Run the trained BiLSTM on test.csv and write a Kaggle submission CSV."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import ReviewDataset, load_vocab, read_csv
from model import BiLSTMClassifier, median_decode

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
_DEFAULT_OUTPUT_DIR = ROOT / "submissions"


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    all_preds = []
    for batch in loader:
        x, lengths = batch[0].to(device), batch[1].to(device)
        all_preds.append(median_decode(model(x, lengths)).cpu().numpy())
    return np.concatenate(all_preds)


def local_score(preds: np.ndarray, truth_path: Path) -> float:
    truth = pd.read_csv(truth_path)
    mae = np.abs(preds - truth["label"].values).mean()
    return 1.0 - mae / 4.0


def main(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    vocab = load_vocab(args.vocab)
    ckpt_args = checkpoint["args"]

    model = BiLSTMClassifier(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=ckpt_args["embed_dim"],
        hidden_dim=ckpt_args["hidden_dim"],
        num_layers=ckpt_args["num_layers"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded model from {args.checkpoint}")

    texts, _, ids = read_csv(data_dir / "test.csv")
    dataset = ReviewDataset(texts, vocab, max_len=ckpt_args["max_len"])
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2)

    preds = predict(model, loader, device)

    solved_path = data_dir / "test_solved.csv"
    if solved_path.exists():
        score = local_score(preds, solved_path)
        print(f"Local test score (vs test_solved.csv): {score:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "29_bilstm_oll_submission.csv"
    pd.DataFrame({"id": ids, "label": preds}).to_csv(out_path, index=False)
    print(f"Kaggle submission saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(_DEFAULT_ARTIFACT_DIR / "best_model.pt"))
    parser.add_argument("--vocab",      default=str(_DEFAULT_ARTIFACT_DIR / "vocab.json"))
    parser.add_argument("--data_dir",   default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--output_dir", default=str(_DEFAULT_OUTPUT_DIR))
    main(parser.parse_args())
