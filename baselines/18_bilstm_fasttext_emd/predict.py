"""Generate submission CSV from a trained FastText BiLSTM checkpoint."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TwoStreamDataset, load_vocab, read_csv
from model import TwoStreamBiLSTM, median_decode

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
_DEFAULT_SUBMISSION_DIR = ROOT / "submissions"


def kaggle_score(preds, labels):
    return 1.0 - np.abs(preds - labels).mean() / 4.0


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = Path(args.artifact_dir)

    ckpt = torch.load(artifact_dir / "best_model.pt", map_location=device)
    saved_args = ckpt["args"]
    vocab = load_vocab(artifact_dir / "vocab.json")

    model = TwoStreamBiLSTM(
        vocab_size=len(vocab),
        embed_dim=saved_args["embed_dim"],
        hidden_dim=saved_args["hidden_dim"],
        num_layers=saved_args["num_layers"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data_dir = Path(args.data_dir)
    _, titles, bodies, _, ids = read_csv(data_dir / "test.csv")
    ds = TwoStreamDataset(titles, bodies, vocab, saved_args["max_len_title"], saved_args["max_len_body"])
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    all_preds = []
    with torch.no_grad():
        for x_t, l_t, x_b, l_b in loader:
            x_t, l_t, x_b, l_b = x_t.to(device), l_t.to(device), x_b.to(device), l_b.to(device)
            main_logits, _ = model(x_t, l_t, x_b, l_b)
            all_preds.append(median_decode(main_logits).cpu().numpy())
    preds = np.concatenate(all_preds)

    sub_dir = Path(args.submission_dir)
    sub_dir.mkdir(parents=True, exist_ok=True)
    out = sub_dir / "submission_18_bilstm_fasttext_emd.csv"
    pd.DataFrame({"id": ids, "label": preds}).to_csv(out, index=False)
    print(f"Saved: {out}")

    solved = data_dir / "test_solved.csv"
    if solved.exists():
        truth = pd.read_csv(solved)["label"].values
        print(f"Local score: {kaggle_score(preds, truth):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir",   default=str(_DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--data_dir",       default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--submission_dir", default=str(_DEFAULT_SUBMISSION_DIR))
    main(parser.parse_args())
