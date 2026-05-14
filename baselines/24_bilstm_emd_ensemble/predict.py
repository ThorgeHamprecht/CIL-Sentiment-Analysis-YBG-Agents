"""Ensemble 3 seed checkpoints, average softmax probs, median decode."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import TwoStreamDataset, load_vocab, read_csv
from model import TwoStreamBiLSTM

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
_DEFAULT_OUTPUT_DIR = ROOT / "submissions"
DEFAULT_SEEDS = [42, 1337, 2024]


@torch.no_grad()
def predict_probs(model, loader, device) -> np.ndarray:
    model.eval()
    all_probs = []
    for batch in loader:
        x_t, l_t, x_b, l_b = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        main_logits, _ = model(x_t, l_t, x_b, l_b)
        all_probs.append(F.softmax(main_logits.float(), dim=1).cpu().numpy())
    return np.concatenate(all_probs)


def median_decode_np(avg_probs: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(avg_probs, axis=1)[:, :-1]
    return (cdf < 0.5).sum(axis=1).clip(0, avg_probs.shape[1] - 1)


def main(args):
    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    seeds = [int(s) for s in args.seeds]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = load_vocab(artifact_dir / "vocab.json")
    _, titles, bodies, _, ids = read_csv(data_dir / "test.csv")

    first_ckpt = torch.load(artifact_dir / f"best_model_seed{seeds[0]}.pt", map_location="cpu", weights_only=False)
    ckpt_args = first_ckpt["args"]

    test_ds = TwoStreamDataset(titles, bodies, vocab, ckpt_args["max_len_title"], ckpt_args["max_len_body"])
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    all_probs = []
    for seed in seeds:
        ckpt_path = artifact_dir / f"best_model_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"Warning: checkpoint for seed {seed} not found, skipping.")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = TwoStreamBiLSTM(
            vocab_size=ckpt["vocab_size"],
            embed_dim=ckpt_args["embed_dim"],
            hidden_dim=ckpt_args["hidden_dim"],
            num_layers=ckpt_args["num_layers"],
            dropout=0.0,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded seed {seed} checkpoint")
        all_probs.append(predict_probs(model, test_loader, device))
        del model

    avg_probs = np.mean(all_probs, axis=0)
    preds = median_decode_np(avg_probs)

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = "_".join(str(s) for s in seeds)
    out_path = output_dir / f"24_bilstm_emd_ensemble_seed{seed_tag}_submission.csv"
    pd.DataFrame({"id": ids, "label": preds}).to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",        nargs="+", default=[str(s) for s in DEFAULT_SEEDS])
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--data_dir",     default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--output_dir",   default=str(_DEFAULT_OUTPUT_DIR))
    main(parser.parse_args())
