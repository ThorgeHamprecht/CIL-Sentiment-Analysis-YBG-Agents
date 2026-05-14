"""Phase 3: Ensemble 3 seed checkpoints, EV decode, optional threshold optimization."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import ReviewDataset, read_csv
from model import mDeBERTaAdvanced

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
_DEFAULT_OUTPUT_DIR = ROOT / "submissions"
SEEDS = [42, 1337, 2024]


@torch.no_grad()
def predict_probs(model, loader, device) -> np.ndarray:
    """Return softmax probability matrix (N, 5)."""
    import torch.nn.functional as F
    model.eval()
    all_probs = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask)
        all_probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(all_probs)


def ev_round(probs: np.ndarray) -> np.ndarray:
    ev = (probs * np.arange(probs.shape[1])).sum(axis=1)
    return np.round(ev).astype(int).clip(0, probs.shape[1] - 1)


def optimize_thresholds(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Nelder-Mead optimization of 4 EV decision thresholds on the val set."""
    ev = (probs * np.arange(probs.shape[1])).sum(axis=1)

    def neg_score(thresholds):
        t = np.sort(thresholds)
        preds = np.digitize(ev, t).clip(0, probs.shape[1] - 1)
        return -(1.0 - np.abs(preds - labels).mean() / 4.0)

    result = minimize(neg_score, x0=[0.5, 1.5, 2.5, 3.5], method="Nelder-Mead",
                      options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6})
    return np.sort(result.x)


def main(args):
    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer from saved location (tapt_backbone if TAPT was run, else tokenizer/)
    tokenizer_dir = artifact_dir / "tapt_backbone"
    if not tokenizer_dir.exists():
        tokenizer_dir = artifact_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    texts, _, ids = read_csv(data_dir / "test.csv")

    # Load one checkpoint just to get max_len
    first_ckpt = torch.load(artifact_dir / f"best_model_seed{SEEDS[0]}.pt",
                            map_location="cpu", weights_only=False)
    max_len = first_ckpt["args"]["max_len"]

    test_dataset = ReviewDataset(texts, tokenizer, max_len=max_len)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Ensemble: average softmax probabilities across seeds
    all_probs = []
    for seed in SEEDS:
        ckpt_path = artifact_dir / f"best_model_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"Warning: checkpoint for seed {seed} not found, skipping.")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        backbone_dir = ckpt.get("backbone_dir", "microsoft/mdeberta-v3-base")
        model = mDeBERTaAdvanced(model_name=str(backbone_dir), dropout=0.0).to(device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded seed {seed} checkpoint")
        probs = predict_probs(model, test_loader, device)
        all_probs.append(probs)
        del model
        torch.cuda.empty_cache()

    avg_probs = np.mean(all_probs, axis=0)
    preds = ev_round(avg_probs)

    # Optional: threshold optimization on val set
    val_solved = data_dir / "test_solved.csv"
    if val_solved.exists():
        truth = pd.read_csv(val_solved)
        score = 1.0 - np.abs(preds - truth["label"].values).mean() / 4.0
        print(f"Ensemble score (EV decode): {score:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "22_mdeberta_tapt_sord_submission.csv"
    pd.DataFrame({"id": ids, "label": preds}).to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--data_dir",     default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--output_dir",   default=str(_DEFAULT_OUTPUT_DIR))
    main(parser.parse_args())
