"""Run inference with best CE checkpoint and save submission CSV."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import ReviewDataset, read_csv
from model import mDeBERTaEMD, median_decode

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds = []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask)
        all_preds.append(median_decode(logits).cpu().numpy())
    return np.concatenate(all_preds)


def main(args):
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = Path(args.artifact_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt      = torch.load(artifact_dir / "best_model_ce.pt", map_location=device, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(ckpt["backbone"])
    model     = mDeBERTaEMD(model_name=ckpt["backbone"], dropout=0.0).to(device)
    model.load_state_dict(ckpt["model"])

    texts, _, ids = read_csv(Path(args.data_dir) / "test.csv")
    test_ds     = ReviewDataset(texts, tokenizer, max_len=args.max_len)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    preds = predict(model, test_loader, device)
    out   = output_dir / "28_mdeberta_ce_submission.csv"
    pd.DataFrame({"id": ids, "label": preds}).to_csv(out, index=False)
    print(f"Saved: {out}  ({len(preds):,} predictions)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     default=str(ROOT / "data"))
    parser.add_argument("--artifact_dir", default=str(Path(__file__).parent / "artifacts"))
    parser.add_argument("--output_dir",   default=str(ROOT / "submissions"))
    parser.add_argument("--max_len",      type=int, default=128)
    main(parser.parse_args())
