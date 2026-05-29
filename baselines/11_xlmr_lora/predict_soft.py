"""Generate teacher soft labels (T=1 probs) on all train + test examples.

Outputs:
  <artifact_dir>/train_soft.csv  — id, p0, p1, p2, p3, p4
  <artifact_dir>/test_soft.csv   — id, p0, p1, p2, p3, p4
These are loaded by baselines/12_distilled_bilstm/train.py.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import ReviewDataset, read_csv
from model import XLMRLoRA

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_NAME = "xlm-roberta-base"
SCRATCH = Path("/work/scratch") / os.environ.get("USER", "<user>") / "cil"


@torch.no_grad()
def get_soft_labels(model, loader, device):
    model.eval()
    all_probs = []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask)
        all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(all_probs)


def run_split(split_name, data_path, tokenizer, model, device, args, out_dir):
    texts, labels, ids = read_csv(data_path)
    print(f"  {split_name}: {len(texts)} examples")
    ds = ReviewDataset(texts, tokenizer, max_len=args.max_len, labels=labels)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    probs = get_soft_labels(model, loader, device)
    df = pd.DataFrame(probs, columns=["p0", "p1", "p2", "p3", "p4"])
    df.insert(0, "id", ids)
    out_path = out_dir / f"{split_name}_soft.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


def main(args):
    os.environ.setdefault("HF_HOME", str(SCRATCH / ".cache/huggingface"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = Path(args.artifact_dir)
    data_dir = Path(args.data_dir)

    print("Loading tokenizer and model...")
    tokenizer_path = artifact_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path) if tokenizer_path.exists() else MODEL_NAME
    )

    ckpt = torch.load(artifact_dir / "best_model.pt", map_location=device)
    saved_args = ckpt["args"]

    model = XLMRLoRA(MODEL_NAME, dropout=0.0).to(device)
    model.encoder.load_adapter(str(artifact_dir / "lora_adapter"), adapter_name="default")
    model.classifier.load_state_dict(ckpt["classifier"])

    print("Generating soft labels...")
    run_split("train", data_dir / "train.csv", tokenizer, model, device, args, artifact_dir)
    run_split("test",  data_dir / "test.csv",  tokenizer, model, device, args, artifact_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len",     type=int, default=128)
    parser.add_argument("--batch_size",  type=int, default=128)
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--data_dir",    default=str(_DEFAULT_DATA_DIR))
    main(parser.parse_args())
