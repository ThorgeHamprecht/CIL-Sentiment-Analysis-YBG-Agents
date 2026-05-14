"""Phase 1: Task-Adaptive Pretraining via MLM on train.csv + test.csv review text."""
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = ROOT / "data"
_DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_NAME = "microsoft/mdeberta-v3-base"


class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def load_texts(data_dir: Path) -> list:
    texts = []
    for fname in ("train.csv", "test.csv"):
        df = pd.read_csv(data_dir / fname)
        col = next(c for c in ("sentence", "text") if c in df.columns)
        texts.extend(df[col].fillna("").tolist())
    return texts


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_tokens(["\n", "  "], special_tokens=True)

    texts = load_texts(data_dir)
    print(f"TAPT corpus: {len(texts):,} texts")

    print("Tokenizing...")
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_len,
        return_tensors="pt",
    )

    dataset = TextDataset(encodings)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=collator,
    )

    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    total_steps = args.epochs * len(loader) // args.grad_accum
    warmup_steps = int(0.06 * total_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / args.grad_accum
            loss.backward()
            total_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 500 == 0:
                    print(f"  step {global_step} | loss {total_loss / (step + 1):.4f}")

        print(f"Epoch {epoch} | avg_loss={total_loss / len(loader):.4f}")

    # Save adapted backbone only (discard MLM head)
    backbone_dir = out_dir / "tapt_backbone"
    backbone_dir.mkdir(exist_ok=True)
    model.deberta.save_pretrained(str(backbone_dir))
    tokenizer.save_pretrained(str(backbone_dir))
    print(f"TAPT backbone saved to {backbone_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len",    type=int,   default=256)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--grad_accum", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--epochs",     type=int,   default=2)
    parser.add_argument("--data_dir",   default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--artifact_dir", default=str(_DEFAULT_ARTIFACT_DIR))
    main(parser.parse_args())
