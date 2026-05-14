"""HuggingFace tokenizer dataset for mDeBERTa fine-tuning."""
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def read_csv(path):
    df = pd.read_csv(path)
    if "sentence" in df.columns:
        texts = df["sentence"].fillna("").tolist()
    elif "title" in df.columns and "paragraph" in df.columns:
        texts = (df["title"].fillna("") + " " + df["paragraph"].fillna("")).tolist()
    elif "text" in df.columns:
        texts = df["text"].fillna("").tolist()
    else:
        raise ValueError(f"No text column found in {path}.")

    labels = None
    for col in ("label", "rating", "stars"):
        if col in df.columns:
            labels = df[col].astype(int).tolist()
            break

    ids = df["id"].tolist() if "id" in df.columns else list(range(len(df)))
    return texts, labels, ids


class ReviewDataset(Dataset):
    def __init__(self, texts: list, tokenizer, max_len: int = 128, labels: list = None):
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
