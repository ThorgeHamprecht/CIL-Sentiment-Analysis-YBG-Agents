"""Tokenization, vocabulary, and Dataset — shared with BiLSTM baseline."""
import json
import re
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
PAD_IDX = 0
UNK_IDX = 1


def tokenize(text: str) -> list:
    return re.findall(r"\b\w+\b|[^\w\s]", str(text).lower(), re.UNICODE)


def build_vocab(texts: list, max_vocab: int = 30_000) -> dict:
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for tok, _ in counter.most_common(max_vocab - 2):
        vocab.setdefault(tok, len(vocab))
    return vocab


def save_vocab(vocab: dict, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)


def load_vocab(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_csv(path):
    """Return (texts, labels_or_None, ids). Handles sentence or title+paragraph columns."""
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
    def __init__(self, texts: list, vocab: dict, max_len: int = 256, labels: list = None):
        self.labels = labels
        self.encoded = []
        self.lengths = []
        for text in texts:
            tokens = tokenize(text)[:max_len]
            ids = [vocab.get(t, UNK_IDX) for t in tokens]
            self.lengths.append(max(len(ids), 1))
            ids += [PAD_IDX] * (max_len - len(ids))
            self.encoded.append(ids)

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded[idx], dtype=torch.long)
        length = torch.tensor(self.lengths[idx], dtype=torch.long)
        if self.labels is not None:
            return x, length, torch.tensor(self.labels[idx], dtype=torch.long)
        return x, length
