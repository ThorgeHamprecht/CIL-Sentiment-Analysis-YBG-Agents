"""Two-stream dataset: splits each review into title and body for gated fusion."""
import json
import re
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset

PAD_TOKEN, UNK_TOKEN, PAD_IDX, UNK_IDX = "[PAD]", "[UNK]", 0, 1


def tokenize(text: str) -> list:
    return re.findall(r"\b\w+\b|[^\w\s]", str(text).lower(), re.UNICODE)


def split_title_body(text: str):
    """Split on first '. ' — Amazon reviews start with a short title sentence."""
    text = str(text)
    idx = text.find(". ")
    if idx > 0:
        return text[:idx], text[idx + 2:]
    return text, text  # fallback: use full text for both streams


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
    """Return (texts, titles, bodies, labels_or_None, ids)."""
    df = pd.read_csv(path)
    if "title" in df.columns and "paragraph" in df.columns:
        titles = df["title"].fillna("").tolist()
        bodies = df["paragraph"].fillna("").tolist()
        texts = [t + " " + b for t, b in zip(titles, bodies)]
    elif "sentence" in df.columns:
        texts = df["sentence"].fillna("").tolist()
        splits = [split_title_body(t) for t in texts]
        titles = [s[0] for s in splits]
        bodies = [s[1] for s in splits]
    elif "text" in df.columns:
        texts = df["text"].fillna("").tolist()
        splits = [split_title_body(t) for t in texts]
        titles = [s[0] for s in splits]
        bodies = [s[1] for s in splits]
    else:
        raise ValueError(f"No text column found in {path}.")

    labels = None
    for col in ("label", "rating", "stars"):
        if col in df.columns:
            labels = df[col].astype(int).tolist()
            break

    ids = df["id"].tolist() if "id" in df.columns else list(range(len(df)))
    return texts, titles, bodies, labels, ids


def _encode(tokens: list, vocab: dict, max_len: int):
    ids = [vocab.get(t, UNK_IDX) for t in tokens[:max_len]]
    length = max(len(ids), 1)
    ids += [PAD_IDX] * (max_len - len(ids))
    return ids, length


class TwoStreamDataset(Dataset):
    def __init__(
        self,
        titles: list,
        bodies: list,
        vocab: dict,
        max_len_title: int = 64,
        max_len_body: int = 192,
        labels: list = None,
    ):
        self.labels = labels
        self.title_enc, self.title_len = [], []
        self.body_enc, self.body_len = [], []

        for title, body in zip(titles, bodies):
            t_ids, t_len = _encode(tokenize(title), vocab, max_len_title)
            b_ids, b_len = _encode(tokenize(body), vocab, max_len_body)
            self.title_enc.append(t_ids)
            self.title_len.append(t_len)
            self.body_enc.append(b_ids)
            self.body_len.append(b_len)

    def __len__(self) -> int:
        return len(self.title_enc)

    def __getitem__(self, idx):
        x_t = torch.tensor(self.title_enc[idx], dtype=torch.long)
        l_t = torch.tensor(self.title_len[idx], dtype=torch.long)
        x_b = torch.tensor(self.body_enc[idx], dtype=torch.long)
        l_b = torch.tensor(self.body_len[idx], dtype=torch.long)
        if self.labels is not None:
            return x_t, l_t, x_b, l_b, torch.tensor(self.labels[idx], dtype=torch.long)
        return x_t, l_t, x_b, l_b
