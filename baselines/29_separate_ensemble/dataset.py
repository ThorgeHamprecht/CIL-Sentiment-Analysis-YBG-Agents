"""Dataset utilities for mDeBERTa review-rating experiments."""
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def read_csv(path):
    """Read a repo review CSV and return texts, optional labels, and ids."""
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


def _batch_tokenize(texts, tokenizer, max_len, batch_size, show_progress):
    """Tokenize in chunks to keep peak Python-side memory lower."""
    batches = []
    indices = range(0, len(texts), batch_size)
    if show_progress and tqdm is not None:
        indices = tqdm(indices, desc="Tokenizing", miniters=max(1, len(indices) // 20))

    for start in indices:
        chunk = texts[start:start + batch_size]
        batches.append(
            tokenizer(
                chunk,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
        )

    encodings = {}
    for batch in batches:
        for key, value in batch.items():
            encodings.setdefault(key, []).append(value)
    return {key: torch.cat(values, dim=0) for key, values in encodings.items()}


class ReviewDataset(Dataset):
    """Torch dataset wrapping tokenized review text and optional labels."""

    def __init__(
        self,
        texts: list,
        tokenizer,
        max_len: int = 256,
        labels: list = None,
        show_progress: bool = False,
        batch_size: int = 1024,
    ):
        self.labels = labels
        if show_progress and len(texts) > batch_size:
            self.encodings = _batch_tokenize(texts, tokenizer, max_len, batch_size, show_progress)
        else:
            self.encodings = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )

    def __len__(self):
        """Return the number of tokenized examples."""
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        """Return one tokenized example, including label when available."""
        item = {key: value[idx] for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
