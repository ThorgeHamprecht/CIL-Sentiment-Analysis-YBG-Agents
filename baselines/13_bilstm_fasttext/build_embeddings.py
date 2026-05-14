"""Build vocab and extract FastText CC embedding matrix for the training vocab.

Reads train.csv, builds a 30k vocab, then scans cc.en.300.vec.gz and
cc.de.300.vec.gz to extract vectors for every vocab token found.
Tokens not found in either file get a random vector with the same std
as the pretrained vectors (~0.1).

Outputs:
  <artifact_dir>/vocab.json       — token → index
  <artifact_dir>/embeddings.npy   — float32 (vocab_size, 300)

Run once before submitting the training job (or let submit.sh call it).
"""
import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DIM = 300
PAD_TOKEN, UNK_TOKEN, PAD_IDX, UNK_IDX = "[PAD]", "[UNK]", 0, 1


def tokenize(text: str) -> list:
    return re.findall(r"\b\w+\b|[^\w\s]", str(text).lower(), re.UNICODE)


def build_vocab(texts: list, max_vocab: int) -> dict:
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for tok, _ in counter.most_common(max_vocab - 2):
        vocab.setdefault(tok, len(vocab))
    return vocab


def load_vectors_for_vocab(vec_gz_path: Path, vocab: dict, matrix: np.ndarray, filled: set):
    """Scan a .vec.gz file and fill matrix rows for tokens in vocab."""
    path = str(vec_gz_path)
    opener = gzip.open if path.endswith(".gz") else open
    found = 0
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # skip header line
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab and word not in filled:
                try:
                    vec = np.array(parts[1:], dtype=np.float32)
                    if len(vec) == DIM:
                        matrix[vocab[word]] = vec
                        filled.add(word)
                        found += 1
                except ValueError:
                    pass
    return found


def main(args):
    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ft_dir = Path(args.fasttext_dir)

    # --- build vocab from train.csv ---
    print("Reading train.csv...")
    df = pd.read_csv(Path(args.data_dir) / "train.csv")
    if "sentence" in df.columns:
        texts = df["sentence"].fillna("").tolist()
    elif "title" in df.columns:
        texts = (df["title"].fillna("") + " " + df["paragraph"].fillna("")).tolist()
    else:
        texts = df["text"].fillna("").tolist()

    print(f"Building vocab (max {args.max_vocab})...")
    vocab = build_vocab(texts, args.max_vocab)
    print(f"Vocab size: {len(vocab)}")

    # --- init matrix with small random values ---
    matrix = np.random.normal(0, 0.1, (len(vocab), DIM)).astype(np.float32)
    matrix[PAD_IDX] = 0.0  # padding stays zero

    filled = set()

    # --- fill from English vectors ---
    en_path = ft_dir / "cc.en.300.vec.gz"
    if en_path.exists():
        print(f"Loading EN vectors from {en_path}...")
        found = load_vectors_for_vocab(en_path, vocab, matrix, filled)
        print(f"  Filled {found} tokens from EN")
    else:
        print(f"WARNING: {en_path} not found — skipping EN vectors")

    # --- fill remaining tokens from German vectors ---
    de_path = ft_dir / "cc.de.300.vec.gz"
    if de_path.exists():
        print(f"Loading DE vectors from {de_path}...")
        found = load_vectors_for_vocab(de_path, vocab, matrix, filled)
        print(f"  Filled {found} additional tokens from DE")
    else:
        print(f"WARNING: {de_path} not found — skipping DE vectors")

    coverage = len(filled) / (len(vocab) - 2) * 100  # exclude PAD/UNK
    print(f"Coverage: {len(filled)}/{len(vocab)-2} tokens ({coverage:.1f}%)")
    print(f"Random init: {len(vocab)-2-len(filled)} tokens")

    # --- save ---
    vocab_path = out_dir / "vocab.json"
    emb_path   = out_dir / "embeddings.npy"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    np.save(emb_path, matrix)
    print(f"Saved vocab   -> {vocab_path}")
    print(f"Saved embeddings -> {emb_path}  ({matrix.nbytes / 1024**2:.1f} MB)")


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_vocab",    type=int, default=50_000)
    parser.add_argument("--data_dir",     default=str(ROOT / "data"))
    parser.add_argument("--artifact_dir", default=str(Path(__file__).parent / "artifacts"))
    parser.add_argument("--fasttext_dir", default="/work/scratch/thamprecht/cil/fasttext")
    main(parser.parse_args())
