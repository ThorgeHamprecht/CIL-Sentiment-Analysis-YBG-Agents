"""TF-IDF + Next Word Negation + Linear SVM for ETHZ CIL.

This follows the main idea from arXiv:1806.06407:
preprocess negation by marking only the next word, vectorize with TF-IDF,
then train a Linear Support Vector Machine.

Run:
    python svm_nwn.py
    python svm_nwn.py --sample 20000
    python svm_nwn.py --no-submit
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


SEED = 42
ID_COL = "id"
TEXT_COL = "sentence"
LABEL_COL = "label"
DEFAULT_DATA_DIR = "ethz-cil-text-class-2026"

TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", flags=re.UNICODE)
NEGATION_WORDS = {
    "aint",
    "aren't",
    "cannot",
    "cant",
    "can't",
    "couldn't",
    "didn't",
    "doesn't",
    "don't",
    "hadn't",
    "hasn't",
    "haven't",
    "isn't",
    "neither",
    "never",
    "no",
    "none",
    "nor",
    "not",
    "nothing",
    "wasn't",
    "weren't",
    "won't",
    "wouldn't",
    "kein",
    "keine",
    "keinem",
    "keinen",
    "keiner",
    "keines",
    "nicht",
    "nichts",
    "nie",
    "niemals",
    "ohne",
}


def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path} and {test_path}. "
            f"Pass --data-dir if your data lives elsewhere."
        )

    print(f"[data] reading {train_path} and {test_path}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train[TEXT_COL] = train[TEXT_COL].fillna("").astype(str)
    test[TEXT_COL] = test[TEXT_COL].fillna("").astype(str)
    train[LABEL_COL] = train[LABEL_COL].astype(int)
    test[ID_COL] = test[ID_COL].astype(int)

    print(f"[data] train rows: {len(train):,}  test rows: {len(test):,}")
    print(f"[data] label distribution:\n{train[LABEL_COL].value_counts().sort_index()}")
    return train, test


def normalize_token(token: str) -> str:
    return token.lower().replace("’", "'")


def apply_next_word_negation(text: str) -> str:
    """Replace only the word after a negation marker with not_<word>."""
    output: list[str] = []
    negate_next = False

    for match in TOKEN_RE.finditer(text):
        token = normalize_token(match.group(0))

        if negate_next:
            output.append(f"not_{token}")
            negate_next = False
            continue

        if token in NEGATION_WORDS:
            negate_next = True
            continue

        output.append(token)

    return " ".join(output)


def preprocess_texts(texts: Iterable[str], use_nwn: bool) -> list[str]:
    if use_nwn:
        return [apply_next_word_negation(text) for text in texts]
    return [" ".join(normalize_token(match.group(0)) for match in TOKEN_RE.finditer(text)) for text in texts]


def build_vectorizer(
    max_features: int,
    ngram_max: int,
    min_df: int,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_df=0.95,
        sublinear_tf=True,
        max_features=max_features,
        lowercase=False,
        dtype=np.float32,
    )


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "mae": mean_absolute_error(y_true, y_pred),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }
    print(
        f"[{name}] acc={metrics['accuracy']:.4f}  "
        f"macro-F1={metrics['macro_f1']:.4f}  "
        f"MAE={metrics['mae']:.4f}  "
        f"QWK={metrics['qwk']:.4f}"
    )
    return metrics


def write_submission(ids: np.ndarray, preds: np.ndarray, out_path: Path) -> None:
    sub = pd.DataFrame(
        {ID_COL: np.asarray(ids, dtype=int), LABEL_COL: np.asarray(preds, dtype=int)}
    )
    sub.to_csv(out_path, index=False)
    print(f"[submission] wrote {len(sub):,} rows -> {out_path}")


def train_svm(C: float, max_iter: int) -> LinearSVC:
    return LinearSVC(
        C=C,
        class_weight=None,
        dual="auto",
        max_iter=max_iter,
        random_state=SEED,
        verbose=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=6000)
    parser.add_argument("--max-features", type=int, default=10_000)
    parser.add_argument("--ngram-max", type=int, default=1)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--out", type=str, default="submission_svm_nwn.csv")
    parser.add_argument("--no-nwn", action="store_true", help="Disable NWN preprocessing.")
    parser.add_argument("--no-submit", action="store_true", help="Skip full refit/submission.")
    args = parser.parse_args()

    train, test = load_data(args.data_dir)
    if args.sample is not None and args.sample < len(train):
        train = train.sample(n=args.sample, random_state=SEED).reset_index(drop=True)
        print(f"[svm-nwn] using sub-sample: {len(train):,} rows")

    X_text = train[TEXT_COL].to_numpy(dtype=str)
    y = train[LABEL_COL].to_numpy(dtype=int)
    use_nwn = not args.no_nwn

    X_tr_text, X_val_text, y_tr, y_val = train_test_split(
        X_text,
        y,
        test_size=args.val_size,
        stratify=y,
        random_state=SEED,
    )
    print(f"[svm-nwn] train={len(X_tr_text):,}  val={len(X_val_text):,}")

    print(f"[svm-nwn] preprocessing text (NWN={use_nwn}) ...")
    t0 = time.time()
    X_tr_pre = preprocess_texts(X_tr_text, use_nwn=use_nwn)
    X_val_pre = preprocess_texts(X_val_text, use_nwn=use_nwn)
    print(f"[svm-nwn] preprocessed in {time.time() - t0:.1f}s")

    print(
        "[svm-nwn] fitting TF-IDF "
        f"(word 1-{args.ngram_max}, max_features={args.max_features:,}) ..."
    )
    t0 = time.time()
    vectorizer = build_vectorizer(args.max_features, args.ngram_max, args.min_df)
    X_tr = vectorizer.fit_transform(X_tr_pre)
    X_val = vectorizer.transform(X_val_pre)
    print(f"[svm-nwn] features: {X_tr.shape[1]:,}  (fit {time.time() - t0:.1f}s)")

    print("[svm-nwn] training LinearSVC ...")
    t0 = time.time()
    clf = train_svm(args.C, args.max_iter)
    clf.fit(X_tr, y_tr)
    print(f"[svm-nwn] trained in {time.time() - t0:.1f}s")

    y_val_pred = clf.predict(X_val)
    evaluate("val", np.asarray(y_val), np.asarray(y_val_pred))

    if args.no_submit:
        return

    print("[svm-nwn] refitting on full training set for submission ...")
    t0 = time.time()
    X_full_pre = preprocess_texts(X_text, use_nwn=use_nwn)
    test_pre = preprocess_texts(test[TEXT_COL].values, use_nwn=use_nwn)
    vectorizer_full = build_vectorizer(args.max_features, args.ngram_max, args.min_df)
    X_full = vectorizer_full.fit_transform(X_full_pre)
    X_test = vectorizer_full.transform(test_pre)
    print(f"[svm-nwn] full features: {X_full.shape[1]:,}  (fit {time.time() - t0:.1f}s)")

    t0 = time.time()
    clf_full = train_svm(args.C, args.max_iter)
    clf_full.fit(X_full, y)
    print(f"[svm-nwn] refit in {time.time() - t0:.1f}s")

    test_pred = clf_full.predict(X_test)
    write_submission(
        np.asarray(test[ID_COL].values), np.asarray(test_pred), Path(args.out)
    )


if __name__ == "__main__":
    main()
