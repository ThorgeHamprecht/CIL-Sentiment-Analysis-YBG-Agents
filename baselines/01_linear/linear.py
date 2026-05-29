"""Linear baseline for ETHZ CIL Text Classification 2026.

Pipeline
--------
1.  Combine word (1-2 grams) + character (3-5 grams) TF-IDF features,
    plus capped binary word count features. Char n-grams are
    language-agnostic which helps for the EN/DE mix.
2.  Multinomial Logistic Regression (saga solver, L2).
3.  Stratified hold-out validation -> report accuracy, macro-F1, MAE,
    and quadratic-weighted kappa (the standard metric for ordinal ratings).
4.  Refit on the *full* training set with the same hyperparameters and
    write a Kaggle submission CSV.

Data layout (local; download from Kaggle once and unzip here):
    ethz-cil-text-class-2026/train.csv  columns: id, sentence, label
    ethz-cil-text-class-2026/test.csv   columns: id, sentence

Run:
    python linear.py
    python linear.py --tune --no-submit
    python linear.py --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split


SEED = 42
ID_COL = "id"
TEXT_COL = "sentence"
LABEL_COL = "label"
DEFAULT_DATA_DIR = "ethz-cil-text-class-2026"
MAX_LABEL_DISTANCE = 4.0


@dataclass(frozen=True)
class TfIdfConfig:
    name: str
    c: float
    word_ngram_max: int
    char_ngram_min: int
    char_ngram_max: int
    min_df: int
    strip_accents: str | None
    max_word_features: int
    max_char_features: int
    use_count_features: bool
    max_count_features: int
    binary_counts: bool


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


def build_vectorizers(
    max_word_features: int = 200_000,
    max_char_features: int = 200_000,
    max_count_features: int = 50_000,
    word_ngram_max: int = 2,
    char_ngram_min: int = 3,
    char_ngram_max: int = 5,
    min_df: int = 3,
    strip_accents: str | None = "unicode",
    use_count_features: bool = True,
    binary_counts: bool = True,
) -> tuple[TfidfVectorizer, TfidfVectorizer, CountVectorizer | None]:
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, word_ngram_max),
        min_df=min_df,
        max_df=0.95,
        sublinear_tf=True,
        max_features=max_word_features,
        lowercase=True,
        strip_accents=strip_accents,
        dtype=np.float32,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(char_ngram_min, char_ngram_max),
        min_df=min_df,
        max_df=0.95,
        sublinear_tf=True,
        max_features=max_char_features,
        lowercase=True,
        strip_accents=strip_accents,
        dtype=np.float32,
    )
    count_vec = None
    if use_count_features:
        count_vec = CountVectorizer(
            analyzer="word",
            ngram_range=(1, word_ngram_max),
            min_df=min_df,
            max_df=0.95,
            max_features=max_count_features,
            lowercase=True,
            strip_accents=strip_accents,
            binary=binary_counts,
            dtype=np.int64,
        )
    return word_vec, char_vec, count_vec


def fit_features(
    texts,
    word_vec: TfidfVectorizer,
    char_vec: TfidfVectorizer,
    count_vec: CountVectorizer | None,
):
    Xw = word_vec.fit_transform(texts)
    Xc = char_vec.fit_transform(texts)
    blocks = [Xw, Xc]
    if count_vec is not None:
        blocks.append(count_vec.fit_transform(texts))
    return hstack(blocks, format="csr", dtype=np.float32)


def transform_features(
    texts,
    word_vec: TfidfVectorizer,
    char_vec: TfidfVectorizer,
    count_vec: CountVectorizer | None,
):
    Xw = word_vec.transform(texts)
    Xc = char_vec.transform(texts)
    blocks = [Xw, Xc]
    if count_vec is not None:
        blocks.append(count_vec.transform(texts))
    return hstack(blocks, format="csr", dtype=np.float32)


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    metrics = {
        "accuracy": 1.0 - (mae / MAX_LABEL_DISTANCE),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "mae": mae,
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


def make_config(args: argparse.Namespace) -> TfIdfConfig:
    strip_accents = None if args.keep_accents else "unicode"
    return TfIdfConfig(
        name="single",
        c=args.C,
        word_ngram_max=args.word_ngram_max,
        char_ngram_min=args.char_ngram_min,
        char_ngram_max=args.char_ngram_max,
        min_df=args.min_df,
        strip_accents=strip_accents,
        max_word_features=args.max_word_features,
        max_char_features=args.max_char_features,
    use_count_features=args.count_features,
        max_count_features=args.max_count_features,
        binary_counts=not args.raw_counts,
    )


def tune_configs(args: argparse.Namespace) -> list[TfIdfConfig]:
    strip_options: list[tuple[str, str | None]] = [("strip", "unicode")]
    if args.tune_keep_accents:
        strip_options.append(("keep", None))

    base_feature_configs = [
        ("base", 2, 3, 5, 3),
        ("word3", 3, 3, 5, 3),
        ("char6", 2, 3, 6, 3),
        ("lowdf", 3, 3, 6, 2),
    ]
    configs = []
    for c in args.tune_C:
        for feature_name, word_max, char_min, char_max, min_df in base_feature_configs:
            for strip_name, strip_accents in strip_options:
                configs.append(
                    TfIdfConfig(
                        name=f"{feature_name}_C{c:g}_{strip_name}",
                        c=c,
                        word_ngram_max=word_max,
                        char_ngram_min=char_min,
                        char_ngram_max=char_max,
                        min_df=min_df,
                        strip_accents=strip_accents,
                        max_word_features=args.max_word_features,
                        max_char_features=args.max_char_features,
                        use_count_features=args.count_features,
                        max_count_features=args.max_count_features,
                        binary_counts=not args.raw_counts,
                    )
                )
    return configs


def build_classifier(c: float, max_iter: int) -> LogisticRegression:
    return LogisticRegression(
        C=c,
        solver="saga",
        penalty="l2",
        max_iter=max_iter,
        n_jobs=-1,
        verbose=0,
        random_state=SEED,
    )


def predict_min_squared_cdf_distance(clf: LogisticRegression, X) -> np.ndarray:
    proba = clf.predict_proba(X)
    classes = np.asarray(clf.classes_)
    order = np.argsort(classes)
    classes = classes[order]
    proba = proba[:, order]

    proba_cdf = np.cumsum(proba, axis=1)
    target_cdf = (classes[None, :] >= classes[:, None]).astype(np.float32)
    squared_cdf_distance = np.sum(
        (proba_cdf[:, None, :] - target_cdf[None, :, :]) ** 2, axis=2
    )
    return classes[np.argmin(squared_cdf_distance, axis=1)]


def fit_and_evaluate(
    config: TfIdfConfig,
    X_tr_text: np.ndarray,
    X_val_text: np.ndarray,
    y_tr: np.ndarray,
    y_val: np.ndarray,
    max_iter: int,
) -> dict:
    print(
        "[linear] fitting TF-IDF "
        f"{config.name}: word 1-{config.word_ngram_max}, "
        f"char_wb {config.char_ngram_min}-{config.char_ngram_max}, "
        f"min_df={config.min_df}, "
        f"strip_accents={config.strip_accents}, "
        f"count_features={config.use_count_features}, "
        f"max_count_features={config.max_count_features:,}, "
        f"binary_counts={config.binary_counts}, "
        f"C={config.c:g}"
    )
    t0 = time.time()
    word_vec, char_vec, count_vec = build_vectorizers(
        max_word_features=config.max_word_features,
        max_char_features=config.max_char_features,
        max_count_features=config.max_count_features,
        word_ngram_max=config.word_ngram_max,
        char_ngram_min=config.char_ngram_min,
        char_ngram_max=config.char_ngram_max,
        min_df=config.min_df,
        strip_accents=config.strip_accents,
        use_count_features=config.use_count_features,
        binary_counts=config.binary_counts,
    )
    X_tr = fit_features(X_tr_text, word_vec, char_vec, count_vec)
    X_val = transform_features(X_val_text, word_vec, char_vec, count_vec)
    print(f"[linear] features: {X_tr.shape[1]:,}  (fit {time.time() - t0:.1f}s)")

    print("[linear] training LogisticRegression (saga, multinomial) ...")
    t0 = time.time()
    clf = build_classifier(config.c, max_iter)
    clf.fit(X_tr, y_tr)
    print(f"[linear] trained in {time.time() - t0:.1f}s")

    train_metrics = evaluate(
        f"{config.name} train", np.asarray(y_tr), predict_min_squared_cdf_distance(clf, X_tr)
    )
    val_metrics = evaluate(
        f"{config.name} val", np.asarray(y_val), predict_min_squared_cdf_distance(clf, X_val)
    )
    return {"config": config, "train": train_metrics, "val": val_metrics}


def refit_and_submit(
    config: TfIdfConfig,
    X_text: np.ndarray,
    y: np.ndarray,
    test: pd.DataFrame,
    max_iter: int,
    out_path: Path,
) -> None:
    print(f"[linear] refitting best config on full training set: {config.name}")
    t0 = time.time()
    word_vec_full, char_vec_full, count_vec_full = build_vectorizers(
        max_word_features=config.max_word_features,
        max_char_features=config.max_char_features,
        max_count_features=config.max_count_features,
        word_ngram_max=config.word_ngram_max,
        char_ngram_min=config.char_ngram_min,
        char_ngram_max=config.char_ngram_max,
        min_df=config.min_df,
        strip_accents=config.strip_accents,
        use_count_features=config.use_count_features,
        binary_counts=config.binary_counts,
    )
    X_full = fit_features(X_text, word_vec_full, char_vec_full, count_vec_full)
    clf_full = build_classifier(config.c, max_iter)
    clf_full.fit(X_full, y)
    print(f"[linear] refit in {time.time() - t0:.1f}s")

    X_test = transform_features(
        test[TEXT_COL].to_numpy(dtype=str), word_vec_full, char_vec_full, count_vec_full
    )
    test_pred = predict_min_squared_cdf_distance(clf_full, X_test)
    write_submission(test[ID_COL].to_numpy(dtype=int), np.asarray(test_pred), out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument(
        "--C", type=float, default=0.5, help="Inverse regularization strength."
    )
    parser.add_argument("--max-iter", type=int, default=600)
    parser.add_argument("--max-word-features", type=int, default=200_000)
    parser.add_argument("--max-char-features", type=int, default=200_000)
    parser.add_argument("--max-count-features", type=int, default=50_000)
    parser.add_argument("--word-ngram-max", type=int, default=2)
    parser.add_argument("--char-ngram-min", type=int, default=3)
    parser.add_argument("--char-ngram-max", type=int, default=5)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument(
        "--count-features",
        action="store_true",
        help="Add extra CountVectorizer word features.",
    )
    parser.add_argument(
        "--raw-counts",
        action="store_true",
        help="Use raw counts instead of binary count indicators.",
    )
    parser.add_argument(
        "--keep-accents",
        action="store_true",
        help="Do not strip accents/umlauts before TF-IDF.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run a compact TF-IDF/C sweep and select the best validation QWK.",
    )
    parser.add_argument(
        "--tune-C",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0, 2.0],
        help="C values to try in --tune mode.",
    )
    parser.add_argument(
        "--tune-keep-accents",
        action="store_true",
        help="Also try accent-preserving TF-IDF configs in --tune mode.",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Skip full refit/submission after validation.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="If set, train on only this many examples (for fast smoke tests).",
    )
    parser.add_argument("--out", type=str, default="submission_linear.csv")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    train, test = load_data(args.data_dir)

    if args.sample is not None and args.sample < len(train):
        train = train.sample(n=args.sample, random_state=SEED).reset_index(drop=True)
        print(f"[linear] using sub-sample: {len(train):,} rows")

    X_text = train[TEXT_COL].to_numpy(dtype=str)
    y = train[LABEL_COL].to_numpy(dtype=int)

    X_tr_text, X_val_text, y_tr, y_val = train_test_split(
        X_text,
        y,
        test_size=args.val_size,
        stratify=y,
        random_state=SEED,
    )
    print(f"[linear] train={len(X_tr_text):,}  val={len(X_val_text):,}")

    configs = tune_configs(args) if args.tune else [make_config(args)]
    print(f"[linear] configs to run: {len(configs)}")

    results = [
        fit_and_evaluate(config, X_tr_text, X_val_text, y_tr, y_val, args.max_iter)
        for config in configs
    ]
    best = max(results, key=lambda item: item["val"]["qwk"])

    if args.tune:
        print("[linear:tune] validation summary (sorted by QWK):")
        for result in sorted(results, key=lambda item: item["val"]["qwk"], reverse=True):
            config = result["config"]
            train = result["train"] 
            val = result["val"]
            print(
                f"  {config.name}: "
                f"val_qwk={val['qwk']:.4f} val_acc={val['accuracy']:.4f} "
                f"train_acc={train['accuracy']:.4f} "
                f"word=1-{config.word_ngram_max} "
                f"char={config.char_ngram_min}-{config.char_ngram_max} "
                f"min_df={config.min_df} C={config.c:g} "
                f"count={config.use_count_features} "
                f"strip_accents={config.strip_accents}"
            )

    best_config = best["config"]
    print(
        f"[linear] best={best_config.name} "
        f"val_qwk={best['val']['qwk']:.4f} "
        f"val_acc={best['val']['accuracy']:.4f} "
        f"train_acc={best['train']['accuracy']:.4f}"
    )

    if not args.no_submit:
        refit_and_submit(best_config, X_text, y, test, args.max_iter, Path(args.out))


if __name__ == "__main__":
    main()
