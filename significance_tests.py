"""
Bootstrap confidence intervals + Wilcoxon signed-rank tests for submission comparisons.
Bootstrap: 10,000 paired resamples for CIs and p-values.
Wilcoxon: non-parametric paired test on per-sample |error| differences.
Bonferroni correction applied for multiple comparisons (n_tests = 9).

Usage:
    python3 bootstrap_significance.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SUBS    = Path(__file__).parent / "submissions"
GT_PATH = Path(__file__).parent / "testlabels" / "test_solved.csv"
N_BOOT    = 10_000
N_TESTS   = 9        # number of pairwise comparisons — for Bonferroni
ALPHA     = 0.05
ALPHA_BON = ALPHA / N_TESTS   # 0.0056
RNG       = np.random.default_rng(42)

gt = pd.read_csv(GT_PATH).sort_values("id").reset_index(drop=True)["label"].values


def load(fname):
    return pd.read_csv(SUBS / fname).sort_values("id").reset_index(drop=True)["label"].values


def kaggle_score(preds, labels):
    return 1.0 - np.abs(preds - labels).mean() / 4.0


def bootstrap_ci(preds, labels, n=N_BOOT):
    """Return (score, lower_95, upper_95) via percentile bootstrap."""
    n_samples = len(labels)
    scores = np.empty(n)
    for i in range(n):
        idx = RNG.integers(0, n_samples, n_samples)
        scores[i] = kaggle_score(preds[idx], labels[idx])
    return kaggle_score(preds, labels), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def bootstrap_diff(preds_a, preds_b, labels, n=N_BOOT):
    """
    Test H0: score(A) == score(B).
    Returns (observed_diff, p_value, lower_95_diff, upper_95_diff).
    p_value: fraction of bootstrap samples where diff favours the other direction.
    """
    n_samples = len(labels)
    obs_diff  = kaggle_score(preds_a, labels) - kaggle_score(preds_b, labels)
    diffs = np.empty(n)
    for i in range(n):
        idx = RNG.integers(0, n_samples, n_samples)
        diffs[i] = kaggle_score(preds_a[idx], labels[idx]) - kaggle_score(preds_b[idx], labels[idx])
    # Two-sided p-value
    if obs_diff >= 0:
        p = (diffs <= 0).mean()
    else:
        p = (diffs >= 0).mean()
    p_two_sided = min(2 * p, 1.0)
    return obs_diff, p_two_sided, np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def sig_stars(p, bonferroni=False):
    threshold = ALPHA_BON if bonferroni else ALPHA
    if p < 0.001 and p < threshold: return "***"
    if p < 0.01  and p < threshold: return "**"
    if p < 0.05  and p < threshold: return "*"
    if p < 0.05:                    return "(*)n.s.†"  # significant uncorrected, not after Bonferroni
    return "n.s."


def wilcoxon_test(preds_a, preds_b, labels):
    """
    Wilcoxon signed-rank test on per-sample absolute error differences.
    d_i = |e_A_i| - |e_B_i|  (positive = A makes larger error = B is better)
    We test H0: median(d) = 0.
    Returns (statistic, p_value, direction) where direction = "A>B" if A has lower errors.
    """
    err_a = np.abs(preds_a - labels)
    err_b = np.abs(preds_b - labels)
    d = err_a - err_b   # positive means A is worse
    # Only test where there is a difference
    d_nonzero = d[d != 0]
    if len(d_nonzero) == 0:
        return 0.0, 1.0, "tie"
    stat, p = wilcoxon(d_nonzero, alternative="two-sided", method="approx")
    direction = "A>B" if d.mean() < 0 else "B>A"
    return stat, p, direction


# ── Load submissions ──────────────────────────────────────────────────────────
MODELS = {
    "BiLSTM CE":         "06_rnn_bilstm_submission.csv",
    "BiLSTM EMD²":       "15_bilstm_emd_submission.csv",
    "BiLSTM OLL":        "29_bilstm_oll_submission.csv",
    "Transformer CE":    "07_transformer_custom_submission.csv",
    "Transformer EMD²":  "16_transformer_emd_submission.csv",
    "Transformer OLL":   "30_transformer_oll_submission.csv",
    "mDeBERTa CE":       "28_mdeberta_ce_submission.csv",
    "mDeBERTa EMD²":     "20_mdeberta_emd_v2_submission.csv",
    "mDeBERTa LLRD+EMA": "23_mdeberta_llrd_ema_seed42_submission.csv",
    "Best (B27)":        "27_mdeberta_seed_split_seed42_1337_2024_submission.csv",
}

preds = {}
for name, fname in MODELS.items():
    p = SUBS / fname
    if p.exists():
        preds[name] = load(fname)
    else:
        print(f"MISSING: {fname}")

# ── Individual scores with 95% CIs ───────────────────────────────────────────
print("\n" + "="*65)
print("Individual scores with 95% bootstrap CI (n=10,000)")
print("="*65)
print(f"{'Model':<24} {'Score':>7}  {'95% CI':>18}")
print("-"*65)
for name, p in preds.items():
    s, lo, hi = bootstrap_ci(p, gt)
    print(f"{name:<24} {s:.4f}  [{lo:.4f}, {hi:.4f}]")

# ── Key pairwise comparisons ──────────────────────────────────────────────────
COMPARISONS = [
    # (A, B, label)  — tests A vs B, positive diff means A > B
    ("BiLSTM EMD²",      "BiLSTM CE",        "BiLSTM: EMD² vs CE"),
    ("BiLSTM OLL",       "BiLSTM CE",        "BiLSTM: OLL vs CE"),
    ("BiLSTM EMD²",      "BiLSTM OLL",       "BiLSTM: EMD² vs OLL"),
    ("Transformer EMD²", "Transformer CE",   "Transformer: EMD² vs CE"),
    ("Transformer OLL",  "Transformer CE",   "Transformer: OLL vs CE"),
    ("Transformer EMD²", "Transformer OLL",  "Transformer: EMD² vs OLL"),
    ("mDeBERTa EMD²",    "mDeBERTa CE",      "mDeBERTa: EMD² vs CE"),
    ("mDeBERTa LLRD+EMA","mDeBERTa EMD²",    "mDeBERTa: LLRD+EMA vs baseline"),
    ("Best (B27)",       "mDeBERTa LLRD+EMA","Best ensemble vs single LLRD+EMA"),
]

print("\n" + "="*100)
print(f"Pairwise tests — Bootstrap (paired) + Wilcoxon signed-rank  |  Bonferroni α={ALPHA_BON:.4f} (n={N_TESTS} tests)")
print("="*100)
print(f"{'Comparison':<42} {'Δ':>7}  {'Boot p':>8}  {'95% CI of Δ':>20}  {'Wilcox p':>9}  {'sig (Bonf.)':>12}")
print("-"*100)
for a, b, label in COMPARISONS:
    if a not in preds or b not in preds:
        print(f"{label:<42}  MISSING")
        continue
    diff, p_boot, lo, hi = bootstrap_diff(preds[a], preds[b], gt)
    _, p_wil, direction   = wilcoxon_test(preds[a], preds[b], gt)
    stars = sig_stars(p_boot, bonferroni=True)
    print(f"{label:<42} {diff:+.4f}  {p_boot:>8.4f}  [{lo:+.4f}, {hi:+.4f}]  {p_wil:>9.4f}  {stars:>12}")

print(f"\n*** p<0.001  ** p<0.01  * p<0.05 (all Bonferroni-corrected, α={ALPHA_BON:.4f})")
print(f"(*)n.s.† = significant before Bonferroni correction (p<0.05) but not after")
