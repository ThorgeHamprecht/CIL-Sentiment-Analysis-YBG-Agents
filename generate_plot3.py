"""
Plot 3: Median vs Argmax decode — which is actually better?

Compares decoder performance against ground truth across two architectures:
  - submissions/seed2024_test_probs.npy       (mDeBERTa-v3-base, seed 2024)
  - submissions/bilstm_seed42_test_probs.npy  (TwoStream BiLSTM, seed 42)

Panels:
  A: Test scores — median vs argmax for both architectures (the key result)
  B: On the ~3-5% disagreement cases: which decoder is right more often?
  C: Error distribution |error|=0..4, median vs argmax, mDeBERTa
  D: Error distribution |error|=0..4, median vs argmax, BiLSTM

Saves to report/figures/plot3_decoder_comparison.{pdf,png}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT    = Path(__file__).resolve().parent
SUBS    = ROOT / "submissions"
GT_PATH = ROOT / "testlabels" / "test_solved.csv"
OUT_DIR = ROOT / "report" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MDEBERTA_PROBS = SUBS / "seed2024_test_probs.npy"
BILSTM_PROBS   = SUBS / "bilstm_seed42_test_probs.npy"

C_MEDIAN = "#2166ac"
C_ARGMAX = "#d6604d"
N_CLASSES = 5


def load_gt():
    df = pd.read_csv(GT_PATH).sort_values("id").reset_index(drop=True)
    return df["label"].values   # 1-indexed (1–5)


def decode(probs):
    cdf        = np.cumsum(probs, axis=1)[:, :-1]
    median_pred = (cdf < 0.5).sum(axis=1).clip(0, 4)   # 0-indexed
    argmax_pred = probs.argmax(axis=1)                  # 0-indexed
    return median_pred, argmax_pred


def kaggle_score(preds_0idx, gt_0idx):
    """Both preds and gt are 0-indexed (0–4)."""
    return 1.0 - np.abs(preds_0idx - gt_0idx).mean() / 4.0


def error_dist(preds_0idx, gt_0idx):
    e = np.abs(preds_0idx - gt_0idx)
    return [(e == k).mean() * 100 for k in range(N_CLASSES)]


def main():
    gt = load_gt()   # 0-indexed (0–4)

    probs_m = np.load(MDEBERTA_PROBS)
    probs_b = np.load(BILSTM_PROBS)
    print(f"mDeBERTa probs: {probs_m.shape}")
    print(f"BiLSTM   probs: {probs_b.shape}")

    med_m, arg_m = decode(probs_m)
    med_b, arg_b = decode(probs_b)

    # ── Scores ───────────────────────────────────────────────────────────────
    scores = {
        "mDeBERTa\nmedian": kaggle_score(med_m, gt),
        "mDeBERTa\nargmax": kaggle_score(arg_m, gt),
        "BiLSTM\nmedian":   kaggle_score(med_b, gt),
        "BiLSTM\nargmax":   kaggle_score(arg_b, gt),
    }
    for k, v in scores.items():
        print(f"{k.replace(chr(10),' ')}: {v:.4f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))

    vals_a   = list(scores.values())
    colors_a = [C_MEDIAN, C_ARGMAX, C_MEDIAN, C_ARGMAX]
    bars = ax.bar(range(4), vals_a, color=colors_a, edgecolor="white",
                  width=0.55, zorder=3)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel("Test Score  (1 − MAE/4)")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Truncate y-axis to highlight differences
    y_min = min(vals_a) - 0.003
    y_max = max(vals_a) + 0.004
    ax.set_ylim(y_min, y_max)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)
    d = 0.012
    kw = dict(transform=ax.transAxes, color="k", clip_on=False, lw=1)
    ax.plot((-d, +d), (-d*0.6, +d*0.6), **kw)
    ax.plot((-d, +d), (-d*0.6+0.022, +d*0.6+0.022), **kw)

    for bar, v in zip(bars, vals_a):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.00015,
                f"{v:.4f}", ha="center", fontsize=8.5)

    # Group separators + labels below the x-axis
    ax.axvline(1.5, color="grey", lw=0.8, linestyle="--", alpha=0.5)
    for gx, gname in [(0.5, "mDeBERTa"), (2.5, "BiLSTM")]:
        ax.text(gx, -0.06, gname, ha="center", fontsize=9, color="grey",
                fontstyle="italic", transform=ax.get_xaxis_transform())

    legend_patches = [mpatches.Patch(color=C_MEDIAN, label="Median decode"),
                      mpatches.Patch(color=C_ARGMAX, label="Argmax decode")]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper right")

    out = OUT_DIR / "plot3_decoder_comparison.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
