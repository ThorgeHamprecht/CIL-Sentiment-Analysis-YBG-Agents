"""
Generate Plots 2, 4, 5 for the CIL report.
Run locally — only needs submissions/ and testlabels/.
Plot 3 (median vs argmax) generated separately after cluster job finishes.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns

ROOT        = Path(__file__).parent
SUBMISSIONS = ROOT / "submissions"
LABELS_PATH = ROOT / "testlabels" / "test_solved.csv"
FIGURES_DIR = ROOT / "report" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":     150,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
})

NUM_CLASSES = 5
CLASSES     = ["1*", "2*", "3*", "4*", "5*"]

gt        = pd.read_csv(LABELS_PATH).sort_values("id").reset_index(drop=True)
gt_labels = gt["label"].values

def score(fname):
    df = pd.read_csv(SUBMISSIONS / fname).sort_values("id").reset_index(drop=True)
    return 1.0 - np.abs(df["label"].values - gt_labels).mean() / 4.0

def preds(fname):
    df = pd.read_csv(SUBMISSIONS / fname).sort_values("id").reset_index(drop=True)
    return df["label"].values

def confusion_norm(p, l, n=NUM_CLASSES):
    cm = np.zeros((n, n), dtype=int)
    for t, pr in zip(l, p): cm[t][pr] += 1
    return cm.astype(float) / cm.sum(axis=1, keepdims=True)

def error_dist(p, l):
    e = np.abs(np.array(p) - np.array(l))
    return [(e == k).mean() * 100 for k in range(NUM_CLASSES)]


# ════════════════════════════════════════════════════════════════════════════
# PLOT 2: Loss function comparison — single panel, no ensemble models
# ════════════════════════════════════════════════════════════════════════════

loss_color_map = {
    "Cross-Entropy": "#e74c3c",
    "OLL":           "#e67e22",
    "EMD²":            "#27ae60",
}

PANEL_A = [
    # (file, loss_type, family_label)
    ("06_rnn_bilstm_submission.csv",         "Cross-Entropy", "BiLSTM"),
    ("29_bilstm_oll_submission.csv",         "OLL",           "BiLSTM"),
    ("15_bilstm_emd_submission.csv",         "EMD²",            "BiLSTM"),
    ("07_transformer_custom_submission.csv", "Cross-Entropy", "Custom\nTransformer"),
    ("30_transformer_oll_submission.csv",    "OLL",           "Custom\nTransformer"),
    ("16_transformer_emd_submission.csv",    "EMD²",            "Custom\nTransformer"),
    ("28_mdeberta_ce_submission.csv",        "Cross-Entropy", "mDeBERTa"),
    ("20_mdeberta_emd_v2_submission.csv",     "EMD²",            "mDeBERTa"),
]

fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

labels_a, scores_a, colors_a = [], [], []
for fname, loss, fam in PANEL_A:
    if not (SUBMISSIONS / fname).exists(): continue
    labels_a.append(f"{fam}\n{loss}")
    scores_a.append(score(fname))
    colors_a.append(loss_color_map[loss])

x    = np.arange(len(labels_a))
bars = ax1.bar(x, scores_a, color=colors_a, edgecolor="white", width=0.65, zorder=3)
ax1.set_xticks(x)
ax1.set_xticklabels(labels_a, fontsize=8)
ax1.set_ylabel("Test Score  (1 − MAE/4)")
ax1.set_ylim(0.855, 0.915)
ax1.axhline(0.9, color="gray", linestyle="--", lw=0.8, alpha=0.5)
ax1.grid(axis="y", alpha=0.3, zorder=0)
ax1.set_axisbelow(True)

for bar, s in zip(bars, scores_a):
    ax1.text(bar.get_x() + bar.get_width()/2, s + 0.0005,
             f"{s:.4f}", ha="center", fontsize=7, rotation=90)

# Group separators
group_bounds = []
prev_fam = None
for i, (_, _, fam) in enumerate([(f, l, fm) for f, l, fm in PANEL_A
                                   if (SUBMISSIONS / f).exists()]):
    if prev_fam and fam != prev_fam:
        group_bounds.append(i - 0.5)
    prev_fam = fam
for gb in group_bounds:
    ax1.axvline(gb, color="grey", lw=0.8, linestyle="--", alpha=0.5)

legend_patches = [mpatches.Patch(color=c, label=l) for l, c in loss_color_map.items()]
ax1.legend(handles=legend_patches, fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "plot2_loss_comparison.pdf")
plt.close()
print("Saved: plot2_loss_comparison")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 4: Fine-tuning strategy + Ensembling as human rater aggregation
#   (A) LLRD + EMA ablation — new panel
#   (B) Learning curves (seed-split B27)
#   (C) Ensemble gain
#   (D) Conceptual — ensemble as human rater aggregation
# ════════════════════════════════════════════════════════════════════════════

fig, (ax_ablation, ax_ens) = plt.subplots(1, 2, figsize=(13, 5))

# ── Panel A: LLRD + EMA ablation ────────────────────────────────────────────
# BiLSTM:   B17 (EMD², no EMA)  vs  B24 (EMD², EMA, ×3 ens)
# mDeBERTa: B19 (no LLRD/EMA)  vs  B23-seed42 (LLRD+EMA, single seed)
ABLATION = [
    # (label, file, group, has_technique)
    ("BiLSTM\nno EMA",           "submission_17_bilstm_ordinal_emd.csv",       "BiLSTM",   False),
    ("BiLSTM\n+EMA (×3 ens.)",   "24_bilstm_emd_ensemble_seed42_1337_2024_submission.csv", "BiLSTM", True),
    ("mDeBERTa\nno LLRD/EMA",    "19_mdeberta_emd_submission.csv",             "mDeBERTa", False),
    ("mDeBERTa\n+LLRD+EMA",      "23_mdeberta_llrd_ema_seed42_submission.csv", "mDeBERTa", True),
]

abl_labels, abl_scores, abl_colors, abl_groups = [], [], [], []
COLOR_BASE = {"BiLSTM": "#ef8f3c", "mDeBERTa": "#2196F3"}
COLOR_TECH = {"BiLSTM": "#b35b00", "mDeBERTa": "#0d47a1"}
for lbl, fname, grp, has_tech in ABLATION:
    if not (SUBMISSIONS / fname).exists():
        continue
    abl_labels.append(lbl)
    abl_scores.append(score(fname))
    abl_colors.append(COLOR_TECH[grp] if has_tech else COLOR_BASE[grp])
    abl_groups.append(grp)

x_abl  = np.arange(len(abl_labels))
bars_a = ax_ablation.bar(x_abl, abl_scores, color=abl_colors,
                         edgecolor="white", width=0.55, zorder=3)
ax_ablation.set_xticks(x_abl)
ax_ablation.set_xticklabels(abl_labels, fontsize=8.5)
ax_ablation.set_ylabel("Test Score  (1 − MAE/4)")
ax_ablation.grid(True, alpha=0.2, axis="y", zorder=0)
ax_ablation.set_axisbelow(True)
# Zoom in to make gains visible — y-axis starts just below the minimum score
y_min = min(abl_scores) - 0.003
y_max = max(abl_scores) + 0.006
ax_ablation.set_ylim(y_min, y_max)
# Broken-axis indicator at the bottom
ax_ablation.annotate("", xy=(0, 0), xytext=(0, 0))  # dummy; spine break drawn below
for spine in ["left", "bottom"]:
    ax_ablation.spines[spine].set_linewidth(0.8)
ax_ablation.spines["bottom"].set_visible(False)
ax_ablation.tick_params(bottom=False)
# Draw double-slash break marker on y-axis
d = 0.012
kwargs = dict(transform=ax_ablation.transAxes, color="k", clip_on=False, lw=1)
ax_ablation.plot((-d, +d), (-d*0.6, +d*0.6), **kwargs)
ax_ablation.plot((-d, +d), (-d*0.6 + 0.022, +d*0.6 + 0.022), **kwargs)

# Score labels on bars
for bar, s in zip(bars_a, abl_scores):
    ax_ablation.text(bar.get_x() + bar.get_width()/2, s + 0.00015,
                     f"{s:.4f}", ha="center", fontsize=8.5)

# Delta annotations between paired bars (within each group)
pair_indices = [(0, 1), (2, 3)]  # BiLSTM pair, mDeBERTa pair
for i_base, i_tech in pair_indices:
    if i_tech >= len(abl_scores): continue
    diff = abl_scores[i_tech] - abl_scores[i_base]
    mid_x = (i_base + i_tech) / 2
    top_y = max(abl_scores[i_base], abl_scores[i_tech])
    ax_ablation.annotate(
        "", xy=(i_tech, abl_scores[i_tech]),
        xytext=(i_base, abl_scores[i_base]),
        arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax_ablation.text(mid_x, top_y + 0.0012,
                     f"+{diff:.4f}", ha="center", fontsize=8.5,
                     color="#1a1a1a", fontweight="bold")

# Group separators and labels
ax_ablation.axvline(1.5, color="grey", lw=0.8, linestyle="--", alpha=0.5)
for grp_x, grp_name in [(0.5, "BiLSTM"), (2.5, "mDeBERTa")]:
    ax_ablation.text(grp_x, y_min + (y_max - y_min)*0.03,
                     grp_name, ha="center", fontsize=8, color="grey",
                     fontstyle="italic")



# ── Panel C: Ensemble gain with diversity annotation ────────────────────────
ENS_DATA = [
    ("Best single model\n(seed 2024)",      "27_mdeberta_seed_split_seed2024_submission.csv",         "#64b5f6"),
    ("Fixed-split ×3\n(same data, B23)",    "23_mdeberta_llrd_ema_submission.csv",                    "#1976d2"),
    ("Seed-split ×3\n(indep. data, B27)",   "27_mdeberta_seed_split_seed42_1337_2024_submission.csv", "#0d47a1"),
]
ens_labels, ens_scores = [], []
for lbl, fname, _ in ENS_DATA:
    if not (SUBMISSIONS / fname).exists(): continue
    ens_labels.append(lbl)
    ens_scores.append(score(fname))

ens_colors = [d[2] for d in ENS_DATA[:len(ens_labels)]]
bars_e = ax_ens.bar(range(len(ens_labels)), ens_scores, color=ens_colors,
                    edgecolor="white", width=0.5)

for bar, s in zip(bars_e, ens_scores):
    ax_ens.text(bar.get_x() + bar.get_width()/2, s + 0.00003,
                f"{s:.4f}", ha="center", fontsize=9)

if len(ens_scores) >= 2:
    for i in range(1, len(ens_scores)):
        diff = ens_scores[i] - ens_scores[i-1]
        mid  = (i - 1 + i) / 2
        ax_ens.annotate("", xy=(i, ens_scores[i]), xytext=(i-1, ens_scores[i-1]),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=1))
        ax_ens.text(mid, max(ens_scores[i], ens_scores[i-1]) + 0.0003,
                    f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}",
                    ha="center", fontsize=8, color="#333")

ax_ens.set_xticks(range(len(ens_labels)))
ax_ens.set_xticklabels(ens_labels, fontsize=8.5)
ax_ens.set_ylabel("Test Score  (1 − MAE/4)")
ax_ens.set_ylim(0.9065, 0.9092)
ax_ens.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "plot4_finetuning_ensemble.pdf")
plt.close()
print("Saved: plot4_finetuning_ensemble")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 5: The diagonal — ordinal structure in errors
# ════════════════════════════════════════════════════════════════════════════

best_preds   = preds("27_mdeberta_seed_split_seed42_1337_2024_submission.csv")
mdeberta_ce_preds = preds("28_mdeberta_ce_submission.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ── Panel A: Confusion matrix best model ────────────────────────────────────
cm_best = confusion_norm(best_preds, gt_labels)
sns.heatmap(cm_best, annot=True, fmt=".2f", cmap="Blues", ax=axes[0],
            xticklabels=CLASSES, yticklabels=CLASSES, cbar=True,
            vmin=0, vmax=1, linewidths=0.5)
axes[0].set_xlabel("Predicted rating")
axes[0].set_ylabel("True rating")

# ── Panel B: Error distribution — best model vs mDeBERTa CE ─────────────────
err_best = error_dist(best_preds,       gt_labels)
err_ce   = error_dist(mdeberta_ce_preds, gt_labels)

x  = np.arange(NUM_CLASSES)
w  = 0.35
axes[1].bar(x - w/2, err_best, w, label="Best model (EMD²)", color="#27ae60")
axes[1].bar(x + w/2, err_ce,   w, label="mDeBERTa (CE)",     color="#e74c3c")

axes[1].set_xticks(x)
axes[1].set_xticklabels([f"|error|={k}" for k in range(NUM_CLASSES)])
axes[1].set_ylabel("% of test predictions")
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="y")

# Annotate exact@0
exact_best = err_best[0]
exact_ce   = err_ce[0]
axes[1].text(0 - w/2, exact_best + 0.5, f"{exact_best:.1f}%",
             ha="center", fontsize=8, color="#27ae60", fontweight="bold")
axes[1].text(0 + w/2, exact_ce + 0.5, f"{exact_ce:.1f}%",
             ha="center", fontsize=8, color="#e74c3c", fontweight="bold")


plt.tight_layout()
plt.savefig(FIGURES_DIR / "plot5_ordinal_errors.pdf")
plt.close()
print("Saved: plot5_ordinal_errors")

print(f"\nAll report plots saved to {FIGURES_DIR}/")
