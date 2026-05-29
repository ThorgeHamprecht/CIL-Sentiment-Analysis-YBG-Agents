"""
Generate all report figures for the CIL Sentiment Analysis project.
Saves to report/figures/. Run from the project root.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import combinations

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
SUBMISSIONS   = ROOT / "submissions"
LABELS_PATH   = ROOT / "testlabels" / "test_solved.csv"
FIGURES_DIR   = ROOT / "report" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

NUM_CLASSES = 5
CLASSES     = [1, 2, 3, 4, 5]   # display labels (1-indexed for report)

# ── Ground truth ─────────────────────────────────────────────────────────────
gt        = pd.read_csv(LABELS_PATH).sort_values("id").reset_index(drop=True)
gt_labels = gt["label"].values   # 0-indexed (0..4)

def kaggle_score(preds, labels):
    return 1.0 - np.abs(np.array(preds) - np.array(labels)).mean() / 4.0

def load_sub(filename):
    path = SUBMISSIONS / filename
    df   = pd.read_csv(path).sort_values("id").reset_index(drop=True)
    return df["label"].values

def confusion_matrix_np(preds, labels, n=NUM_CLASSES):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(labels, preds):
        cm[t][p] += 1
    return cm

def error_dist(preds, labels):
    errors = np.abs(np.array(preds) - np.array(labels))
    return [(errors == k).sum() / len(labels) for k in range(NUM_CLASSES)]

def per_class_stats(preds, labels, n=NUM_CLASSES):
    preds, labels = np.array(preds), np.array(labels)
    acc, mae = [], []
    for c in range(n):
        mask = labels == c
        acc.append((preds[mask] == c).mean() if mask.sum() else 0)
        mae.append(np.abs(preds[mask] - c).mean() if mask.sum() else 0)
    return acc, mae

# ── All models ───────────────────────────────────────────────────────────────
# (filename_stem, display_label, family, param_millions)
ALL_MODELS = [
    # RNN / BiLSTM
    ("06_rnn_bilstm_submission",                    "BiLSTM baseline",             "RNN",       5),
    ("08_rnn_improved_submission",                  "BiLSTM improved",             "RNN",       5),
    ("submission_10_bilstm_ordinal",                "BiLSTM + ordinal loss",       "RNN",       5),
    ("15_bilstm_emd_submission",                    "BiLSTM + W²",                 "RNN",       5),
    ("submission_13_bilstm_fasttext",               "BiLSTM + FastText",           "RNN",       8),
    ("submission_18_bilstm_fasttext_emd",           "BiLSTM + FastText + W²",      "RNN",       8),
    ("submission_17_bilstm_ordinal_emd",            "BiLSTM + ordinal + W²",       "RNN",       5),
    ("24_bilstm_emd_ensemble_seed42_1337_2024_submission", "BiLSTM + W² + EMA ×3", "RNN",       5),
    # Custom Transformer
    ("07_transformer_custom_submission",            "Custom Transformer",           "Transformer", 10),
    ("submission_14_transformer_fasttext",          "Custom Transf. + FastText",   "Transformer", 13),
    ("16_transformer_emd_submission",               "Custom Transf. + W²",         "Transformer", 10),
    # Pre-trained
    ("09_mdeberta_coral_submission",                "mDeBERTa + CORAL",            "Pretrained",  278),
    ("19_mdeberta_emd_submission",                  "mDeBERTa + W²",               "Pretrained",  278),
    ("20_mdeberta_emd_v2_submission",               "mDeBERTa + W² + MSD",         "Pretrained",  278),
    ("21_xlmr_large_emd_submission",                "XLM-R Large + W²",            "Pretrained",  560),
    ("23_mdeberta_llrd_ema_seed42_submission",      "mDeBERTa LLRD+EMA seed42",    "Pretrained",  278),
    ("23_mdeberta_llrd_ema_seed2024_submission",    "mDeBERTa LLRD+EMA seed2024",  "Pretrained",  278),
    ("23_mdeberta_llrd_ema_submission",             "mDeBERTa LLRD+EMA ×3 seeds",  "Pretrained",  278),
    ("26_mdeberta_kfold_5fold_submission",          "mDeBERTa 5-fold CV",          "Pretrained",  278),
    ("27_mdeberta_seed_split_seed2024_submission",  "mDeBERTa seed-split s2024",   "Pretrained",  278),
    ("27_mdeberta_seed_split_seed42_1337_2024_submission", "mDeBERTa seed-split ×3 (ours)", "Pretrained", 278),
]

# Load all models
records = []
for fname, label, family, params in ALL_MODELS:
    path = SUBMISSIONS / f"{fname}.csv"
    if not path.exists():
        print(f"  [skip] {fname}")
        continue
    preds = load_sub(f"{fname}.csv")
    score = kaggle_score(preds, gt_labels)
    records.append({"file": fname, "label": label, "family": family,
                    "params": params, "score": score, "preds": preds})

df_models = pd.DataFrame([{k: v for k, v in r.items() if k != "preds"} for r in records])
preds_map  = {r["file"]: r["preds"] for r in records}

FAMILY_COLORS = {
    "RNN":         "#4C72B0",
    "Transformer": "#DD8452",
    "Pretrained":  "#55A868",
}

print(f"Loaded {len(records)} models.")
print(df_models[["label","score"]].sort_values("score", ascending=False).to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Score progression — all models grouped by family
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))

df_sorted = df_models.sort_values("score")
colors    = [FAMILY_COLORS[f] for f in df_sorted["family"]]
bars      = ax.barh(df_sorted["label"], df_sorted["score"], color=colors, edgecolor="white", height=0.7)

# Highlight best model
best_idx = df_sorted["score"].idxmax()
bars[df_sorted.index.get_loc(best_idx)].set_edgecolor("gold")
bars[df_sorted.index.get_loc(best_idx)].set_linewidth(2)

ax.set_xlabel("Test Score  (1 − MAE/4)")
ax.set_title("Test Scores Across All Submitted Models")
ax.set_xlim(0.85, 0.915)
ax.axvline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

for bar, (_, row) in zip(bars, df_sorted.iterrows()):
    ax.text(row["score"] + 0.0002, bar.get_y() + bar.get_height()/2,
            f"{row['score']:.4f}", va="center", fontsize=8)

legend_patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
ax.legend(handles=legend_patches, loc="lower right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_score_progression.pdf")
plt.savefig(FIGURES_DIR / "01_score_progression.png")
plt.close()
print("Saved: 01_score_progression")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Confusion matrices — best model vs key ablations
# ════════════════════════════════════════════════════════════════════════════
CM_MODELS = [
    ("27_mdeberta_seed_split_seed42_1337_2024_submission", "Best: mDeBERTa seed-split ×3"),
    ("23_mdeberta_llrd_ema_submission",                    "mDeBERTa LLRD+EMA ×3 seeds"),
    ("19_mdeberta_emd_submission",                         "mDeBERTa + W² (no LLRD/EMA)"),
    ("24_bilstm_emd_ensemble_seed42_1337_2024_submission", "BiLSTM + W² + EMA ×3"),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (fname, title) in zip(axes, CM_MODELS):
    if fname not in preds_map:
        ax.axis("off"); continue
    cm   = confusion_matrix_np(preds_map[fname], gt_labels)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)   # row-normalised
    sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                xticklabels=CLASSES, yticklabels=CLASSES,
                cbar=False, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.suptitle("Row-Normalised Confusion Matrices (Test Set)", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_confusion_matrices.pdf")
plt.savefig(FIGURES_DIR / "02_confusion_matrices.png")
plt.close()
print("Saved: 02_confusion_matrices")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Error distribution — key models
# ════════════════════════════════════════════════════════════════════════════
ERR_MODELS = [
    ("27_mdeberta_seed_split_seed42_1337_2024_submission", "mDeBERTa seed-split ×3 (best)"),
    ("23_mdeberta_llrd_ema_submission",                    "mDeBERTa LLRD+EMA ×3"),
    ("26_mdeberta_kfold_5fold_submission",                 "mDeBERTa 5-fold CV"),
    ("19_mdeberta_emd_submission",                         "mDeBERTa + W² (no LLRD)"),
    ("24_bilstm_emd_ensemble_seed42_1337_2024_submission", "BiLSTM ×3"),
    ("09_mdeberta_coral_submission",                       "mDeBERTa + CORAL"),
]
ERR_MODELS = [(f, l) for f, l in ERR_MODELS if f in preds_map]

x      = np.arange(NUM_CLASSES)
width  = 0.8 / len(ERR_MODELS)
fig, ax = plt.subplots(figsize=(10, 5))

for i, (fname, label) in enumerate(ERR_MODELS):
    dist = error_dist(preds_map[fname], gt_labels)
    offset = (i - len(ERR_MODELS)/2 + 0.5) * width
    ax.bar(x + offset, [d*100 for d in dist], width, label=label)

ax.set_xlabel("|Error| (absolute distance from true class)")
ax.set_ylabel("Percentage of predictions (%)")
ax.set_title("Error Distribution by Model")
ax.set_xticks(x)
ax.set_xticklabels([f"|error|={k}" for k in range(NUM_CLASSES)])
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_error_distribution.pdf")
plt.savefig(FIGURES_DIR / "03_error_distribution.png")
plt.close()
print("Saved: 03_error_distribution")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Per-class accuracy and MAE — best model vs ablations
# ════════════════════════════════════════════════════════════════════════════
PC_MODELS = [
    ("27_mdeberta_seed_split_seed42_1337_2024_submission", "mDeBERTa seed-split ×3 (best)"),
    ("23_mdeberta_llrd_ema_submission",                    "mDeBERTa LLRD+EMA ×3"),
    ("19_mdeberta_emd_submission",                         "mDeBERTa + W² (no LLRD)"),
    ("24_bilstm_emd_ensemble_seed42_1337_2024_submission", "BiLSTM ×3"),
]
PC_MODELS = [(f, l) for f, l in PC_MODELS if f in preds_map]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
x     = np.arange(NUM_CLASSES)
width = 0.8 / len(PC_MODELS)

for i, (fname, label) in enumerate(PC_MODELS):
    acc, mae = per_class_stats(preds_map[fname], gt_labels)
    offset   = (i - len(PC_MODELS)/2 + 0.5) * width
    ax1.bar(x + offset, acc, width, label=label)
    ax2.bar(x + offset, mae, width, label=label)

ax1.set_title("Per-Class Accuracy")
ax1.set_xlabel("True Class (star rating)")
ax1.set_ylabel("Accuracy")
ax1.set_xticks(x); ax1.set_xticklabels(CLASSES)
ax1.legend(fontsize=8)

ax2.set_title("Per-Class MAE")
ax2.set_xlabel("True Class (star rating)")
ax2.set_ylabel("Mean Absolute Error")
ax2.set_xticks(x); ax2.set_xticklabels(CLASSES)
ax2.legend(fontsize=8)

plt.suptitle("Per-Class Performance on Test Set", fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_per_class_stats.pdf")
plt.savefig(FIGURES_DIR / "04_per_class_stats.png")
plt.close()
print("Saved: 04_per_class_stats")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Pairwise prediction agreement heatmap
# ════════════════════════════════════════════════════════════════════════════
AGR_MODELS = [
    ("27_mdeberta_seed_split_seed42_1337_2024_submission", "27 seed-split ×3"),
    ("27_mdeberta_seed_split_seed2024_submission",         "27 seed 2024"),
    ("23_mdeberta_llrd_ema_submission",                    "23 ×3 seeds"),
    ("23_mdeberta_llrd_ema_seed42_submission",             "23 seed 42"),
    ("23_mdeberta_llrd_ema_seed2024_submission",           "23 seed 2024"),
    ("26_mdeberta_kfold_5fold_submission",                 "26 5-fold CV"),
    ("21_xlmr_large_emd_submission",                       "XLM-R Large"),
    ("20_mdeberta_emd_v2_submission",                      "mDeBERTa no LLRD"),
    ("24_bilstm_emd_ensemble_seed42_1337_2024_submission", "BiLSTM ×3"),
]
AGR_MODELS = [(f, l) for f, l in AGR_MODELS if f in preds_map]

n    = len(AGR_MODELS)
agr  = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        pi = preds_map[AGR_MODELS[i][0]]
        pj = preds_map[AGR_MODELS[j][0]]
        agr[i, j] = (pi == pj).mean()

labels_agr = [l for _, l in AGR_MODELS]
fig, ax    = plt.subplots(figsize=(9, 7))
mask       = np.eye(n, dtype=bool)
sns.heatmap(agr, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
            xticklabels=labels_agr, yticklabels=labels_agr,
            vmin=0.85, vmax=1.0, mask=mask, linewidths=0.5)
# diagonal — fill separately
for i in range(n):
    ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color="#dddddd", lw=0))
    ax.text(i+0.5, i+0.5, "1.000", ha="center", va="center", fontsize=8, color="black")

ax.set_title("Pairwise Prediction Agreement on Test Set")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_pairwise_agreement.pdf")
plt.savefig(FIGURES_DIR / "05_pairwise_agreement.png")
plt.close()
print("Saved: 05_pairwise_agreement")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Learning curves (hardcoded from training logs)
# ════════════════════════════════════════════════════════════════════════════
CURVES = {
    # baseline 27 — seed-split, independent val sets
    "27 seed 42  (split)":   [0.9053, 0.9075, 0.9076, 0.9075],
    "27 seed 1337 (split)":  [0.9038, 0.9067, 0.9073, 0.9072],
    "27 seed 2024 (split)":  [0.9055, 0.9077, 0.9079, 0.9085, 0.9078],
    # baseline 26 — k-fold
    "26 fold 0":  [0.9039, 0.9062, 0.9068, 0.9065],
    "26 fold 1":  [0.9044, 0.9067, 0.9073, 0.9072],
    "26 fold 2":  [0.9038, 0.9061, 0.9066, 0.9066],
    "26 fold 3":  [0.9051, 0.9078, 0.9081, 0.9083, 0.9081],
}

CURVE_STYLES = {
    "27 seed 42  (split)":   dict(color="#2196F3", ls="-",  lw=2),
    "27 seed 1337 (split)":  dict(color="#03A9F4", ls="--", lw=2),
    "27 seed 2024 (split)":  dict(color="#00BCD4", ls=":",  lw=2.5),
    "26 fold 0":  dict(color="#FF5722", ls="-",  lw=1.5, alpha=0.7),
    "26 fold 1":  dict(color="#FF7043", ls="--", lw=1.5, alpha=0.7),
    "26 fold 2":  dict(color="#FF8A65", ls=":",  lw=1.5, alpha=0.7),
    "26 fold 3":  dict(color="#BF360C", ls="-",  lw=2,   alpha=0.9),
}

fig, ax = plt.subplots(figsize=(9, 5))
for name, vals in CURVES.items():
    epochs = list(range(1, len(vals)+1))
    best_e = int(np.argmax(vals)) + 1
    style  = CURVE_STYLES[name]
    ax.plot(epochs, vals, marker="o", markersize=4, label=name, **style)
    ax.plot(best_e, max(vals), marker="*", markersize=10,
            color=style["color"], zorder=5)

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Score (1 − MAE/4)")
ax.set_title("Learning Curves — Baselines 26 (k-fold) and 27 (seed-split)")
ax.legend(fontsize=8, ncol=2)
ax.set_xticks(range(1, 6))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "06_learning_curves.pdf")
plt.savefig(FIGURES_DIR / "06_learning_curves.png")
plt.close()
print("Saved: 06_learning_curves")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Score vs model size (params)
# ════════════════════════════════════════════════════════════════════════════
SIZE_MODELS = [
    ("06_rnn_bilstm_submission",                    "BiLSTM",              5,   "RNN"),
    ("24_bilstm_emd_ensemble_seed42_1337_2024_submission", "BiLSTM ×3",    5,   "RNN"),
    ("07_transformer_custom_submission",            "Custom Transf.",      10,  "Transformer"),
    ("09_mdeberta_coral_submission",                "mDeBERTa CORAL",      278, "Pretrained"),
    ("19_mdeberta_emd_submission",                  "mDeBERTa +W²",        278, "Pretrained"),
    ("20_mdeberta_emd_v2_submission",               "mDeBERTa +W²+MSD",    278, "Pretrained"),
    ("23_mdeberta_llrd_ema_submission",             "mDeBERTa LLRD ×3",    278, "Pretrained"),
    ("27_mdeberta_seed_split_seed42_1337_2024_submission", "mDeBERTa split ×3", 278, "Pretrained"),
    ("21_xlmr_large_emd_submission",                "XLM-R Large",         560, "Pretrained"),
]
SIZE_MODELS = [(f, l, p, fam) for f, l, p, fam in SIZE_MODELS if f in preds_map]

fig, ax = plt.subplots(figsize=(8, 5))
for fname, label, params, family in SIZE_MODELS:
    score = kaggle_score(preds_map[fname], gt_labels)
    color = FAMILY_COLORS[family]
    ax.scatter(params, score, color=color, s=80, zorder=5)
    ax.annotate(label, (params, score), textcoords="offset points",
                xytext=(6, 3), fontsize=7.5)

ax.set_xscale("log")
ax.set_xlabel("Model Parameters (millions, log scale)")
ax.set_ylabel("Test Score (1 − MAE/4)")
ax.set_title("Score vs. Model Size")
legend_patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
ax.legend(handles=legend_patches)
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "07_score_vs_size.pdf")
plt.savefig(FIGURES_DIR / "07_score_vs_size.png")
plt.close()
print("Saved: 07_score_vs_size")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 8: Ablation — bar chart isolating each component
# ════════════════════════════════════════════════════════════════════════════
ABLATION = [
    ("09_mdeberta_coral_submission",                       "CORAL loss",                   "Loss"),
    ("19_mdeberta_emd_submission",                         "W² loss, no LLRD/EMA",         "Loss"),
    ("20_mdeberta_emd_v2_submission",                      "+ MSD + mean pool",            "Architecture"),
    ("23_mdeberta_llrd_ema_seed42_submission",             "+ LLRD + EMA (1 seed)",        "Fine-tuning"),
    ("23_mdeberta_llrd_ema_submission",                    "+ 3-seed ensemble",            "Ensemble"),
    ("26_mdeberta_kfold_5fold_submission",                 "5-fold CV ensemble",           "Ensemble"),
    ("27_mdeberta_seed_split_seed2024_submission",         "seed-split single",            "Ensemble"),
    ("27_mdeberta_seed_split_seed42_1337_2024_submission", "seed-split ×3 (best)",         "Ensemble"),
]
ABLATION = [(f, l, g) for f, l, g in ABLATION if f in preds_map]

GROUP_COLORS = {
    "Loss":         "#E74C3C",
    "Architecture": "#F39C12",
    "Fine-tuning":  "#27AE60",
    "Ensemble":     "#2980B9",
}

scores_abl = [kaggle_score(preds_map[f], gt_labels) for f, _, _ in ABLATION]
labels_abl = [l for _, l, _ in ABLATION]
colors_abl = [GROUP_COLORS[g] for _, _, g in ABLATION]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(labels_abl, scores_abl, color=colors_abl, edgecolor="white", height=0.6)
ax.set_xlabel("Test Score  (1 − MAE/4)")
ax.set_title("Ablation Study: Component Contributions")
ax.set_xlim(0.85, 0.912)
ax.axvline(scores_abl[-1], color="gold", linestyle="--", linewidth=1.2, label=f"Best: {scores_abl[-1]:.4f}")

for bar, score in zip(bars, scores_abl):
    ax.text(score + 0.0002, bar.get_y() + bar.get_height()/2,
            f"{score:.4f}", va="center", fontsize=8.5)

legend_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "08_ablation.pdf")
plt.savefig(FIGURES_DIR / "08_ablation.png")
plt.close()
print("Saved: 08_ablation")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 9: Architecture family comparison — best per family
# ════════════════════════════════════════════════════════════════════════════
FAMILY_BEST = [
    ("06_rnn_bilstm_submission",                    "BiLSTM (baseline)",      "RNN"),
    ("24_bilstm_emd_ensemble_seed42_1337_2024_submission", "BiLSTM best ×3",   "RNN"),
    ("07_transformer_custom_submission",            "Custom Transf. (base)",  "Transformer"),
    ("submission_14_transformer_fasttext",          "Custom Transf. best",    "Transformer"),
    ("19_mdeberta_emd_submission",                  "mDeBERTa (base)",        "Pretrained"),
    ("21_xlmr_large_emd_submission",                "XLM-R Large",            "Pretrained"),
    ("27_mdeberta_seed_split_seed42_1337_2024_submission", "mDeBERTa (best)", "Pretrained"),
]
FAMILY_BEST = [(f, l, fam) for f, l, fam in FAMILY_BEST if f in preds_map]

fig, ax = plt.subplots(figsize=(8, 4.5))
scores_fb = [kaggle_score(preds_map[f], gt_labels) for f, _, _ in FAMILY_BEST]
labels_fb = [l for _, l, _ in FAMILY_BEST]
colors_fb = [FAMILY_COLORS[fam] for _, _, fam in FAMILY_BEST]

bars = ax.bar(range(len(FAMILY_BEST)), scores_fb, color=colors_fb, edgecolor="white", width=0.6)
ax.set_xticks(range(len(FAMILY_BEST)))
ax.set_xticklabels(labels_fb, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Test Score  (1 − MAE/4)")
ax.set_title("Best Model per Architecture Family")
ax.set_ylim(0.86, 0.915)
ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

for bar, score in zip(bars, scores_fb):
    ax.text(bar.get_x() + bar.get_width()/2, score + 0.0003,
            f"{score:.4f}", ha="center", fontsize=8.5)

legend_patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
ax.legend(handles=legend_patches)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "09_family_comparison.pdf")
plt.savefig(FIGURES_DIR / "09_family_comparison.png")
plt.close()
print("Saved: 09_family_comparison")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 10: Ensemble gain — single seed vs ensemble
# ════════════════════════════════════════════════════════════════════════════
ENS_PAIRS = [
    # (single, ensemble, label)
    ("23_mdeberta_llrd_ema_seed42_submission",             "23_mdeberta_llrd_ema_submission",
     "23: fixed split\n(seed42 vs ×3)"),
    ("27_mdeberta_seed_split_seed2024_submission",         "27_mdeberta_seed_split_seed42_1337_2024_submission",
     "27: seed-split\n(seed2024 vs ×3)"),
]
ENS_PAIRS = [(s, e, l) for s, e, l in ENS_PAIRS if s in preds_map and e in preds_map]

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(ENS_PAIRS))
w = 0.35
singles   = [kaggle_score(preds_map[s], gt_labels) for s, _, _ in ENS_PAIRS]
ensembles = [kaggle_score(preds_map[e], gt_labels) for _, e, _ in ENS_PAIRS]

b1 = ax.bar(x - w/2, singles,   w, label="Best single model", color="#5b9bd5")
b2 = ax.bar(x + w/2, ensembles, w, label="Ensemble",          color="#ed7d31")

for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f"{bar.get_height():.4f}", ha="center", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([l for _, _, l in ENS_PAIRS])
ax.set_ylabel("Test Score  (1 − MAE/4)")
ax.set_title("Ensemble Gain: Single Model vs. 3-Model Ensemble")
ax.set_ylim(0.906, 0.9095)
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "10_ensemble_gain.pdf")
plt.savefig(FIGURES_DIR / "10_ensemble_gain.png")
plt.close()
print("Saved: 10_ensemble_gain")


print(f"\nAll figures saved to {FIGURES_DIR}/")
print("PDF + PNG for each figure.")
