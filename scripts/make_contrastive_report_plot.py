from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


OUT_DIR = Path(__file__).resolve().parents[1] / "figures"


def annotate_bars(ax, bars, offset=0.00016, ha="center", inside=False, skip_first=False):
    for idx, bar in enumerate(bars):
        if skip_first and idx == 0:
            continue
        height = bar.get_height()
        y = height - 0.0010 if inside else height + offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{height:.4f}",
            ha=ha,
            va="top" if inside else "bottom",
            fontsize=6.8,
            color="white" if inside else "black",
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    categories = [
        "KNN-1",
        "KNN-7",
        "KNN-101",
        "Medoid\ndistribution",
        "Classification-\nSupCon ensemble",
        "Best\nclassification",
    ]

    standard_scores = np.array([0.8787, 0.9008, 0.9069, 0.9068]) 
    distance_scores = np.array([0.8767, 0.8997, 0.9066, 0.9056]) 
   
    ensemble_score = 0.907361607142857
    best_classification_score = 0.9078

    retrieval_color = "#2563eb"
    distance_color = "#dc2626"
    ensemble_color = retrieval_color
    classification_color = "#6b7280"

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.1, 2.8), constrained_layout=True)
    x = np.array([0.0, 0.86, 1.72, 2.58, 3.50, 4.34])
    width = 0.36
    standard_bars = ax.bar(
        x[:4] - width / 2,
        standard_scores,
        color=retrieval_color,
        width=width,
    )
    distance_bars = ax.bar(
        x[:4] + width / 2,
        distance_scores,
        color=distance_color,
        width=width,
    )
    ensemble_bar = ax.bar(x[4], ensemble_score, color=ensemble_color, width=0.62)
    classification_bar = ax.bar(x[5], best_classification_score, color=classification_color, width=0.62)

    annotate_bars(ax, [standard_bars[0]], ha="center")
    annotate_bars(ax, [distance_bars[0]], ha="center")
    annotate_bars(ax, standard_bars, ha="center", inside=True, skip_first=True)
    annotate_bars(ax, distance_bars, ha="center", inside=True, skip_first=True)
    annotate_bars(ax, ensemble_bar, inside=True)
    annotate_bars(ax, classification_bar, inside=True)

    ax.axvline(3.04, color="#9ca3af", linewidth=0.8, linestyle=":", alpha=0.9)
    ax.axvline(3.92, color="#9ca3af", linewidth=0.8, linestyle=":", alpha=0.9)

    ax.set_ylabel(r"Score ($1-\mathrm{MAE}/4$)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0.870, 0.910)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Patch(facecolor=retrieval_color, label="Standard SupCon"),
        Patch(facecolor=distance_color, label="Distance-weighted SupCon"),
        Patch(facecolor=classification_color, label="Best classification model"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        columnspacing=1.0,
        handlelength=1.4,
    )

    fig.savefig(OUT_DIR / "contrastive_results.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "contrastive_results.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "contrastive_results.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
