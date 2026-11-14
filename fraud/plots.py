"""Charts for the report and notebook. One function per figure; each returns the Figure."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import average_precision_score, precision_recall_curve

from .config import FIGURES_DIR, TARGET
from .features import clock_features

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]  # fixed order; validated colour-blind safe as adjacent series
BLUE_RAMP = LinearSegmentedColormap.from_list("blue_ramp", ["#eef4fd", "#9ec5f4", "#2a78d6", "#0d366b"])


def use_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })  # fmt: skip


def save(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    return path


def _titles(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="semibold", pad=26 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")


def _percent_axis(axis) -> None:
    axis.set_major_formatter(lambda v, _: f"{v:g}%")


# ---------------------------------------------------------------------- the data


def fraud_rate_by_hour(raw: pd.DataFrame):
    hour, _ = clock_features(raw["unix_time"].to_numpy())
    rate = raw.groupby(hour)[TARGET].mean() * 100
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(rate.index, rate.values, width=0.62, color=BLUE)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.set_xlim(-0.7, 23.7)
    ax.grid(axis="x", visible=False)
    _percent_axis(ax.yaxis)
    ax.set_xlabel("Hour of day")
    _titles(ax, "Fraud is concentrated in six night-time hours", "Share of training transactions that are fraud, by hour of day")
    return fig


def fraud_rate_by_category(raw: pd.DataFrame):
    rate = (raw.groupby("category")[TARGET].mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.barh(rate.index, rate.values, height=0.55, color=BLUE)
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in rate.values], padding=4, color=INK_2, fontsize=9)
    ax.set_xlim(0, rate.max() * 1.12)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelcolor=INK_2)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    _percent_axis(ax.xaxis)
    _titles(ax, "Online shopping and grocery purchases carry the most fraud", "Share of training transactions that are fraud, by merchant category")
    return fig


def amount_distribution(raw: pd.DataFrame):
    bins = np.logspace(0, np.log10(raw["amt"].max()), 60)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for label, flag, color in [("Legitimate", 0, BLUE), ("Fraud", 1, ORANGE)]:
        amounts = raw.loc[raw[TARGET] == flag, "amt"]
        share = np.histogram(amounts, bins=bins)[0] / len(amounts) * 100
        ax.stairs(share, bins, color=color, linewidth=2, label=label)
        ax.stairs(share, bins, color=color, alpha=0.10, fill=True)
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1_000, 10_000], ["$1", "$10", "$100", "$1,000", "$10,000"])
    ax.minorticks_off()
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    _percent_axis(ax.yaxis)
    ax.set_xlabel("Transaction amount (log scale)")
    ax.legend(loc="upper right")
    _titles(ax, "Fraud clusters around specific amounts, mostly above $250", "Share of each class's transactions per amount bin")
    return fig


# -------------------------------------------------------------------- experiments


def pr_curves(y, scores: dict[str, np.ndarray], title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for (name, p), color in zip(scores.items(), SERIES):
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, color=color, label=f"{name}  (PR-AUC {average_precision_score(y, p):.3f})")
    ax.set_xlim(0, 1.005)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.legend(loc="lower left")
    _titles(ax, title, subtitle)
    return fig


def strategy_dotplot(table: pd.DataFrame, title: str, subtitle: str):
    """One panel per metric; a dot per strategy. Axes are zoomed, so dots rather than bars."""
    metrics = [("pr_auc", "PR-AUC"), ("precision", "Precision"), ("recall", "Recall")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), sharey=True)
    names = list(table.index)[::-1]
    rows = np.arange(len(names))
    for ax, (column, label) in zip(axes, metrics):
        values = table[column].loc[names].to_numpy()
        lo, hi = values.min(), values.max()
        pad = max(hi - lo, 0.004)
        ax.scatter(values, rows, s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        for value, row in zip(values, rows):
            ax.annotate(f"{value:.3f}", (value, row), xytext=(8, 0), textcoords="offset points", va="center", color=INK_2, fontsize=9)
        ax.set_xlim(lo - pad * 0.35, hi + pad * 0.75)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.set_yticks(rows, names)
        ax.set_xlabel(label)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(axis="y", labelcolor=INK_2)
    _titles(axes[0], title, subtitle)
    fig.subplots_adjust(wspace=0.12)
    return fig


def threshold_tradeoff(y, p, chosen: float):
    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for values, label, color in [(precision, "Precision", BLUE), (recall, "Recall", ORANGE), (f1, "F1", AQUA)]:
        ax.plot(thresholds, values, color=color, label=label)
    ax.axvline(chosen, color=INK_2, linewidth=1)
    ax.annotate(f"chosen threshold {chosen:.3f}", (chosen, 0.705), xytext=(6, 0), textcoords="offset points", color=INK_2, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.7, 1.005)
    ax.set_xlabel("Decision threshold on the model's fraud probability")
    ax.legend(loc="lower right")
    _titles(ax, "Raising the threshold trades missed fraud for fewer false alarms", "Validation data (transactions the model did not train on)")
    return fig


# --------------------------------------------------------------------- final test


def confusion_matrix(metrics: dict, title: str):
    counts = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    share = counts / counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    ax.imshow(share, cmap=BLUE_RAMP, vmin=0, vmax=1)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(2):
        for j in range(2):
            color = "#ffffff" if share[i, j] > 0.5 else INK
            ax.text(j, i - 0.08, f"{counts[i, j]:,}", ha="center", va="center", color=color, fontsize=15, fontweight="semibold")
            ax.text(j, i + 0.22, f"{share[i, j]:.2%} of row", ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks([0, 1], ["Approved", "Flagged as fraud"])
    ax.set_yticks([0, 1], ["Legitimate", "Fraud"])
    ax.tick_params(colors=INK_2)
    ax.set_xlabel("Model decision")
    ax.set_ylabel("Actual")
    # 2px surface gap between the four cells
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    _titles(ax, title)
    return fig


def test_pr_curve(y, p, metrics: dict):
    precision, recall, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(recall, precision, color=BLUE)
    ax.scatter([metrics["recall"]], [metrics["precision"]], s=80, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(
        f"threshold {metrics['threshold']:.3f}\nprecision {metrics['precision']:.1%}, recall {metrics['recall']:.1%}",
        (metrics["recall"], metrics["precision"]), xytext=(-12, -14), textcoords="offset points", ha="right", va="top", color=INK_2, fontsize=9,
    )  # fmt: skip
    ax.set_xlim(0, 1.005)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    _titles(ax, "Precision-recall on the held-out test set", f"PR-AUC {metrics['pr_auc']:.3f}; the dot is the deployed operating point")
    return fig


def feature_importance(importance: pd.Series, top: int = 15):
    top_features = importance.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.barh(top_features.index, top_features.values, height=0.55, color=BLUE)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in top_features.values], padding=4, color=INK_2, fontsize=9)
    ax.set_xlim(0, top_features.max() * 1.1)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelcolor=INK_2)
    ax.set_xlabel("Mean absolute SHAP contribution (log-odds)")
    _titles(ax, "What the model relies on", f"Average size of each feature's contribution to a prediction, top {top} features")
    return fig
