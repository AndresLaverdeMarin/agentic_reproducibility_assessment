"""Plot ReScience C score, length, and disagreement distributions.

Run this file from Spyder or another environment with pandas and matplotlib.
ARA scores are transformed from the unit interval to the 1--4 ordinal scale
used by the human labels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCORE_CSV = (
    ROOT
    / "data"
    / "experiments"
    / "batch_runs"
    / "rescience_c_gemini3_1_pro_T0_result.csv"
)
PAPER_SUMMARY_CSV = ROOT / "data" / "resciencec" / "paper_summary.csv"
GROUND_TRUTH_CSV = ROOT / "data" / "resciencec" / "reproducibility_scores.csv"

SCORE_METRICS = (
    "R_sources",
    "R_methods",
    "R_experiments",
    "R_sinks",
    "R_c",
    "R_s",
    "R",
)

COMPARABLE_METRICS = (
    "all_nodes",
    "R_sources",
    "R_methods",
    "R_experiments",
    "R_sinks",
)

METRIC_LABELS = {
    "R_sources": "Sources",
    "R_methods": "Methods",
    "R_experiments": "Experiments",
    "R_sinks": "Sinks",
    "R_c": "Content",
    "R_s": "Structure",
    "R": "Composite",
    "all_nodes": "All nodes",
}

GROUND_TRUTH_COLUMNS = {
    "R_sources": "score_Sources",
    "R_methods": "score_Methods",
    "R_experiments": "score_Experiments",
    "R_sinks": "score_Sinks",
}

SAVE_FIGURE = False
OUTPUT_PATH = ROOT / "src" / "_analysis" / "figure_repro_score_distribution.png"
FIGSIZE = (15, 8)


def to_ordinal_scale(values: pd.DataFrame) -> pd.DataFrame:
    return 1.0 + (3.0 * values)


def load_scores(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    scores = data.loc[:, ["paperid", *SCORE_METRICS]].copy()
    scores = scores.rename(columns={"paperid": "paper_id"})
    scores.loc[:, SCORE_METRICS] = to_ordinal_scale(
        scores.loc[:, SCORE_METRICS].apply(pd.to_numeric, errors="coerce")
    )
    return scores


def load_paper_lengths(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    lengths = data.loc[:, ["paper_id", "num_pages", "num_words"]].copy()
    lengths.loc[:, ["num_pages", "num_words"]] = lengths.loc[
        :, ["num_pages", "num_words"]
    ].apply(pd.to_numeric, errors="coerce")
    return lengths.dropna()


def load_ground_truth(path: Path) -> pd.DataFrame:
    truth = pd.read_csv(path, sep=";")
    truth["paper_id"] = truth["paper"].str.replace(".pdf", "", regex=False)
    for column in GROUND_TRUTH_COLUMNS.values():
        truth[column] = pd.to_numeric(truth[column], errors="coerce")
    return truth


def build_analysis_table(
    scores: pd.DataFrame,
    lengths: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    data = scores.merge(lengths, on="paper_id", how="inner")
    data = data.merge(truth, on="paper_id", how="inner")

    stage_ara_columns = list(GROUND_TRUTH_COLUMNS)
    stage_truth_columns = list(GROUND_TRUTH_COLUMNS.values())
    data["all_nodes"] = data[stage_ara_columns].mean(axis=1)
    data["score_All_nodes"] = data[stage_truth_columns].mean(axis=1)

    for metric in COMPARABLE_METRICS:
        truth_column = "score_All_nodes" if metric == "all_nodes" else GROUND_TRUTH_COLUMNS[metric]
        data[f"disagreement_{metric}"] = (
            data[metric] - data[truth_column]
        )
    data["avg_disagreement"] = data["disagreement_all_nodes"]
    data["num_words_k"] = data["num_words"] / 1000.0
    return data


def plot_score_boxplot(ax, scores: pd.DataFrame) -> None:
    scores.loc[:, SCORE_METRICS].rename(columns=METRIC_LABELS).boxplot(
        ax=ax,
        grid=False,
        showmeans=True,
        patch_artist=True,
        medianprops={"color": "#1f2937", "linewidth": 1.5},
        meanprops={
            "marker": "o",
            "markerfacecolor": "#ffffff",
            "markeredgecolor": "#1f2937",
            "markersize": 4,
        },
        boxprops={"facecolor": "#dbeafe", "edgecolor": "#2563eb", "alpha": 0.85},
        whiskerprops={"color": "#2563eb"},
        capprops={"color": "#2563eb"},
    )
    ax.set_title("Reproducibility Scores")
    ax.set_ylabel("Score")
    ax.set_ylim(0.9, 4.1)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)


def plot_length_kde(ax, lengths: pd.DataFrame) -> None:
    lengths["num_pages"].clip(lower=0).plot.kde(
        ax=ax,
        bw_method=0.25,
        linewidth=1.8,
        label="Pages",
        color="#2563eb",
    )
    lengths["num_words_k"].clip(lower=0).plot.kde(
        ax=ax,
        bw_method=0.25,
        linewidth=1.8,
        label="Words (thousands)",
        color="#ea580c",
    )
    ax.set_title("Paper Lengths")
    ax.set_xlabel("Pages or thousand words")
    ax.set_ylabel("Density")
    ax.set_xlim(left=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)


def plot_disagreement_boxplot(ax, data: pd.DataFrame) -> None:
    columns = [f"disagreement_{metric}" for metric in COMPARABLE_METRICS]
    labels = {f"disagreement_{metric}": METRIC_LABELS[metric] for metric in COMPARABLE_METRICS}
    data.loc[:, columns].rename(columns=labels).boxplot(
        ax=ax,
        grid=False,
        showmeans=True,
        patch_artist=True,
        medianprops={"color": "#1f2937", "linewidth": 1.5},
        meanprops={
            "marker": "o",
            "markerfacecolor": "#ffffff",
            "markeredgecolor": "#1f2937",
            "markersize": 4,
        },
        boxprops={"facecolor": "#fee2e2", "edgecolor": "#dc2626", "alpha": 0.85},
        whiskerprops={"color": "#dc2626"},
        capprops={"color": "#dc2626"},
    )
    ax.set_title("Prediction Disagreement")
    ax.set_ylabel("ARA score - ground truth")
    max_abs = data.loc[:, columns].abs().max().max()
    max_abs = 1.0 if pd.isna(max_abs) else max(1.0, float(max_abs))
    ax.set_ylim(-max_abs - 0.1, max_abs + 0.1)
    ax.axhline(0, color="#6b7280", linewidth=1, alpha=0.7)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)


def plot_disagreement_over_length(ax, data: pd.DataFrame) -> None:
    ax.scatter(
        data["num_words_k"],
        data["avg_disagreement"],
        s=22,
        alpha=0.65,
        linewidths=0,
        color="#dc2626",
    )
    ax.set_title("Disagreement over Length")
    ax.set_xlabel("Words (thousands)")
    ax.set_ylabel("Mean disagreement")
    ax.set_xlim(left=0)
    max_abs = data["avg_disagreement"].abs().max()
    max_abs = 1.0 if pd.isna(max_abs) else max(1.0, float(max_abs))
    ax.set_ylim(-max_abs - 0.1, max_abs + 0.1)
    ax.axhline(0, color="#6b7280", linewidth=1, alpha=0.7)
    ax.grid(alpha=0.25)


def plot_disagreement_over_pages(ax, data: pd.DataFrame) -> None:
    ax.hist(
        data["num_words_k"],
        bins=20,
        color="#ea580c",
        alpha=0.82,
        edgecolor="#9a3412",
        linewidth=0.6,
    )
    ax.set_title("Paper Length Histogram")
    ax.set_xlabel("Words (thousands)")
    ax.set_ylabel("Number of papers")
    ax.set_xlim(left=0)
    ax.grid(axis="y", alpha=0.25)


def plot_score_over_words(ax, data: pd.DataFrame) -> None:
    ax.scatter(
        data["num_words_k"],
        data["R"],
        s=22,
        alpha=0.65,
        linewidths=0,
        color="#2563eb",
    )
    ax.set_title("Composite Score over Words")
    ax.set_xlabel("Words (thousands)")
    ax.set_ylabel("Composite score")
    ax.set_xlim(left=0)
    ax.set_ylim(0.9, 4.1)
    ax.grid(alpha=0.25)


def plot_score_over_pages(ax, data: pd.DataFrame) -> None:
    ax.scatter(
        data["num_pages"],
        data["R"],
        s=22,
        alpha=0.65,
        linewidths=0,
        color="#2563eb",
    )
    ax.set_title("Composite Score over Pages")
    ax.set_xlabel("Pages")
    ax.set_ylabel("Composite score")
    ax.set_xlim(left=0)
    ax.set_ylim(0.9, 4.1)
    ax.grid(alpha=0.25)


def plot_composite_score_bars(ax, data: pd.DataFrame) -> None:
    ordered = data.sort_values("R", ascending=True)
    ax.barh(
        ordered["paper_id"],
        ordered["R"],
        color="#2563eb",
        alpha=0.82,
    )
    ax.set_title("Composite Score by Paper")
    ax.set_xlabel("Composite score")
    ax.set_ylabel("Paper")
    ax.set_xlim(1.0, 4.0)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.25)


def plot_placeholder(ax) -> None:
    ax.axis("off")


scores = load_scores(SCORE_CSV)
paper_lengths = load_paper_lengths(PAPER_SUMMARY_CSV)
ground_truth = load_ground_truth(GROUND_TRUTH_CSV)
analysis = build_analysis_table(scores, paper_lengths, ground_truth)

fig, axes = plt.subplots(2, 4, figsize=FIGSIZE, constrained_layout=True)
plot_score_boxplot(axes[0, 0], scores)
plot_length_kde(axes[0, 1], analysis)
plot_disagreement_boxplot(axes[0, 2], analysis)
plot_composite_score_bars(axes[0, 3], analysis)
plot_disagreement_over_length(axes[1, 0], analysis)
plot_disagreement_over_pages(axes[1, 1], analysis)
plot_score_over_words(axes[1, 2], analysis)
plot_score_over_pages(axes[1, 3], analysis)

if SAVE_FIGURE:
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")

plt.show()
