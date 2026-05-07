"""Plot ReScience C score, length, and disagreement distributions.

Run this file from Spyder or another environment with pandas and matplotlib.
ARA scores are transformed from the unit interval to the 1--4 ordinal scale
used by the human labels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


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

NODE_COUNT_METRICS = (
    "n_sources",
    "n_methods",
    "n_experiments",
    "n_sinks",
)

COMPARABLE_METRICS = (
    "all_nodes",
    "R_sources",
    "R_methods",
    "R_experiments",
    "R_sinks",
)

DISAGREEMENT_LABELS = {
    "R_sources": r"$V_{\mathrm{sources}}$",
    "R_methods": r"$V_{\mathrm{methods}}$",
    "R_experiments": r"$V_{\mathrm{experiments}}$",
    "R_sinks": r"$V_{\mathrm{sinks}}$",
    "all_nodes": r"$V$",
}

METRIC_LABELS = {
    "R_sources": r"$R_{\mathrm{sources}}$",
    "R_methods": r"$R_{\mathrm{methods}}$",
    "R_experiments": r"$R_{\mathrm{experiments}}$",
    "R_sinks": r"$R_{\mathrm{sinks}}$",
    "R_c": r"$R_c$",
    "R_s": r"$R_s$",
    "R": r"$R$",
    "all_nodes": r"$V$",
}

GROUND_TRUTH_COLUMNS = {
    "R_sources": "score_Sources",
    "R_methods": "score_Methods",
    "R_experiments": "score_Experiments",
    "R_sinks": "score_Sinks",
}

FIG_RES = 5.5
FIGSIZE = (FIG_RES*2, 3.1)

SUBPLOT_COLORS = {
    1: (255 / 255, 0 / 255, 0 / 255),
    2: (146 / 255, 208 / 255, 80 / 255),
    3: (0 / 255, 176 / 255, 240 / 255),
    4: (0 / 255, 112 / 255, 192 / 255),
    5: "black",
    6: "black",
}


def to_ordinal_scale(values: pd.DataFrame) -> pd.DataFrame:
    return 1.0 + (3.0 * values)


def load_scores(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    scores = data.loc[:, ["paperid", *SCORE_METRICS, *NODE_COUNT_METRICS]].copy()
    scores = scores.rename(columns={"paperid": "paper_id"})
    scores.loc[:, SCORE_METRICS] = to_ordinal_scale(
        scores.loc[:, SCORE_METRICS].apply(pd.to_numeric, errors="coerce")
    )
    scores.loc[:, NODE_COUNT_METRICS] = scores.loc[:, NODE_COUNT_METRICS].apply(
        pd.to_numeric,
        errors="coerce",
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
    data["n_nodes"] = data.loc[:, NODE_COUNT_METRICS].sum(axis=1)

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


def set_square_subplot() -> None:
    plt.gca().set_box_aspect(1)


def regression_stats(x_values: pd.Series, y_values: pd.Series) -> dict[str, float]:
    regression_data = pd.DataFrame({"x": x_values, "y": y_values}).dropna()
    result = stats.linregress(regression_data["x"], regression_data["y"])
    t_statistic = result.slope / result.stderr if result.stderr else np.nan
    f_statistic = t_statistic**2
    return {
        "slope": result.slope,
        "intercept": result.intercept,
        "r_squared": result.rvalue**2,
        "t_statistic": t_statistic,
        "f_statistic": f_statistic,
        "p_value": result.pvalue,
    }


def regression_label(values: dict[str, float]) -> str:
    return (
        f"$R^2$ = {values['r_squared']:.2f}\n"
        f"F = {values['f_statistic']:.2f}\n"
        f"t = {values['t_statistic']:.2f}\n"
        f"p = {values['p_value']:.3f}"
    )


def plot_regression_line(
    x_values: pd.Series,
    values: dict[str, float],
    color: str | tuple[float, float, float] = "#111827",
) -> None:
    x_line = np.linspace(float(x_values.min()), float(x_values.max()), 100)
    y_line = values["intercept"] + values["slope"] * x_line
    plt.plot(x_line, y_line, color=color, linewidth=1.4)


def plot_paper_length_kde(data: pd.DataFrame) -> None:
    data["num_pages"].clip(lower=0).plot.kde(
        bw_method=0.25,
        linewidth=1.8,
        color=SUBPLOT_COLORS[1],
        alpha=0.8,
        label="Pages",
        linestyle="--",
    )
    data["num_words_k"].clip(lower=0).plot.kde(
        bw_method=0.25,
        linewidth=1.8,
        color=SUBPLOT_COLORS[1],
        alpha=0.8,
        label="Words (thousands)",
        linestyle="-",
    )
    data["n_nodes"].clip(lower=0).plot.kde(
        bw_method=0.25,
        linewidth=1.8,
        color=SUBPLOT_COLORS[1],
        alpha=0.8,
        label="Nodes",
        linestyle=":",
    )
    plt.title("Paper Length", fontweight="bold")
    plt.xlabel("Pages (or thousand words)")
    plt.ylabel("")
    plt.xticks([0, 10, 20, 30, 40])
    plt.yticks([])
    plt.xlim(0, 40)
    plt.legend(fontsize="small", loc="upper right")
    set_square_subplot()


def plot_score_boxplot(scores: pd.DataFrame) -> None:
    labels = [METRIC_LABELS[metric] for metric in SCORE_METRICS]
    values = [scores[metric].dropna() for metric in SCORE_METRICS]
    plt.boxplot(
        values,
        labels=labels,
        showmeans=True,
        patch_artist=True,
        medianprops={"color": "#1f2937", "linewidth": 1.5},
        meanprops={
            "marker": "o",
            "markerfacecolor": "#ffffff",
            "markeredgecolor": "#1f2937",
            "markersize": 4,
        },
        boxprops={"facecolor": SUBPLOT_COLORS[1], "edgecolor": "#7f1d1d", "alpha": 0.85},
        whiskerprops={"color": "#7f1d1d"},
        capprops={"color": "#7f1d1d"},
    )
    plt.title("Reproducibility Score", fontweight="bold")
    # plt.xlabel("Score component")
    # plt.ylabel("Score")
    plt.xticks(range(1, len(labels) + 1), labels, rotation=35, ha="right")
    plt.yticks([1, 2, 3, 4])
    plt.ylim(0.9, 4.1)
    set_square_subplot()


def plot_disagreement_boxplot(data: pd.DataFrame) -> None:
    columns = [f"disagreement_{metric}" for metric in COMPARABLE_METRICS]
    labels = [METRIC_LABELS[metric] for metric in COMPARABLE_METRICS]
    labels = [DISAGREEMENT_LABELS[metric] for metric in COMPARABLE_METRICS]
    values = [data[column].dropna() for column in columns]
    plt.boxplot(
        values,
        labels=labels,
        showmeans=True,
        patch_artist=True,
        medianprops={"color": "#1f2937", "linewidth": 1.5},
        meanprops={
            "marker": "o",
            "markerfacecolor": "#ffffff",
            "markeredgecolor": "#1f2937",
            "markersize": 4,
        },
        boxprops={"facecolor": SUBPLOT_COLORS[2], "edgecolor": "#365314", "alpha": 0.85},
        whiskerprops={"color": "#365314"},
        capprops={"color": "#365314"},
    )
    plt.title("Human-Agent Disagreement", fontweight="bold")
    # plt.xlabel("Score component")
    plt.ylabel(r"Stage-Level Score Difference $R_k$")
    max_abs = data.loc[:, columns].abs().max().max()
    max_abs = 0.1 if pd.isna(max_abs) else float(max_abs)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=35, ha="right")
    plt.yticks([-2, -1, 0, 1, 2])
    plt.ylim(-2.2, max_abs + 0.05)
    set_square_subplot()


def plot_disagreement_over_length(data: pd.DataFrame) -> None:
    plt.scatter(
        data["num_words_k"],
        data["avg_disagreement"],
        s=22,
        alpha=0.65,
        linewidths=0,
        color=SUBPLOT_COLORS[4],
    )
    regression = regression_stats(data["num_words_k"], data["avg_disagreement"])
    plot_regression_line(data["num_words_k"], regression, SUBPLOT_COLORS[4])
    plt.title("Disagreement (Over Length)", fontweight="bold")
    plt.xlabel("Paper Length (thousand words)")
    plt.ylabel(r"Stage-Level Score Difference $R_k$")
    plt.xticks([0, 5, 10, 15, 20, 25])
    plt.yticks([-2, -1, 0, 1, 2])
    plt.xlim(0, 15)
    max_abs = data["avg_disagreement"].abs().max()
    max_abs = 0.1 if pd.isna(max_abs) else float(max_abs)
    plt.ylim(-1, max_abs + 0.05)
    plt.text(
        0.95,
        0.05,
        regression_label(regression),
        transform=plt.gca().transAxes,
        va="bottom",
        ha="right",
        # fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.85},
    )


def plot_disagreement_boxplot_over_page_bins(
    data: pd.DataFrame,
    n_bins: int = 20,
) -> None:
    binned_data = data.loc[:, ["num_pages", "avg_disagreement"]].dropna().copy()
    binned_data["page_bin"] = pd.cut(
        binned_data["num_pages"],
        bins=n_bins,
        include_lowest=True,
    )

    bin_groups = []
    bin_labels = []
    for page_bin, group in binned_data.groupby("page_bin", observed=True):
        if group.empty:
            continue
        bin_groups.append(group["avg_disagreement"])
        bin_labels.append(f"{page_bin.left:.0f}-{page_bin.right:.0f}")

    plt.boxplot(
        bin_groups,
        labels=bin_labels,
        showmeans=True,
        patch_artist=True,
        medianprops={"color": "#1f2937", "linewidth": 1.5},
        meanprops={
            "marker": "o",
            "markerfacecolor": "#ffffff",
            "markeredgecolor": "#1f2937",
            "markersize": 4,
        },
        boxprops={"facecolor": SUBPLOT_COLORS[3], "edgecolor": "#075985", "alpha": 0.85},
        whiskerprops={"color": "#075985"},
        capprops={"color": "#075985"},
    )
    plt.title("Disagreement by Page Length", fontweight="bold")
    plt.xlabel("Page length bin")
    plt.ylabel(r"Mean Disagreement ($R$)")
    plt.xticks(range(1, len(bin_labels) + 1), bin_labels, rotation=90)
    plt.yticks([-2, -1, 0, 1, 2])
    max_abs = binned_data["avg_disagreement"].abs().max()
    max_abs = 1.0 if pd.isna(max_abs) else max(1.0, float(max_abs))
    plt.ylim(-max_abs - 0.1, max_abs + 0.1)
    set_square_subplot()


def plot_disagreement_boxplot_over_word_bins(
    data: pd.DataFrame,
    n_bins: int = 20,
) -> None:
    binned_data = data.loc[:, ["num_words_k", "avg_disagreement"]].dropna().copy()
    binned_data["word_bin"] = pd.cut(
        binned_data["num_words_k"],
        bins=n_bins,
        include_lowest=True,
    )

    bin_groups = []
    bin_labels = []
    for word_bin, group in binned_data.groupby("word_bin", observed=True):
        if group.empty:
            continue
        bin_groups.append(group["avg_disagreement"])
        bin_labels.append(f"{word_bin.left:.1f}")

    plt.boxplot(
        bin_groups,
        labels=bin_labels,
        showmeans=True,
        patch_artist=True,
        medianprops={"color": "#1f2937", "linewidth": 1.5},
        meanprops={
            "marker": "o",
            "markerfacecolor": "#ffffff",
            "markeredgecolor": "#1f2937",
            "markersize": 4,
        },
        boxprops={"facecolor": SUBPLOT_COLORS[3], "edgecolor": "#075985", "alpha": 0.85},
        whiskerprops={"color": "#075985"},
        capprops={"color": "#075985"},
    )
    plt.title("Disagreement (by Length)", fontweight="bold")
    plt.xlabel("")
    plt.ylabel(r"Mean Disagreement ($R$)")
    plt.xticks(range(1, len(bin_labels) + 1), bin_labels, rotation=90)
    plt.yticks([-1, 0, 1, 2, 3])
    plt.ylim(-1, 3.1)
    plt.text(
        0.5,
        0.03,
        "Length (thousand words)",
        transform=plt.gca().transAxes,
        ha="center",
        va="bottom",
        # fontsize=9,
    )
    set_square_subplot()


def plot_disagreement_boxplot_over_node_bins(
    data: pd.DataFrame,
    n_bins: int = 20,
) -> None:
    binned_data = data.loc[:, ["n_nodes", "avg_disagreement"]].dropna().copy()
    binned_data["node_bin"] = pd.cut(
        binned_data["n_nodes"],
        bins=n_bins,
        include_lowest=True,
    )

    bin_groups = []
    bin_labels = []
    for node_bin, group in binned_data.groupby("node_bin", observed=True):
        if group.empty:
            continue
        bin_groups.append(group["avg_disagreement"])
        bin_labels.append(f"{node_bin.left:.0f}-{node_bin.right:.0f}")

    plt.boxplot(
        bin_groups,
        labels=bin_labels,
        showmeans=True,
        patch_artist=True,
        medianprops={"color": "#1f2937", "linewidth": 1.5},
        meanprops={
            "marker": "o",
            "markerfacecolor": "#ffffff",
            "markeredgecolor": "#1f2937",
            "markersize": 4,
        },
        boxprops={"facecolor": SUBPLOT_COLORS[4], "edgecolor": "#1e3a8a", "alpha": 0.85},
        whiskerprops={"color": "#1e3a8a"},
        capprops={"color": "#1e3a8a"},
    )
    plt.title("Disagreement by Node Count", fontweight="bold")
    plt.xlabel("Node count bin")
    plt.ylabel(r"Mean Disagreement ($R$)")
    plt.xticks(range(1, len(bin_labels) + 1), bin_labels, rotation=90)
    plt.yticks([-2, -1, 0, 1, 2])
    max_abs = binned_data["avg_disagreement"].abs().max()
    max_abs = 1.0 if pd.isna(max_abs) else max(1.0, float(max_abs))
    plt.ylim(-max_abs - 0.1, max_abs + 0.1)
    set_square_subplot()


def plot_reproscore_boxplot_over_page_bins(
    data: pd.DataFrame,
    n_bins: int = 20,
) -> None:
    binned_data = data.loc[:, ["num_pages", "R"]].dropna().copy()
    binned_data["page_bin"] = pd.cut(
        binned_data["num_pages"],
        bins=n_bins,
        include_lowest=True,
    )

    bin_groups = []
    bin_labels = []
    for page_bin, group in binned_data.groupby("page_bin", observed=True):
        if group.empty:
            continue
        bin_groups.append(group["R"])
        bin_labels.append(f"{page_bin.left:.0f}")

    plt.boxplot(
        bin_groups,
        labels=bin_labels,
        showmeans=True,
        patch_artist=True,
        medianprops={"color": "#1f2937", "linewidth": 1.5},
        meanprops={
            "marker": "o",
            "markerfacecolor": "#ffffff",
            "markeredgecolor": "#1f2937",
            "markersize": 4,
        },
        boxprops={"facecolor": SUBPLOT_COLORS[4], "edgecolor": "#1e3a8a", "alpha": 0.85},
        whiskerprops={"color": "#1e3a8a"},
        capprops={"color": "#1e3a8a"},
    )
    plt.title("Reproducibility (by Pages)", fontweight="bold")
    plt.xlabel("Page length bin")
    plt.ylabel(r"Reproducibility Score ($R$)")
    plt.xticks(range(1, len(bin_labels) + 1), bin_labels, rotation=90)
    plt.yticks([1, 2, 3, 4])
    plt.ylim(1.9, 4.1)
    set_square_subplot()


def plot_reproscore_boxplot_over_word_bins(
    data: pd.DataFrame,
    n_bins: int = 20,
) -> None:
    binned_data = data.loc[:, ["num_words_k", "R"]].dropna().copy()
    binned_data["word_bin"] = pd.cut(
        binned_data["num_words_k"],
        bins=n_bins,
        include_lowest=True,
    )

    bin_groups = []
    bin_labels = []
    for word_bin, group in binned_data.groupby("word_bin", observed=True):
        if group.empty:
            continue
        bin_groups.append(group["R"])
        bin_labels.append(f"{word_bin.left:.1f}")

    plt.boxplot(
        bin_groups,
        labels=bin_labels,
        showmeans=True,
        patch_artist=True,
        medianprops={"color": "#1f2937", "linewidth": 1.5},
        meanprops={
            "marker": "o",
            "markerfacecolor": "#ffffff",
            "markeredgecolor": "#1f2937",
            "markersize": 4,
        },
        boxprops={"facecolor": SUBPLOT_COLORS[4], "edgecolor": "#075985", "alpha": 0.85},
        whiskerprops={"color": "#075985"},
        capprops={"color": "#075985"},
    )
    plt.title("Reproducibility (by Length)", fontweight="bold")
    plt.xlabel("")
    plt.ylabel("Reproducibility score")
    plt.xticks(range(1, len(bin_labels) + 1), bin_labels, rotation=90)
    plt.yticks([1, 2, 3, 4])
    plt.ylim(0.9, 4.1)
    plt.text(
        0.5,
        0.03,
        "Word length bin (thousands)",
        transform=plt.gca().transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    set_square_subplot()


def plot_reproscore_boxplot_over_node_bins(
    data: pd.DataFrame,
    n_bins: int = 20,
) -> None:
    binned_data = data.loc[:, ["n_nodes", "R"]].dropna().copy()
    binned_data["node_bin"] = pd.cut(
        binned_data["n_nodes"],
        bins=n_bins,
        include_lowest=True,
    )

    bin_groups = []
    bin_labels = []
    for node_bin, group in binned_data.groupby("node_bin", observed=True):
        if group.empty:
            continue
        bin_groups.append(group["R"])
        bin_labels.append(f"{node_bin.left:.0f}")

    plt.boxplot(
        bin_groups,
        labels=bin_labels,
        showmeans=True,
        patch_artist=True,
        medianprops={"color": "#1f2937", "linewidth": 1.5},
        meanprops={
            "marker": "o",
            "markerfacecolor": "#ffffff",
            "markeredgecolor": "#1f2937",
            "markersize": 4,
        },
        boxprops={"facecolor": SUBPLOT_COLORS[4], "edgecolor": "#075985", "alpha": 0.85},
        whiskerprops={"color": "#075985"},
        capprops={"color": "#075985"},
    )
    plt.title("Reproducibility (by Nodes)", fontweight="bold")
    plt.xlabel("Node count bin")
    plt.ylabel("Reproducibility score")
    plt.xticks(range(1, len(bin_labels) + 1), bin_labels, rotation=90)
    plt.yticks([1, 2, 3, 4])
    plt.ylim(0.9, 4.1)
    set_square_subplot()


scores = load_scores(SCORE_CSV)
paper_lengths = load_paper_lengths(PAPER_SUMMARY_CSV)
ground_truth = load_ground_truth(GROUND_TRUTH_CSV)
analysis = build_analysis_table(scores, paper_lengths, ground_truth)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.default"] = "regular"
plt.figure(figsize=FIGSIZE)

plt.subplot(1, 4, 1)
plot_score_boxplot(scores)

plt.subplot(1, 4, 2)
plot_disagreement_boxplot(analysis)

plt.subplot(1, 4, 3)
plot_disagreement_boxplot_over_word_bins(analysis)

plt.subplot(1, 4, 4)
plot_reproscore_boxplot_over_page_bins(analysis)

plt.tight_layout()

plt.show()
