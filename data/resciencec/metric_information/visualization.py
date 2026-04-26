from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUTS_DIR = Path("outputs")
SCORES_CSV = OUTPUTS_DIR / "scores.csv"
PLOT_PATH = OUTPUTS_DIR / "score_distributions.png"
TABLE_PATH = OUTPUTS_DIR / "score_summary.csv"
DIMENSIONS = ["Sources", "Methods", "Experiments", "Sinks"]


def load_scores() -> pd.DataFrame:
    df = pd.read_csv(SCORES_CSV, sep=";")

    for dimension in DIMENSIONS:
        df[f"score_{dimension}"] = pd.to_numeric(df[f"score_{dimension}"], errors="coerce")
        df[f"confidence_{dimension}"] = pd.to_numeric(
            df[f"confidence_{dimension}"], errors="coerce"
        )

    return df


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dimension in DIMENSIONS:
        score_col = f"score_{dimension}"
        confidence_col = f"confidence_{dimension}"

        rows.append(
            {
                "dimension": dimension,
                "count": int(df[score_col].notna().sum()),
                "mean_score": df[score_col].mean(),
                "std_score": df[score_col].std(),
                "mean_confidence": df[confidence_col].mean(),
                "std_confidence": df[confidence_col].std(),
            }
        )

    summary = pd.DataFrame(rows)
    return summary.round(
        {
            "mean_score": 3,
            "std_score": 3,
            "mean_confidence": 3,
            "std_confidence": 3,
        }
    )


def plot_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()

    for ax, dimension in zip(axes, DIMENSIONS):
        score_col = f"score_{dimension}"
        confidence_col = f"confidence_{dimension}"

        grouped = (
            df.groupby(score_col, dropna=True)
            .agg(
                count=(score_col, "size"),
                mean_confidence=(confidence_col, "mean"),
            )
            .reset_index()
            .sort_values(score_col)
        )

        if grouped.empty:
            ax.set_visible(False)
            continue

        scores = grouped[score_col].astype(int)
        counts = grouped["count"]
        mean_confidence = grouped["mean_confidence"]

        bars = ax.bar(scores, counts, color="#4C78A8", alpha=0.85, width=0.65)
        ax.set_title(dimension)
        ax.set_xlabel("Score")
        ax.set_ylabel("Number of papers")
        ax.set_xticks(sorted(scores.unique()))
        ax.set_ylim(0, max(counts) * 1.2 if len(counts) else 1)

        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                str(int(count)),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax2 = ax.twinx()
        ax2.plot(
            scores,
            mean_confidence,
            color="#F58518",
            marker="o",
            linewidth=2,
        )
        ax2.set_ylabel("Mean confidence")
        ax2.set_ylim(0, 100)

    fig.suptitle(
        "ReScience C score distributions and confidence by assessment category",
        fontsize=14,
    )
    fig.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


df = load_scores()
summary = build_summary_table(df)

# plot_distributions(df)
# summary.to_csv(TABLE_PATH, index=False)

# print(summary.to_string(index=False))
# print(f"\nSaved plot to {PLOT_PATH}")
# print(f"Saved summary table to {TABLE_PATH}")

