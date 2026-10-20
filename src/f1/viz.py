"""Charts used in the notebook and README."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TOP_TIERS = ["Dominant Legends", "Podium Contenders"]
NOTABLE_WIN_RATE = 0.1
NOTABLE_PACE_STD = 0.0135


def _finish(path: str | Path | None) -> None:
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_metronome(full_stats: pd.DataFrame, path: str | Path | None = None) -> None:
    top = full_stats[full_stats["Total Races"] > 50].nsmallest(10, "Pace_StdDev")
    plt.figure(figsize=(12, 5))
    sns.barplot(data=top, x="Pace_StdDev", y="Driver Name", hue="Driver Name",
                palette="mako", legend=False)
    plt.title("The Human Metronome: Top 10 Most Consistent Drivers", fontsize=14, weight="bold")
    plt.xlabel("Std Dev of Pace vs Race Median (lower is better)")
    plt.ylabel(None)
    _finish(path)


def plot_reliability(master: pd.DataFrame, driver_stats: pd.DataFrame,
                     path: str | Path | None = None) -> None:
    top_names = driver_stats.loc[driver_stats["Category"].isin(TOP_TIERS), "Driver Name"]
    subset = master[master["driver_name"].isin(top_names)]
    counts = subset.groupby(["driver_name", "DNF_Reason"]).size().unstack().fillna(0)
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    cols = ["Accident/Driver Error", "Car/Mechanical Failure"]
    pct = pct.reindex(columns=cols, fill_value=0).sort_values(cols[0], ascending=False).head(10)
    pct.plot(kind="bar", stacked=True, figsize=(12, 5), color=["#e74c3c", "#95a5a6"])
    plt.title("Reliability Analysis: Accident vs Mechanical Failure (Top Drivers)",
              fontsize=14, weight="bold")
    plt.ylabel("Percentage of Races (%)")
    plt.legend(title="Cause of Issue", bbox_to_anchor=(1.0, 1.0))
    plt.xticks(rotation=45)
    _finish(path)


def plot_matrix(full_stats: pd.DataFrame, path: str | Path | None = None) -> None:
    data = full_stats.rename(columns={"Total Races": "Experience (Total Races)"})
    plt.figure(figsize=(14, 9))
    sns.scatterplot(data=data, x="Avg Finish Pct", y="Win Rate", hue="Pace_StdDev",
                    palette="viridis_r", size="Experience (Total Races)", sizes=(50, 450),
                    alpha=0.8, edgecolor="black")
    notable = data[(data["Win Rate"] > NOTABLE_WIN_RATE) | (data["Pace_StdDev"] < NOTABLE_PACE_STD)]
    for _, row in notable.iterrows():
        plt.text(row["Avg Finish Pct"] + 0.005, row["Win Rate"] + 0.005, row["Driver Name"],
                 fontsize=9, weight="bold",
                 bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=0.5))
    plt.title("F1 Performance Matrix: Speed vs Consistency (1950-2024)", fontsize=16, weight="bold")
    plt.xlabel("Average Finish Percentile (Speed, era-neutral)")
    plt.ylabel("Win Rate")
    plt.legend(title="Pace Std Dev (Lower=Better)", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    _finish(path)
