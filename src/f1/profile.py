"""Multi-criteria driver profile: era-neutral performance plus racecraft, reliability and
teammate-relative criteria, with a redundancy check (correlation + PCA)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from f1.dnf import classify_dnf
from f1.features import MIN_RACES, driver_performance_table

MIN_PAIRED_RACES = 20
PROFILE_CRITERIA = [
    "Avg Finish Pct",
    "Win Rate",
    "Podium Rate",
    "Positions Gained",
    "Driver-Error DNF Rate",
    "Mechanical DNF Rate",
    "Teammate Delta",
]


def _with_reason(master: pd.DataFrame) -> pd.DataFrame:
    return master.assign(reason=master["status"].map(classify_dnf))


def positions_gained(master: pd.DataFrame) -> pd.DataFrame:
    """Mean `grid - finish` over classified finishes that started from a real grid slot."""
    data = _with_reason(master)
    data = data[(data["reason"] == "Finished") & (data["grid"] > 0)]
    gained = (data["grid"] - data["positionOrder"]).groupby(data["driver_name"]).mean()
    return gained.rename("Positions Gained").reset_index().rename(
        columns={"driver_name": "Driver Name"}
    )


def dnf_rates(master: pd.DataFrame) -> pd.DataFrame:
    """Share of starts ending in a driver-error or mechanical retirement."""
    data = _with_reason(master)
    flags = pd.DataFrame(
        {
            "driver_name": data["driver_name"],
            "Driver-Error DNF Rate": data["reason"] == "Accident/Driver Error",
            "Mechanical DNF Rate": data["reason"] == "Car/Mechanical Failure",
        }
    )
    return flags.groupby("driver_name").mean().reset_index().rename(
        columns={"driver_name": "Driver Name"}
    )


def teammate_delta(master: pd.DataFrame) -> pd.DataFrame:
    """Mean finish percentile minus the other driver(s) in the same car, race by race.

    Separates driver from car: a positive value means beating the teammate on average.
    """
    data = master.dropna(subset=["finish_pct"])
    keys = ["raceId", "constructorId"]
    size = data.groupby(keys)["finish_pct"].transform("count")
    total = data.groupby(keys)["finish_pct"].transform("sum")
    paired = data[size > 1]
    others = (total[size > 1] - paired["finish_pct"]) / (size[size > 1] - 1)
    delta = (paired["finish_pct"] - others).groupby(paired["driver_name"])
    out = pd.DataFrame({"Teammate Delta": delta.mean(), "Teammate Races": delta.count()})
    return out.reset_index().rename(columns={"driver_name": "Driver Name"})


def build_profile(
    master: pd.DataFrame, min_races: int = MIN_RACES, min_paired: int = MIN_PAIRED_RACES
) -> pd.DataFrame:
    """One row per driver with all criteria; teammate delta is NaN below `min_paired` races."""
    profile = driver_performance_table(master, min_races=min_races)
    for extra in (positions_gained(master), dnf_rates(master), teammate_delta(master)):
        profile = profile.merge(extra, on="Driver Name", how="left")
    thin = profile["Teammate Races"].fillna(0) < min_paired
    return profile.assign(**{"Teammate Delta": profile["Teammate Delta"].mask(thin)})


def redundancy(
    profile: pd.DataFrame, criteria: list[str] = PROFILE_CRITERIA
) -> tuple[pd.DataFrame, np.ndarray]:
    """Correlation matrix and PCA explained-variance ratios on complete, robust-scaled rows."""
    complete = profile[criteria].dropna()
    scaled = RobustScaler().fit_transform(complete)
    variance = PCA().fit(scaled).explained_variance_ratio_
    return complete.corr(), variance


def plot_correlation(corr: pd.DataFrame, path: str | Path | None = None) -> None:
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, square=True)
    plt.title("Driver criteria: correlation", fontsize=14, weight="bold")
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.show()
    plt.close()
