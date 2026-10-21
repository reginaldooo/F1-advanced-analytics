"""Era-neutral performance and consistency features, plus driver clustering."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

from f1.dnf import non_starter_mask

MAX_PACE_RATIO = 1.07
MIN_RACES = 20
CLUSTER_FEATURES = ["Avg Finish Pct", "Podium Rate", "Win Rate"]
CATEGORY_NAMES = ["Dominant Legends", "Podium Contenders", "Midfielders", "Backmarkers"]
MODERN_POINTS = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)


def add_finish_pct(results: pd.DataFrame) -> pd.DataFrame:
    """Add `finish_pct`: 1 for the winner, 0 for last, comparable across eras and grid sizes."""
    starters = results.groupby("raceId")["positionOrder"].transform("count")
    denominator = (starters - 1).where(starters > 1)
    return results.assign(
        starters=starters,
        finish_pct=1 - (results["positionOrder"] - 1) / denominator,
    )


def add_rescored_points(
    results: pd.DataFrame, scheme: tuple[int, ...] = MODERN_POINTS
) -> pd.DataFrame:
    """Add `rescored_points`: every race scored with one points scheme, whatever the era.

    Ignores the original `points` column. No fastest-lap bonus, no sprints.
    """
    awarded = {place: pts for place, pts in enumerate(scheme, start=1)}
    return results.assign(rescored_points=results["positionOrder"].map(awarded).fillna(0))


def build_master(
    results: pd.DataFrame,
    drivers: pd.DataFrame,
    status: pd.DataFrame,
    races: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join results, drivers and status; drop non-starters; add driver_name, points, finish_pct.

    Pass `races` to also get the season `year` of each result.
    """
    master = results.merge(drivers, on="driverId").merge(status, on="statusId")
    if races is not None:
        master = master.merge(races[["raceId", "year"]], on="raceId")
    master = master[~non_starter_mask(master)].reset_index(drop=True)
    master = master.assign(
        driver_name=master["forename"] + " " + master["surname"],
        points=pd.to_numeric(master["points"], errors="coerce").fillna(0),
    )
    return add_rescored_points(add_finish_pct(master))


def driver_performance_table(master: pd.DataFrame, min_races: int = MIN_RACES) -> pd.DataFrame:
    """Per-driver era-neutral stats for drivers with more than `min_races` races."""
    position = master["positionOrder"]
    grouped = master.assign(
        win=position == 1, podium=position <= 3, top5=position <= 5
    ).groupby("driver_name")
    stats = grouped.agg(
        **{
            "Total Races": ("raceId", "count"),
            "Avg Finish Pct": ("finish_pct", "mean"),
            "Total Wins": ("win", "sum"),
            "Total Podiums": ("podium", "sum"),
            "Total Top5": ("top5", "sum"),
            "Rescored Points": ("rescored_points", "sum"),
        }
    ).reset_index()
    stats = stats.rename(columns={"driver_name": "Driver Name"})
    stats = stats[stats["Total Races"] > min_races].copy()
    races = stats["Total Races"]
    return stats.assign(
        **{
            "Win Rate": stats["Total Wins"] / races,
            "Podium Rate": stats["Total Podiums"] / races,
            "Top5 Rate": stats["Total Top5"] / races,
            "Rescored PPR": stats["Rescored Points"] / races,
        }
    ).reset_index(drop=True)


def cluster_drivers(stats: pd.DataFrame, k: int = 4, seed: int = 42) -> pd.DataFrame:
    """K-Means on robust-scaled features; clusters are named by mean win rate."""
    scaled = RobustScaler().fit_transform(stats[CLUSTER_FEATURES])
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(scaled)
    out = stats.assign(cluster_id=labels)
    ranked = out.groupby("cluster_id")["Win Rate"].mean().sort_values(ascending=False).index
    names = dict(zip(ranked, CATEGORY_NAMES))
    return out.assign(Category=out["cluster_id"].map(names))


def pace_consistency(laps: pd.DataFrame) -> pd.DataFrame:
    """Std of lap pace relative to the race median, laps above 107% of median dropped.

    `laps` needs raceId, driver_name and milliseconds. Returns Driver Name, Pace_StdDev.
    """
    pace = laps["milliseconds"] / laps.groupby("raceId")["milliseconds"].transform("median")
    clean = laps.assign(pace=pace)
    clean = clean[clean["pace"] <= MAX_PACE_RATIO]
    per_race = clean.groupby(["driver_name", "raceId"])["pace"].std().reset_index()
    career = per_race.groupby("driver_name")["pace"].mean().reset_index()
    career.columns = ["Driver Name", "Pace_StdDev"]
    return career
