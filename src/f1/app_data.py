"""Build the small precomputed driver table that the Streamlit app reads."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from f1.clustering import pca_coordinates, scale_features
from f1.data import load_tables
from f1.features import build_master, cluster_drivers, pace_consistency
from f1.profile import build_profile
from f1.rating import MIN_RACES, bootstrap_ratings, build_pairs

APP_DATA_PATH = Path("app/data/drivers.csv")
APP_COLUMNS = [
    "Driver Name",
    "Nationality",
    "First Season",
    "Last Season",
    "Total Races",
    "Total Wins",
    "Total Podiums",
    "Category",
    "Avg Finish Pct",
    "Win Rate",
    "Podium Rate",
    "Top5 Rate",
    "Positions Gained",
    "Driver-Error DNF Rate",
    "Mechanical DNF Rate",
    "Teammate Delta",
    "Teammate Races",
    "Pace_StdDev",
    "Rating",
    "Rating Low",
    "Rating High",
    "Rating Rank",
    "PC1",
    "PC2",
]


def assemble_app_table(
    profile: pd.DataFrame,
    consistency: pd.DataFrame,
    ratings: pd.DataFrame,
    seasons: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """One row per profiled driver; consistency and rating stay NaN when not available."""
    rated = ratings.rename(columns={"Low": "Rating Low", "High": "Rating High"})
    rated = rated[["Driver Name", "Rating", "Rating Low", "Rating High"]]
    table = (
        profile.merge(consistency, on="Driver Name", how="left")
        .merge(rated, on="Driver Name", how="left")
        .merge(seasons, on="Driver Name", how="left")
        .merge(meta.drop_duplicates("Driver Name"), on="Driver Name", how="left")
    )
    coords = pca_coordinates(scale_features(table))
    table = table.assign(
        **{
            "Rating Rank": table["Rating"].rank(ascending=False, method="first"),
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
        }
    )
    return table[APP_COLUMNS]


def build_app_table(data_dir: str | Path | None = None, n_boot: int = 30) -> pd.DataFrame:
    tables = load_tables(data_dir)
    master = build_master(tables["results"], tables["drivers"], tables["status"], tables["races"])
    profile = cluster_drivers(build_profile(master))

    laps = tables["lap_times"].merge(
        tables["drivers"][["driverId", "forename", "surname"]], on="driverId"
    )
    laps = laps.assign(driver_name=laps["forename"] + " " + laps["surname"])
    consistency = pace_consistency(laps)

    ratings = bootstrap_ratings(build_pairs(master), n_boot=n_boot, seed=0)
    ratings = ratings[ratings["Races"] > MIN_RACES]

    seasons = (
        master.groupby("driver_name")["year"]
        .agg(["min", "max"])
        .rename(columns={"min": "First Season", "max": "Last Season"})
        .reset_index()
        .rename(columns={"driver_name": "Driver Name"})
    )
    drivers = tables["drivers"]
    meta = pd.DataFrame(
        {
            "Driver Name": drivers["forename"] + " " + drivers["surname"],
            "Nationality": drivers["nationality"],
        }
    )
    return assemble_app_table(profile, consistency, ratings, seasons, meta)


def write_app_table(table: pd.DataFrame, path: str | Path = APP_DATA_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(path, index=False)
    return path


def main() -> None:
    path = write_app_table(build_app_table())
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
