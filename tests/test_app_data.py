import numpy as np
import pandas as pd
import pytest

from f1.app_data import APP_COLUMNS, assemble_app_table, write_app_table
from f1.clustering import pca_coordinates


def _inputs():
    names = [f"d{i}" for i in range(6)]
    profile = pd.DataFrame(
        {
            "Driver Name": names,
            "Total Races": [30, 40, 50, 60, 70, 80],
            "Total Wins": [0, 1, 2, 3, 4, 5],
            "Total Podiums": [1, 2, 3, 4, 5, 6],
            "Avg Finish Pct": np.linspace(0.3, 0.8, 6),
            "Win Rate": np.linspace(0.0, 0.3, 6),
            "Podium Rate": np.linspace(0.05, 0.5, 6),
            "Top5 Rate": np.linspace(0.1, 0.7, 6),
            "Positions Gained": np.linspace(-1, 3, 6),
            "Driver-Error DNF Rate": np.linspace(0.05, 0.1, 6),
            "Mechanical DNF Rate": np.linspace(0.1, 0.2, 6),
            "Teammate Delta": [0.1, np.nan, 0.0, -0.05, 0.02, 0.08],
            "Teammate Races": [30, 5, 50, 60, 70, 80],
            "Category": ["Backmarkers"] * 3 + ["Midfielders"] * 2 + ["Dominant Legends"],
        }
    )
    consistency = pd.DataFrame({"Driver Name": names[3:], "Pace_StdDev": [0.015, 0.016, 0.017]})
    ratings = pd.DataFrame(
        {
            "Driver Name": names[:5],
            "Rating": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Low": [0.0, 10.0, 20.0, 30.0, 40.0],
            "High": [20.0, 30.0, 40.0, 50.0, 60.0],
            "Races": [30, 40, 50, 60, 70],
        }
    )
    seasons = pd.DataFrame(
        {"Driver Name": names, "First Season": range(1990, 1996), "Last Season": range(2000, 2006)}
    )
    meta = pd.DataFrame({"Driver Name": names, "Nationality": ["British"] * 6})
    return profile, consistency, ratings, seasons, meta


def test_assemble_app_table_has_documented_columns_and_one_row_per_driver():
    table = assemble_app_table(*_inputs())
    assert list(table.columns) == APP_COLUMNS
    assert table["Driver Name"].is_unique
    assert len(table) == 6


def test_assemble_app_table_keeps_drivers_without_consistency_or_rating():
    table = assemble_app_table(*_inputs()).set_index("Driver Name")
    assert np.isnan(table.loc["d0", "Pace_StdDev"])  # no lap data before 1996
    assert np.isnan(table.loc["d5", "Rating"])  # not rated


def test_assemble_app_table_ranks_ratings_from_best():
    table = assemble_app_table(*_inputs()).set_index("Driver Name")
    assert table.loc["d4", "Rating Rank"] == 1
    assert table.loc["d0", "Rating Rank"] == 5


def test_assemble_app_table_adds_pca_coordinates_for_every_driver():
    table = assemble_app_table(*_inputs())
    assert table[["PC1", "PC2"]].notna().all().all()


def test_pca_coordinates_returns_two_columns():
    X = np.random.default_rng(0).normal(size=(10, 3))
    assert pca_coordinates(X).shape == (10, 2)


def test_write_app_table_round_trips(tmp_path):
    table = assemble_app_table(*_inputs())
    path = write_app_table(table, tmp_path / "out" / "drivers.csv")
    back = pd.read_csv(path)
    assert list(back.columns) == APP_COLUMNS
    assert len(back) == len(table)
    assert back["Rating"].round(2).tolist()[:5] == pytest.approx(
        table["Rating"].round(2).tolist()[:5]
    )
