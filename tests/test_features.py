import numpy as np
import pandas as pd
import pytest

from f1.features import (
    MAX_PACE_RATIO,
    add_finish_pct,
    build_master,
    cluster_drivers,
    driver_performance_table,
    pace_consistency,
)


def test_finish_pct_is_one_for_winner_and_zero_for_last(results):
    out = add_finish_pct(results)
    assert out.loc[out.positionOrder == 1, "finish_pct"].eq(1.0).all()
    assert out.loc[(out.raceId == 1) & (out.positionOrder == 4), "finish_pct"].iloc[0] == 0.0


def test_finish_pct_stays_within_unit_interval(results):
    out = add_finish_pct(results)
    assert out["finish_pct"].between(0, 1).all()


def test_finish_pct_does_not_mutate_input(results):
    before = results.copy()
    add_finish_pct(results)
    pd.testing.assert_frame_equal(results, before)


def test_finish_pct_single_starter_is_nan():
    single = pd.DataFrame({"raceId": [9], "positionOrder": [1]})
    assert np.isnan(add_finish_pct(single)["finish_pct"].iloc[0])


def test_build_master_adds_name_numeric_points_and_finish_pct(results, drivers, status):
    master = build_master(results, drivers, status)
    assert {"driver_name", "status", "finish_pct"} <= set(master.columns)
    assert master["points"].dtype.kind == "f"
    assert master.loc[master.driverId == 1, "driver_name"].iloc[0] == "Ann A"
    assert master["points"].notna().all()


def _master(n_races):
    rows = []
    for race in range(n_races):
        for pos, driver in enumerate(["Winner", "Mid", "Back"], start=1):
            rows.append({"raceId": race, "driver_name": driver, "positionOrder": pos})
    return add_finish_pct(pd.DataFrame(rows))


def test_driver_performance_table_filters_by_min_races():
    table = driver_performance_table(_master(10), min_races=20)
    assert table.empty


def test_driver_performance_table_rates():
    table = driver_performance_table(_master(25), min_races=20).set_index("Driver Name")
    assert table.loc["Winner", "Win Rate"] == 1.0
    assert table.loc["Winner", "Podium Rate"] == 1.0
    assert table.loc["Back", "Podium Rate"] == 1.0
    assert table.loc["Mid", "Win Rate"] == 0.0
    assert table.loc["Winner", "Avg Finish Pct"] == 1.0
    assert table.loc["Back", "Top5 Rate"] == 1.0


def test_cluster_drivers_names_ordered_by_win_rate():
    rng = np.random.default_rng(0)
    base = pd.DataFrame(
        {
            "Driver Name": [f"d{i}" for i in range(40)],
            "Avg Finish Pct": np.repeat([0.9, 0.7, 0.5, 0.2], 10) + rng.normal(0, 0.01, 40),
            "Podium Rate": np.repeat([0.8, 0.4, 0.1, 0.0], 10) + rng.normal(0, 0.01, 40),
            "Win Rate": np.repeat([0.4, 0.15, 0.02, 0.0], 10) + rng.normal(0, 0.005, 40),
        }
    )
    out = cluster_drivers(base, k=4, seed=42)
    means = out.groupby("Category")["Win Rate"].mean()
    assert means["Dominant Legends"] > means["Podium Contenders"]
    assert means["Podium Contenders"] > means["Midfielders"]
    assert means["Midfielders"] > means["Backmarkers"]
    assert "Category" not in base.columns


def _laps(scale):
    """Driver A steady, driver B erratic, one slow pit lap; times scaled per circuit."""
    a = [90_000, 90_100, 89_900, 90_000, 150_000]
    b = [90_000, 91_500, 88_500, 92_000, 88_000]
    return pd.DataFrame(
        {
            "raceId": 1,
            "driver_name": ["A"] * 5 + ["B"] * 5,
            "milliseconds": [int(x * scale) for x in a + b],
        }
    )


def test_pace_consistency_drops_laps_above_threshold():
    out = pace_consistency(_laps(1)).set_index("Driver Name")
    assert out.loc["A", "Pace_StdDev"] < out.loc["B", "Pace_StdDev"]
    assert MAX_PACE_RATIO == pytest.approx(1.07)


def test_pace_consistency_is_independent_of_circuit_length():
    short = pace_consistency(_laps(1)).set_index("Driver Name")
    long_ = pace_consistency(_laps(2.5)).set_index("Driver Name")
    pd.testing.assert_frame_equal(short, long_, rtol=1e-3)


def test_build_master_excludes_non_starters_from_finish_percentile():
    results = pd.DataFrame(
        {
            "raceId": [1] * 5,
            "driverId": [1, 2, 3, 4, 5],
            "statusId": [1, 1, 1, 3, 3],
            "positionOrder": [1, 2, 3, 4, 5],
            "grid": [1, 2, 3, 0, 0],
            "points": ["10", "6", "4", "0", "0"],
        }
    )
    drivers = pd.DataFrame(
        {"driverId": [1, 2, 3, 4, 5], "forename": list("abcde"), "surname": list("vwxyz")}
    )
    status = pd.DataFrame({"statusId": [1, 3], "status": ["Finished", "Did not qualify"]})
    master = build_master(results, drivers, status)
    assert len(master) == 3
    assert master["starters"].eq(3).all()
    assert master.loc[master.positionOrder == 3, "finish_pct"].iloc[0] == 0.0
