import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from f1.features import MODERN_POINTS  # noqa: E402
from f1.profile import (  # noqa: E402
    PROFILE_CRITERIA,
    build_profile,
    dnf_rates,
    plot_correlation,
    positions_gained,
    redundancy,
    teammate_delta,
)


def _rows(driver, constructor, race, grid, pos, status="Finished", finish_pct=None):
    return {
        "driver_name": driver,
        "constructorId": constructor,
        "raceId": race,
        "grid": grid,
        "positionOrder": pos,
        "status": status,
        "finish_pct": finish_pct if finish_pct is not None else 1 - (pos - 1) / 3,
        "rescored_points": dict(enumerate(MODERN_POINTS, start=1)).get(pos, 0),
    }


@pytest.fixture
def master():
    return pd.DataFrame(
        [
            # race 1: A gains 2 places, B loses 1, C crashes, D pit-lane start (grid 0) ignored
            _rows("A", 1, 1, 4, 2),
            _rows("B", 1, 1, 1, 2 + 1 - 1 + 1),
            _rows("C", 2, 1, 2, 4, "Collision"),
            _rows("D", 2, 1, 0, 1),
            # race 2: A engine failure, B finishes
            _rows("A", 1, 2, 3, 4, "Engine"),
            _rows("B", 1, 2, 5, 1),
        ]
    )


def test_positions_gained_uses_classified_finishes_with_a_real_grid_slot(master):
    out = positions_gained(master).set_index("Driver Name")["Positions Gained"]
    assert out["A"] == 2  # race 2 retired, ignored
    assert out["B"] == pytest.approx(((1 - 3) + (5 - 1)) / 2)
    assert "C" not in out.index  # never classified
    assert "D" not in out.index  # pit-lane start only


def test_dnf_rates_split_accident_and_mechanical_as_share_of_starts(master):
    out = dnf_rates(master).set_index("Driver Name")
    assert out.loc["A", "Mechanical DNF Rate"] == pytest.approx(0.5)
    assert out.loc["A", "Driver-Error DNF Rate"] == 0
    assert out.loc["C", "Driver-Error DNF Rate"] == 1.0
    assert out.loc["B", "Mechanical DNF Rate"] == 0


def test_teammate_delta_is_difference_to_other_driver_in_same_car(master):
    out = teammate_delta(master).set_index("Driver Name")
    pct = master.set_index(["driver_name", "raceId"])["finish_pct"]
    expected_a = ((pct["A", 1] - pct["B", 1]) + (pct["A", 2] - pct["B", 2])) / 2
    assert out.loc["A", "Teammate Delta"] == pytest.approx(expected_a)
    assert out.loc["A", "Teammate Delta"] == pytest.approx(-out.loc["B", "Teammate Delta"])
    assert out.loc["A", "Teammate Races"] == 2


def test_teammate_delta_skips_solo_entries():
    solo = pd.DataFrame([_rows("Z", 9, 1, 1, 1)])
    assert teammate_delta(solo).empty


def test_build_profile_masks_teammate_delta_below_min_paired_races():
    rows = []
    for race in range(30):
        rows += [
            _rows("Fast", 1, race, 2, 1, finish_pct=0.9),
            _rows("Slow", 1, race, 1, 2, finish_pct=0.5),
        ]
    for race in range(30, 33):  # only 3 races as teammates
        rows += [_rows("Fast", 1, race, 2, 1), _rows("Newbie", 1, race, 1, 2)]
    for race in range(33, 60):  # Newbie gets enough starts elsewhere
        rows += [_rows("Newbie", 5, race, 3, 3), _rows("Other", 6, race, 4, 4)]
    built = build_profile(pd.DataFrame(rows), min_races=20, min_paired=20)
    profile = built.set_index("Driver Name")
    assert profile.loc["Fast", "Teammate Delta"] > 0
    assert np.isnan(profile.loc["Newbie", "Teammate Delta"])
    assert set(PROFILE_CRITERIA) <= set(profile.columns)


def test_redundancy_returns_correlation_and_explained_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    profile = pd.DataFrame({"a": x, "b": x * 2 + rng.normal(0, 0.01, 60), "c": rng.normal(size=60)})
    corr, variance = redundancy(profile, ["a", "b", "c"])
    assert corr.loc["a", "b"] > 0.99
    assert abs(corr.loc["a", "c"]) < 0.5
    assert variance.sum() == pytest.approx(1.0)
    assert variance[0] >= variance[1] >= variance[2]


def test_redundancy_drops_rows_with_missing_values():
    profile = pd.DataFrame({"a": [1.0, 2, 3, 4, np.nan], "b": [2.0, 4, 6, 9, 1]})
    corr, _ = redundancy(profile, ["a", "b"])
    assert not corr.isna().any().any()


def test_plot_correlation_saves_file(tmp_path):
    corr = pd.DataFrame([[1, 0.2], [0.2, 1]], columns=["a", "b"], index=["a", "b"])
    out = tmp_path / "c.png"
    plot_correlation(corr, out)
    assert out.exists()
