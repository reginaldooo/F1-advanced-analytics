import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from f1.rating import (  # noqa: E402
    ELO_SCALE,
    bootstrap_ratings,
    build_pairs,
    cross_validate,
    fit_ratings,
    plot_ratings,
    select_c,
)


def _entry(race, year, driver, team, pos, status="Finished"):
    return {
        "raceId": race,
        "year": year,
        "driver_name": driver,
        "constructorId": team,
        "positionOrder": pos,
        "status": status,
    }


def test_build_pairs_makes_every_ordered_pair_and_drops_mechanical_retirements():
    master = pd.DataFrame(
        [
            _entry(1, 2000, "A", 1, 1),
            _entry(1, 2000, "B", 2, 2),
            _entry(1, 2000, "C", 3, 3, "Engine"),  # mechanical: not a skill signal
            _entry(1, 2000, "D", 4, 4, "Collision"),  # driver error: counts as a loss
        ]
    )
    pairs = build_pairs(master)
    assert pairs.X.shape[0] == 3  # A>B, A>D, B>D
    assert set(pairs.y) == {0, 1} or pairs.X.shape[0] < 2
    assert "C" not in pairs.drivers


def test_build_pairs_rows_are_antisymmetric_and_teammates_cancel_the_car_term():
    master = pd.DataFrame(
        [_entry(1, 2000, "A", 1, 1), _entry(1, 2000, "B", 1, 2), _entry(1, 2000, "C", 2, 3)]
    )
    pairs = build_pairs(master)
    dense = pairs.X.toarray()
    assert np.allclose(dense.sum(axis=1), 0)
    car_block = dense[:, pairs.n_drivers :]
    teammate_row = np.where((np.abs(dense[:, : pairs.n_drivers]).sum(axis=1) == 2)
                            & (np.abs(car_block).sum(axis=1) == 0))[0]
    assert len(teammate_row) == 1  # the A-vs-B pair has no constructor signal


def _simulate(seed=0, seasons=4, races=40):
    """8 drivers rotating across 4 cars each season; skill and car effects are known."""
    rng = np.random.default_rng(seed)
    skill = {f"D{i}": s for i, s in enumerate([1.2, 0.8, 0.4, 0.0, 0.0, -0.4, -0.8, -1.2])}
    car_strength = np.array([1.5, 0.5, -0.5, -1.5])
    rows, race_id = [], 0
    for season in range(seasons):
        for _ in range(races):
            race_id += 1
            perf = {}
            for i, (driver, s) in enumerate(skill.items()):
                team = (i // 2 + season) % 4
                perf[(driver, team)] = s + car_strength[team] + rng.gumbel(0, 0.7)
            order = sorted(perf, key=perf.get, reverse=True)
            for pos, (driver, team) in enumerate(order, start=1):
                rows.append(_entry(race_id, 2000 + season, driver, team, pos))
    return pd.DataFrame(rows), skill


def test_fit_ratings_recovers_driver_skill_despite_car_differences():
    master, skill = _simulate()
    table = fit_ratings(build_pairs(master)).set_index("Driver Name")
    truth = pd.Series(skill)
    corr = np.corrcoef(table.loc[truth.index, "Rating"], truth)[0, 1]
    assert corr > 0.95
    assert table["Rating"].idxmax() == "D0"
    assert table["Rating"].idxmin() == "D7"


def test_fit_ratings_uses_elo_like_scale():
    master, _ = _simulate(seasons=2, races=20)
    table = fit_ratings(build_pairs(master))
    assert ELO_SCALE == pytest.approx(400 / np.log(10))
    assert table["Rating"].abs().max() > 1  # logits times ~174 gives tens of points


def test_bootstrap_ratings_returns_ordered_intervals():
    master, _ = _simulate(seasons=2, races=25)
    out = bootstrap_ratings(build_pairs(master), n_boot=8, seed=1)
    assert (out["Low"] <= out["Rating"]).mean() > 0.7
    assert (out["Low"] <= out["High"]).all()


def test_cross_validate_shows_driver_term_adds_information():
    master, _ = _simulate()
    out = cross_validate(build_pairs(master), folds=4).set_index("model")
    assert {"Constant", "Car only", "Driver only", "Driver + car"} <= set(out.index)
    assert out.loc["Driver + car", "log_loss"] < out.loc["Car only", "log_loss"]
    assert out.loc["Car only", "log_loss"] < out.loc["Constant", "log_loss"]


def test_plot_ratings_saves_file(tmp_path):
    table = pd.DataFrame(
        {
            "Driver Name": ["A", "B", "C"],
            "Rating": [100.0, 20.0, -50.0],
            "Low": [60.0, -10.0, -90.0],
            "High": [140.0, 50.0, -10.0],
        }
    )
    out = tmp_path / "r.png"
    plot_ratings(table, out, highlight=["A"])
    assert out.exists()


def test_select_c_reports_log_loss_for_each_strength():
    master, _ = _simulate(seasons=2, races=25)
    out = select_c(build_pairs(master), c_grid=(0.1, 1.0), folds=3)
    assert list(out["C"]) == [0.1, 1.0]
    assert out["log_loss"].between(0, 0.7).all()
