"""Cross-era driver rating: Bradley-Terry on race finishing order with a car effect.

Every pair of starters in a race is one observation (who finished ahead). The latent strength of
an entry is `driver skill + constructor-season effect`, fitted jointly with L2-regularised
logistic regression. Teammates and drivers who change teams separate skill from car.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from f1.dnf import classify_dnf

ELO_SCALE = 400 / np.log(10)
MECHANICAL = "Car/Mechanical Failure"
DEFAULT_C = 0.1  # chosen by cross-validated log-loss over C in {0.03, 0.1, 0.3, 1}
C_GRID = (0.03, 0.1, 0.3, 1.0)
MIN_RACES = 20


@dataclass(frozen=True)
class Pairs:
    X: sparse.csr_matrix
    y: np.ndarray
    race_codes: np.ndarray
    drivers: list[str]
    n_drivers: int
    entries: pd.Series


def build_pairs(master: pd.DataFrame) -> Pairs:
    """One row per (winner, loser) pair of eligible entries in each race.

    Mechanical retirements are not a skill signal and are dropped; driver-error retirements
    stay and lose to every finisher ahead of them. Needs raceId, year, driver_name,
    constructorId, positionOrder and status.
    """
    data = master[master["status"].map(classify_dnf) != MECHANICAL]
    drivers = sorted(data["driver_name"].unique())
    driver_index = {name: i for i, name in enumerate(drivers)}
    car_key = data["constructorId"].astype(str) + "-" + data["year"].astype(str)
    cars = sorted(car_key.unique())
    car_index = {key: i for i, key in enumerate(cars)}
    n_drivers = len(drivers)

    data = data.assign(d=data["driver_name"].map(driver_index), c=car_key.map(car_index))
    rows, cols, vals, labels, races = [], [], [], [], []
    pair_count = 0
    for race_code, (_, race) in enumerate(data.groupby("raceId")):
        race = race.sort_values("positionOrder")
        if len(race) < 2:
            continue
        d, c = race["d"].to_numpy(), race["c"].to_numpy()
        ahead, behind = np.triu_indices(len(race), 1)
        n = len(ahead)
        ids = np.arange(pair_count, pair_count + n)
        sign = np.where(ids % 2 == 0, 1.0, -1.0)  # flip every other pair so y has both classes
        driver_cols = (d[ahead], d[behind])
        car_cols = (n_drivers + c[ahead], n_drivers + c[behind])
        for col_ahead, col_behind in (driver_cols, car_cols):
            rows += [ids, ids]
            cols += [col_ahead, col_behind]
            vals += [sign, -sign]
        labels.append((sign > 0).astype(int))
        races.append(np.full(n, race_code))
        pair_count += n

    X = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(pair_count, n_drivers + len(cars)),
    ).tocsr()
    X.eliminate_zeros()
    entries = data.groupby("driver_name").size().reindex(drivers)
    return Pairs(X, np.concatenate(labels), np.concatenate(races), drivers, n_drivers, entries)


def _fit(pairs: Pairs, columns: slice, c: float, weight: np.ndarray | None) -> LogisticRegression:
    model = LogisticRegression(C=c, fit_intercept=False, max_iter=1000)
    return model.fit(pairs.X[:, columns], pairs.y, sample_weight=weight)


def fit_ratings(
    pairs: Pairs, c: float = DEFAULT_C, sample_weight: np.ndarray | None = None
) -> pd.DataFrame:
    """Driver skill on an Elo-like scale (logit * 400 / ln 10), with the car effect removed."""
    model = _fit(pairs, slice(None), c, sample_weight)
    skill = model.coef_[0][: pairs.n_drivers] * ELO_SCALE
    return pd.DataFrame(
        {"Driver Name": pairs.drivers, "Rating": skill, "Races": pairs.entries.to_numpy()}
    )


def bootstrap_ratings(
    pairs: Pairs, n_boot: int = 30, seed: int = 0, c: float = DEFAULT_C
) -> pd.DataFrame:
    """Full-data rating plus a 5-95% interval from resampling races with replacement."""
    rng = np.random.default_rng(seed)
    n_races = int(pairs.race_codes.max()) + 1
    base = fit_ratings(pairs, c)
    draws = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_races, n_races), minlength=n_races)
        draws.append(fit_ratings(pairs, c, counts[pairs.race_codes].astype(float))["Rating"])
    stacked = np.vstack(draws)
    return base.assign(
        Low=np.percentile(stacked, 5, axis=0), High=np.percentile(stacked, 95, axis=0)
    )


def cross_validate(pairs: Pairs, folds: int = 5, c: float = DEFAULT_C) -> pd.DataFrame:
    """Held-out pairwise log-loss/accuracy by race fold: does the driver term add information?"""
    nd = pairs.n_drivers
    models = {
        "Car only": slice(nd, None),
        "Driver only": slice(0, nd),
        "Driver + car": slice(None),
    }
    scores = {name: {"log_loss": [], "accuracy": []} for name in ["Constant", *models]}
    for train, test in GroupKFold(n_splits=folds).split(pairs.X, pairs.y, pairs.race_codes):
        y_test = pairs.y[test]
        constant = np.full(len(test), 0.5)
        scores["Constant"]["log_loss"].append(log_loss(y_test, constant, labels=[0, 1]))
        scores["Constant"]["accuracy"].append(0.5)
        for name, columns in models.items():
            model = LogisticRegression(C=c, fit_intercept=False, max_iter=1000)
            model.fit(pairs.X[train][:, columns], pairs.y[train])
            proba = model.predict_proba(pairs.X[test][:, columns])[:, 1]
            scores[name]["log_loss"].append(log_loss(y_test, proba, labels=[0, 1]))
            scores[name]["accuracy"].append(float(((proba > 0.5) == y_test).mean()))
    return pd.DataFrame(
        [
            {"model": name, **{k: float(np.mean(v)) for k, v in s.items()}}
            for name, s in scores.items()
        ]
    )


def select_c(pairs: Pairs, c_grid: tuple[float, ...] = C_GRID, folds: int = 5) -> pd.DataFrame:
    """Held-out log-loss of the driver + car model for each regularisation strength."""
    rows = []
    for c in c_grid:
        scores = cross_validate(pairs, folds=folds, c=c).set_index("model").loc["Driver + car"]
        rows.append({"C": c, "log_loss": scores["log_loss"], "accuracy": scores["accuracy"]})
    return pd.DataFrame(rows)


def plot_ratings(
    table: pd.DataFrame,
    path: str | Path | None = None,
    highlight: tuple[str, ...] | list[str] = (),
    top: int = 15,
) -> None:
    shown = pd.concat(
        [table.nlargest(top, "Rating"), table[table["Driver Name"].isin(highlight)]]
    ).drop_duplicates("Driver Name").sort_values("Rating")
    colors = ["#E10600" if n in highlight else "#4C72B0" for n in shown["Driver Name"]]
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(shown) + 1.5))
    ax.errorbar(
        shown["Rating"], range(len(shown)),
        xerr=[shown["Rating"] - shown["Low"], shown["High"] - shown["Rating"]],
        fmt="none", ecolor="#888888", capsize=3,
    )
    ax.scatter(shown["Rating"], range(len(shown)), c=colors, zorder=3)
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels(shown["Driver Name"])
    ax.set_xlabel("Driver skill rating (Elo-like points, car effect removed, 5-95% interval)")
    ax.set_title("Cross-era driver rating", fontsize=14, weight="bold")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.show()
    plt.close(fig)
