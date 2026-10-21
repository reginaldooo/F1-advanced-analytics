import json

import numpy as np
import pandas as pd
import pytest

from f1.app_metrics import (
    clustering_metrics,
    profile_metrics,
    rating_metrics,
    read_metrics,
    reliability_metrics,
    write_metrics,
)
from f1.rating import build_pairs


@pytest.fixture
def features():
    rng = np.random.default_rng(0)
    centers = np.array([[0.2, 0.0, 0.0], [0.5, 0.1, 0.02], [0.8, 0.5, 0.2], [0.65, 0.3, 0.08]])
    return np.vstack([c + rng.normal(0, 0.02, size=(15, 3)) for c in centers])


@pytest.fixture
def master():
    rows = []
    for race in range(1, 9):
        for pos, (driver, team) in enumerate([("A", 1), ("B", 1), ("C", 2), ("D", 2)], start=1):
            order = pos if race % 2 else 5 - pos
            rows.append(
                {
                    "raceId": race,
                    "year": 2000 + race // 4,
                    "driver_name": driver,
                    "constructorId": team,
                    "positionOrder": order,
                    "status": "Finished",
                }
            )
    return pd.DataFrame(rows)


def test_clustering_metrics_covers_k2_to_k10_and_three_algorithms(features):
    out = clustering_metrics(features)
    assert [row["k"] for row in out["k_table"]] == list(range(2, 11))
    assert {row["method"] for row in out["algorithms"]} == {"K-Means", "GMM", "HDBSCAN"}
    assert out["chosen_k"] == 4
    assert all(-1 <= row["silhouette"] <= 1 for row in out["k_table"])


def test_reliability_metrics_compares_keywords_with_trained_model():
    out = reliability_metrics()
    methods = {row["method"]: row for row in out["classifiers"]}
    assert set(methods) == {"Keywords", "Trained (leave-one-out)"}
    assert methods["Trained (leave-one-out)"]["accuracy"] > methods["Keywords"]["accuracy"]
    assert 0 <= methods["Keywords"]["accuracy"] <= 1
    assert out["n_statuses"] == sum(out["label_counts"].values())


def test_reliability_metrics_weights_by_status_counts():
    plain = reliability_metrics()
    weighted = reliability_metrics(pd.Series({"Finished": 1000, "Engine": 10}))
    assert weighted["classifiers"][0]["weighted_accuracy"] != plain["classifiers"][0][
        "weighted_accuracy"
    ]


def test_rating_metrics_reports_models_and_regularisation_grid(master):
    out = rating_metrics(build_pairs(master), folds=2)
    models = {row["model"]: row for row in out["cv"]}
    assert set(models) == {"Constant", "Car only", "Driver only", "Driver + car"}
    assert models["Constant"]["accuracy"] == 0.5
    assert [row["C"] for row in out["c_grid"]] == [0.03, 0.1, 0.3, 1.0]
    assert out["n_drivers"] == 4
    assert out["n_pairs"] == 8 * 6


def test_profile_metrics_reports_pc1_share_and_correlations():
    rng = np.random.default_rng(1)
    base = rng.normal(size=60)
    profile = pd.DataFrame(
        {
            "Avg Finish Pct": base,
            "Win Rate": base + rng.normal(0, 0.1, 60),
            "Podium Rate": base + rng.normal(0, 0.1, 60),
            "Positions Gained": rng.normal(size=60),
            "Driver-Error DNF Rate": rng.normal(size=60),
            "Mechanical DNF Rate": rng.normal(size=60),
            "Teammate Delta": rng.normal(size=60),
        }
    )
    out = profile_metrics(profile)
    assert 0 < out["pc1_share"] <= 1
    assert out["correlation"]["Win Rate"]["Avg Finish Pct"] > 0.9


def test_write_and_read_metrics_round_trip_with_nan_as_null(tmp_path):
    data = {"a": [{"x": float("nan"), "y": 1.5}], "b": 3}
    path = write_metrics(data, tmp_path / "out" / "metrics.json")
    assert json.loads(path.read_text())["a"][0]["x"] is None
    assert read_metrics(path)["b"] == 3
