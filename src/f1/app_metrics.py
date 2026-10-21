"""Model-quality numbers shown in the app's Methodology tab, written to a JSON file.

Only existing evaluation functions are called; nothing here fits a new model.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from f1.clustering import compare_algorithms, evaluate_k
from f1.dnf import evaluate_classifiers, load_status_labels
from f1.profile import redundancy
from f1.rating import C_GRID, DEFAULT_C, Pairs, cross_validate, select_c

METRICS_PATH = Path("app/data/metrics.json")
CHOSEN_K = 4
RATING_FOLDS = 5


def _records(table: pd.DataFrame) -> list[dict]:
    return table.to_dict("records")


def clustering_metrics(X: np.ndarray) -> dict:
    """Validity scores per k (2..10) and K-Means vs GMM vs HDBSCAN at the chosen k."""
    return {
        "chosen_k": CHOSEN_K,
        "k_table": _records(evaluate_k(X)),
        "algorithms": _records(compare_algorithms(X, k=CHOSEN_K)),
        "n_drivers": int(len(X)),
    }


def reliability_metrics(counts: pd.Series | None = None) -> dict:
    """Keyword rules vs trained classifier against the hand labels, plus label counts."""
    label_counts = dict(Counter(load_status_labels().values()))
    return {
        "classifiers": _records(evaluate_classifiers(counts)),
        "label_counts": label_counts,
        "n_statuses": sum(label_counts.values()),
    }


def rating_metrics(pairs: Pairs, folds: int = RATING_FOLDS) -> dict:
    """Held-out pairwise log-loss/accuracy per model and per regularisation strength."""
    return {
        "c": DEFAULT_C,
        "folds": folds,
        "cv": _records(cross_validate(pairs, folds=folds)),
        "c_grid": _records(select_c(pairs, C_GRID, folds=folds)),
        "n_pairs": int(pairs.X.shape[0]),
        "n_drivers": int(pairs.n_drivers),
        "n_car_seasons": int(pairs.X.shape[1] - pairs.n_drivers),
    }


def profile_metrics(profile: pd.DataFrame) -> dict:
    """Criteria correlation matrix and the variance share of the first principal component."""
    corr, variance = redundancy(profile)
    return {"pc1_share": float(variance[0]), "correlation": corr.to_dict()}


def data_metrics(tables: dict, profile: pd.DataFrame, consistency: pd.DataFrame) -> dict:
    return {
        "results_rows": int(len(tables["results"])),
        "lap_rows": int(len(tables["lap_times"])),
        "drivers_profiled": int(len(profile)),
        "drivers_with_consistency": int(
            profile["Driver Name"].isin(consistency["Driver Name"]).sum()
        ),
        "drivers_with_teammate_delta": int(profile["Teammate Delta"].notna().sum()),
    }


def build_metrics(
    tables: dict,
    profile: pd.DataFrame,
    consistency: pd.DataFrame,
    pairs: Pairs,
    X: np.ndarray,
) -> dict:
    counts = tables["results"].merge(tables["status"], on="statusId")["status"].value_counts()
    return {
        "data": data_metrics(tables, profile, consistency),
        "clustering": clustering_metrics(X),
        "reliability": reliability_metrics(counts),
        "rating": rating_metrics(pairs),
        "profile": profile_metrics(profile),
    }


def _clean(value):
    """JSON has no NaN: turn it into null, recursively."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def write_metrics(metrics: dict, path: str | Path = METRICS_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(metrics), indent=2, allow_nan=False) + "\n")
    return path


def read_metrics(path: str | Path = METRICS_PATH) -> dict:
    return json.loads(Path(path).read_text())
