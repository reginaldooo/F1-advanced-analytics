"""Load the Kaggle F1 tables from a local folder or the kagglehub cache."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

KAGGLE_DATASET = "rohanrao/formula-1-world-championship-1950-2020"
DATA_DIR_ENV = "F1_DATA_DIR"
REQUIRED_TABLES = ("drivers", "results", "status", "lap_times", "races")


def _resolve_data_dir(data_dir: str | Path | None) -> Path:
    """Explicit arg, then $F1_DATA_DIR, then ./data, then a kagglehub download."""
    if data_dir is not None:
        return Path(data_dir)
    if os.environ.get(DATA_DIR_ENV):
        return Path(os.environ[DATA_DIR_ENV])
    local = Path("data")
    if local.is_dir():
        return local
    import kagglehub

    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def load_tables(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Return the required tables as DataFrames; `\\N` is read as missing."""
    root = _resolve_data_dir(data_dir)
    tables = {}
    for name in REQUIRED_TABLES:
        path = root / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}.csv in {root}")
        tables[name] = pd.read_csv(path, na_values="\\N")
    return tables
