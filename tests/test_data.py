import pandas as pd
import pytest

from f1.data import REQUIRED_TABLES, load_tables


def _write_tables(path):
    for name in REQUIRED_TABLES:
        pd.DataFrame({"a": [1, None]}).to_csv(path / f"{name}.csv", index=False)


def test_load_tables_from_explicit_dir(tmp_path):
    _write_tables(tmp_path)
    tables = load_tables(tmp_path)
    assert set(tables) == set(REQUIRED_TABLES)


def test_load_tables_reads_backslash_n_as_missing(tmp_path):
    _write_tables(tmp_path)
    (tmp_path / "results.csv").write_text("a,b\n1,\\N\n")
    assert load_tables(tmp_path)["results"]["b"].isna().all()


def test_load_tables_uses_env_var(tmp_path, monkeypatch):
    _write_tables(tmp_path)
    monkeypatch.setenv("F1_DATA_DIR", str(tmp_path))
    assert set(load_tables()) == set(REQUIRED_TABLES)


def test_load_tables_missing_file_raises_clear_error(tmp_path):
    _write_tables(tmp_path)
    (tmp_path / "results.csv").unlink()
    with pytest.raises(FileNotFoundError, match="results"):
        load_tables(tmp_path)
