from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
DATA = APP.parent / "data" / "drivers.csv"


@pytest.fixture
def app():
    return AppTest.from_file(str(APP), default_timeout=60).run()


def test_app_data_is_committed_and_complete():
    table = pd.read_csv(DATA)
    assert len(table) > 200
    assert table["Driver Name"].is_unique
    assert table[["Avg Finish Pct", "Win Rate", "PC1", "PC2", "Category"]].notna().all().all()


def test_app_renders_without_exceptions_and_has_four_tabs(app):
    assert not app.exception
    assert [t.label for t in app.tabs] == [
        "Driver Explorer",
        "Cluster View",
        "Compare",
        "How to Read This",
    ]


def test_compare_defaults_to_fangio_and_verstappen(app):
    values = {s.label: s.value for s in app.selectbox}
    assert values["Driver A"] == "Juan Fangio"
    assert values["Driver B"] == "Max Verstappen"


def test_compare_handles_a_driver_without_lap_data_or_rating(app):
    table = pd.read_csv(DATA)
    unrated = table.loc[table["Rating"].isna(), "Driver Name"].iloc[0]
    driver_a = next(s for s in app.selectbox if s.label == "Driver A")
    driver_a.select(unrated).run()
    assert not app.exception


def test_filters_that_match_nothing_show_empty_state_not_an_error(app):
    search = next(t for t in app.sidebar.text_input if t.label == "Search driver")
    search.input("zzzz-no-such-driver").run()
    assert not app.exception


def test_theme_config_uses_palette_and_tiers_are_colored():
    config = (APP.parents[1] / ".streamlit" / "config.toml").read_text()
    for hex_color in ("#FF6A00", "#0D0D0D", "#2B2D30", "#F5F5F5"):
        assert hex_color in config
    source = APP.read_text()
    for tier in ("Dominant Legends", "Podium Contenders", "Midfielders", "Backmarkers"):
        assert f'"{tier}": ' in source.split("TIER_COLOR = {")[1].split("}")[0]
