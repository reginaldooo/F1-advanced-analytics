import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from f1.viz import plot_matrix, plot_metronome, plot_reliability  # noqa: E402


def _stats():
    return pd.DataFrame(
        {
            "Driver Name": [f"d{i}" for i in range(12)],
            "Total Races": [60 + i for i in range(12)],
            "Avg Finish Pct": [0.5 + i * 0.03 for i in range(12)],
            "Win Rate": [0.01 * i for i in range(12)],
            "Pace_StdDev": [0.012 + 0.001 * i for i in range(12)],
        }
    )


def test_plot_metronome_saves_file(tmp_path):
    out = tmp_path / "m.png"
    plot_metronome(_stats(), out)
    assert out.exists()


def test_plot_matrix_saves_file(tmp_path):
    out = tmp_path / "x.png"
    plot_matrix(_stats(), out)
    assert out.exists()


def test_plot_reliability_saves_file(tmp_path):
    rows = []
    for name in ("a", "b"):
        for reason in ("Accident/Driver Error", "Car/Mechanical Failure", "Finished"):
            rows.append({"driver_name": name, "DNF_Reason": reason})
    master = pd.DataFrame(rows)
    stats = pd.DataFrame({"Driver Name": ["a", "b"], "Category": ["Dominant Legends"] * 2})
    out = tmp_path / "r.png"
    plot_reliability(master, stats, out)
    assert out.exists()
