import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from f1.clustering import (  # noqa: E402
    compare_algorithms,
    evaluate_k,
    plot_k_selection,
    plot_pca,
    scale_features,
)


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[0.9, 0.8, 0.4], [0.6, 0.3, 0.1], [0.2, 0.0, 0.0]])
    rows = np.vstack([c + rng.normal(0, 0.01, (30, 3)) for c in centers])
    return pd.DataFrame(rows, columns=["Avg Finish Pct", "Podium Rate", "Win Rate"])


def test_scale_features_returns_array_with_one_column_per_feature(blobs):
    assert scale_features(blobs).shape == (90, 3)


def test_evaluate_k_has_one_row_per_k_with_expected_columns(blobs):
    out = evaluate_k(scale_features(blobs), ks=range(2, 6), seeds=range(3))
    assert list(out["k"]) == [2, 3, 4, 5]
    assert {"silhouette", "davies_bouldin", "inertia", "stability"} <= set(out.columns)


def test_evaluate_k_finds_true_cluster_count_and_perfect_stability(blobs):
    out = evaluate_k(scale_features(blobs), ks=range(2, 6), seeds=range(3)).set_index("k")
    assert out.loc[3, "stability"] == pytest.approx(1.0)
    assert out.loc[[3, 4, 5], "davies_bouldin"].idxmin() == 3


def test_compare_algorithms_reports_each_method_and_noise(blobs):
    out = compare_algorithms(scale_features(blobs), k=3).set_index("method")
    assert {"K-Means", "GMM", "HDBSCAN"} <= set(out.index)
    assert out.loc["K-Means", "clusters"] == 3
    assert out.loc["K-Means", "noise"] == 0


def test_plots_save_files(blobs, tmp_path):
    X = scale_features(blobs)
    table = evaluate_k(X, ks=range(2, 5), seeds=range(2))
    plot_k_selection(table, tmp_path / "k.png")
    labels = np.repeat([0, 1, 2], 30)
    plot_pca(X, labels, tmp_path / "pca.png")
    assert (tmp_path / "k.png").exists()
    assert (tmp_path / "pca.png").exists()
