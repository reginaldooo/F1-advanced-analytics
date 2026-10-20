"""Model selection for the driver clustering: choose k, compare algorithms, project to 2D."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from f1.features import CLUSTER_FEATURES

DEFAULT_SEED = 42
HDBSCAN_MIN_CLUSTER_SIZE = 5


def scale_features(stats: pd.DataFrame) -> np.ndarray:
    return RobustScaler().fit_transform(stats[CLUSTER_FEATURES])


def evaluate_k(
    X: np.ndarray, ks: Iterable[int] = range(2, 11), seeds: Iterable[int] = range(10)
) -> pd.DataFrame:
    """Silhouette, Davies-Bouldin, inertia and seed stability (mean ARI) for each k."""
    seeds = list(seeds)
    rows = []
    for k in ks:
        base = KMeans(k, random_state=DEFAULT_SEED, n_init=10).fit(X)
        others = [KMeans(k, random_state=s, n_init=10).fit_predict(X) for s in seeds]
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(X, base.labels_),
                "davies_bouldin": davies_bouldin_score(X, base.labels_),
                "inertia": base.inertia_,
                "stability": float(np.mean([adjusted_rand_score(base.labels_, o) for o in others])),
            }
        )
    return pd.DataFrame(rows)


def _score(X: np.ndarray, labels: np.ndarray) -> dict:
    mask = labels != -1
    clusters = len(set(labels[mask]))
    scorable = clusters > 1 and mask.sum() > clusters
    return {
        "clusters": clusters,
        "noise": int((~mask).sum()),
        "silhouette": silhouette_score(X[mask], labels[mask]) if scorable else np.nan,
        "davies_bouldin": davies_bouldin_score(X[mask], labels[mask]) if scorable else np.nan,
    }


def compare_algorithms(X: np.ndarray, k: int = 4, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """K-Means vs GMM vs HDBSCAN; noise = points HDBSCAN leaves unassigned."""
    labelings = {
        "K-Means": KMeans(k, random_state=seed, n_init=10).fit_predict(X),
        "GMM": GaussianMixture(k, random_state=seed, n_init=5).fit(X).predict(X),
        "HDBSCAN": HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE).fit_predict(X),
    }
    return pd.DataFrame([{"method": m, **_score(X, labels)} for m, labels in labelings.items()])


def plot_k_selection(table: pd.DataFrame, path: str | Path | None = None) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    panels = [
        ("inertia", "Elbow (inertia)"),
        ("silhouette", "Silhouette (higher is better)"),
        ("davies_bouldin", "Davies-Bouldin (lower is better)"),
        ("stability", "Seed stability (ARI)"),
    ]
    for ax, (col, title) in zip(axes, panels):
        ax.plot(table["k"], table[col], marker="o")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("k")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_pca(
    X: np.ndarray,
    labels: np.ndarray,
    path: str | Path | None = None,
    names: np.ndarray | None = None,
) -> None:
    pcs = PCA(n_components=2, random_state=DEFAULT_SEED).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(pcs[:, 0], pcs[:, 1], c=labels, cmap="viridis", alpha=0.8)
    ax.legend(*scatter.legend_elements(), title="cluster")
    if names is not None:
        for (x, y), name in zip(pcs, names):
            ax.annotate(name, (x, y), fontsize=7, alpha=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Drivers in 2D (PCA of scaled features)")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.show()
    plt.close(fig)
