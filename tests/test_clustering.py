"""Unit tests for the K-Means clustering and segmentation module."""

import numpy as np
import pandas as pd
import pytest

from smartbizlq.analytics.clustering import ClusterResult, run_clustering


def test_run_clustering_basic():
    np.random.seed(42)
    # 3 distinct clusters in 2D space
    c1 = np.random.normal(loc=[10.0, 10.0], scale=1.0, size=(20, 2))
    c2 = np.random.normal(loc=[50.0, 50.0], scale=1.0, size=(20, 2))
    c3 = np.random.normal(loc=[90.0, 90.0], scale=1.0, size=(20, 2))

    data = np.vstack([c1, c2, c3])
    df = pd.DataFrame(data, columns=["feature_x", "feature_y"])

    result = run_clustering(df, feature_cols=["feature_x", "feature_y"], n_clusters=3, max_k=5)

    assert isinstance(result, ClusterResult)
    assert len(result.labels) == 60
    assert len(np.unique(result.labels)) == 3
    assert result.silhouette > 0.6  # High separation
    assert len(result.elbow_data) == 4  # k=2, 3, 4, 5
    assert isinstance(result.profiles, pd.DataFrame)
    assert len(result.profiles) == 3
    assert len(result.descriptions) == 3
    assert all("Cluster" in desc for desc in result.descriptions)


def test_run_clustering_missing_column():
    df = pd.DataFrame({"feat1": [1, 2, 3]})
    with pytest.raises(KeyError):
        run_clustering(df, feature_cols=["missing_feat"], n_clusters=2)


def test_run_clustering_insufficient_samples():
    df = pd.DataFrame({"feat1": [1.0, 2.0], "feat2": [3.0, 4.0]})
    with pytest.raises(ValueError):
        run_clustering(df, feature_cols=["feat1", "feat2"], n_clusters=5)
