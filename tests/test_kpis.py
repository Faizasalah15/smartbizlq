"""Unit tests for the KPI computation and classification metrics module."""

import numpy as np
import pandas as pd
import pytest

from smartbizlq.analytics.kpis import (
    compute_classification_metrics,
    compute_kpi_strip,
)


def test_compute_kpi_strip_standard():
    df = pd.DataFrame({
        "order_id": ["O1", "O1", "O2", "O3"],
        "quantity": [2, 3, 5, 10],
        "revenue": [200.0, 300.0, 500.0, 1000.0],
    })

    kpis = compute_kpi_strip(df, revenue_col="revenue", qty_col="quantity", id_col="order_id")

    assert kpis["total_revenue"] == 2000.0
    assert kpis["total_orders"] == 3.0
    assert kpis["total_units"] == 20.0
    assert kpis["avg_price"] == 100.0


def test_compute_kpi_strip_empty():
    df = pd.DataFrame(columns=["order_id", "quantity", "revenue"])
    kpis = compute_kpi_strip(df, revenue_col="revenue", qty_col="quantity", id_col="order_id")

    assert kpis["total_revenue"] == 0.0
    assert kpis["total_orders"] == 0.0
    assert kpis["total_units"] == 0.0
    assert kpis["avg_price"] == 0.0


def test_compute_kpi_strip_missing_column():
    df = pd.DataFrame({"order_id": ["O1"], "revenue": [100.0]})
    with pytest.raises(KeyError):
        compute_kpi_strip(df, revenue_col="revenue", qty_col="missing_qty", id_col="order_id")


def test_compute_classification_metrics_binary():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.15])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob=y_prob)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert metrics["accuracy"] == pytest.approx(5 / 6, 0.01)
    assert metrics["roc_auc"] > 0.8


def test_compute_classification_metrics_2d_proba():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 0]
    y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.85, 0.15]])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob=y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_classification_metrics_no_prob():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert "roc_auc" not in metrics
