"""Key Performance Indicator (KPI) and model evaluation metrics module.

This module provides enterprise-grade computation functions for top-line
business indicators (e.g. revenue, order volume, average price) and
machine learning classification evaluation (accuracy, precision, recall,
F1 score, and ROC-AUC).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_kpi_strip(
    df: pd.DataFrame,
    revenue_col: str,
    qty_col: str,
    id_col: str,
) -> dict[str, float]:
    """Compute aggregate top-level business KPIs from a transaction DataFrame.

    Calculates:
    - total_revenue: Gross monetary value across transactions.
    - total_orders: Count of distinct order identifiers.
    - total_units: Total volume of units sold.
    - avg_price: Blended average selling price per unit (total_revenue / total_units).

    Parameters
    ----------
    df : pd.DataFrame
        Input transaction dataset.
    revenue_col : str
        Column name representing monetary transaction amounts.
    qty_col : str
        Column name representing transaction quantities / volume.
    id_col : str
        Column name identifying individual orders or transactions.

    Returns
    -------
    dict[str, float]
        Dictionary with keys: 'total_revenue', 'total_orders', 'total_units', 'avg_price'.

    Raises
    ------
    KeyError
        If any of revenue_col, qty_col, or id_col are not present in df.
    """
    logger.debug(
        "Computing KPI strip for columns: rev=%s, qty=%s, id=%s",
        revenue_col,
        qty_col,
        id_col,
    )

    required_cols = [revenue_col, qty_col, id_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        msg = f"Missing required column(s) for KPI computation: {missing}"
        logger.error(msg)
        raise KeyError(msg)

    if df.empty:
        logger.warning("Empty DataFrame provided to compute_kpi_strip. Returning zero metrics.")
        return {
            "total_revenue": 0.0,
            "total_orders": 0.0,
            "total_units": 0.0,
            "avg_price": 0.0,
        }

    rev_series = pd.to_numeric(df[revenue_col], errors="coerce").fillna(0.0)
    qty_series = pd.to_numeric(df[qty_col], errors="coerce").fillna(0.0)

    total_revenue = float(rev_series.sum())
    total_orders = float(df[id_col].dropna().nunique())
    total_units = float(qty_series.sum())
    avg_price = float(total_revenue / total_units) if total_units > 0.0 else 0.0

    kpis = {
        "total_revenue": round(total_revenue, 2),
        "total_orders": total_orders,
        "total_units": round(total_units, 2),
        "avg_price": round(avg_price, 2),
    }

    logger.info(
        "KPI strip computed successfully: Revenue=%.2f, Orders=%.0f, Units=%.2f, Avg Price=%.2f",
        kpis["total_revenue"],
        kpis["total_orders"],
        kpis["total_units"],
        kpis["avg_price"],
    )
    return kpis


def compute_classification_metrics(
    y_true: Sequence[Any] | np.ndarray | pd.Series,
    y_pred: Sequence[Any] | np.ndarray | pd.Series,
    y_prob: Sequence[float] | np.ndarray | pd.Series | None = None,
) -> dict[str, float]:
    """Calculate comprehensive classification evaluation metrics.

    Computes Accuracy, Precision, Recall, F1 score, and optionally ROC-AUC
    when predicted class probabilities are supplied. Dynamically selects
    binary or weighted averaging based on the number of unique classes.

    Parameters
    ----------
    y_true : Sequence[Any] | np.ndarray | pd.Series
        Ground-truth target labels.
    y_pred : Sequence[Any] | np.ndarray | pd.Series
        Predicted target labels from classifier.
    y_prob : Sequence[float] | np.ndarray | pd.Series | None, optional
        Predicted positive class probabilities (or 2D probability matrix).
        Default is None.

    Returns
    -------
    dict[str, float]
        Dictionary containing metric scores:
        - 'accuracy': Overall classification accuracy.
        - 'precision': Precision (binary or weighted).
        - 'recall': Recall (binary or weighted).
        - 'f1': F1 harmonic mean of precision and recall.
        - 'roc_auc': (Optional) Area Under the Receiver Operating Characteristic
          Curve if y_prob is supplied and both positive and negative classes exist.
    """
    logger.debug("Computing classification metrics on %d samples", len(y_true))

    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if len(y_t) == 0:
        logger.warning("Empty ground truth passed to compute_classification_metrics.")
        res = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        if y_prob is not None:
            res["roc_auc"] = 0.0
        return res

    unique_classes = np.unique(y_t)
    average_mode = "binary" if len(unique_classes) <= 2 else "weighted"

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, average=average_mode, zero_division=0))
    rec = float(recall_score(y_t, y_p, average=average_mode, zero_division=0))
    f1 = float(f1_score(y_t, y_p, average=average_mode, zero_division=0))

    metrics: dict[str, float] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob)
        try:
            if len(unique_classes) == 2:
                # If y_prob_arr is 2D shape (N, 2), extract column 1 (positive class)
                if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 2:
                    auc_score = float(roc_auc_score(y_t, y_prob_arr[:, 1]))
                elif y_prob_arr.ndim == 1:
                    auc_score = float(roc_auc_score(y_t, y_prob_arr))
                else:
                    auc_score = float(roc_auc_score(y_t, y_prob_arr, multi_class="ovr"))
                metrics["roc_auc"] = round(auc_score, 4)
            elif len(unique_classes) > 2:
                auc_score = float(roc_auc_score(y_t, y_prob_arr, multi_class="ovr", average="weighted"))
                metrics["roc_auc"] = round(auc_score, 4)
            else:
                logger.warning("Only 1 unique class present in y_true; ROC-AUC is undefined.")
                metrics["roc_auc"] = 0.0
        except Exception as exc:
            logger.warning("Failed to compute ROC-AUC: %s", exc)
            metrics["roc_auc"] = 0.0

    logger.info("Classification metrics computed: %s", metrics)
    return metrics
