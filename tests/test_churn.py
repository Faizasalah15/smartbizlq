"""Unit tests for the churn prediction model and pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from smartbizlq.analytics.churn import ChurnResult, train_churn_model


def test_train_churn_model_basic():
    np.random.seed(42)
    n_samples = 120

    tenure = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
    contract_type = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples)
    # Synthesize churn with strong signal from tenure and contract
    churn_prob = np.where(contract_type == "Month-to-month", 0.6, 0.1)
    churn = (np.random.rand(n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "contract_type": contract_type,
        "churn": churn,
    })

    result = train_churn_model(
        df=df,
        churn_col="churn",
        feature_cols=["tenure", "monthly_charges", "contract_type"],
        n_estimators=10,
        test_size=0.25,
        cv_folds=3,
    )

    assert isinstance(result, ChurnResult)
    assert isinstance(result.model, Pipeline)
    assert "accuracy" in result.metrics
    assert "precision" in result.metrics
    assert "recall" in result.metrics
    assert "f1" in result.metrics
    assert "roc_auc" in result.metrics
    assert result.confusion_matrix.shape == (2, 2)
    assert len(result.feature_importance) == 3
    assert len(result.X_test) == len(result.y_test)
    assert len(result.y_pred) == len(result.y_test)
    assert len(result.y_prob) == len(result.y_test)
    assert "mean" in result.cv_scores


def test_train_churn_model_missing_target():
    df = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
    with pytest.raises(KeyError):
        train_churn_model(df, churn_col="target", feature_cols=["feat1", "feat2"])


def test_train_churn_model_single_class():
    df = pd.DataFrame({
        "feat1": [1, 2, 3, 4],
        "target": [0, 0, 0, 0],
    })
    with pytest.raises(ValueError):
        train_churn_model(df, churn_col="target", feature_cols=["feat1"])


def test_train_churn_model_accepts_yes_no_labels():
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame({
        "tenure": rng.integers(1, 72, n),
        "contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
    })
    df["Churn"] = np.where((df["contract"] == "Month-to-month") & (df["tenure"] < 24), "Yes", "No")
    res = train_churn_model(df, "Churn", ["tenure", "contract"], n_estimators=50, cv_folds=3)
    assert set(np.unique(res.y_test)) == {0, 1}
    assert res.metrics["roc_auc"] > 0.9
