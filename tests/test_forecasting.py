"""Unit tests for time-series forecasting, naive baseline, and evaluation metrics."""

import numpy as np
import pandas as pd
from prophet import Prophet
import pytest

from smartbizlq.analytics.forecasting import (
    ForecastResult,
    compute_forecast_metrics,
    compute_naive_baseline,
    run_forecast,
)


def test_compute_forecast_metrics():
    actual = [100.0, 200.0, 300.0]
    predicted = [110.0, 190.0, 300.0]

    metrics = compute_forecast_metrics(actual, predicted)

    # Errors: +10, -10, 0 -> MAE = 20 / 3 = 6.6667
    assert metrics["mae"] == pytest.approx(6.6667, 0.001)
    # Squared errors: 100, 100, 0 -> mean = 200/3 -> sqrt = 8.16496
    assert metrics["rmse"] == pytest.approx(8.1650, 0.001)
    # Percentage errors: 10%, 5%, 0% -> mean = 5.0%
    assert metrics["mape"] == pytest.approx(5.0, 0.001)


def test_compute_naive_baseline():
    train = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    test = pd.Series([60.0, 70.0], index=[5, 6])

    # Should repeat the last 2 elements: [40.0, 50.0]
    baseline = compute_naive_baseline(train, test)

    assert isinstance(baseline, pd.Series)
    assert len(baseline) == 2
    # Plain naive (season length 1) repeats the last value
    assert list(baseline.values) == [50.0, 50.0]
    # Seasonal naive with a 2-period cycle repeats the last full cycle
    seasonal = compute_naive_baseline(train, test, season_length=2)
    assert list(seasonal.values) == [40.0, 50.0]
    assert list(baseline.index) == [5, 6]


def test_run_forecast_pipeline():
    # Generate 60 days of daily time series
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    trend = np.linspace(100, 200, 60)
    seasonal = 10 * np.sin(np.linspace(0, 8 * np.pi, 60))
    values = trend + seasonal

    df = pd.DataFrame({"timestamp": dates, "sales": values})

    result = run_forecast(
        df=df,
        date_col="timestamp",
        target_col="sales",
        periods=14,
    )

    assert isinstance(result, ForecastResult)
    assert isinstance(result.model, Prophet)
    assert isinstance(result.forecast_df, pd.DataFrame)
    assert "yhat" in result.forecast_df.columns
    assert "yhat_lower" in result.forecast_df.columns
    assert "yhat_upper" in result.forecast_df.columns

    # Verify interval ordering
    assert (result.forecast_df["yhat_lower"] <= result.forecast_df["yhat_upper"]).all()

    # Metrics check
    assert "model" in result.metrics
    assert "baseline" in result.metrics
    assert "mae" in result.metrics
    assert "rmse" in result.metrics
    assert "mape" in result.metrics
    assert result.metrics["mae"] >= 0.0


def test_run_forecast_with_regressors():
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    promo = np.random.choice([0, 1], size=50)
    sales = 100 + promo * 25 + np.random.normal(0, 2, size=50)

    df = pd.DataFrame({"timestamp": dates, "sales": sales, "promo": promo})

    result = run_forecast(
        df=df,
        date_col="timestamp",
        target_col="sales",
        periods=7,
        regressor_cols=["promo"],
    )

    assert isinstance(result, ForecastResult)
    assert "yhat" in result.forecast_df.columns
    assert result.model.extra_regressors.get("promo") is not None


def test_run_forecast_invalid_columns():
    df = pd.DataFrame({"date": ["2024-01-01"], "target": [10]})
    with pytest.raises(KeyError):
        run_forecast(df, date_col="missing_date", target_col="target")


def test_run_forecast_insufficient_data():
    df = pd.DataFrame({
        "ds": pd.date_range("2024-01-01", periods=3),
        "y": [1.0, 2.0, 3.0],
    })
    with pytest.raises(ValueError):
        run_forecast(df, date_col="ds", target_col="y")


def test_infer_frequency_and_season_length():
    from smartbizlq.analytics.forecasting import infer_frequency, season_length_for

    weekly = pd.Series(pd.date_range("2024-01-07", periods=20, freq="W"))
    assert infer_frequency(weekly).startswith("W")
    assert season_length_for(infer_frequency(weekly)) == 52
    assert season_length_for("D") == 7
    assert season_length_for("MS") == 12


def test_run_forecast_weekly_resample_produces_weekly_future():
    from smartbizlq.analytics.forecasting import run_forecast

    rng = np.random.default_rng(0)
    days = pd.date_range("2023-01-01", periods=300, freq="D")
    orders = pd.DataFrame({"order_date": np.repeat(days, 2), "revenue": rng.uniform(50, 150, 600)})
    res = run_forecast(orders, "order_date", "revenue", periods=4, resample="W", agg="sum")
    future = res.forecast_df[res.forecast_df["ds"] > res.test_df["ds"].max()]
    assert len(future) == 4
    assert (future["ds"].diff().dropna() == pd.Timedelta(days=7)).all()
    assert res.metrics["frequency"] == "W"
