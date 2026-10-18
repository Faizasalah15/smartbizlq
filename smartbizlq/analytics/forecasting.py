"""Prophet-based time-series forecasting and evaluation module.

This module provides production-grade time-series forecasting leveraging
Facebook Prophet, including:
- Chronological train/test splits.
- Seasonality and optional exogenous regressor integration.
- Prediction interval estimation (yhat_lower, yhat_upper).
- Model evaluation against a seasonal naive baseline (MAE, RMSE, MAPE).
- Optional rolling cross-validation diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Dataclass holding complete forecasting outputs and diagnostics.

    Attributes
    ----------
    forecast_df : pd.DataFrame
        Complete Prophet forecast DataFrame containing ds, yhat, yhat_lower,
        yhat_upper, and component breakdown.
    train_df : pd.DataFrame
        Historical training partition (ds, y, and regressors).
    test_df : pd.DataFrame
        Held-out evaluation partition (ds, y, and regressors).
    metrics : dict[str, Any]
        Dictionary comparing model vs baseline MAE, RMSE, and MAPE.
    cv_metrics : dict[str, float] | None
        Summary cross-validation metrics across rolling horizons, if computed.
    model : Prophet
        Fitted Facebook Prophet model instance.
    """

    forecast_df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    metrics: dict[str, Any]
    cv_metrics: dict[str, float] | None
    model: Prophet


def compute_forecast_metrics(
    actual: Sequence[float] | np.ndarray | pd.Series,
    predicted: Sequence[float] | np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute standard regression/forecasting error metrics: MAE, RMSE, and MAPE.

    Parameters
    ----------
    actual : Sequence[float] | np.ndarray | pd.Series
        Ground-truth actual target values.
    predicted : Sequence[float] | np.ndarray | pd.Series
        Forecasted/predicted target values.

    Returns
    -------
    dict[str, float]
        Dictionary with keys 'mae', 'rmse', 'mape'.
    """
    act = np.asarray(actual, dtype=float)
    pred = np.asarray(predicted, dtype=float)

    if len(act) == 0 or len(pred) == 0:
        logger.warning("Empty series passed to compute_forecast_metrics.")
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

    # Mean Absolute Error
    mae = float(np.mean(np.abs(act - pred)))

    # Root Mean Squared Error
    rmse = float(np.sqrt(np.mean((act - pred) ** 2)))

    # Mean Absolute Percentage Error (%) - avoid divide by zero
    non_zero_mask = np.abs(act) > 1e-8
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs((act[non_zero_mask] - pred[non_zero_mask]) / act[non_zero_mask])) * 100.0)
    else:
        mape = 0.0

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
    }


def compute_naive_baseline(
    train: Sequence[float] | np.ndarray | pd.Series,
    test: Sequence[float] | np.ndarray | pd.Series,
    season_length: int = 1,
) -> pd.Series:
    """Seasonal naive benchmark: each test point copies the value one season earlier.

    With ``season_length=1`` this is the plain naive forecast (repeat the last
    training value). With e.g. 7 for daily data it repeats the same weekday from
    the last observed week, which is the standard benchmark a forecasting model
    should beat.

    Args:
        train: Training values in time order.
        test: Test values (only the length and index are used).
        season_length: Number of periods in one seasonal cycle.

    Returns:
        Baseline predictions aligned to the test index.
    """
    train_vals = np.asarray(train, dtype=float)
    test_index = test.index if isinstance(test, pd.Series) else pd.RangeIndex(len(test))
    n = len(test_index)
    if n == 0 or len(train_vals) == 0:
        return pd.Series(dtype=float, index=test_index, name="baseline")
    m = season_length if 1 <= season_length <= len(train_vals) else 1
    last_cycle = train_vals[-m:]
    preds = np.array([last_cycle[h % m] for h in range(n)])
    return pd.Series(preds, index=test_index, name="baseline")


_SEASON_BY_FREQ = {"H": 24, "D": 7, "B": 5, "W": 52, "M": 12, "MS": 12, "Q": 4, "QS": 4}


def infer_frequency(ds: pd.Series) -> str:
    """Infer a pandas frequency alias from a sorted datetime series (fallback: median gap)."""
    freq = pd.infer_freq(ds) if len(ds) >= 3 else None
    if freq:
        return freq
    gap = ds.diff().median()
    if pd.isna(gap) or gap <= pd.Timedelta(hours=1):
        return "H"
    if gap <= pd.Timedelta(days=1):
        return "D"
    if gap <= pd.Timedelta(days=8):
        return "W"
    if gap <= pd.Timedelta(days=31):
        return "MS"
    return "QS"


def season_length_for(freq: str) -> int:
    """Map a frequency alias (e.g. 'W-SUN', 'D', 'MS') to a seasonal cycle length."""
    base = freq.split("-")[0].upper()
    for key in (base, base.rstrip("S"), base[:1]):
        if key in _SEASON_BY_FREQ:
            return _SEASON_BY_FREQ[key]
    return 1


def run_forecast(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    periods: int = 30,
    regressor_cols: list[str] | None = None,
    resample: str | None = None,
    agg: str = "sum",
) -> ForecastResult:
    """Run an enterprise Prophet forecast pipeline with evaluation and baseline comparison.

    Pipeline execution steps:
    1. Prepare and validate time series data (ds, y, and regressors).
    2. Chronological train/test split (last 20% held out for out-of-sample validation).
    3. Fit Prophet on training data (with optional exogenous regressors).
    4. Predict on test set + future horizon periods.
    5. Compute evaluation metrics (MAE, RMSE, MAPE) on test partition.
    6. Compute seasonal naive baseline and its error metrics.
    7. Compare model performance against baseline.
    8. Run Prophet rolling cross-validation if data volume permits.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing timestamps, target metric, and optional regressors.
    date_col : str
        Name of the column containing timestamp / date values.
    target_col : str
        Name of the continuous target variable to forecast.
    periods : int, default=30
        Number of future periods to forecast beyond the test partition, in the
        data's own frequency (days for daily data, weeks for weekly data, ...).
    regressor_cols : list[str] | None, default=None
        Optional list of external feature column names to include as Prophet regressors.
        Future regressor values are unknown, so they are held at their last observed value.
    resample : str | None, default=None
        Optional pandas frequency ('D', 'W', 'MS') to aggregate irregular data such as
        individual orders into a regular series before modelling.
    agg : str, default='sum'
        How to aggregate the target when resampling or when timestamps repeat
        ('sum' for sales volumes, 'mean' for prices or levels).

    Returns
    -------
    ForecastResult
        Dataclass containing full forecast DataFrame, train/test splits,
        metrics dict, cross-validation metrics, and fitted Prophet model.

    Raises
    ------
    KeyError
        If date_col, target_col, or any specified regressor_cols are missing.
    ValueError
        If clean dataset contains fewer than 5 valid historical timestamps.
    """
    logger.info("Initializing Prophet forecasting for target='%s' on date='%s'", target_col, date_col)

    # 1. Validation and data preparation
    required_cols = [date_col, target_col]
    reg_cols = regressor_cols if regressor_cols is not None else []
    all_needed = required_cols + reg_cols

    missing = [c for c in all_needed if c not in df.columns]
    if missing:
        msg = f"Missing required column(s) in input DataFrame: {missing}"
        logger.error(msg)
        raise KeyError(msg)

    # Clean and rename to Prophet conventions
    work_df = df[all_needed].copy()
    work_df = work_df.rename(columns={date_col: "ds", target_col: "y"})
    work_df["ds"] = pd.to_datetime(work_df["ds"], errors="coerce")
    work_df["y"] = pd.to_numeric(work_df["y"], errors="coerce")

    for reg in reg_cols:
        work_df[reg] = pd.to_numeric(work_df[reg], errors="coerce").fillna(0.0)

    work_df = work_df.dropna(subset=["ds", "y"])

    if agg not in ("sum", "mean"):
        raise ValueError("agg must be 'sum' or 'mean'")
    agg_rules = {"y": agg, **{reg: "mean" for reg in reg_cols}}

    if resample:
        work_df = (
            work_df.set_index("ds").resample(resample).agg(agg_rules).reset_index()
        )
        if agg == "mean":
            work_df = work_df.dropna(subset=["y"])
        for reg in reg_cols:
            work_df[reg] = work_df[reg].ffill().fillna(0.0)
    elif work_df["ds"].duplicated().any():
        logger.info("Duplicate timestamps found in dataset; aggregating with '%s'.", agg)
        work_df = work_df.groupby("ds", as_index=False).agg(agg_rules)

    work_df = work_df.sort_values("ds").reset_index(drop=True)
    freq = resample or infer_frequency(work_df["ds"])
    season_len = season_length_for(freq)

    if len(work_df) < 5:
        msg = f"Insufficient time series points for modeling. Minimum 5 required, found {len(work_df)}."
        logger.error(msg)
        raise ValueError(msg)

    # 2. Time-based train/test split (hold out last 20% of chronological observations)
    n_total = len(work_df)
    split_idx = int(n_total * 0.8)

    # Ensure train has at least 3 points and test has at least 1
    if split_idx >= n_total:
        split_idx = n_total - 1
    if split_idx < 3:
        split_idx = max(2, n_total - 1)

    train_df = work_df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = work_df.iloc[split_idx:].copy().reset_index(drop=True)

    logger.info(
        "Chronological split completed: Train=%d samples (%s to %s), Test=%d samples (%s to %s)",
        len(train_df),
        train_df["ds"].min().strftime("%Y-%m-%d"),
        train_df["ds"].max().strftime("%Y-%m-%d"),
        len(test_df),
        test_df["ds"].min().strftime("%Y-%m-%d"),
        test_df["ds"].max().strftime("%Y-%m-%d"),
    )

    # 3. Fit Prophet model on training data
    model = Prophet(interval_width=0.95)
    for reg in reg_cols:
        model.add_regressor(reg)

    model.fit(train_df)

    # 4. Predict on test set + future periods
    # Construct prediction dataframe covering train, test, and future horizons
    last_known_date = test_df["ds"].max()
    future_oos_dates = pd.date_range(start=last_known_date, periods=periods + 1, freq=freq)[1:]
    future_oos_df = pd.DataFrame({"ds": future_oos_dates})

    # Impute future exogenous regressors with last observed values
    for reg in reg_cols:
        fill_val = float(test_df[reg].iloc[-1]) if not test_df.empty else 0.0
        future_oos_df[reg] = fill_val

    eval_columns = ["ds"] + reg_cols
    full_future = pd.concat(
        [
            train_df[eval_columns],
            test_df[eval_columns],
            future_oos_df[eval_columns],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["ds"]).sort_values("ds").reset_index(drop=True)

    forecast_df = model.predict(full_future)

    # 5. Compute evaluation metrics on held-out test set
    test_eval = pd.merge(test_df[["ds", "y"]], forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="inner")
    model_metrics = compute_forecast_metrics(test_eval["y"], test_eval["yhat"])

    # 6. Compute naive seasonal baseline
    baseline_pred = compute_naive_baseline(train_df["y"], test_df["y"], season_length=season_len)
    baseline_metrics = compute_forecast_metrics(test_df["y"], baseline_pred)

    # 7. Compare model vs baseline
    mae_diff = baseline_metrics["mae"] - model_metrics["mae"]
    improvement_pct = round((mae_diff / baseline_metrics["mae"] * 100.0) if baseline_metrics["mae"] > 0 else 0.0, 2)

    metrics: dict[str, Any] = {
        "model": model_metrics,
        "baseline": baseline_metrics,
        "mae": model_metrics["mae"],
        "rmse": model_metrics["rmse"],
        "mape": model_metrics["mape"],
        "baseline_mae": baseline_metrics["mae"],
        "baseline_rmse": baseline_metrics["rmse"],
        "baseline_mape": baseline_metrics["mape"],
        "mae_improvement_pct": improvement_pct,
        "frequency": freq,
        "season_length": season_len,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }
    logger.info(
        "Forecast evaluation: Model MAE=%.4f (Baseline MAE=%.4f, Improvement=%.2f%%)",
        model_metrics["mae"],
        baseline_metrics["mae"],
        improvement_pct,
    )

    # 8. Prophet cross-validation if sufficient historical span
    cv_metrics: dict[str, float] | None = None
    date_span_days = (train_df["ds"].max() - train_df["ds"].min()).days

    # Cross validation requires substantial span (e.g. at least 60 days) and no future regressors conflict
    if date_span_days >= 60 and not reg_cols:
        try:
            horizon_days = max(7, date_span_days // 5)
            initial_days = max(14, date_span_days // 2)
            period_days = max(7, date_span_days // 4)

            cv_df = cross_validation(
                model,
                initial=f"{initial_days} days",
                period=f"{period_days} days",
                horizon=f"{horizon_days} days",
                parallel=None,
                disable_tqdm=True,
            )
            perf_df = performance_metrics(cv_df)
            cv_metrics = {
                "mae": round(float(perf_df["mae"].mean()), 4),
                "rmse": round(float(perf_df["rmse"].mean()), 4),
            }
            # Prophet omits MAPE when actual values include zeros
            if "mape" in perf_df.columns:
                cv_metrics["mape"] = round(float(perf_df["mape"].mean()) * 100.0, 4)
            logger.info("Cross-validation completed: %s", cv_metrics)
        except Exception as cv_err:
            logger.warning("Prophet cross-validation skipped or encountered error: %s", cv_err)
            cv_metrics = None
    else:
        logger.info("Skipping cross-validation: span is %d days (needs >=60) or regressors present.", date_span_days)

    return ForecastResult(
        forecast_df=forecast_df,
        train_df=train_df,
        test_df=test_df,
        metrics=metrics,
        cv_metrics=cv_metrics,
        model=model,
    )
