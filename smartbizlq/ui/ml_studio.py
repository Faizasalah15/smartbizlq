"""ML Studio page: Prophet forecasting, churn classification and K-Means segmentation.

Every number shown on this page comes from the analytics modules in
``smartbizlq.analytics``; nothing is estimated or filled in by the UI.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..analytics.churn import train_churn_model
from ..analytics.clustering import run_clustering
from ..analytics.forecasting import run_forecast
from ..data.cleaning import coerce_numeric_columns
from ..data.loaders import find_column, list_available_datasets, load_dataset
from ..db.models import ModelRun, User, UserBadge
from ..db.session import SessionLocal

logger = logging.getLogger(__name__)

RESAMPLE_OPTIONS = {"Use data as is": None, "Daily totals": "D", "Weekly totals": "W", "Monthly totals": "MS"}
ID_HINTS = ("id", "uuid", "key", "code")


# ---------------------------------------------------------------- persistence
def _save_model_artifact(model_obj: Any, name_prefix: str) -> str:
    """Save a fitted model to ``models/`` with joblib and return its path."""
    os.makedirs("models", exist_ok=True)
    path = os.path.join("models", f"{uuid.uuid4().hex[:8]}_{name_prefix}.joblib")
    joblib.dump(model_obj, path)
    return path


def _record_model_run(model_name: str, version: str, metrics: dict[str, Any], params: dict[str, Any],
                      file_path: str, dataset_name: str) -> None:
    """Log a model run (and award the user a badge) in the database. Failures are logged, not raised."""
    try:
        with SessionLocal() as db:
            db.add(ModelRun(
                model_name=model_name,
                version=version,
                metrics_json=json.dumps(metrics, default=float),
                params_json=json.dumps(params, default=str),
                file_path=file_path,
                dataset_name=dataset_name,
            ))
            user = st.session_state.get("user") or {}
            if user.get("id"):
                record = db.get(User, user["id"])
                if record is not None:
                    record.score = (record.score or 0) + 50
                db.add(UserBadge(user_id=user["id"], badge=f"Trained {model_name} on {dataset_name}"))
            db.commit()
    except Exception as exc:
        logger.warning("Could not save model run: %s", exc)


def _looks_like_id(df: pd.DataFrame, col: str) -> bool:
    name = col.lower()
    return any(name == h or name.endswith(h) or name.endswith("_" + h) for h in ID_HINTS) or (
        df[col].dtype == object and df[col].nunique() > 0.9 * len(df)
    )


# ---------------------------------------------------------------- forecasting
def _forecast_section(df: pd.DataFrame, dataset_name: str) -> None:
    st.subheader("Time-series forecasting (Prophet)")
    st.caption(
        "The last 20% of the history is held back as a test period. Prophet is scored on it and compared "
        "with a seasonal naive benchmark (repeat the last seasonal cycle). A model is only useful if it beats that benchmark."
    )
    date_guess = find_column(list(df.columns), ["date", "time", "ds", "week", "month"])
    cols = list(df.columns)
    date_col = st.selectbox("Date column", cols, index=cols.index(date_guess) if date_guess else 0, key="fc_date")
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric:
        st.warning("This dataset has no numeric column to forecast.")
        return
    target_col = st.selectbox("Value to forecast", numeric, key="fc_target")
    c1, c2, c3 = st.columns(3)
    resample_label = c1.selectbox("Aggregate to", list(RESAMPLE_OPTIONS), key="fc_resample",
                                  help="Use daily/weekly/monthly totals for transaction-level data such as orders.")
    agg = c2.selectbox("Combine values by", ["sum", "mean"], key="fc_agg",
                       help="'sum' for volumes such as sales or MWh; 'mean' for prices or levels.")
    periods = c3.number_input("Periods to forecast ahead", min_value=1, max_value=365, value=12, key="fc_periods")
    regressors = st.multiselect(
        "External drivers (optional regressors)", [c for c in numeric if c != target_col], key="fc_regs",
        help="Extra variables such as temperature or CPI. Future values are held at their last observed value.",
    )

    if not st.button("Train and evaluate forecast", key="fc_run"):
        return
    with st.spinner("Fitting Prophet and scoring it on the held-out period..."):
        try:
            res = run_forecast(df, date_col, target_col, periods=int(periods),
                               regressor_cols=regressors or None,
                               resample=RESAMPLE_OPTIONS[resample_label], agg=agg)
        except Exception as exc:
            logger.exception("Forecast failed")
            st.error(f"Forecast failed: {exc}")
            return

    m = res.metrics
    fc = res.forecast_df
    test_start = res.test_df["ds"].min()
    fig = go.Figure()
    fig.add_scatter(x=fc["ds"], y=fc["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=fc["ds"], y=fc["yhat_lower"], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(79,129,189,0.18)", name="95% prediction interval", hoverinfo="skip")
    fig.add_scatter(x=fc["ds"], y=fc["yhat"], mode="lines", line=dict(color="#4F81BD", width=2), name="Prophet forecast")
    fig.add_scatter(x=res.train_df["ds"], y=res.train_df["y"], mode="markers", marker=dict(size=4, color="#555"), name="Actual (training)")
    fig.add_scatter(x=res.test_df["ds"], y=res.test_df["y"], mode="markers", marker=dict(size=5, color="#C0504D"), name="Actual (test)")
    fig.add_vline(x=test_start, line_dash="dot", line_color="#888")
    fig.update_layout(height=430, hovermode="x unified", yaxis_title=target_col, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Test-period accuracy** ({m['n_train']} training points, {m['n_test']} test points, frequency `{m['frequency']}`, season length {m['season_length']})")
    table = pd.DataFrame(
        {"Prophet": [m["mae"], m["rmse"], m["mape"]], "Seasonal naive benchmark": [m["baseline_mae"], m["baseline_rmse"], m["baseline_mape"]]},
        index=["MAE", "RMSE", "MAPE (%)"],
    )
    st.dataframe(table.style.format("{:,.2f}"), use_container_width=True)
    if m["mape"] > 100:
        st.caption("MAPE is very large here because some periods have values close to zero; MAE and RMSE are more reliable for this series.")
    if m["mae_improvement_pct"] > 0:
        st.success(f"Prophet's MAE is {m['mae_improvement_pct']:.1f}% lower than the benchmark on the test period.")
    else:
        st.warning(f"Prophet did not beat the benchmark (MAE {abs(m['mae_improvement_pct']):.1f}% worse). "
                   "Try a different aggregation, more history, or keep the simple benchmark.")
    if res.cv_metrics:
        cv_text = f"MAE {res.cv_metrics['mae']:,.2f}, RMSE {res.cv_metrics['rmse']:,.2f}"
        if "mape" in res.cv_metrics:
            cv_text += f", MAPE {res.cv_metrics['mape']:.2f}%"
        st.caption(f"Rolling cross-validation on the training period: {cv_text}")

    future = fc[fc["ds"] > res.test_df["ds"].max()][["ds", "yhat", "yhat_lower", "yhat_upper"]]
    st.download_button("Download forecast (CSV)", future.to_csv(index=False).encode(), "forecast.csv", "text/csv", key="fc_dl")

    path = _save_model_artifact(res.model, "prophet")
    _record_model_run("prophet", "v2", {k: v for k, v in m.items() if not isinstance(v, dict)},
                      {"date_col": date_col, "target_col": target_col, "periods": int(periods),
                       "regressors": regressors, "resample": RESAMPLE_OPTIONS[resample_label], "agg": agg},
                      path, dataset_name)


# ---------------------------------------------------------------- churn
def _churn_section(df: pd.DataFrame, dataset_name: str) -> None:
    st.subheader("Churn classification (Random Forest)")
    st.caption(
        "Stratified 80/20 train/test split, 5-fold cross-validation on the training set, class weights to handle "
        "imbalance, and permutation importance on the test set."
    )
    cols = list(df.columns)
    churn_guess = find_column(cols, ["churn", "attrition", "exited", "left", "response"])
    target = st.selectbox("Target column (1/0, Yes/No or True/False)", cols,
                          index=cols.index(churn_guess) if churn_guess else 0, key="ch_target")
    default_feats = [c for c in cols if c != target and not _looks_like_id(df, c)]
    feats = st.multiselect("Features", [c for c in cols if c != target], default=default_feats, key="ch_feats",
                           help="ID columns are left out by default because they carry no real signal.")
    trees = st.slider("Number of trees", 50, 500, 200, step=50, key="ch_trees")
    if not st.button("Train churn model", key="ch_run"):
        return
    if not feats:
        st.error("Select at least one feature.")
        return
    with st.spinner("Training and cross-validating..."):
        try:
            res = train_churn_model(df, target, feats, n_estimators=trees)
        except Exception as exc:
            logger.exception("Churn training failed")
            st.error(f"Training failed: {exc}")
            return

    mt = res.metrics
    k = st.columns(5)
    k[0].metric("ROC-AUC", f"{mt.get('roc_auc', float('nan')):.3f}")
    k[1].metric("Accuracy", f"{mt['accuracy']:.1%}")
    k[2].metric("Precision", f"{mt['precision']:.1%}")
    k[3].metric("Recall", f"{mt['recall']:.1%}")
    k[4].metric("F1", f"{mt['f1']:.3f}")
    if res.cv_scores.get("folds"):
        st.caption(f"Cross-validated ROC-AUC on the training set: {res.cv_scores['mean']:.3f} ± {res.cv_scores['std']:.3f} "
                   f"({len(res.cv_scores['folds'])} folds)")
    churn_rate = float(res.y_test.mean())
    st.caption(f"Churn rate in the test set: {churn_rate:.1%}. Always compare accuracy with simply predicting 'no churn' "
               f"({1 - churn_rate:.1%}); ROC-AUC, precision and recall are the more informative measures.")

    c1, c2 = st.columns(2)
    with c1:
        cm = np.asarray(res.confusion_matrix)
        labels = ["Stayed (0)", "Churned (1)"][: cm.shape[0]]
        fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual", color="Customers"), title="Confusion matrix (test set)")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        imp = pd.DataFrame(res.feature_importance, columns=["feature", "importance"]).head(12).sort_values("importance")
        fig = px.bar(imp, x="importance", y="feature", orientation="h", title="Permutation importance (drop in ROC-AUC)")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    path = _save_model_artifact(res.model, "churn")
    _record_model_run("churn_random_forest", "v2", {**mt, "cv_roc_auc_mean": res.cv_scores.get("mean")},
                      {"target": target, "features": feats, "n_estimators": trees}, path, dataset_name)


# ---------------------------------------------------------------- clustering
def _cluster_section(df: pd.DataFrame, dataset_name: str) -> None:
    st.subheader("Segmentation (K-Means)")
    st.caption("Features are standardised before clustering. Use the elbow and silhouette charts to justify k.")
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if not _looks_like_id(df, c)]
    if not numeric:
        st.warning("This dataset has no numeric columns to cluster on.")
        return
    feats = st.multiselect("Features", numeric, default=numeric[: min(4, len(numeric))], key="km_feats")
    k = st.slider("Number of clusters (k)", 2, 10, 3, key="km_k")
    if not st.button("Run clustering", key="km_run"):
        return
    if not feats:
        st.error("Select at least one feature.")
        return
    with st.spinner("Clustering..."):
        try:
            res = run_clustering(df, feats, n_clusters=k)
        except Exception as exc:
            logger.exception("Clustering failed")
            st.error(f"Clustering failed: {exc}")
            return

    st.metric(f"Silhouette score at k={k}", f"{res.silhouette:.3f}", help="From -1 to 1; higher means better-separated clusters.")
    c1, c2 = st.columns(2)
    if res.elbow_data:
        elbow = pd.DataFrame(res.elbow_data, columns=["k", "inertia"])
        fig = px.line(elbow, x="k", y="inertia", markers=True, title="Elbow curve (inertia)")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        c1.plotly_chart(fig, use_container_width=True)
    if res.silhouette_by_k:
        sil = pd.DataFrame(list(res.silhouette_by_k.items()), columns=["k", "silhouette"])
        fig = px.bar(sil, x="k", y="silhouette", title="Silhouette score by k")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        c2.plotly_chart(fig, use_container_width=True)
    st.markdown("**Cluster profiles** (average of each feature, and % difference from the overall average)")
    st.dataframe(res.profiles, use_container_width=True)
    for text in res.descriptions:
        st.markdown(f"- {text}")
    out = df.copy()
    out["cluster"] = res.labels
    st.download_button("Download data with cluster labels (CSV)", out.to_csv(index=False).encode(), "clusters.csv", "text/csv", key="km_dl")

    path = _save_model_artifact(res.model, "kmeans")
    _record_model_run("kmeans", "v2", {"silhouette": res.silhouette, "k": k}, {"features": feats, "n_clusters": k}, path, dataset_name)


# ---------------------------------------------------------------- page
def render_ml_studio_page() -> None:
    """Render ML Studio."""
    st.header("ML Studio")
    options = list_available_datasets()
    if not options:
        st.info("Upload a dataset first, or add CSV files to data/sample.")
        return
    sel = st.selectbox("Dataset", options, key="ml_dataset")
    df = load_dataset(sel)
    if df is None or df.empty:
        st.error("That dataset is empty or could not be read.")
        return
    df = coerce_numeric_columns(df)
    with st.expander(f"Preview ({len(df):,} rows × {df.shape[1]} columns)"):
        st.dataframe(df.head(20), use_container_width=True)

    mode = st.radio("Task", ["Forecasting", "Churn prediction", "Segmentation"], horizontal=True, key="ml_mode")
    if mode == "Forecasting":
        _forecast_section(df, sel)
    elif mode == "Churn prediction":
        _churn_section(df, sel)
    else:
        _cluster_section(df, sel)
