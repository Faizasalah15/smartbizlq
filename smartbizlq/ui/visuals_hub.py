"""Visuals Hub: Power BI-like interactive charting and exploratory data analysis engine."""

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

from ..data.loaders import list_available_datasets, load_dataset


def safe_plotly(fig: go.Figure) -> None:
    """Render a Plotly figure safely within Streamlit containers."""
    try:
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Unable to render chart: {e}")


def _render_visuals(df: pd.DataFrame) -> None:
    """Render Power BI-like visual controls and dynamic charts.

    Supports 24 visual types:
    - Bar: Stacked Bar Chart, Clustered Bar Chart, 100% Stacked Bar Chart
    - Time-series: Line Chart, Area Chart, Stacked Area Chart, Combo Chart, Ribbon Chart
    - Share & Distribution: Pie Chart, Donut Chart, Treemap, Histogram
    - Correlation: Scatter Chart, Heatmap (correlation)
    - Tabular: Table, Matrix (pivot)
    - KPI Cards: Card, Multi-row Card, Gauge, KPI
    - Flows: Waterfall Chart, Funnel Chart
    - Geospatial: Map, Choropleth

    Args:
        df: The active pandas DataFrame to visualize.
    """
    if df is None or df.empty:
        st.warning("Upload or select a dataset first.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    date_cols: list[str] = [
        c for c in df.columns if "date" in c.lower() or np.issubdtype(df[c].dtype, np.datetime64)
    ]
    if not date_cols:
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notnull().sum() > 0.3 * len(df):
                    date_cols.append(c)
            except Exception:
                pass

    st.sidebar.subheader("Visual controls")
    visual = st.sidebar.selectbox(
        "Visual",
        [
            "Stacked Bar Chart",
            "Clustered Bar Chart",
            "100% Stacked Bar Chart",
            "Line Chart",
            "Area Chart",
            "Stacked Area Chart",
            "Combo Chart",
            "Pie Chart",
            "Donut Chart",
            "Scatter Chart",
            "Treemap",
            "Table",
            "Matrix",
            "Card",
            "Multi-row Card",
            "Gauge",
            "KPI",
            "Waterfall Chart",
            "Funnel Chart",
            "Ribbon Chart",
            "Map",
            "Choropleth",
            "Heatmap",
            "Histogram",
        ],
        index=0,
        key="vh_visual",
    )

    x = st.sidebar.selectbox("X", options=[None] + cat_cols + date_cols + numeric_cols, key="vh_x")
    y = st.sidebar.selectbox("Y (numeric)", options=[None] + numeric_cols, key="vh_y")
    group = st.sidebar.selectbox("Group / color", options=[None] + cat_cols, key="vh_group")
    size = st.sidebar.selectbox("Size", options=[None] + numeric_cols, key="vh_size")
    color = st.sidebar.selectbox("Color", options=[None] + cat_cols + numeric_cols, key="vh_color")
    top_n = st.sidebar.number_input("Top N", min_value=1, max_value=200, value=10, key="vh_topn")

    # 1-3. Bar Charts
    if visual in ["Stacked Bar Chart", "Clustered Bar Chart", "100% Stacked Bar Chart"]:
        st.subheader(visual)
        if not x or not y:
            st.info("Choose X (category) and Y (measure) in sidebar.")
        else:
            try:
                if group:
                    if "100%" in visual:
                        df_agg = df.groupby([x, group])[y].sum().reset_index()
                        df_sum = df_agg.groupby(x)[y].transform("sum")
                        df_agg["pct"] = df_agg[y] / df_sum
                        fig = px.bar(df_agg, x=x, y="pct", color=group, title=f"100% Stacked {y} by {x}")
                    else:
                        barmode = "stack" if "Stacked" in visual else "group"
                        fig = px.bar(df, x=x, y=y, color=group, barmode=barmode, title=f"{visual} of {y} by {x}")
                else:
                    fig = px.bar(df, x=x, y=y, title=f"{visual} of {y} by {x}")
                safe_plotly(fig)
            except Exception as e:
                st.warning(f"Failed to render bar chart: {e}")

    # 4-7. Line, Area, Stacked Area, Combo Charts
    elif visual in ["Line Chart", "Area Chart", "Stacked Area Chart", "Combo Chart"]:
        st.subheader(visual)
        if not x or not y:
            st.info("Select X and Y axes in sidebar.")
        else:
            try:
                if visual == "Line Chart":
                    fig = px.line(df, x=x, y=y, color=group, markers=True, title=f"Line Chart of {y} vs {x}")
                elif visual == "Area Chart":
                    fig = px.area(df, x=x, y=y, color=group, title=f"Area Chart of {y} vs {x}")
                elif visual == "Stacked Area Chart":
                    fig = px.area(df, x=x, y=y, color=group, line_group=group, title=f"Stacked Area of {y} vs {x}")
                else:
                    agg_ts = df.groupby(x)[y].sum().reset_index()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=agg_ts[x], y=agg_ts[y], name=f"Sum of {y}"))
                    fig.add_trace(go.Scatter(x=agg_ts[x], y=agg_ts[y], name=f"Trend {y}", yaxis="y2"))
                    fig.update_layout(
                        title=f"Combo Chart ({y} by {x})",
                        yaxis2=dict(overlaying="y", side="right", title="Trend Scale"),
                    )
                safe_plotly(fig)
            except Exception as e:
                st.warning(f"Failed to render time-series chart: {e}")

    # 8-9. Pie & Donut Charts
    elif visual in ["Pie Chart", "Donut Chart"]:
        st.subheader(visual)
        if not x or not y:
            st.info("Select category (X) and measure (Y) in sidebar.")
        else:
            fig = px.pie(
                df,
                names=x,
                values=y,
                hole=0.4 if visual == "Donut Chart" else 0.0,
                title=f"{visual} of {y} by {x}",
            )
            safe_plotly(fig)

    # 10. Scatter Chart
    elif visual == "Scatter Chart":
        st.subheader("Scatter Chart")
        if not x or not y:
            st.info("Select numeric X and Y columns.")
        else:
            fig = px.scatter(
                df,
                x=x,
                y=y,
                color=color,
                size=size,
                hover_data=df.columns,
                title=f"Scatter Analysis ({y} vs {x})",
            )
            safe_plotly(fig)

    # 11. Treemap
    elif visual == "Treemap":
        st.subheader("Treemap")
        if not x:
            st.info("Select a categorical X column.")
        else:
            path_cols = [x] + ([group] if group else [])
            fig = px.treemap(df, path=path_cols, values=y, title=f"Hierarchical Treemap by {x}")
            safe_plotly(fig)

    # 12. Table
    elif visual == "Table":
        st.subheader("Tabular View")
        st.dataframe(df, use_container_width=True)

    # 13. Matrix (Pivot Table)
    elif visual == "Matrix":
        st.subheader("Matrix (Pivot Table)")
        rows = st.sidebar.selectbox("Rows", options=[None] + cat_cols, key="vh_matrix_rows")
        cols = st.sidebar.selectbox("Cols", options=[None] + cat_cols, key="vh_matrix_cols")
        vals = st.sidebar.selectbox("Values", options=[None] + numeric_cols, key="vh_matrix_vals")
        if rows and cols and vals:
            pivot = pd.pivot_table(df, values=vals, index=rows, columns=cols, aggfunc="sum", fill_value=0)
            st.dataframe(pivot, use_container_width=True)
        else:
            st.info("Pick rows, columns, and numeric value to compute pivot matrix.")

    # 14-17. Cards, Multi-row Card, Gauge, KPI
    elif visual in ["Card", "Multi-row Card", "Gauge", "KPI"]:
        st.subheader(visual)
        if not y:
            st.info("Choose numeric measure Y in sidebar.")
        else:
            total = float(df[y].sum(skipna=True))
            if visual == "Card":
                st.metric(f"Total {y}", f"{total:,.0f}")
            elif visual == "Multi-row Card":
                c1, c2, c3 = st.columns(3)
                c1.metric("Sum", f"{total:,.0f}")
                c2.metric("Mean", f"{df[y].mean():.2f}")
                c3.metric("Total Records", f"{len(df):,}")
            elif visual == "Gauge":
                target = st.sidebar.number_input(
                    "Target",
                    value=total * 1.1 if total > 0 else 1000.0,
                    key="vh_gauge_target",
                )
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=total,
                        delta={"reference": float(target)},
                        gauge={"axis": {"range": [None, float(max(target, total, 1))]}},
                        title={"text": f"Gauge: {y}"},
                    )
                )
                safe_plotly(fig)
            else:
                st.metric(f"KPI {y}", f"{total:,.0f}")

    # 18-20. Waterfall, Funnel, Ribbon
    elif visual in ["Waterfall Chart", "Funnel Chart", "Ribbon Chart"]:
        st.subheader(visual)
        if not x or not y:
            st.info("Choose X and Y in sidebar.")
        else:
            agg_df = (
                df.groupby(x)[y]
                .sum()
                .reset_index()
                .sort_values(y, ascending=False)
                .head(top_n)
            )
            if visual == "Waterfall Chart":
                fig = go.Figure(
                    go.Waterfall(
                        x=agg_df[x].astype(str),
                        y=agg_df[y],
                        measure=["relative"] * len(agg_df),
                    )
                )
                fig.update_layout(title=f"Waterfall Decomposition: {y} by {x}")
                safe_plotly(fig)
            elif visual == "Funnel Chart":
                fig = go.Figure(go.Funnel(y=agg_df[x].astype(str), x=agg_df[y]))
                fig.update_layout(title=f"Funnel Conversion: {y} by {x}")
                safe_plotly(fig)
            else:  # Ribbon Chart
                if date_cols and y:
                    dc = date_cols[0]
                    df_ribbon = df.copy()
                    df_ribbon[dc] = pd.to_datetime(df_ribbon[dc], errors="coerce")
                    ts = df_ribbon.groupby(pd.Grouper(key=dc, freq="ME"))[y].sum().reset_index()
                    fig = px.area(ts, x=dc, y=y, title=f"Ribbon Over Time ({y})")
                    safe_plotly(fig)
                else:
                    st.info("Ribbon approximates over time; provide a valid date column.")

    # 21-22. Maps
    elif visual in ["Map", "Choropleth"]:
        st.subheader(visual)
        geo_cols = [c for c in df.columns if any(k in c.lower() for k in ["country", "region", "state", "location"])]
        if not geo_cols or not y:
            st.info("Need a geographic column and a numeric measure Y.")
        else:
            geo = geo_cols[0]
            agg_geo = df.groupby(geo)[y].sum().reset_index()
            try:
                fig = px.choropleth(
                    agg_geo,
                    locations=geo,
                    locationmode="country names",
                    color=y,
                    title=f"Choropleth: {y} by {geo}",
                )
                safe_plotly(fig)
            except Exception:
                st.warning("Choropleth rendering failed, displaying regional bar chart.")
                safe_plotly(px.bar(agg_geo.sort_values(y, ascending=False).head(top_n), x=geo, y=y))

    # 23. Heatmap
    elif visual == "Heatmap":
        st.subheader("Correlation Heatmap")
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Matrix")
            safe_plotly(fig)
        else:
            st.info("Need at least 2 numeric columns to compute correlations.")

    # 24. Histogram
    elif visual == "Histogram":
        st.subheader("Distribution Histogram")
        if not y:
            st.info("Choose a numeric column Y in sidebar.")
        else:
            bins = st.sidebar.slider("Bins", 5, 200, 30, key="vh_hist_bins")
            fig = px.histogram(df, x=y, nbins=bins, title=f"Distribution of {y}")
            safe_plotly(fig)

    st.markdown("---")
    st.write(
        "Tip: Visual controls are located in the sidebar (X/Y/Group/Size/Color/Agg/TopN). "
        "Filter and drill into datasets interactively."
    )


def render_visuals_hub_page() -> None:
    """Render the Visuals Hub page with dataset selection and Power BI-like visualizations."""
    st.header("Visuals Hub — Power BI-Style Analytics")

    ds_files = list_available_datasets()
    if not ds_files:
        st.info("Upload a CSV dataset in the 'Upload Data' tab first.")
        return

    sel = st.selectbox("Choose dataset", ds_files, key="vh_dataset_select")
    if not sel:
        return

    df = load_dataset(sel)
    _render_visuals(df)
