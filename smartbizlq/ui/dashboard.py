"""Executive KPI Dashboard with interactive filters, visual charts, drill-through, and report exports."""

import logging
import re
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..reporting.excel_export import export_kpi_workbook
from ..reporting.pptx_export import export_dashboard_to_pptx

logger = logging.getLogger(__name__)

from ..data.loaders import find_column, list_available_datasets, load_dataset

from ..analytics.kpis import compute_kpi_strip


def _safe_compute_kpis(
    df: pd.DataFrame,
    revenue_col: str | None,
    qty_col: str | None,
    id_col: str | None,
) -> dict[str, float]:
    """Safely compute top-level KPI metrics with fallback handling for missing columns."""
    if df.empty:
        return {"total_revenue": 0.0, "total_orders": 0.0, "total_units": 0.0, "avg_price": 0.0}

    # If all three columns exist, attempt to use the official analytics function
    if (
        compute_kpi_strip is not None
        and revenue_col
        and qty_col
        and id_col
        and all(c in df.columns for c in [revenue_col, qty_col, id_col])
    ):
        try:
            return compute_kpi_strip(df, revenue_col=revenue_col, qty_col=qty_col, id_col=id_col)
        except Exception as e:
            logger.warning("compute_kpi_strip call failed, falling back to manual computation: %s", e)

    # Manual robust aggregation
    total_rev = 0.0
    if revenue_col and revenue_col in df.columns:
        s = pd.to_numeric(df[revenue_col], errors="coerce")
        total_rev = float(s.sum(skipna=True))

    total_qty = 0.0
    if qty_col and qty_col in df.columns:
        q = pd.to_numeric(df[qty_col], errors="coerce")
        total_qty = float(q.sum(skipna=True)) if q.notnull().any() else 0.0

    avg_price = (total_rev / total_qty) if total_qty > 0.0 and not np.isnan(total_rev) else 0.0
    orders = float(len(df)) if id_col is None or id_col not in df.columns else float(df[id_col].dropna().nunique())

    return {
        "total_revenue": round(total_rev, 2),
        "total_orders": orders,
        "total_units": round(total_qty, 2),
        "avg_price": round(avg_price, 2),
    }


def safe_plotly(fig: go.Figure) -> None:
    """Render a Plotly figure safely within Streamlit containers."""
    try:
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Unable to render chart: {e}")


def render_dashboard_page() -> None:
    """Render the main KPI dashboard with interactive charts, multi-level filters, drill-through, and export buttons."""
    st.header("Executive Dashboard")

    ds_files = list_available_datasets()
    if not ds_files:
        st.info("No datasets found. Please upload a CSV in the 'Upload Data' tab first.")
        return

    sel = st.selectbox("Dataset (select)", ds_files, key="dashboard_dataset_select")
    # Filter widgets are keyed per dataset so switching datasets resets them cleanly
    ds_key = re.sub(r"\W+", "_", str(sel))
    if not sel:
        return

    df = load_dataset(sel)
    if df is None or df.empty:
        st.error("Failed to load dataset or dataset is empty.")
        return

    # Basic cleaning and session store
    cleaned = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    st.session_state["active_df"] = cleaned.copy()

    st.markdown("**Dataset sample (cleaned)**")
    st.dataframe(cleaned.head(10), use_container_width=True)

    # Column pattern detection
    all_col_names = cleaned.columns.tolist()
    date_col = find_column(all_col_names, ["date", "day", "ds", "timestamp", "time"])
    revenue_col = find_column(all_col_names, ["revenue", "sales", "amount", "total", "net"])
    qty_col = find_column(all_col_names, ["quantity", "qty", "units", "volume"])
    category_col = find_column(all_col_names, ["category", "product", "item", "sku"])
    region_col = find_column(all_col_names, ["country", "region", "state", "location"])
    id_col = find_column(all_col_names, ["id", "order", "invoice"])

    work = cleaned.copy()
    if date_col:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    if revenue_col:
        work[revenue_col] = pd.to_numeric(work[revenue_col], errors="coerce")
    if qty_col:
        work[qty_col] = pd.to_numeric(work[qty_col], errors="coerce")

    # Sidebar Dashboard Filters
    st.sidebar.subheader("Dashboard Filters (page)")

    dr = None
    if date_col:
        try:
            valid_dates = work[date_col].dropna()
            min_d = valid_dates.min().date() if not valid_dates.empty else datetime.now(timezone.utc).date()
            max_d = valid_dates.max().date() if not valid_dates.empty else datetime.now(timezone.utc).date()
        except Exception:
            min_d = datetime.now(timezone.utc).date()
            max_d = datetime.now(timezone.utc).date()
        dr = st.sidebar.date_input("Date range", [min_d, max_d], key=f"dash_date_range_{ds_key}")

    region_sel = ["All"]
    if region_col:
        regions = sorted(work[region_col].dropna().astype(str).unique().tolist())
        region_sel = st.sidebar.multiselect(
            "Region / Country",
            ["All"] + regions,
            default=["All"],
            key=f"dash_region_sel_{ds_key}",
        )

    cat_sel = ["All"]
    if category_col:
        cats = sorted(work[category_col].dropna().astype(str).unique().tolist())
        cat_sel = st.sidebar.multiselect(
            "Category",
            ["All"] + cats,
            default=["All"],
            key=f"dash_cat_sel_{ds_key}",
        )

    initial_rev = float(work[revenue_col].sum()) if (revenue_col in work.columns and work[revenue_col].notnull().any()) else 0.0
    target_rev = st.sidebar.number_input(
        "Revenue target (gauge)",
        min_value=0.0,
        value=initial_rev,
        key=f"dash_target_rev_{ds_key}",
    )
    st.sidebar.markdown("---")

    # Apply interactive filters
    filtered = work.copy()
    if date_col and dr:
        try:
            if len(dr) == 2:
                start = pd.to_datetime(dr[0])
                end = pd.to_datetime(dr[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                filtered[date_col] = pd.to_datetime(filtered[date_col], errors="coerce")
                mask = (filtered[date_col] >= start) & (filtered[date_col] <= end)
                filtered = filtered.loc[mask.fillna(False)]
        except Exception as filter_err:
            logger.warning("Date filtering encountered issue: %s", filter_err)

    if region_col and "All" not in region_sel:
        chosen_regions = [r for r in region_sel if r != "All"]
        filtered = filtered[filtered[region_col].astype(str).isin(chosen_regions)]

    if category_col and "All" not in cat_sel:
        chosen_cats = [c for c in cat_sel if c != "All"]
        filtered = filtered[filtered[category_col].astype(str).isin(chosen_cats)]

    # Compute and render KPI Strip
    kpis_data = _safe_compute_kpis(filtered, revenue_col=revenue_col, qty_col=qty_col, id_col=id_col)
    total_rev = kpis_data["total_revenue"]
    orders = int(kpis_data["total_orders"])
    total_qty = kpis_data["total_units"]
    avg_price = kpis_data["avg_price"]

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Revenue", f"${total_rev:,.0f}" if total_rev else "$0")
    kpi_cols[1].metric("Total Orders/Rows", f"{orders:,}")
    kpi_cols[2].metric("Total Units", f"{total_qty:,.0f}" if total_qty else "0")
    kpi_cols[3].metric("Avg Price", f"${avg_price:,.2f}" if avg_price > 0 else "N/A")

    st.markdown("---")

    # Visuals Row 1: Revenue Trend | Top Categories | Donut + Gauge
    row1_col1, row1_col2, row1_col3 = st.columns([3, 2, 2])
    chart_assets: list[tuple[str, bytes]] = []

    with row1_col1:
        st.subheader("Revenue Trend")
        if date_col and revenue_col and filtered[date_col].notnull().any() and filtered[revenue_col].notnull().any():
            freq = "D" if filtered[date_col].nunique() > 60 else "M"
            ts = filtered.groupby(pd.Grouper(key=date_col, freq=freq))[revenue_col].sum().reset_index()
            fig_ts = px.line(ts, x=date_col, y=revenue_col, title="Revenue Over Time", markers=True)
            safe_plotly(fig_ts)
            try:
                chart_assets.append(("Revenue Trend", fig_ts.to_image(format="png", scale=2)))
            except Exception:
                pass
        else:
            st.info("Provide valid Date and Revenue columns to display the trend.")

        st.markdown("**Recent top rows**")
        sort_col = revenue_col if (revenue_col in filtered.columns and filtered[revenue_col].notnull().any()) else filtered.columns[0]
        st.dataframe(filtered.sort_values(by=sort_col, ascending=False).head(8), use_container_width=True)

    with row1_col2:
        st.subheader("Top Categories")
        topc = None
        if category_col and revenue_col and category_col in filtered.columns:
            topc = (
                filtered.groupby(category_col)[revenue_col]
                .sum()
                .reset_index()
                .sort_values(revenue_col, ascending=False)
                .head(12)
            )
            fig_bar = px.bar(topc, x=category_col, y=revenue_col, title="Top Categories by Revenue")
            safe_plotly(fig_bar)
            try:
                chart_assets.append(("Top Categories", fig_bar.to_image(format="png", scale=2)))
            except Exception:
                pass
        else:
            st.info("Category column not detected or revenue values missing.")

        if topc is not None and not topc.empty:
            sel_cat = st.selectbox(
                "Drill: select category",
                options=["(all)"] + topc[category_col].astype(str).tolist(),
                key="dash_sel_cat",
            )
            if sel_cat and sel_cat != "(all)":
                drilled = filtered[filtered[category_col].astype(str) == sel_cat]
                st.write(f"Drilled rows: **{len(drilled)}**")
                st.dataframe(drilled.head(10), use_container_width=True)

    with row1_col3:
        st.subheader("Share & Target")
        if revenue_col and category_col and revenue_col in filtered.columns:
            share = (
                filtered.groupby(category_col)[revenue_col]
                .sum()
                .reset_index()
                .sort_values(revenue_col, ascending=False)
                .head(8)
            )
            fig_donut = px.pie(
                share,
                names=category_col,
                values=revenue_col,
                hole=0.45,
                title="Revenue Share (Top Categories)",
            )
            safe_plotly(fig_donut)
            try:
                chart_assets.append(("Revenue Share", fig_donut.to_image(format="png", scale=2)))
            except Exception:
                pass
        else:
            st.info("Category and Revenue columns required.")

        st.markdown("**Revenue vs Target**")
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=total_rev,
                delta={"reference": float(target_rev)},
                gauge={"axis": {"range": [None, float(max(target_rev, total_rev, 1))]}},
            )
        )
        safe_plotly(fig_gauge)
        try:
            chart_assets.append(("Revenue vs Target", fig_gauge.to_image(format="png", scale=2)))
        except Exception:
            pass

    # Visuals Row 2: Geo Map | Mini KPIs | Details Drill-through
    btm_col1, btm_col2, btm_col3 = st.columns([3, 2, 2])

    with btm_col1:
        st.subheader("Geo / Region Map")
        if region_col and revenue_col and region_col in filtered.columns:
            agg_geo = filtered.groupby(region_col)[revenue_col].sum().reset_index()
            try:
                fig_map = px.choropleth(
                    agg_geo,
                    locations=region_col,
                    locationmode="country names",
                    color=revenue_col,
                    title="Revenue by Country",
                )
                safe_plotly(fig_map)
                try:
                    chart_assets.append(("Revenue by Geo", fig_map.to_image(format="png", scale=2)))
                except Exception:
                    pass
            except Exception:
                st.warning("Map view unavailable; displaying regional bar chart.")
                bar_fig = px.bar(
                    agg_geo.sort_values(revenue_col, ascending=False).head(10),
                    x=region_col,
                    y=revenue_col,
                )
                safe_plotly(bar_fig)
                try:
                    chart_assets.append(("Top Regions", bar_fig.to_image(format="png", scale=2)))
                except Exception:
                    pass
        else:
            st.info("Provide region and revenue columns for geographic visualization.")

    with btm_col2:
        st.subheader("Mini KPIs")
        if "ts" in locals() and not ts.empty:
            recent = ts.tail(12)
            safe_plotly(px.area(recent, x=date_col, y=revenue_col, title="Recent 12-Period Velocity"))
            try:
                chart_assets.append(("Mini Trend", px.area(recent, x=date_col, y=revenue_col).to_image(format="png", scale=2)))
            except Exception:
                pass
        else:
            st.info("Date and Revenue columns required for velocity sparkline.")

        if revenue_col in filtered.columns:
            hist_fig = px.histogram(filtered, x=revenue_col, nbins=30, title="Revenue Distribution")
            safe_plotly(hist_fig)
            try:
                chart_assets.append(("Revenue Distribution", hist_fig.to_image(format="png", scale=2)))
            except Exception:
                pass

    with btm_col3:
        st.subheader("Details / Drill-through")
        search_query = st.text_input("Search (text in any column)", key="dash_search")
        if search_query:
            mask = pd.Series(False, index=filtered.index)
            for c in filtered.columns:
                mask = mask | filtered[c].astype(str).str.contains(search_query, case=False, na=False)
            table_view = filtered[mask].copy()
        else:
            table_view = filtered.copy()

        st.markdown(f"**Rows available after filters:** {len(table_view)}")
        show_all = st.checkbox("Show all rows (disable pagination)", value=False, key="dash_show_all")
        default_page_size = 50
        page_size = st.number_input(
            "Rows per page",
            min_value=10,
            max_value=2000,
            value=default_page_size,
            step=10,
            key="dash_page_size",
        )
        total_rows = len(table_view)

        if total_rows == 0:
            st.info("No rows to display.")
        else:
            if show_all:
                start_idx = 0
                end_idx = total_rows
                current_page = 1
                total_pages = 1
            else:
                total_pages = max(1, math.ceil(total_rows / page_size))
                current_page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    key="dash_page_num",
                )
                start_idx = (current_page - 1) * page_size
                end_idx = min(start_idx + page_size, total_rows)

            view = table_view.iloc[start_idx:end_idx]
            st.markdown(f"Showing rows **{start_idx}** — **{end_idx - 1}** (page {current_page} of {total_pages})")
            st.dataframe(view, use_container_width=True)

            st.markdown("**Select rows**")
            sel_mode = st.radio(
                "Selection mode",
                ["None", "Select by index (from current view)", "Select by column value (from current view)"],
                index=0,
                key="dash_sel_mode",
            )
            selected_df = pd.DataFrame()

            if sel_mode == "Select by index (from current view)":
                index_options = [str(i) for i in view.index.tolist()]
                preview_cols = view.columns.tolist()[:2]
                if preview_cols:
                    labeled_options = [
                        f"{i} | " + " | ".join([str(view.loc[int(i), c])[:40] for c in preview_cols])
                        for i in index_options
                    ]
                    pick = st.multiselect("Pick row indices (multi-select)", labeled_options, key="dash_pick_indices")
                    picked_indices = []
                    for p in pick:
                        try:
                            idx = int(p.split("|")[0].strip())
                            picked_indices.append(idx)
                        except Exception:
                            pass
                else:
                    picked = st.multiselect("Pick row indices (multi-select)", index_options, key="dash_pick_indices_simple")
                    picked_indices = [int(x) for x in picked]

                if st.button("Select all rows in current view", key="dash_select_all_view"):
                    picked_indices = [int(x) for x in index_options]

                if picked_indices:
                    selected_df = table_view.loc[picked_indices]

            elif sel_mode == "Select by column value (from current view)":
                col_choice = st.selectbox("Column to select by", options=view.columns.tolist(), key="dash_sel_col")
                if col_choice:
                    unique_vals = view[col_choice].dropna().astype(str).unique().tolist()
                    selected_vals = st.multiselect(
                        f"Choose {col_choice} values (from current page)",
                        options=unique_vals,
                        key="dash_sel_vals",
                    )
                    if st.button("Select all rows with chosen values (current page)", key="dash_select_vals_all"):
                        selected_vals = unique_vals
                    if selected_vals:
                        selected_df = table_view[table_view[col_choice].astype(str).isin(selected_vals)].copy()

            if not selected_df.empty:
                st.markdown(f"**Selected rows:** {len(selected_df)}")
                st.dataframe(selected_df, use_container_width=True)
                csv_bytes = selected_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download selected rows (CSV)",
                    csv_bytes,
                    file_name=f"selected_rows_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                    key="dash_download_selected",
                )
                idx_list = selected_df.index.tolist()
                st.text_area(
                    "Selected row indices (copy/paste)",
                    value=", ".join(map(str, idx_list)),
                    height=80,
                    key="dash_selected_indices_text",
                )

    st.markdown("---")
    st.write(
        "Tip: Use the **sidebar boxes** (Dataset, Date range, Region, Category, Revenue target) "
        "to cross-filter all visualizations across the dashboard."
    )

    # -------------------------------------------------------------
    # REPORT EXPORTS: PowerPoint (.pptx) & Excel (.xlsx)
    # -------------------------------------------------------------
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        if st.button("Export Report (PPTX)", key="export_pptx"):
            try:
                kpis = {
                    "Total Revenue": f"${total_rev:,.2f}",
                    "Total Orders/Rows": f"{orders:,}",
                    "Total Units": f"{total_qty:,.0f}",
                    "Avg Price": f"${avg_price:.2f}" if avg_price > 0 else "N/A",
                }
                sample_df = filtered.head(100)
                out_dir = "reports"
                os.makedirs(out_dir, exist_ok=True)
                fname = f"SmartBizLQ_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pptx"
                out_path = os.path.join(out_dir, fname)

                # Fallback charts if none were captured
                if not chart_assets:
                    try:
                        if date_col and revenue_col and filtered[date_col].notnull().any() and filtered[revenue_col].notnull().any():
                            ts_exp = filtered.groupby(pd.Grouper(key=date_col, freq="M"))[revenue_col].sum().reset_index()
                            fig_ts_exp = px.line(ts_exp, x=date_col, y=revenue_col, title="Revenue Over Time")
                            chart_assets.append(("Revenue Trend", fig_ts_exp.to_image(format="png", scale=2)))
                    except Exception:
                        pass

                export_dashboard_to_pptx(kpis, chart_assets, sample_df, out_path)
                with open(out_path, "rb") as f:
                    data = f.read()
                st.success(f"PowerPoint report exported: {fname}")
                st.download_button(
                    "Download PPTX",
                    data,
                    file_name=fname,
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    key="download_pptx_btn",
                )
            except Exception as e:
                st.error(f"PowerPoint export failed: {e}")
                logger.exception("PPTX export error")

    with exp_col2:
        if st.button("Export KPI & Data (Excel)", key="export_excel_kpis"):
            try:
                kpis = {
                    "Total Revenue": f"${total_rev:,.2f}",
                    "Total Orders/Rows": f"{orders:,}",
                    "Total Units": f"{total_qty:,.0f}",
                    "Avg Price": f"${avg_price:.2f}" if avg_price > 0 else "N/A",
                }
                out_dir = "reports"
                os.makedirs(out_dir, exist_ok=True)
                fname_xlsx = f"SmartBizLQ_KPIs_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.xlsx"
                out_path_xlsx = os.path.join(out_dir, fname_xlsx)

                export_kpi_workbook(kpis, filtered, out_path_xlsx)
                with open(out_path_xlsx, "rb") as f:
                    data_xlsx = f.read()
                st.success(f"Excel KPI workbook exported: {fname_xlsx}")
                st.download_button(
                    "Download Excel Workbook",
                    data_xlsx,
                    file_name=fname_xlsx,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_xlsx_kpi_btn",
                )
            except Exception as e:
                st.error(f"Excel export failed: {e}")
                logger.exception("Excel KPI export error")
