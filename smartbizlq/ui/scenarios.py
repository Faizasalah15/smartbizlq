"""Scenario & sensitivity page: Low/Base/High cases, tornado chart, price x volume grid, Excel export."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..analytics.scenarios import (
    DEFAULT_CASES,
    INPUT_LABELS,
    OUTPUT_LABELS,
    ScenarioInputs,
    run_scenarios,
    scenarios_to_frame,
    sensitivity_table,
    tornado_data,
)
from ..reporting.excel_export import export_scenario_workbook

logger = logging.getLogger(__name__)

PRICE_STEPS = (0.8, 0.9, 1.0, 1.1, 1.2)
VOLUME_STEPS = (0.8, 0.9, 1.0, 1.1, 1.2)


def _gbp(x: float) -> str:
    return f"-£{abs(x):,.0f}" if x < 0 else f"£{x:,.0f}"


def _read_inputs() -> ScenarioInputs:
    """Collect base-case assumptions from the page."""
    st.caption("Illustrative default assumptions, not real company data. Change any value to see the effect.")
    c1, c2, c3, c4 = st.columns(4)
    price = c1.number_input(INPUT_LABELS["price"], min_value=0.0, value=120.0, step=5.0, key="sc_price")
    volume = c2.number_input(INPUT_LABELS["volume"], min_value=0.0, value=50_000.0, step=1_000.0, key="sc_volume")
    unit_cost = c3.number_input(INPUT_LABELS["unit_cost"], min_value=0.0, value=70.0, step=1.0, key="sc_cost")
    growth_pct = c4.number_input("Volume growth rate (%)", min_value=-100.0, max_value=200.0, value=6.0, step=0.5, key="sc_growth")
    c5, c6, c7, _ = st.columns(4)
    energy = c5.number_input(INPUT_LABELS["energy_mwh"], min_value=0.0, value=800.0, step=50.0, key="sc_energy")
    wholesale = c6.number_input(INPUT_LABELS["wholesale_price_mwh"], min_value=0.0, value=95.0, step=5.0, key="sc_wholesale")
    fixed = c7.number_input(INPUT_LABELS["fixed_costs"], min_value=0.0, value=900_000.0, step=10_000.0, key="sc_fixed")
    return ScenarioInputs(
        price=price,
        volume=volume,
        unit_cost=unit_cost,
        growth_rate=growth_pct / 100.0,
        energy_mwh=energy,
        wholesale_price_mwh=wholesale,
        fixed_costs=fixed,
    )


def _tornado_chart(base: ScenarioInputs, swing: float) -> go.Figure:
    records = tornado_data(base, swing_pct=swing, output_name="operating_profit")
    df = pd.DataFrame(records).sort_values("swing")  # largest bar on top
    fig = go.Figure()
    fig.add_bar(y=df["label"], x=df["downside_change"], orientation="h", name=f"Input -{swing:.0%}", marker_color="#C0504D")
    fig.add_bar(y=df["label"], x=df["upside_change"], orientation="h", name=f"Input +{swing:.0%}", marker_color="#4F81BD")
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Change in operating profit (£)",
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def render_scenarios_page() -> None:
    """Render the scenario & sensitivity studio."""
    st.header("Scenario & Sensitivity Studio")
    st.markdown(
        "A transparent one-period profit model: **revenue = price × volume**; costs are variable cost, "
        "energy (MWh × wholesale £/MWh) and fixed costs. Growth scales volume and energy use."
    )
    base = _read_inputs()

    results = run_scenarios(base)
    _, base_out = results["Base"]

    st.subheader("Base case")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Revenue", _gbp(base_out.revenue))
    k2.metric("Total cost", _gbp(base_out.total_cost))
    k3.metric("Operating profit", _gbp(base_out.operating_profit))
    k4.metric(
        "Profit after growth",
        _gbp(base_out.operating_profit_after_growth),
        delta=_gbp(base_out.operating_profit_after_growth - base_out.operating_profit),
    )

    st.subheader("Low / Base / High cases")
    case_desc = []
    for name, mult in DEFAULT_CASES.items():
        if mult:
            parts = [f"{INPUT_LABELS[k].split(' (')[0].lower()} ×{v:.2f}" for k, v in mult.items()]
            case_desc.append(f"**{name}:** " + ", ".join(parts))
    st.markdown("  \n".join(case_desc))
    table = scenarios_to_frame(results)
    money_rows = [INPUT_LABELS["price"], INPUT_LABELS["unit_cost"], INPUT_LABELS["wholesale_price_mwh"]]
    pct_rows = [INPUT_LABELS["growth_rate"]]
    shown = table.copy().astype(object)
    for idx in shown.index:
        for col in shown.columns:
            v = table.loc[idx, col]
            if idx in pct_rows:
                shown.loc[idx, col] = f"{v:.1%}"
            elif idx == OUTPUT_LABELS["gross_margin_pct"]:
                shown.loc[idx, col] = f"{v:.1f}%"
            elif idx in money_rows:
                shown.loc[idx, col] = f"£{v:,.2f}"
            elif idx in (INPUT_LABELS["volume"], INPUT_LABELS["energy_mwh"]):
                shown.loc[idx, col] = f"{v:,.0f}"
            else:
                shown.loc[idx, col] = _gbp(v)
    st.dataframe(shown, use_container_width=True)

    col_t, col_s = st.columns(2)
    with col_t:
        st.subheader("Tornado: what moves profit most?")
        swing = st.select_slider("Flex each input by", options=[0.05, 0.10, 0.20], value=0.10, format_func=lambda v: f"±{v:.0%}", key="sc_swing")
        st.plotly_chart(_tornado_chart(base, swing), use_container_width=True)
    with col_s:
        st.subheader("Price × volume sensitivity")
        sens = sensitivity_table(
            base,
            "price", [round(base.price * m, 2) for m in PRICE_STEPS],
            "volume", [round(base.volume * m) for m in VOLUME_STEPS],
            output_name="operating_profit",
        )
        fig = px.imshow(
            sens.values,
            x=[f"{v:,.0f}" for v in sens.columns],
            y=[f"£{v:,.2f}" for v in sens.index],
            text_auto=".3s",
            # Red/blue only when some cells are loss-making; otherwise one colour from light to dark
            color_continuous_scale="RdBu" if sens.values.min() < 0 else "Blues",
            color_continuous_midpoint=0 if sens.values.min() < 0 else None,
            labels=dict(x="Sales volume (units)", y="Selling price", color="Operating profit (£)"),
            aspect="auto",
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Export to Excel")
    st.write(
        "Download the same model as an Excel workbook. Every calculated cell is a live formula, "
        "so you can change the yellow input cells in Excel and the whole model recalculates."
    )
    if st.button("Build Excel model", key="sc_export"):
        try:
            os.makedirs("reports", exist_ok=True)
            fname = f"SmartBizLQ_scenario_model_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.xlsx"
            path = export_scenario_workbook(base, os.path.join("reports", fname), price_steps=PRICE_STEPS, volume_steps=VOLUME_STEPS)
            with open(path, "rb") as fh:
                st.download_button(
                    "Download Excel model",
                    fh.read(),
                    file_name=fname,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="sc_download",
                )
        except Exception as exc:  # pragma: no cover - UI feedback path
            logger.exception("Scenario export failed")
            st.error(f"Excel export failed: {exc}")
