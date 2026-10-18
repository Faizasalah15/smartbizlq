"""Unit tests for SmartBizLQ UI modules and components."""

import json
import pandas as pd
import pytest

from smartbizlq.ui.admin import render_admin_page
from smartbizlq.ui.dashboard import _safe_compute_kpis, render_dashboard_page, safe_plotly as safe_plotly_dash
from smartbizlq.ui.home import render_home_page
from smartbizlq.ui.ml_studio import _looks_like_id, render_ml_studio_page
from smartbizlq.ui.model_history import _format_json_field, render_model_history_page
from smartbizlq.ui.scenarios import render_scenarios_page
from smartbizlq.ui.upload import render_upload_page
from smartbizlq.ui.visuals_hub import _render_visuals, render_visuals_hub_page


def test_ui_imports():
    """Verify all public page render functions are callable and properly typed."""
    assert callable(render_dashboard_page)
    assert callable(render_visuals_hub_page)
    assert callable(_render_visuals)
    assert callable(render_ml_studio_page)
    assert callable(render_scenarios_page)
    assert callable(render_model_history_page)
    assert callable(render_admin_page)
    assert callable(render_upload_page)
    assert callable(render_home_page)


def test_format_json_field():
    """Verify JSON parsing helper in model history."""
    assert _format_json_field(None) == {}
    assert _format_json_field("") == {}

    valid_json = json.dumps({"accuracy": 0.95, "model": "rf"})
    parsed = _format_json_field(valid_json)
    assert parsed["accuracy"] == 0.95
    assert parsed["model"] == "rf"

    invalid_json = "not-a-json-string"
    fallback = _format_json_field(invalid_json)
    assert "raw" in fallback
    assert fallback["raw"] == "not-a-json-string"


def test_safe_compute_kpis():
    """Verify robust KPI calculation for dashboard."""
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 3],
        "sales": [100.0, 200.0, 300.0, 50.0],
        "quantity": [2, 4, 6, 1],
    })

    kpis = _safe_compute_kpis(df, revenue_col="sales", qty_col="quantity", id_col="order_id")
    assert kpis["total_revenue"] == 650.0
    assert kpis["total_units"] == 13.0
    assert kpis["total_orders"] == 3.0  # distinct order_ids: 1, 2, 3
    assert kpis["avg_price"] == 50.0  # 650 / 13

    # Empty DataFrame
    empty_kpis = _safe_compute_kpis(pd.DataFrame(), "sales", "quantity", "order_id")
    assert empty_kpis["total_revenue"] == 0.0
    assert empty_kpis["total_orders"] == 0.0

    # Missing columns fallback
    partial_df = pd.DataFrame({"sales": [50.0]})
    partial_kpis = _safe_compute_kpis(partial_df, "sales", None, None)
    assert partial_kpis["total_revenue"] == 50.0
    assert partial_kpis["total_orders"] == 1.0


def test_looks_like_id():
    df = pd.DataFrame({"customerID": [f"C{i}" for i in range(20)], "tenure": range(20), "plan": ["a", "b"] * 10})
    assert _looks_like_id(df, "customerID")
    assert not _looks_like_id(df, "tenure")
    assert not _looks_like_id(df, "plan")


def test_safe_plotly_handles_none_without_exception():
    """Verify safe_plotly gracefully handles errors."""
    try:
        safe_plotly_dash(None)
    except Exception as e:
        pytest.fail(f"safe_plotly raised an unhandled exception: {e}")
