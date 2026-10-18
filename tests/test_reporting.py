"""Unit tests for SmartBizLQ reporting and export engines (PowerPoint and Excel)."""

import os
import tempfile
import pandas as pd
import pytest
from openpyxl import load_workbook
from pptx import Presentation

from smartbizlq.reporting.excel_export import (
    export_kpi_workbook,
    export_scenario_workbook,
)
from smartbizlq.reporting.pptx_export import export_dashboard_to_pptx


# -------------------------------------------------------------
# PPTX Export Tests
# -------------------------------------------------------------
def test_export_dashboard_to_pptx_success():
    """Verify PowerPoint export creates a valid multi-slide presentation."""
    kpis = {
        "Total Revenue": "$1,250,000",
        "Total Orders": "4,500",
        "Total Units": "85,000",
        "Avg Price": "$14.71",
    }

    # Generate a tiny 1x1 valid PNG in bytes
    # Minimal 1x1 transparent PNG byte signature
    minimal_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00"
        b"\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    charts = [
        ("Revenue Velocity Trend", minimal_png),
        ("Top Categories Share", minimal_png),
    ]

    sample_df = pd.DataFrame({
        "order_id": [101, 102, 103],
        "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "revenue": [250.0, 450.0, 120.0],
        "category": ["Electronics", "Furniture", "Supplies"],
    })

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "test_report.pptx")
        res_path = export_dashboard_to_pptx(kpis, charts, sample_df, out_path)

        assert res_path == out_path
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0

        # Validate PPTX structure using python-pptx
        prs = Presentation(out_path)
        # 1 title + 1 KPI + 2 charts + 1 sample data = 5 slides
        assert len(prs.slides) == 5

        # Check title slide
        title_slide = prs.slides[0]
        assert "SmartBizLQ" in title_slide.shapes.title.text

        # Check KPI slide
        kpi_slide = prs.slides[1]
        assert "Key Performance Indicators" in kpi_slide.shapes.title.text


def test_export_dashboard_to_pptx_empty_charts():
    """Verify PPTX exporter functions even when no chart images are provided."""
    kpis = {"Metric A": "100"}
    sample_df = pd.DataFrame({"A": [1, 2]})

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "test_empty_charts.pptx")
        export_dashboard_to_pptx(kpis, [], sample_df, out_path)

        assert os.path.exists(out_path)
        prs = Presentation(out_path)
        # 1 title + 1 KPI + 1 sample data = 3 slides
        assert len(prs.slides) == 3


def test_export_dashboard_to_pptx_empty_path_error():
    """Verify ValueError is raised if out_path is empty."""
    with pytest.raises(ValueError, match="out_path"):
        export_dashboard_to_pptx({}, [], pd.DataFrame(), "")


# -------------------------------------------------------------
# Excel Scenario Export Tests (LIVE FORMULAS)
# -------------------------------------------------------------
def _base():
    from smartbizlq.analytics.scenarios import ScenarioInputs
    return ScenarioInputs(price=120.0, volume=50_000.0, unit_cost=70.0, growth_rate=0.06,
                          energy_mwh=800.0, wholesale_price_mwh=95.0, fixed_costs=900_000.0)


def test_export_scenario_workbook_live_formulas(tmp_path):
    """Every calculated cell must be a formula that references the Inputs sheet."""
    out_path = str(tmp_path / "scenario_model.xlsx")
    assert export_scenario_workbook(_base(), out_path) == out_path

    wb = load_workbook(out_path, data_only=False)
    assert wb.sheetnames == ["Inputs", "Calculations", "Outputs", "Sensitivity"]

    ws_in = wb["Inputs"]
    assert ws_in["B2"].value == 120.0 and ws_in["B3"].value == 50_000.0 and ws_in["B8"].value == 900_000.0

    ws_c = wb["Calculations"]
    assert ws_c["C2"].value == "=Inputs!B2"
    assert ws_c["B3"].value == "=Inputs!B3*Inputs!C3"
    assert ws_c["C10"].value == "=C2*C3"
    assert ws_c["C12"].value == "=C6*C7"
    assert ws_c["C17"].value == "=C15-C13"
    assert ws_c["C19"].value == "=(C11+C12)*(1+C5)+C13"

    assert wb["Outputs"]["C6"].value == "=Calculations!C17"
    sens = wb["Sensitivity"]
    assert sens["C3"].value == "=Inputs!$B$3*C$2"
    assert sens["C4"].value.startswith("=($B4-Inputs!$B$4)*C$3")


def test_export_scenario_workbook_recalculates_to_python_values(tmp_path):
    """Recalculate the workbook in LibreOffice (if installed) and compare with the Python model."""
    import shutil
    import subprocess

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice is None:
        pytest.skip("LibreOffice not installed")
    from smartbizlq.analytics.scenarios import run_scenarios

    src = tmp_path / "model.xlsx"
    export_scenario_workbook(_base(), str(src))
    out_dir = tmp_path / "recalc"
    subprocess.run([soffice, "--headless", "--convert-to", "xlsx", "--outdir", str(out_dir), str(src)],
                   check=True, capture_output=True, timeout=120)
    ws = load_workbook(out_dir / "model.xlsx", data_only=True)["Calculations"]
    for col, case in zip("BCD", ["Low", "Base", "High"]):
        out = run_scenarios(_base())[case][1]
        assert ws[f"{col}10"].value == pytest.approx(out.revenue)
        assert ws[f"{col}17"].value == pytest.approx(out.operating_profit)
        assert ws[f"{col}20"].value == pytest.approx(out.operating_profit_after_growth)


def test_export_scenario_workbook_empty_path_error():
    with pytest.raises(ValueError, match="out_path"):
        export_scenario_workbook(_base(), "")


# -------------------------------------------------------------
# Excel KPI Workbook Export Tests
# -------------------------------------------------------------
def test_export_kpi_workbook_structure():
    """Verify KPI workbook export produces KPI_Summary, Data_Profile, and Dataset_Extract."""
    kpis = {"Revenue": "$50,000", "Orders": "120"}
    df = pd.DataFrame({
        "product_id": [1, 2, 3],
        "name": ["Widget A", "Widget B", "Widget C"],
        "price": [10.5, 20.0, 15.75],
    })

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "kpis.xlsx")
        res_path = export_kpi_workbook(kpis, df, out_path)

        assert res_path == out_path
        assert os.path.exists(out_path)

        wb = load_workbook(out_path)
        assert "KPI_Summary" in wb.sheetnames
        assert "Data_Profile" in wb.sheetnames
        assert "Dataset_Extract" in wb.sheetnames

        ws_kpi = wb["KPI_Summary"]
        # KPI headers and items
        assert ws_kpi.cell(row=4, column=1).value == "Revenue"
        assert ws_kpi.cell(row=4, column=2).value == "$50,000"

        ws_data = wb["Dataset_Extract"]
        assert ws_data.cell(row=1, column=1).value == "product_id"
        assert ws_data.cell(row=2, column=1).value == 1
