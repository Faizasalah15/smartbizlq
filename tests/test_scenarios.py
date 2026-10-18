"""Tests for the commercial scenario model."""

import pytest

from smartbizlq.analytics.scenarios import (
    DEFAULT_CASES,
    ScenarioInputs,
    apply_case,
    compute_scenario,
    run_scenarios,
    scenarios_to_frame,
    sensitivity_table,
    tornado_data,
)

BASE = ScenarioInputs(price=100.0, volume=1000.0, unit_cost=60.0, growth_rate=0.10,
                      energy_mwh=50.0, wholesale_price_mwh=80.0, fixed_costs=10_000.0)


def test_compute_scenario_hand_calculated():
    out = compute_scenario(BASE)
    assert out.revenue == 100_000.0
    assert out.variable_cost == 60_000.0
    assert out.energy_cost == 4_000.0          # 50 MWh x £80/MWh
    assert out.total_cost == 74_000.0
    assert out.gross_margin == 36_000.0
    assert out.gross_margin_pct == 36.0
    assert out.operating_profit == 26_000.0
    # growth scales volume-driven items only; fixed costs unchanged
    assert out.revenue_after_growth == pytest.approx(110_000.0)
    assert out.total_cost_after_growth == pytest.approx(64_000.0 * 1.1 + 10_000.0)
    assert out.operating_profit_after_growth == pytest.approx(110_000.0 - 80_400.0)


def test_zero_revenue_does_not_divide_by_zero():
    out = compute_scenario(ScenarioInputs(price=0.0, volume=0.0, unit_cost=0.0))
    assert out.gross_margin_pct == 0.0


def test_run_scenarios_uses_case_multipliers():
    results = run_scenarios(BASE)
    assert list(results) == list(DEFAULT_CASES)
    low_inputs, low_out = results["Low"]
    assert low_inputs.volume == pytest.approx(BASE.volume * DEFAULT_CASES["Low"]["volume"])
    assert low_inputs.fixed_costs == BASE.fixed_costs
    assert results["High"][1].operating_profit > results["Base"][1].operating_profit > low_out.operating_profit


def test_apply_case_rejects_unknown_input():
    with pytest.raises(AttributeError):
        apply_case(BASE, {"not_a_field": 1.1})


def test_scenarios_to_frame_has_one_column_per_case():
    frame = scenarios_to_frame(run_scenarios(BASE))
    assert list(frame.columns) == ["Low", "Base", "High"]
    assert "Operating profit" in frame.index


def test_sensitivity_table_matches_direct_calculation():
    table = sensitivity_table(BASE, "price", [90.0, 100.0], "volume", [900.0, 1000.0])
    assert table.shape == (2, 2)
    assert table.loc[100.0, 1000.0] == compute_scenario(BASE).operating_profit
    assert table.loc[90.0, 900.0] == pytest.approx((90 - 60) * 900 - 4_000 - 10_000)


def test_sensitivity_table_invalid_names():
    with pytest.raises(AttributeError):
        sensitivity_table(BASE, "bogus", [1], "volume", [1])
    with pytest.raises(AttributeError):
        sensitivity_table(BASE, "price", [1], "volume", [1], output_name="bogus")


def test_tornado_sorted_and_skips_zero_inputs():
    records = tornado_data(ScenarioInputs(price=100.0, volume=1000.0, unit_cost=60.0), swing_pct=0.1)
    swings = [r["swing"] for r in records]
    assert swings == sorted(swings, reverse=True)
    names = {r["variable"] for r in records}
    assert "energy_mwh" not in names and "growth_rate" not in names
    price = next(r for r in records if r["variable"] == "price")
    assert price["upside_change"] == pytest.approx(10.0 * 1000)
