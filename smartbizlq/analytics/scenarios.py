"""Commercial scenario and sensitivity modelling for SmartBizLQ.

A deliberately simple, transparent one-period profit model that can be
explained line by line and mirrored exactly in Excel
(see ``smartbizlq.reporting.excel_export.export_scenario_workbook``).

Model logic (all money in GBP):

    revenue          = price x volume
    variable_cost    = unit_cost x volume
    energy_cost      = energy_mwh x wholesale_price_mwh
    gross_margin     = revenue - variable_cost - energy_cost
    operating_profit = gross_margin - fixed_costs

Growth projection: ``growth_rate`` scales sales volume and energy use
(both are volume-driven); price, unit cost, wholesale price and fixed
costs are held constant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import logging
from typing import Any, Iterable

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScenarioInputs:
    """Input assumptions for one scenario.

    Attributes:
        price: Selling price per unit (GBP).
        volume: Units sold in the period.
        unit_cost: Variable cost per unit, excluding energy (GBP).
        growth_rate: Projected volume growth as a decimal (0.05 = 5%).
        energy_mwh: Electricity consumed in the period (MWh).
        wholesale_price_mwh: Wholesale electricity price (GBP per MWh).
        fixed_costs: Fixed costs for the period (GBP).
    """

    price: float
    volume: float
    unit_cost: float
    growth_rate: float = 0.0
    energy_mwh: float = 0.0
    wholesale_price_mwh: float = 0.0
    fixed_costs: float = 0.0


@dataclass(frozen=True)
class ScenarioOutputs:
    """Financial outcomes computed from :class:`ScenarioInputs` (GBP)."""

    revenue: float
    variable_cost: float
    energy_cost: float
    fixed_costs: float
    total_cost: float
    gross_margin: float
    gross_margin_pct: float
    operating_profit: float
    revenue_after_growth: float
    total_cost_after_growth: float
    operating_profit_after_growth: float


INPUT_LABELS: dict[str, str] = {
    "price": "Selling price (£/unit)",
    "volume": "Sales volume (units)",
    "unit_cost": "Variable cost (£/unit)",
    "growth_rate": "Volume growth rate",
    "energy_mwh": "Energy use (MWh)",
    "wholesale_price_mwh": "Wholesale electricity price (£/MWh)",
    "fixed_costs": "Fixed costs (£)",
}

OUTPUT_LABELS: dict[str, str] = {
    "revenue": "Revenue",
    "variable_cost": "Variable cost",
    "energy_cost": "Energy cost",
    "fixed_costs": "Fixed costs",
    "total_cost": "Total cost",
    "gross_margin": "Gross margin",
    "gross_margin_pct": "Gross margin %",
    "operating_profit": "Operating profit",
    "revenue_after_growth": "Revenue after growth",
    "total_cost_after_growth": "Total cost after growth",
    "operating_profit_after_growth": "Operating profit after growth",
}

# Default Low/High cases: multipliers applied to the base inputs.
# Inputs not listed keep their base value.
DEFAULT_CASES: dict[str, dict[str, float]] = {
    "Low": {"price": 0.95, "volume": 0.85, "unit_cost": 1.05, "wholesale_price_mwh": 1.20},
    "Base": {},
    "High": {"price": 1.05, "volume": 1.15, "unit_cost": 0.95, "wholesale_price_mwh": 0.85},
}


def _safe_pct(numerator: float, denominator: float) -> float:
    return numerator / denominator * 100.0 if abs(denominator) > 1e-12 else 0.0


def compute_scenario(inputs: ScenarioInputs) -> ScenarioOutputs:
    """Compute revenue, costs, margin and profit for one set of assumptions.

    Args:
        inputs: Scenario assumptions.

    Returns:
        ScenarioOutputs with values rounded to 2 decimal places.
    """
    revenue = inputs.price * inputs.volume
    variable_cost = inputs.unit_cost * inputs.volume
    energy_cost = inputs.energy_mwh * inputs.wholesale_price_mwh
    total_cost = variable_cost + energy_cost + inputs.fixed_costs
    gross_margin = revenue - variable_cost - energy_cost
    operating_profit = gross_margin - inputs.fixed_costs

    g = 1.0 + inputs.growth_rate
    revenue_g = revenue * g
    total_cost_g = (variable_cost + energy_cost) * g + inputs.fixed_costs
    profit_g = revenue_g - total_cost_g

    return ScenarioOutputs(
        revenue=round(revenue, 2),
        variable_cost=round(variable_cost, 2),
        energy_cost=round(energy_cost, 2),
        fixed_costs=round(inputs.fixed_costs, 2),
        total_cost=round(total_cost, 2),
        gross_margin=round(gross_margin, 2),
        gross_margin_pct=round(_safe_pct(gross_margin, revenue), 2),
        operating_profit=round(operating_profit, 2),
        revenue_after_growth=round(revenue_g, 2),
        total_cost_after_growth=round(total_cost_g, 2),
        operating_profit_after_growth=round(profit_g, 2),
    )


def apply_case(base: ScenarioInputs, multipliers: dict[str, float]) -> ScenarioInputs:
    """Return a copy of ``base`` with each named input multiplied by its factor."""
    valid = {f.name for f in fields(ScenarioInputs)}
    unknown = set(multipliers) - valid
    if unknown:
        raise AttributeError(f"Unknown input(s) in case definition: {sorted(unknown)}")
    return replace(base, **{k: getattr(base, k) * m for k, m in multipliers.items()})


def run_scenarios(
    base: ScenarioInputs,
    cases: dict[str, dict[str, float]] | None = None,
) -> dict[str, tuple[ScenarioInputs, ScenarioOutputs]]:
    """Run several named cases (default Low / Base / High).

    Args:
        base: Base-case assumptions.
        cases: Mapping of case name to input multipliers. Defaults to ``DEFAULT_CASES``.

    Returns:
        Mapping of case name to (inputs used, outputs).
    """
    cases = DEFAULT_CASES if cases is None else cases
    results: dict[str, tuple[ScenarioInputs, ScenarioOutputs]] = {}
    for name, mult in cases.items():
        case_inputs = apply_case(base, mult)
        results[name] = (case_inputs, compute_scenario(case_inputs))
    return results


def scenarios_to_frame(results: dict[str, tuple[ScenarioInputs, ScenarioOutputs]]) -> pd.DataFrame:
    """Turn ``run_scenarios`` output into a tidy comparison table (rows = metrics)."""
    data: dict[str, dict[str, float]] = {}
    for name, (inp, out) in results.items():
        col = {INPUT_LABELS[k]: v for k, v in asdict(inp).items()}
        col.update({OUTPUT_LABELS[k]: v for k, v in asdict(out).items()})
        data[name] = col
    return pd.DataFrame(data)


def sensitivity_table(
    base: ScenarioInputs,
    var1_name: str,
    var1_range: Iterable[float],
    var2_name: str,
    var2_range: Iterable[float],
    output_name: str = "operating_profit",
) -> pd.DataFrame:
    """Two-way sensitivity grid: rows vary ``var1_name``, columns vary ``var2_name``.

    Raises:
        AttributeError: If an input or output name is not a model field.
    """
    valid_inputs = {f.name for f in fields(ScenarioInputs)}
    valid_outputs = {f.name for f in fields(ScenarioOutputs)}
    for name in (var1_name, var2_name):
        if name not in valid_inputs:
            raise AttributeError(f"Invalid input variable '{name}'. Valid: {sorted(valid_inputs)}")
    if output_name not in valid_outputs:
        raise AttributeError(f"Invalid output '{output_name}'. Valid: {sorted(valid_outputs)}")

    v1_list, v2_list = list(var1_range), list(var2_range)
    grid = {
        v2: [
            getattr(compute_scenario(replace(base, **{var1_name: v1, var2_name: v2})), output_name)
            for v1 in v1_list
        ]
        for v2 in v2_list
    }
    table = pd.DataFrame(grid, index=v1_list)
    table.index.name = var1_name
    table.columns.name = var2_name
    return table


def tornado_data(
    base: ScenarioInputs,
    swing_pct: float = 0.10,
    output_name: str = "operating_profit",
) -> list[dict[str, Any]]:
    """One-at-a-time sensitivity: flex each non-zero input by ±``swing_pct``.

    Inputs that do not change the chosen output are left out.

    Returns:
        Records sorted by swing (largest first) with keys ``variable``, ``label``,
        ``base_value``, ``low_output``, ``high_output``, ``base_output``,
        ``downside_change``, ``upside_change`` and ``swing``.
    """
    base_output = getattr(compute_scenario(base), output_name)
    records: list[dict[str, Any]] = []
    for f in fields(ScenarioInputs):
        val = getattr(base, f.name)
        if val == 0:
            continue  # flexing zero by a percentage changes nothing
        low = getattr(compute_scenario(replace(base, **{f.name: val * (1 - swing_pct)})), output_name)
        high = getattr(compute_scenario(replace(base, **{f.name: val * (1 + swing_pct)})), output_name)
        if abs(high - low) < 1e-9:
            continue  # input does not affect this output (e.g. growth vs current-period profit)
        records.append(
            {
                "variable": f.name,
                "label": INPUT_LABELS[f.name],
                "base_value": val,
                "low_output": round(low, 2),
                "high_output": round(high, 2),
                "base_output": round(base_output, 2),
                "downside_change": round(low - base_output, 2),
                "upside_change": round(high - base_output, 2),
                "swing": round(abs(high - low), 2),
            }
        )
    records.sort(key=lambda r: r["swing"], reverse=True)
    return records
