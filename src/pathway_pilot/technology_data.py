"""Technology cost adapters for TechCat-derived Parquet files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pathway_pilot.config import DEV_DATA_DIR
from pathway_pilot.model_config import ModelConfig


@dataclass(frozen=True)
class TechnologyAssumption:
    capital_cost_by_period: dict[int, float]
    unit_capex_by_period: dict[int, float]
    marginal_cost_by_period: dict[int, float]
    lifetime_years: int


FALLBACK_ASSUMPTIONS = {
    "wind": TechnologyAssumption(
        capital_cost_by_period={2030: 90_000, 2040: 85_000, 2050: 80_000},
        unit_capex_by_period={2030: 1_100_000, 2040: 1_050_000, 2050: 1_030_000},
        marginal_cost_by_period={2030: 2, 2040: 2, 2050: 2},
        lifetime_years=30,
    ),
    "solar": TechnologyAssumption(
        capital_cost_by_period={2030: 35_000, 2040: 30_000, 2050: 28_000},
        unit_capex_by_period={2030: 380_000, 2040: 320_000, 2050: 290_000},
        marginal_cost_by_period={2030: 0, 2040: 0, 2050: 0},
        lifetime_years=40,
    ),
    "gas_turbine": TechnologyAssumption(
        capital_cost_by_period={2030: 45_000, 2040: 43_000, 2050: 42_000},
        unit_capex_by_period={2030: 595_000, 2040: 574_000, 2050: 553_000},
        marginal_cost_by_period={2030: 200, 2040: 205, 2050: 205},
        lifetime_years=25,
    ),
}

TECH_FILES = {
    "wind": "onshore_turbines.parquet",
    "solar": "utility_scale_pv.parquet",
    "gas_turbine": "gas_turbine_simple_cycle_large.parquet",
}


def _annuity(discount_rate: float, lifetime_years: float) -> float:
    if discount_rate == 0:
        return 1 / lifetime_years
    return discount_rate / (1 - (1 + discount_rate) ** (-lifetime_years))


def _row_for_period(table: pd.DataFrame, period: int) -> pd.Series:
    rows = table[table["year"] <= period]
    if rows.empty:
        rows = table[table["year"] == table["year"].min()]
    return rows.sort_values("year").iloc[-1]


def _capital_cost(row: pd.Series, discount_rate: float) -> float:
    lifetime = float(row["technical_lifetime_years"])
    investment = _unit_capex(row)
    fixed_om = float(row["fixed_om_eur_per_mw_e_per_year"])
    return investment * _annuity(discount_rate, lifetime) + fixed_om


def _unit_capex(row: pd.Series) -> float:
    return float(row["total_nominal_investment_meur_per_mw_e"]) * 1_000_000


def _renewable_assumption(
    cfg: ModelConfig,
    table: pd.DataFrame,
    periods: list[int],
) -> TechnologyAssumption:
    capital_costs = {}
    unit_capex = {}
    marginal_costs = {}
    for period in periods:
        row = _row_for_period(table, period)
        capital_costs[period] = _capital_cost(row, cfg.discount_rate)
        unit_capex[period] = _unit_capex(row)
        variable_om = row["variable_om_eur_per_mwh_e"]
        marginal_costs[period] = 0 if pd.isna(variable_om) else float(variable_om)

    lifetime = int(round(float(_row_for_period(table, periods[0])["technical_lifetime_years"])))
    return TechnologyAssumption(capital_costs, unit_capex, marginal_costs, lifetime)


def _gas_assumption(
    cfg: ModelConfig,
    table: pd.DataFrame,
    periods: list[int],
) -> TechnologyAssumption:
    capital_costs = {}
    unit_capex = {}
    marginal_costs = {}
    for period in periods:
        row = _row_for_period(table, period)
        capital_costs[period] = _capital_cost(row, cfg.discount_rate)
        unit_capex[period] = _unit_capex(row)
        efficiency = float(row["electrical_efficiency_net_annual_avg"])
        variable_om = float(row["variable_om_eur_per_mwh_e"])
        marginal_costs[period] = cfg.gas_price_eur_per_mwh_fuel[period] / efficiency + variable_om

    lifetime = int(round(float(_row_for_period(table, periods[0])["technical_lifetime_years"])))
    return TechnologyAssumption(capital_costs, unit_capex, marginal_costs, lifetime)


def load_technology_assumptions(
    cfg: ModelConfig,
    tech_dir: str | Path | None = None,
) -> dict[str, TechnologyAssumption]:
    base = Path(tech_dir) if tech_dir is not None else DEV_DATA_DIR / "pathway-pilot" / "tech"
    if not all((base / file_name).exists() for file_name in TECH_FILES.values()):
        return FALLBACK_ASSUMPTIONS

    tables = {name: pd.read_parquet(base / file_name) for name, file_name in TECH_FILES.items()}
    return {
        "wind": _renewable_assumption(cfg, tables["wind"], cfg.investment_periods),
        "solar": _renewable_assumption(cfg, tables["solar"], cfg.investment_periods),
        "gas_turbine": _gas_assumption(cfg, tables["gas_turbine"], cfg.investment_periods),
    }
