"""Configuration loading for the pathway pilot model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelCase:
    demand_zones: tuple[str, ...]
    capacity_factor_zone: str


@dataclass(frozen=True)
class ModelConfig:
    active_model: str
    climate_years: list[int]
    model_cases: dict[str, ModelCase]
    demand_zones: tuple[str, ...]
    capacity_factor_zone: str
    investment_periods: list[int]
    period_weights: dict[int, float]
    discount_rate: float
    eurdkk: float
    gas_price_eur_per_mwh_fuel: dict[int, float]
    load_shedding_variable_cost_eur_per_mwh: float
    load_shedding_max_capacity_mw: float
    capacity_limits_mw: dict[str, float]


def _int_keyed_float_dict(values: dict[Any, Any], name: str) -> dict[int, float]:
    if not isinstance(values, dict):
        raise ValueError(f"{name} must be a mapping")
    return {int(key): float(value) for key, value in values.items()}


def load_config(path: str | Path) -> ModelConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("config must be a mapping")

    investment_periods = [int(period) for period in raw["investment_periods"]]
    active_model = str(raw.get("active_model", "DK"))
    raw_model_cases = raw.get(
        "model_cases",
        {"DK": {"demand_zones": ["DKE1", "DKW1"], "capacity_factor_zone": "DKW1"}},
    )
    model_cases = {
        str(name): ModelCase(
            demand_zones=tuple(str(zone) for zone in case["demand_zones"]),
            capacity_factor_zone=str(case.get("capacity_factor_zone", "DKW1")),
        )
        for name, case in raw_model_cases.items()
    }
    if active_model not in model_cases:
        raise ValueError(f"active_model {active_model!r} not found in model_cases")
    active_case = model_cases[active_model]
    climate_years = [int(year) for year in raw.get("climate_years", [1982])]
    period_weights = _int_keyed_float_dict(raw["period_weights"], "period_weights")
    eurdkk = float(raw.get("eurdkk", 1.0))
    if "gas_price_dkk_per_gj_fuel" in raw:
        gas_prices_dkk_per_gj = _int_keyed_float_dict(
            raw["gas_price_dkk_per_gj_fuel"], "gas_price_dkk_per_gj_fuel"
        )
        gas_prices = {
            period: value * 3.6 / eurdkk for period, value in gas_prices_dkk_per_gj.items()
        }
    elif "gas_price_dkk_per_mwh_fuel" in raw:
        gas_prices_dkk = _int_keyed_float_dict(
            raw["gas_price_dkk_per_mwh_fuel"], "gas_price_dkk_per_mwh_fuel"
        )
        gas_prices = {period: value / eurdkk for period, value in gas_prices_dkk.items()}
    else:
        gas_prices = _int_keyed_float_dict(
            raw["gas_price_eur_per_mwh_fuel"], "gas_price_eur_per_mwh_fuel"
        )
    load_shedding = raw["load_shedding"]
    capacity_limits = {str(key): float(value) for key, value in raw["capacity_limits_mw"].items()}

    missing_weights = set(investment_periods) - set(period_weights)
    missing_gas_prices = set(investment_periods) - set(gas_prices)
    if missing_weights:
        raise ValueError(f"missing period weights for {sorted(missing_weights)}")
    if missing_gas_prices:
        raise ValueError(f"missing gas prices for {sorted(missing_gas_prices)}")

    return ModelConfig(
        active_model=active_model,
        climate_years=climate_years,
        model_cases=model_cases,
        demand_zones=active_case.demand_zones,
        capacity_factor_zone=active_case.capacity_factor_zone,
        investment_periods=investment_periods,
        period_weights=period_weights,
        discount_rate=float(raw["discount_rate"]),
        eurdkk=eurdkk,
        gas_price_eur_per_mwh_fuel=gas_prices,
        load_shedding_variable_cost_eur_per_mwh=float(
            load_shedding["variable_cost_eur_per_mwh"]
        ),
        load_shedding_max_capacity_mw=float(load_shedding["max_capacity_mw"]),
        capacity_limits_mw=capacity_limits,
    )


def with_active_model(cfg: ModelConfig, active_model: str) -> ModelConfig:
    if active_model not in cfg.model_cases:
        raise ValueError(f"active_model {active_model!r} not found in model_cases")
    model_case = cfg.model_cases[active_model]
    return replace(
        cfg,
        active_model=active_model,
        demand_zones=model_case.demand_zones,
        capacity_factor_zone=model_case.capacity_factor_zone,
    )
