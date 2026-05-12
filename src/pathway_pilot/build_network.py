"""Build the PyPSA network for the pathway pilot."""

from __future__ import annotations

import pandas as pd
import pypsa

from pathway_pilot.model_config import ModelConfig
from pathway_pilot.model_inputs import ModelInputs
from pathway_pilot.technology_data import load_technology_assumptions


def _period_value(values: dict[int, float], period: int) -> float:
    if period in values:
        return float(values[period])
    earlier = [year for year in values if year <= period]
    if earlier:
        return float(values[max(earlier)])
    return float(values[min(values)])


def _normalise_string_columns(network: pypsa.Network) -> None:
    for frame in [network.buses, network.loads, network.generators, network.carriers]:
        for column in frame.columns:
            if str(frame[column].dtype) in {"str", "string"}:
                frame[column] = frame[column].astype("object")


def _set_unit_capex(network: pypsa.Network, generator_name: str, value: float) -> None:
    network.generators.loc[generator_name, "unit_capex_eur_per_mw"] = float(value)


def build_network(cfg: ModelConfig, data: ModelInputs) -> pypsa.Network:
    technology_assumptions = load_technology_assumptions(cfg)
    network = pypsa.Network()
    network.set_snapshots(data.snapshots)
    network.set_investment_periods(cfg.investment_periods)
    network.investment_period_weightings.loc[:, "objective"] = pd.Series(
        cfg.period_weights, dtype="float64"
    )

    network.add("Bus", "electricity", carrier="electricity")
    for carrier in ["electricity", "wind", "solar", "gas", "load_shedding"]:
        network.add("Carrier", carrier)
    network.add("Load", "demand", bus="electricity", p_set=data.demand_series)

    for technology, cf_series in [
        ("wind", data.wind_cf_series),
        ("solar", data.solar_cf_series),
    ]:
        tech = technology_assumptions[technology]
        for build_year in cfg.investment_periods:
            network.add(
                "Generator",
                f"{technology}_{build_year}",
                bus="electricity",
                carrier=technology,
                p_nom_extendable=True,
                p_nom_max=cfg.capacity_limits_mw[technology],
                capital_cost=_period_value(tech.capital_cost_by_period, build_year),
                marginal_cost=_period_value(tech.marginal_cost_by_period, build_year),
                p_max_pu=cf_series,
                build_year=build_year,
                lifetime=tech.lifetime_years,
            )
            _set_unit_capex(
                network,
                f"{technology}_{build_year}",
                _period_value(tech.unit_capex_by_period, build_year),
            )

    gas_tech = technology_assumptions["gas_turbine"]
    for build_year in cfg.investment_periods:
        network.add(
            "Generator",
            f"gas_turbine_{build_year}",
            bus="electricity",
            carrier="gas",
            p_nom_extendable=True,
            p_nom_max=cfg.capacity_limits_mw["gas_turbine"],
            capital_cost=_period_value(gas_tech.capital_cost_by_period, build_year),
            marginal_cost=_period_value(gas_tech.marginal_cost_by_period, build_year),
            build_year=build_year,
            lifetime=gas_tech.lifetime_years,
        )
        _set_unit_capex(
            network,
            f"gas_turbine_{build_year}",
            _period_value(gas_tech.unit_capex_by_period, build_year),
        )

    network.add(
        "Generator",
        "load_shedding",
        bus="electricity",
        carrier="load_shedding",
        p_nom=cfg.load_shedding_max_capacity_mw,
        marginal_cost=cfg.load_shedding_variable_cost_eur_per_mwh,
    )
    _set_unit_capex(network, "load_shedding", 0.0)

    _normalise_string_columns(network)
    return network
