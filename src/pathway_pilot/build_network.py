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
    for frame in [
        network.buses,
        network.loads,
        network.generators,
        network.links,
        network.carriers,
    ]:
        for column in frame.columns:
            if str(frame[column].dtype) in {"str", "string"}:
                frame[column] = frame[column].astype("object")


def _set_unit_capex(network: pypsa.Network, generator_name: str, value: float) -> None:
    network.generators.loc[generator_name, "unit_capex_eur_per_mw"] = float(value)


def _component_name(bus: str, technology: str, build_year: int, multi_bus: bool) -> str:
    if multi_bus:
        return f"{bus}_{technology}_{build_year}"
    return f"{technology}_{build_year}"


def build_network(cfg: ModelConfig, data: ModelInputs) -> pypsa.Network:
    technology_assumptions = load_technology_assumptions(cfg)
    network = pypsa.Network()
    network.set_snapshots(data.snapshots)
    network.set_investment_periods(cfg.investment_periods)
    network.investment_period_weightings.loc[:, "objective"] = pd.Series(
        cfg.period_weights, dtype="float64"
    )

    bus_names = data.bus_names
    multi_bus = len(bus_names) > 1
    for bus in bus_names:
        network.add("Bus", bus, carrier="electricity")
    for carrier in [
        "electricity",
        "wind",
        "solar",
        "gas",
        "gas_turbine_cc",
        "load_shedding",
        "interconnector",
    ]:
        network.add("Carrier", carrier)
    for bus in bus_names:
        demand = data.demand_series if data.demand_by_bus is None else data.demand_by_bus[bus]
        load_name = f"demand_{bus}" if multi_bus else "demand"
        network.add("Load", load_name, bus=bus, p_set=demand)

    for bus in bus_names:
        wind_cf = data.wind_cf_series if data.wind_cf_by_bus is None else data.wind_cf_by_bus[bus]
        solar_cf = (
            data.solar_cf_series if data.solar_cf_by_bus is None else data.solar_cf_by_bus[bus]
        )
        for technology, cf_source in [
            ("wind", wind_cf),
            ("solar", solar_cf),
        ]:
            tech = technology_assumptions[technology]
            for build_year in cfg.investment_periods:
                generator_name = _component_name(bus, technology, build_year, multi_bus)
                network.add(
                    "Generator",
                    generator_name,
                    bus=bus,
                    carrier=technology,
                    p_nom_extendable=True,
                    p_nom_max=cfg.capacity_limits_mw[technology],
                    capital_cost=_period_value(tech.capital_cost_by_period, build_year),
                    marginal_cost=_period_value(tech.marginal_cost_by_period, build_year),
                    p_max_pu=cf_source,
                    build_year=build_year,
                    lifetime=tech.lifetime_years,
                )
                _set_unit_capex(
                    network,
                    generator_name,
                    _period_value(tech.unit_capex_by_period, build_year),
                )

    gas_tech = technology_assumptions["gas_turbine"]
    for bus in bus_names:
        for build_year in cfg.investment_periods:
            generator_name = _component_name(bus, "gas_turbine", build_year, multi_bus)
            network.add(
                "Generator",
                generator_name,
                bus=bus,
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
                generator_name,
                _period_value(gas_tech.unit_capex_by_period, build_year),
            )

    gas_cc_tech = technology_assumptions["gas_turbine_cc"]
    for bus in bus_names:
        for build_year in cfg.investment_periods:
            generator_name = _component_name(bus, "gas_turbine_cc", build_year, multi_bus)
            network.add(
                "Generator",
                generator_name,
                bus=bus,
                carrier="gas_turbine_cc",
                p_nom_extendable=True,
                p_nom_max=cfg.capacity_limits_mw["gas_turbine_cc"],
                capital_cost=_period_value(gas_cc_tech.capital_cost_by_period, build_year),
                marginal_cost=_period_value(gas_cc_tech.marginal_cost_by_period, build_year),
                build_year=build_year,
                lifetime=gas_cc_tech.lifetime_years,
            )
            _set_unit_capex(
                network,
                generator_name,
                _period_value(gas_cc_tech.unit_capex_by_period, build_year),
            )

    for bus in bus_names:
        generator_name = f"{bus}_load_shedding" if multi_bus else "load_shedding"
        network.add(
            "Generator",
            generator_name,
            bus=bus,
            carrier="load_shedding",
            p_nom=cfg.load_shedding_max_capacity_mw,
            marginal_cost=cfg.load_shedding_variable_cost_eur_per_mwh,
        )
        _set_unit_capex(network, generator_name, 0.0)

    for interconnector in cfg.interconnectors:
        network.add(
            "Link",
            interconnector.name,
            bus0=interconnector.bus0,
            bus1=interconnector.bus1,
            carrier="interconnector",
            p_nom=interconnector.capacity_mw,
            p_min_pu=-1.0,
            p_max_pu=1.0,
            efficiency=1.0,
        )

    _normalise_string_columns(network)
    return network
