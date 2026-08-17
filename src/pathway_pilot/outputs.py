"""Output tables for solved pathway pilot networks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pypsa


CAPACITY_COLUMNS = [
    "generator",
    "bus",
    "carrier",
    "build_year",
    "lifetime",
    "p_nom_opt",
    "capital_cost",
    "unit_capex_eur_per_mw",
    "marginal_cost",
    "battery_technology",
    "max_hours",
]


def _component_capacity_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.reset_index(names="generator")
    if "p_nom_opt" not in table.columns:
        table["p_nom_opt"] = table["p_nom"]
    else:
        table["p_nom_opt"] = table["p_nom_opt"].fillna(table["p_nom"])
    for column in CAPACITY_COLUMNS:
        if column not in table.columns:
            table[column] = pd.NA
    return table[CAPACITY_COLUMNS].rename(columns={"p_nom_opt": "p_nom_opt_mw"})


def optimal_capacities(network: pypsa.Network) -> pd.DataFrame:
    frames = [_component_capacity_table(network.generators)]
    if not network.storage_units.empty:
        frames.append(_component_capacity_table(network.storage_units))
    table = pd.concat(frames, ignore_index=True)
    table["p_nom_opt_mw"] = table["p_nom_opt_mw"].astype("float64")
    return table


def _component_dispatch_table(power: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    dispatch = power.copy()
    dispatch.index.names = ["period", "timestep"]
    table = (
        dispatch.reset_index()
        .melt(id_vars=["period", "timestep"], var_name="generator", value_name="dispatch_mw")
    )
    return table.merge(metadata, on="generator", how="left")


def hourly_dispatch(network: pypsa.Network) -> pd.DataFrame:
    metadata_columns = ["bus", "carrier", "build_year"]
    frames = [
        _component_dispatch_table(
            network.generators_t.p,
            network.generators[metadata_columns].reset_index(names="generator"),
        )
    ]
    if not network.storage_units.empty:
        frames.append(
            _component_dispatch_table(
                network.storage_units_t.p,
                network.storage_units[metadata_columns].reset_index(names="generator"),
            )
        )
    return pd.concat(frames, ignore_index=True)


def hourly_storage_state_of_charge(network: pypsa.Network) -> pd.DataFrame:
    if network.storage_units.empty:
        return pd.DataFrame(columns=["period", "timestep", "storage_unit", "state_of_charge_mwh"])
    soc = network.storage_units_t.state_of_charge.copy()
    soc.index.names = ["period", "timestep"]
    return soc.reset_index().melt(
        id_vars=["period", "timestep"],
        var_name="storage_unit",
        value_name="state_of_charge_mwh",
    )


def hourly_interconnector_flows(network: pypsa.Network) -> pd.DataFrame:
    if network.links.empty:
        return pd.DataFrame(
            columns=[
                "period",
                "timestep",
                "link",
                "bus0",
                "bus1",
                "flow_bus0_to_bus1_mw",
            ]
        )

    flows = network.links_t.p0.copy()
    flows.index.names = ["period", "timestep"]
    table = flows.reset_index().melt(
        id_vars=["period", "timestep"],
        var_name="link",
        value_name="flow_bus0_to_bus1_mw",
    )
    metadata = network.links[["bus0", "bus1"]].reset_index(names="link")
    return table.merge(metadata, on="link", how="left")


def hourly_prices(network: pypsa.Network) -> pd.DataFrame:
    prices = network.buses_t.marginal_price.copy()
    prices.index.names = ["period", "timestep"]
    return prices.reset_index().melt(
        id_vars=["period", "timestep"],
        var_name="bus",
        value_name="price_eur_per_mwh",
    )


def write_model_outputs(network: pypsa.Network, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    optimal_capacities(network).to_parquet(output_path / "optimal_capacities.parquet", index=False)
    hourly_dispatch(network).to_parquet(output_path / "hourly_dispatch.parquet", index=False)
    hourly_interconnector_flows(network).to_parquet(
        output_path / "hourly_interconnector_flows.parquet", index=False
    )
    hourly_prices(network).to_parquet(output_path / "hourly_prices.parquet", index=False)
    hourly_storage_state_of_charge(network).to_parquet(
        output_path / "hourly_storage_state_of_charge.parquet", index=False
    )
