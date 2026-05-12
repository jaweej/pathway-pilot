"""Output tables for solved pathway pilot networks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pypsa


def optimal_capacities(network: pypsa.Network) -> pd.DataFrame:
    table = network.generators.reset_index(names="generator")
    columns = [
        "generator",
        "carrier",
        "build_year",
        "lifetime",
        "p_nom_opt",
        "capital_cost",
        "unit_capex_eur_per_mw",
        "marginal_cost",
    ]
    if "unit_capex_eur_per_mw" not in table.columns:
        table["unit_capex_eur_per_mw"] = pd.NA
    table = table[columns].rename(columns={"p_nom_opt": "p_nom_opt_mw"})
    table["p_nom_opt_mw"] = table["p_nom_opt_mw"].astype("float64")
    return table


def hourly_dispatch(network: pypsa.Network) -> pd.DataFrame:
    dispatch = network.generators_t.p.copy()
    dispatch.index.names = ["period", "timestep"]
    table = (
        dispatch.reset_index()
        .melt(id_vars=["period", "timestep"], var_name="generator", value_name="dispatch_mw")
    )
    metadata = network.generators[["carrier", "build_year"]].reset_index(names="generator")
    return table.merge(metadata, on="generator", how="left")


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
    hourly_prices(network).to_parquet(output_path / "hourly_prices.parquet", index=False)
