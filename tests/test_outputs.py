from pathlib import Path

import pandas as pd

from pathway_pilot.model_config import load_config
from pathway_pilot.model_inputs import make_synthetic_inputs
from pathway_pilot.outputs import (
    hourly_dispatch,
    hourly_prices,
    optimal_capacities,
    write_model_outputs,
)
from pathway_pilot.solve import solve_model


def test_output_tables_have_expected_columns():
    cfg = load_config(Path("config/model_config.yaml"))
    data = make_synthetic_inputs(periods=cfg.investment_periods, hours_per_period=4)
    network, status, condition = solve_model(cfg, data)

    assert status == "ok"
    assert condition == "optimal"
    assert set(
        [
            "generator",
            "carrier",
            "build_year",
            "p_nom_opt_mw",
            "capital_cost",
            "unit_capex_eur_per_mw",
        ]
    ).issubset(optimal_capacities(network).columns)
    assert set(["period", "timestep", "generator", "dispatch_mw"]).issubset(
        hourly_dispatch(network).columns
    )
    assert set(["period", "timestep", "bus", "price_eur_per_mwh"]).issubset(
        hourly_prices(network).columns
    )


def test_write_model_outputs_creates_three_parquet_files(monkeypatch):
    written = []

    def fake_to_parquet(self, path, index=False):
        written.append((Path(path).name, index, list(self.columns)))

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    cfg = load_config(Path("config/model_config.yaml"))
    data = make_synthetic_inputs(periods=cfg.investment_periods, hours_per_period=4)
    network, _, _ = solve_model(cfg, data)
    write_model_outputs(network, Path("unused"))

    assert [name for name, _, _ in written] == [
        "optimal_capacities.parquet",
        "hourly_dispatch.parquet",
        "hourly_prices.parquet",
    ]
