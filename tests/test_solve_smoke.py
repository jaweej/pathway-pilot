from pathlib import Path

import pandas as pd

from pathway_pilot.model_config import load_config, with_active_model
from pathway_pilot.model_inputs import ModelInputs, make_synthetic_inputs
from pathway_pilot.solve import solve_model


def test_smoke_solve_highs_tiny_fixture():
    cfg = load_config(Path("config/model_config.yaml"))
    data = make_synthetic_inputs(periods=cfg.investment_periods, hours_per_period=24)

    solved, status, condition = solve_model(cfg, data)

    assert status == "ok"
    assert condition == "optimal"
    assert solved.objective > 0


def test_smoke_solve_combined_dk_nl_tiny_fixture():
    cfg = with_active_model(load_config(Path("config/model_config.yaml")), "DK_NL")
    base = make_synthetic_inputs(periods=cfg.investment_periods, hours_per_period=4)
    demand_by_bus = pd.DataFrame(
        {
            "DK": base.demand_series * 0.35,
            "NL": base.demand_series * 0.65,
        }
    )
    wind_cf_by_bus = pd.DataFrame(
        {
            "DK": base.wind_cf_series,
            "NL": (base.wind_cf_series * 0.9).clip(upper=1),
        }
    )
    solar_cf_by_bus = pd.DataFrame(
        {
            "DK": base.solar_cf_series,
            "NL": (base.solar_cf_series * 1.1).clip(upper=1),
        }
    )
    data = ModelInputs(
        demand_series=demand_by_bus.sum(axis=1),
        wind_cf_series=wind_cf_by_bus.mean(axis=1),
        solar_cf_series=solar_cf_by_bus.mean(axis=1),
        demand_by_bus=demand_by_bus,
        wind_cf_by_bus=wind_cf_by_bus,
        solar_cf_by_bus=solar_cf_by_bus,
    )

    solved, status, condition = solve_model(cfg, data)

    assert status == "ok"
    assert condition == "optimal"
    assert "DK_NL_interconnector" in solved.links.index
    assert solved.links.loc["DK_NL_interconnector", "p_nom"] == 1000
