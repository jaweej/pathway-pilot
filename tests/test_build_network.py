from pathlib import Path

import pandas as pd

from pathway_pilot.build_network import build_network
from pathway_pilot.model_config import load_config, with_active_model
from pathway_pilot.model_inputs import build_model_inputs, make_synthetic_inputs


def test_network_contains_core_components():
    cfg = load_config(Path("config/model_config.yaml"))
    data = make_synthetic_inputs(periods=cfg.investment_periods, hours_per_period=24)

    network = build_network(cfg, data)

    assert "electricity" in network.buses.index
    expected = {
        "wind_2030",
        "solar_2030",
        "gas_turbine_2030",
        "gas_turbine_cc_2030",
        "load_shedding",
    }
    assert expected.issubset(network.generators.index)
    assert network.generators.loc["gas_turbine_cc_2030", "carrier"] == "gas_turbine_cc"
    assert list(network.investment_periods) == [2030, 2040, 2050]


def test_combined_dk_nl_network_has_fixed_lossless_interconnector():
    cfg = with_active_model(load_config(Path("config/model_config.yaml")), "DK_NL")
    timestamps = pd.date_range("1982-01-01", periods=2, freq="h")
    cf_rows = []
    demand_rows = []
    for source_year in [2030, 2040]:
        for zone in ["DKE1", "DKW1", "NL00"]:
            for technology in ["onshore_wind", "solar"]:
                for timestamp in timestamps:
                    cf_rows.append(
                        {
                            "target_year": source_year,
                            "weather_year": 1982,
                            "zone": zone,
                            "technology": technology,
                            "timestamp": timestamp,
                            "capacity_factor": {"DKW1": 0.2, "NL00": 0.4}.get(zone, 0.1),
                            "source_file": "cf.csv",
                        }
                    )
            for timestamp in timestamps:
                demand_rows.append(
                    {
                        "target_year": source_year,
                        "weather_year": 1982,
                        "zone": zone,
                        "timestamp": timestamp,
                        "demand_mw": {"DKE1": 10, "DKW1": 20, "NL00": 100}[zone],
                        "source_file": "demand.xlsx",
                    }
                )
    data = build_model_inputs(
        capacity_factors=pd.DataFrame(cf_rows),
        demand=pd.DataFrame(demand_rows),
        periods=cfg.investment_periods,
        weather_year=1982,
        model_regions=cfg.model_regions,
    )

    network = build_network(cfg, data)

    assert {"DK", "NL"}.issubset(network.buses.index)
    assert {"DK_wind_2030", "NL_wind_2030"}.issubset(network.generators.index)
    assert "DK_NL_interconnector" in network.links.index
    link = network.links.loc["DK_NL_interconnector"]
    assert link["bus0"] == "DK"
    assert link["bus1"] == "NL"
    assert link["p_nom"] == 1000
    assert link["p_min_pu"] == -1
    assert link["efficiency"] == 1
