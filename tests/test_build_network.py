from pathlib import Path

from pathway_pilot.build_network import build_network
from pathway_pilot.model_config import load_config
from pathway_pilot.model_inputs import make_synthetic_inputs


def test_network_contains_core_components():
    cfg = load_config(Path("config/model_config.yaml"))
    data = make_synthetic_inputs(periods=cfg.investment_periods, hours_per_period=24)

    network = build_network(cfg, data)

    assert "electricity" in network.buses.index
    expected = {
        "wind_2030",
        "solar_2030",
        "gas_turbine_2030",
        "load_shedding",
    }
    assert expected.issubset(network.generators.index)
    assert list(network.investment_periods) == [2030, 2040, 2050]
