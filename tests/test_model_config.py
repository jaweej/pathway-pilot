from pathlib import Path

from pathway_pilot.model_config import load_config, with_active_model


def test_config_loads_period_weights():
    cfg = load_config(Path("config/model_config.yaml"))

    assert cfg.investment_periods == [2030, 2040, 2050]
    assert cfg.active_model == "NL"
    assert cfg.climate_years == [1995, 2008, 2009]
    assert sorted(cfg.model_cases) == ["DK", "DK_NL", "NL"]
    assert cfg.model_cases["DK"].demand_zones == ("DKE1", "DKW1")
    assert cfg.model_cases["DK_NL"].regions["DK"].demand_zones == ("DKE1", "DKW1")
    assert cfg.model_cases["DK_NL"].regions["NL"].capacity_factor_zone == "NL00"
    assert cfg.model_cases["DK_NL"].interconnectors[0].capacity_mw == 1000
    assert cfg.demand_zones == ("NL00",)
    assert cfg.capacity_factor_zone == "NL00"
    assert cfg.period_weights[2030] == 10
    assert cfg.eurdkk == 7.47
    assert cfg.gas_price_eur_per_mwh_fuel[2050] == 85 * 3.6 / 7.47
    assert cfg.capacity_limits_mw["wind"] > 0
    assert cfg.capacity_limits_mw["gas_turbine_cc"] > 0


def test_config_can_switch_to_combined_dk_nl_case():
    cfg = with_active_model(load_config(Path("config/model_config.yaml")), "DK_NL")

    assert tuple(cfg.model_regions) == ("DK", "NL")
    assert cfg.interconnectors[0].bus0 == "DK"
    assert cfg.interconnectors[0].bus1 == "NL"
